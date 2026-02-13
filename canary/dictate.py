#!/usr/bin/env python3
"""
Dictation client — lightweight, starts instantly.
Connects to dictate-server.py which holds the model in GPU memory.

Just run:  python3 dictate.py
"""

import sounddevice as sd
import numpy as np
import socket
import struct
import threading
import time
import signal
import queue
import subprocess
import sys
import os

import torch

SOCKET_PATH = "/tmp/dictate-canary.sock"
SAMPLE_RATE = 16000
CHANNELS = 1

# Voice activity detection (Silero)
VAD_THRESHOLD = 0.5       # probability to enter speech
VAD_NEG_THRESHOLD = 0.35  # probability to exit speech (hysteresis)
SILENCE_FLUSH = 0.7       # seconds of silence before flushing
BLOCKSIZE = 512           # Silero's required window size at 16kHz
MIN_SPEECH_DURATION = 0.3
MID_FLUSH_INTERVAL = 5.0

# Queue backpressure
MAX_QUEUE_SIZE = 5

# Server communication
SOCKET_TIMEOUT = 30.0


def _is_wayland():
    return os.environ.get("XDG_SESSION_TYPE") == "wayland"


def _check_paste_deps():
    """Check that clipboard paste dependencies are available."""
    if _is_wayland():
        deps = [("wl-copy", "wl-clipboard"), ("wtype", "wtype")]
    else:
        deps = [("xclip", "xclip"), ("xdotool", "xdotool")]

    missing = []
    for cmd, pkg in deps:
        if subprocess.run(["which", cmd], capture_output=True).returncode != 0:
            missing.append(f"  {cmd} (install: sudo apt install {pkg})")

    if missing:
        print("ERROR: Missing dependencies for clipboard paste:")
        for m in missing:
            print(m)
        sys.exit(1)


def clipboard_paste(text):
    """Copy text to clipboard, then simulate Ctrl+V to paste it instantly."""
    if not text:
        return

    if _is_wayland():
        subprocess.run(["wl-copy", "--", text], check=False)
        subprocess.run(["wtype", "-M", "ctrl", "v", "-m", "ctrl"], check=False)
    else:
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(),
                        check=False, capture_output=True)
        subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], check=False)


def press_enter():
    """Simulate pressing Enter."""
    if _is_wayland():
        subprocess.run(["wtype", "-k", "Return"], check=False)
    else:
        subprocess.run(["xdotool", "key", "--clearmodifiers", "Return"], check=False)


# Voice commands: trailing phrases that trigger a key press
VOICE_COMMANDS = [
    ("do it now", press_enter),
    ("press enter", press_enter),
]


def match_voice_command(text: str):
    """Check if text ends with a voice command. Returns (clean_text, command_fn) or (text, None)."""
    lower = text.lower().rstrip(" .!?,")
    for phrase, fn in VOICE_COMMANDS:
        if lower.endswith(phrase):
            # Strip the command phrase from the end
            clean = lower[:len(lower) - len(phrase)].rstrip()
            # Recover original casing for the text portion
            clean = text[:len(clean)].rstrip()
            return clean, fn
    return text, None


def check_server_health():
    """Ping the server with a zero-length request; expects 'OK' back."""
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5.0)
        sock.connect(SOCKET_PATH)
        # Zero-length audio = health check
        sock.sendall(struct.pack("<I", 0))
        header = sock.recv(4)
        if len(header) < 4:
            return False
        length = struct.unpack("<I", header)[0]
        data = b""
        while len(data) < length:
            chunk = sock.recv(min(65536, length - len(data)))
            if not chunk:
                break
            data += chunk
        sock.close()
        return data.decode("utf-8") == "OK"
    except Exception:
        return False


def _is_garbage(text: str) -> bool:
    """Filter out model artifacts like '!!!!', '...', repeated punctuation."""
    stripped = text.strip()
    if not stripped:
        return True
    # All same character repeated (e.g. "!!!!!!!", "......", "??????")
    if len(set(stripped)) <= 2 and not any(c.isalnum() for c in stripped):
        return True
    # Very short with no alphanumeric content
    if len(stripped) < 3 and not any(c.isalnum() for c in stripped):
        return True
    return False


def transcribe_remote(audio: np.ndarray) -> str:
    """Send audio to the server, get text back."""
    audio_bytes = audio.astype(np.float32).tobytes()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(SOCKET_TIMEOUT)
    sock.connect(SOCKET_PATH)
    try:
        sock.sendall(struct.pack("<I", len(audio_bytes)) + audio_bytes)
        header = sock.recv(4)
        if len(header) < 4:
            return ""
        length = struct.unpack("<I", header)[0]
        data = b""
        while len(data) < length:
            chunk = sock.recv(min(65536, length - len(data)))
            if not chunk:
                break
            data += chunk
        text = data.decode("utf-8")
        if text.startswith("ERROR:"):
            print(f"\r\033[K  Server {text}")
            return ""
        return text
    finally:
        sock.close()


class Dictation:
    def __init__(self):
        self.running = True

        # Load Silero VAD
        print("Loading Silero VAD...")
        self.vad_model, _ = torch.hub.load(
            'snakers4/silero-vad', 'silero_vad',
            trust_repo=True,
        )
        self.vad_model.eval()

        # VAD state
        self.speaking = False
        self.audio_buffer = []     # list of audio chunks
        self.vad_probs = []        # parallel list of VAD probabilities
        self.silence_start = None
        self.speech_start = None
        self.last_flush = None

        self.transcribe_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)

    def _get_vad_prob(self, audio_chunk):
        """Get speech probability from Silero VAD for a 512-sample chunk."""
        tensor = torch.from_numpy(audio_chunk).float()
        with torch.no_grad():
            return self.vad_model(tensor, SAMPLE_RATE).item()

    def _enqueue_audio(self, audio):
        """Enqueue audio for transcription, dropping oldest if full."""
        if self.transcribe_queue.full():
            try:
                self.transcribe_queue.get_nowait()  # drop oldest
            except queue.Empty:
                pass
        self.transcribe_queue.put(audio)

    def audio_callback(self, indata, frames, time_info, status):
        audio = indata[:, 0].copy()
        prob = self._get_vad_prob(audio)
        now = time.monotonic()

        if not self.speaking:
            if prob >= VAD_THRESHOLD:
                # Enter speech
                self.speaking = True
                self.speech_start = now
                self.last_flush = now
                self.audio_buffer = [audio]
                self.vad_probs = [prob]
                self.silence_start = None
                print("\r\033[K  [listening...]", end="", flush=True)
        else:
            # Currently speaking
            self.audio_buffer.append(audio)
            self.vad_probs.append(prob)

            if prob < VAD_NEG_THRESHOLD:
                # Below negative threshold — track silence
                if self.silence_start is None:
                    self.silence_start = now
                elif now - self.silence_start >= SILENCE_FLUSH:
                    self._end_flush(now)
                    return
            else:
                self.silence_start = None

            # Mid-flush for long utterances
            if now - self.last_flush >= MID_FLUSH_INTERVAL:
                self._mid_flush(now)

    def _find_split_point(self):
        """Find a good split point by scanning backward for a low-probability chunk."""
        if len(self.vad_probs) < 2:
            return len(self.vad_probs)

        # Scan backward from end, looking for a chunk below negative threshold
        for i in range(len(self.vad_probs) - 1, -1, -1):
            if self.vad_probs[i] < VAD_NEG_THRESHOLD:
                return i + 1  # split after this low-prob chunk
        # No good split point found — flush everything
        return len(self.audio_buffer)

    def _mid_flush(self, now):
        if not self.audio_buffer:
            return

        split = self._find_split_point()
        if split == 0:
            return

        # Split cleanly: flushed portion vs retained portion
        flush_chunks = self.audio_buffer[:split]
        retain_chunks = self.audio_buffer[split:]
        retain_probs = self.vad_probs[split:]

        self.audio_buffer = retain_chunks
        self.vad_probs = retain_probs
        self.last_flush = now

        audio = np.concatenate(flush_chunks)
        if len(audio) / SAMPLE_RATE >= MIN_SPEECH_DURATION:
            self._enqueue_audio(audio)

    def _end_flush(self, now):
        speech_duration = now - self.speech_start if self.speech_start else 0
        self.speaking = False
        self.silence_start = None
        self.speech_start = None
        self.last_flush = None

        # Reset VAD state for next utterance
        self.vad_model.reset_states()

        if speech_duration < MIN_SPEECH_DURATION or not self.audio_buffer:
            self.audio_buffer = []
            self.vad_probs = []
            return

        audio = np.concatenate(self.audio_buffer)
        self.audio_buffer = []
        self.vad_probs = []
        self._enqueue_audio(audio)

    def _transcription_worker(self):
        while self.running:
            try:
                audio = self.transcribe_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            duration = len(audio) / SAMPLE_RATE
            print(f"\r\033[K  [transcribing {duration:.1f}s...]", end="", flush=True)

            try:
                text = transcribe_remote(audio)
                if text and not _is_garbage(text):
                    clean_text, cmd_fn = match_voice_command(text)
                    if cmd_fn and clean_text:
                        print(f"\r\033[K  >> {clean_text}  [+ command]")
                        clipboard_paste(clean_text + " ")
                        cmd_fn()
                    elif cmd_fn:
                        print(f"\r\033[K  >> [command only]")
                        cmd_fn()
                    else:
                        print(f"\r\033[K  >> {text}")
                        clipboard_paste(text + " ")
                else:
                    print("\r\033[K", end="", flush=True)
            except (ConnectionRefusedError, FileNotFoundError):
                print("\r\033[K  ERROR: Server not running! Start dictate-server.py first.")
                self.running = False
            except socket.timeout:
                print("\r\033[K  ERROR: Server timed out (>30s). Is it overloaded?")
            except Exception as e:
                print(f"\r\033[K  ERROR: {e}")

    def run(self):
        # Check dependencies
        _check_paste_deps()

        # Check server health
        if not check_server_health():
            print("ERROR: Cannot reach dictate-server.py.")
            print("Start it first:  python3 dictate-server.py")
            return

        print()
        print("=" * 50)
        print("  DICTATION ACTIVE — just speak!")
        print("  Ctrl+C to quit")
        print("=" * 50)
        print()

        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'running', False))

        worker = threading.Thread(target=self._transcription_worker, daemon=True)
        worker.start()

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype='float32',
            blocksize=BLOCKSIZE,
            callback=self.audio_callback,
        ):
            print("  Listening...\n")
            while self.running:
                time.sleep(0.1)

        print("\nBye! (server still running in background)")


if __name__ == "__main__":
    d = Dictation()
    d.run()

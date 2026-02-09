#!/usr/bin/env python3
"""
voicetype - Realtime voice to text using ElevenLabs Scribe v2
Usage: voicetype [--toggle|--stop|--quit|--status]
"""

import os
import sys
import signal
import subprocess
import asyncio
import json
import base64
import time as _time
from pathlib import Path

CONFIG_DIR = Path.home() / ".config" / "voicetype"
PID_FILE = CONFIG_DIR / "voicetype.pid"
CONFIG_FILE = CONFIG_DIR / "config.json"

# Global state
running = False
listening = False


def notify(title, message):
    subprocess.run(["notify-send", title, message], check=False, capture_output=True)


def load_config():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            return json.load(f)
    config = {"api_key": "", "language": "en"}
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)
    return config


def get_api_key(config):
    return config.get("api_key") or os.environ.get("ELEVENLABS_API_KEY", "")


def _is_wayland():
    return os.environ.get("XDG_SESSION_TYPE") == "wayland"


def clipboard_paste(text):
    """Copy text to clipboard, then simulate Ctrl+V to paste it instantly."""
    if not text:
        return

    if _is_wayland():
        # wl-copy sets clipboard; wtype sends Ctrl+V
        subprocess.run(["wl-copy", "--", text], check=False)
        subprocess.run(["wtype", "-M", "ctrl", "v", "-m", "ctrl"], check=False)
    else:
        # xclip sets clipboard; xdotool sends Ctrl+V
        subprocess.run(["xclip", "-selection", "clipboard"], input=text.encode(),
                        check=False, capture_output=True)
        subprocess.run(["xdotool", "key", "--clearmodifiers", "ctrl+v"], check=False)


def get_pid():
    if not PID_FILE.exists():
        return None
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)
        return pid
    except (ValueError, ProcessLookupError):
        PID_FILE.unlink(missing_ok=True)
        return None


def send_signal(sig):
    pid = get_pid()
    if pid:
        os.kill(pid, sig)
        return True
    return False


class Tray:
    def __init__(self):
        self.proc = None

    def update(self, listening):
        if self.proc:
            self.proc.terminate()
            self.proc.wait()
        icon = "audio-input-microphone" if listening else "audio-input-microphone-muted"
        text = "Listening..." if listening else "Idle"
        try:
            self.proc = subprocess.Popen(
                ["yad", "--notification", f"--image={icon}", f"--text={text}",
                 "--command=voicetype --toggle"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            pass

    def stop(self):
        if self.proc:
            self.proc.terminate()


async def transcribe(config, tray):
    global listening

    try:
        import websockets
        import sounddevice as sd
        import numpy as np
    except ImportError as e:
        notify("Error", f"Missing: {e}")
        return

    api_key = get_api_key(config)
    if not api_key:
        notify("Error", "No API key")
        return

    params = "model_id=scribe_v2_realtime&commit_strategy=vad&vad_silence_threshold_ms=200&audio_format=pcm_16000"
    if config.get("language"):
        params += f"&language_code={config['language']}"

    url = f"wss://api.elevenlabs.io/v1/speech-to-text/realtime?{params}"
    audio_queue = asyncio.Queue(maxsize=100)

    def audio_cb(indata, frames, time, status):
        if listening:
            try:
                audio_queue.put_nowait((indata[:, 0] * 32767).astype(np.int16).tobytes())
            except asyncio.QueueFull:
                pass

    try:
        async with websockets.connect(url, additional_headers={"xi-api-key": api_key}) as ws:
            msg = await ws.recv()
            if json.loads(msg).get("message_type") != "session_started":
                return

            stream = sd.InputStream(samplerate=16000, channels=1, dtype=np.float32,
                                    blocksize=800, callback=audio_cb)
            stream.start()
            tray.update(True)
            notify("Listening", "Speak now...")
            print("Listening...")

            utterance_start = 0.0

            async def send():
                while listening and running:
                    try:
                        data = await asyncio.wait_for(audio_queue.get(), 0.02)
                        await ws.send(json.dumps({
                            "message_type": "input_audio_chunk",
                            "audio_base_64": base64.b64encode(data).decode()
                        }))
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break

            async def recv():
                nonlocal utterance_start
                while listening and running:
                    try:
                        msg = await asyncio.wait_for(ws.recv(), 0.02)
                        recv_t = _time.monotonic()
                        data = json.loads(msg)
                        mtype = data.get("message_type", "")

                        if mtype == "partial_transcript":
                            new_partial = data.get("text", "")
                            if not new_partial:
                                continue
                            if not utterance_start:
                                utterance_start = recv_t
                            since = (recv_t - utterance_start) * 1000
                            print(f"  … {new_partial}  [{since:.0f}ms]", end="\r")

                        elif mtype in ("committed_transcript", "committed_transcript_with_timestamps"):
                            committed = data.get("text", "")
                            if not committed:
                                continue

                            # Paste the whole committed text at once via clipboard
                            final = committed + " "
                            clipboard_paste(final)
                            total = (recv_t - utterance_start) * 1000 if utterance_start else 0
                            print(f"\n→ {committed}  [total:{total:.0f}ms]")

                            utterance_start = 0.0

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        break

            await asyncio.gather(send(), recv())
            stream.stop()
            stream.close()
    except Exception as e:
        notify("Error", str(e))
        print(f"Error: {e}")


def service(config, start_listening=True):
    global running, listening

    tray = Tray()
    running = True
    listening = start_listening

    def on_toggle(sig, frame):
        global listening
        listening = not listening
        tray.update(listening)
        if listening:
            print("Listening...")
        else:
            print("Idle.")

    def on_quit(sig, frame):
        global running, listening
        running = False
        listening = False

    signal.signal(signal.SIGUSR1, on_toggle)
    signal.signal(signal.SIGTERM, on_quit)
    signal.signal(signal.SIGINT, on_quit)

    PID_FILE.write_text(str(os.getpid()))
    tray.update(listening)
    print(f"VoiceType running. {'Listening' if listening else 'Idle'}.")

    try:
        while running:
            if listening:
                asyncio.run(transcribe(config, tray))
                if running and not listening:
                    tray.update(False)
                    notify("Paused", "Idle")
            else:
                signal.pause()
    finally:
        tray.stop()
        PID_FILE.unlink(missing_ok=True)
        print("Quit.")


def main():
    config = load_config()
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""

    if cmd in ("", "--toggle", "-t", "toggle"):
        if get_pid():
            send_signal(signal.SIGUSR1)
        else:
            service(config, start_listening=True)
    elif cmd in ("--start", "-s", "start"):
        if not get_pid():
            service(config, start_listening=True)
        else:
            print("Already running")
    elif cmd in ("--stop", "-x", "stop"):
        send_signal(signal.SIGUSR1) if get_pid() else print("Not running")
    elif cmd in ("--quit", "-q", "quit"):
        send_signal(signal.SIGTERM) if get_pid() else print("Not running")
    elif cmd in ("--status", "status"):
        print(f"Running: {bool(get_pid())}")
        print(f"API Key: {'set' if get_api_key(config) else 'NOT SET'}")
    elif cmd in ("--help", "-h"):
        print(__doc__)
        print("  --toggle   Toggle listening (default)")
        print("  --start    Start service")
        print("  --stop     Stop listening")
        print("  --quit     Quit service")
        print("  --status   Show status")
    else:
        print(f"Unknown: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()

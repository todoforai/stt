#!/usr/bin/env python3
"""
Dictation server — loads Canary-Qwen 2.5B once and keeps it in GPU memory.
Fast loading: ~24s cold start (vs ~55s with stock NeMo).

Start once:  python3 dictate-server.py
The client (dictate.py) connects to it.
"""

import socket
import os
import sys
import struct
import tempfile
import signal
import time
import numpy as np
import soundfile as sf

SOCKET_PATH = "/tmp/dictate-canary.sock"
SAMPLE_RATE = 16000
SAFETENSORS_PATH = None  # auto-detected
MAX_AUDIO_DURATION = 60  # seconds — reject audio longer than this


def find_safetensors():
    base = os.path.expanduser("~/.cache/huggingface/hub/models--nvidia--canary-qwen-2.5b")
    for root, dirs, files in os.walk(base):
        if "model.safetensors" in files:
            return os.path.join(root, "model.safetensors")
    raise FileNotFoundError("Run first: huggingface-cli download nvidia/canary-qwen-2.5b")


def load_model_fast():
    """Fast load: meta-device Qwen + CPU ConformerEncoder + safetensors direct load."""
    import torch
    import json
    from safetensors.torch import load_file

    t0 = time.monotonic()

    from nemo.collections.speechlm2.models.salm import SALM
    import nemo.collections.speechlm2.models.salm as salm_mod
    t1 = time.monotonic()
    print(f"  Imports: {t1-t0:.1f}s")

    # Patch: build Qwen on meta device (0.2s instead of 28s)
    def fast_load_hf(model_path_or_name, pretrained_weights=True, dtype=torch.float32):
        from transformers import AutoModelForCausalLM, AutoConfig
        config = AutoConfig.from_pretrained(model_path_or_name)
        with torch.device('meta'):
            return AutoModelForCausalLM.from_config(config, torch_dtype=dtype)
    salm_mod.load_pretrained_hf = fast_load_hf

    config_path = os.path.expanduser("~/.cache/canary-qwen-fast/config.json")
    with open(config_path) as f:
        cfg = json.load(f)

    t2 = time.monotonic()
    model = SALM(cfg)
    t3 = time.monotonic()
    print(f"  Model shell: {t3-t2:.1f}s")

    # Materialize meta tensors to empty CPU, then load real weights
    model.to_empty(device='cpu')
    safetensors_path = find_safetensors()
    state = load_file(safetensors_path, device='cpu')
    model.load_state_dict(state, strict=False)
    del state
    t4 = time.monotonic()
    print(f"  Load weights: {t4-t3:.1f}s")

    model = model.to('cuda')
    model.eval()
    t5 = time.monotonic()
    print(f"  To CUDA: {t5-t4:.1f}s")

    # Warmup CUDA kernels
    _warmup(model, torch)
    t6 = time.monotonic()
    print(f"  Warmup: {t6-t5:.1f}s")
    print(f"  Total: {t6-t0:.1f}s")
    return model, torch


def load_model_slow():
    """Standard NeMo loading (first run)."""
    import torch
    from nemo.collections.speechlm2.models import SALM

    t0 = time.monotonic()
    model = SALM.from_pretrained('nvidia/canary-qwen-2.5b')
    model = model.to(dtype=torch.bfloat16, device='cuda')
    model.eval()

    # Save config for fast loading next time
    from omegaconf import OmegaConf
    import json
    save_dir = os.path.expanduser("~/.cache/canary-qwen-fast")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(OmegaConf.to_container(model.cfg, resolve=True), f)

    _warmup(model, torch)
    t1 = time.monotonic()
    print(f"  Total (slow path): {t1-t0:.1f}s")
    print(f"  Config saved — next startup will be faster.")
    return model, torch


def _warmup(model, torch_mod):
    silent = np.zeros(SAMPLE_RATE, dtype=np.float32)
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(path, silent, SAMPLE_RATE)
    with torch_mod.no_grad():
        model.generate(
            prompts=[[{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}", "audio": [path]}]],
            max_new_tokens=1,
        )
    os.unlink(path)


def load_model():
    # TODO: fast loading produces bad weights (strict=False skips keys).
    # Use slow path until we fix key mapping.
    return load_model_slow()


def transcribe(model, torch_mod, audio_bytes):
    audio = np.frombuffer(audio_bytes, dtype=np.float32)
    duration = len(audio) / SAMPLE_RATE

    if duration < 0.3:
        return ""

    if duration > MAX_AUDIO_DURATION:
        return f"ERROR: Audio too long ({duration:.1f}s > {MAX_AUDIO_DURATION}s limit)"

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(wav_path, audio, SAMPLE_RATE)
        t0 = time.monotonic()
        with torch_mod.no_grad(), torch_mod.cuda.amp.autocast(dtype=torch_mod.bfloat16):
            answer_ids = model.generate(
                prompts=[[{"role": "user", "content": f"Transcribe the following: {model.audio_locator_tag}", "audio": [wav_path]}]],
                max_new_tokens=256,
            )
        text = model.tokenizer.ids_to_text(answer_ids[0].cpu()).strip()
        elapsed = time.monotonic() - t0
        preview = text[:80] + ('...' if len(text) > 80 else '')
        print(f"  Transcribed {duration:.1f}s audio in {elapsed:.2f}s: {preview}")
        return text
    finally:
        os.unlink(wav_path)


def serve(model, torch_mod):
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    server.listen(2)  # accommodate health check during transcription
    os.chmod(SOCKET_PATH, 0o600)

    def cleanup(*_):
        server.close()
        if os.path.exists(SOCKET_PATH):
            os.unlink(SOCKET_PATH)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f"\nListening on {SOCKET_PATH}")
    print("Server ready — start dictate.py in another terminal.\n")

    while True:
        conn, _ = server.accept()
        try:
            header = conn.recv(4)
            if len(header) < 4:
                continue
            length = struct.unpack("<I", header)[0]

            # Health check: zero-length request
            if length == 0:
                response = b"OK"
                conn.sendall(struct.pack("<I", len(response)) + response)
                continue

            audio_bytes = b""
            while len(audio_bytes) < length:
                chunk = conn.recv(min(65536, length - len(audio_bytes)))
                if not chunk:
                    break
                audio_bytes += chunk
            text = transcribe(model, torch_mod, audio_bytes)
            response = text.encode("utf-8")
            conn.sendall(struct.pack("<I", len(response)) + response)
        except Exception as e:
            print(f"  ERROR: {e}")
            # Try to send error back to client
            try:
                error_msg = f"ERROR: {e}".encode("utf-8")
                conn.sendall(struct.pack("<I", len(error_msg)) + error_msg)
            except Exception:
                pass  # client already disconnected
        finally:
            conn.close()


if __name__ == "__main__":
    print("Starting dictation server...")
    model, torch_mod = load_model()
    serve(model, torch_mod)

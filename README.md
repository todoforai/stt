# stt — Local Speech-to-Text Dictation for Linux

Real-time voice dictation that types into any application. Pure C, no Python, no cloud — runs entirely on your CPU.

Uses [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) with the **Parakeet-TDT 0.6B v3** (int8) model and **Silero VAD** for voice activity detection. Text is injected via `/dev/uinput` with layout-aware key mapping (xkbcommon), so it works on any compositor (X11, Wayland, etc.).

## Quick Start

```bash
./setup.sh   # downloads models (~640MB) + sherpa-onnx libs, builds binaries
./dictate     # start dictating
```

Or use the restart loop (auto-restarts on exit, F9 to restart):
```bash
./run-loop.sh
```

## How It Works

1. **Microphone** → PortAudio captures 16kHz mono audio
2. **VAD** → Silero detects speech segments in real-time
3. **ASR** → Sherpa-ONNX transcribes each segment (parallel worker thread)
4. **Type** → Text is injected as keyboard events via uinput (layout-aware)

An optional X11 **overlay** shows animated audio bars at the bottom of the screen while dictating.

## Voice Commands

Trailing phrases trigger actions instead of being typed:

| Phrase | Action |
|--------|--------|
| `do it now`, `execute` | Press Enter |
| `press enter`, `new line` | Press Enter |
| `interrupt it`, `stop it` | Press Ctrl+C |

## Controls

- **F9** — Pause/resume dictation (sends SIGUSR1)
- **Ctrl+C** — Quit

## Requirements

- Linux (uses `/dev/uinput` for text injection)
- `gcc`, `portaudio19-dev`, `libxkbcommon-dev` (setup.sh checks these)
- `/dev/uinput` write access (`sudo usermod -aG input $USER`, then re-login)
- For the overlay: X11 + Xlib/Xrender/Xext dev headers

## Files

| File | Description |
|------|-------------|
| `dictate.c` | Main dictation engine — audio capture, VAD, ASR, text injection |
| `typer.c/h` | Layout-aware keyboard input via uinput + xkbcommon |
| `overlay.c` | X11 visual indicator with animated audio bars |
| `setup.sh` | Downloads models and libs, checks deps, builds |
| `run-loop.sh` | Auto-restart wrapper with F9 hotkey support |
| `Makefile` | Build rules for `dictate` and `overlay` |

## Experimental

The `experimental/` directory contains alternative STT approaches:

- **canary/** — GPU-accelerated client-server using NVIDIA Canary-Qwen 2.5B (Python, requires CUDA)
- **elevenlabs/** — Cloud-based streaming via ElevenLabs Scribe v2 API (Python, requires API key)

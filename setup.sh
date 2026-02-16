#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"
LIBS_DIR="$SCRIPT_DIR/libs"

MODEL_NAME="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
MODEL_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${MODEL_NAME}.tar.bz2"
VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

SHERPA_VERSION="v1.12.23"
SHERPA_TARBALL="sherpa-onnx-${SHERPA_VERSION}-linux-x64-shared.tar.bz2"
SHERPA_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/${SHERPA_VERSION}/${SHERPA_TARBALL}"

echo "=== stt-sherpa setup ==="

# ── Download sherpa-onnx shared libraries ────────────────────────────────────

if [ -f "$LIBS_DIR/lib/libsherpa-onnx-c-api.so" ]; then
    echo "sherpa-onnx shared libs already exist."
else
    echo "Downloading sherpa-onnx ${SHERPA_VERSION} shared libs..."
    mkdir -p "$LIBS_DIR"
    cd "$LIBS_DIR"
    wget -q --show-progress "$SHERPA_URL" -O "$SHERPA_TARBALL"
    echo "Extracting..."
    tar xjf "$SHERPA_TARBALL" --strip-components=1
    rm "$SHERPA_TARBALL"
    cd "$SCRIPT_DIR"
    echo "sherpa-onnx libs ready."
fi

# ── Check build dependencies ─────────────────────────────────────────────────

echo ""
echo "Checking build dependencies..."

missing=""
command -v gcc >/dev/null 2>&1 || missing="$missing gcc"

# Check for portaudio header
if ! echo '#include <portaudio.h>' | gcc -E -x c - >/dev/null 2>&1; then
    missing="$missing portaudio19-dev"
fi

# Check for xkbcommon header
if ! echo '#include <xkbcommon/xkbcommon.h>' | gcc -E -x c - >/dev/null 2>&1; then
    missing="$missing libxkbcommon-dev"
fi

if [ -n "$missing" ]; then
    echo "  MISSING: $missing"
    echo "  Install with: sudo apt install $missing"
    exit 1
fi
echo "  OK (gcc + portaudio19-dev + libxkbcommon-dev)"

# ── Create models dir ────────────────────────────────────────────────────────

mkdir -p "$MODELS_DIR"

# ── Download Parakeet TDT model ──────────────────────────────────────────────

if [ -d "$MODELS_DIR/$MODEL_NAME" ]; then
    echo "Model already exists: $MODEL_NAME"
else
    echo "Downloading $MODEL_NAME (~640MB)..."
    cd "$MODELS_DIR"
    wget -q --show-progress "$MODEL_URL" -O "${MODEL_NAME}.tar.bz2"
    echo "Extracting..."
    tar xjf "${MODEL_NAME}.tar.bz2"
    rm "${MODEL_NAME}.tar.bz2"
    cd "$SCRIPT_DIR"
    echo "Model ready: $MODEL_NAME"
fi

# ── Download Silero VAD ──────────────────────────────────────────────────────

if [ -f "$MODELS_DIR/silero_vad.onnx" ]; then
    echo "Silero VAD already exists."
else
    echo "Downloading Silero VAD ONNX (~2MB)..."
    wget -q --show-progress "$VAD_URL" -O "$MODELS_DIR/silero_vad.onnx"
    echo "Silero VAD ready."
fi

# ── Check /dev/uinput access ─────────────────────────────────────────────────

echo ""
echo "Checking /dev/uinput access..."
if [ -w /dev/uinput ]; then
    echo "  OK (/dev/uinput writable)"
else
    echo "  WARNING: /dev/uinput not writable"
    echo "  Fix with: sudo usermod -aG input $USER  (then re-login)"
fi

# ── Build C binary ───────────────────────────────────────────────────────────

echo ""
echo "Building dictate binary..."
cd "$SCRIPT_DIR"
make clean
make
echo "Build complete."

echo ""
echo "=== Setup complete! ==="
echo "Run:  $SCRIPT_DIR/dictate"

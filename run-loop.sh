#!/bin/bash
# Run sherpa dictate in a restart loop.
# Press F9 (global hotkey) to restart — it kills dictate, this loop restarts it.
cd "$(dirname "$0")"
while true; do
    ./dictate
    echo ""
    echo "Dictate exited. Restarting in 1s..."
    sleep 1
done

# Scribe v2 Realtime — notes & next steps

Browser test: `scribe-realtime.html` (STT token from the backend `/api/v1/stt/token`;
key stays server-side, browser gets a 15-min `sutkn_…` single-use token).

## Auth (browser)
- WS cannot send custom headers → `xi-api-key` won't work client-side.
- Use single-use token: `POST /v1/single-use-token/realtime_scribe` (server-side),
  then `wss://…/v1/speech-to-text/realtime?token=…&model_id=scribe_v2_realtime`.

## Billing — measured, NOT just docs
Streamed audio is billed by **wall-clock duration, including silence**.
VAD only gates the *transcript* (committed_transcript), not the *meter*.

| streamed | STT credits |
|----------|-------------|
| 20s pure silence | +12 |
| 40s pure silence | +24 |

→ ~0.6 credit/s ≈ 2160 credit/h, regardless of speech vs silence.
→ A always-open WS bills the whole call (silence + while TTS speaks).
→ But close↔reconnect gaps are FREE (measured): 2×1s sent with a 20s no-WS gap
  between → +2 credits only. So only actually-streamed seconds are billed.

## Next step: local VAD gate (cost + barge-in)
Run Silero VAD in the browser (same model as `dictate.c` / `canary/dictate.py`,
~2MB ONNX). Only stream to Scribe while local VAD says "speech".

- Library: `@ricky0123/vad-web` (Silero on onnxruntime-web).
- Reuse our thresholds: positive 0.5 / negative 0.35 / minSpeech ~0.3s / silence ~0.7s.
- Strategy B (cheapest): open WS on `onSpeechStart`, close shortly after `onSpeechEnd`
  → pay only for speech.
- Bonus: same VAD signal solves **barge-in** — `onSpeechStart` during TTS
  → stop TTS + abort LLM, accept new input. `echoCancellation:true` kills self-echo.

## Gotcha: Scribe idle-closes after ~15s of no data (measured)
~15s after the last chunk → WS closes (code 1000), recognition stops (maybe
intentional, to discourage VAD-gating that dodges silence billing). VAD-gated
clients must reconnect — we lazily reopen on next speech, and the idle gap is
free (see billing). So: don't keep the WS open; let it close, reopen on speech.

## Support-bot loop (later)
mic → VAD gate → Scribe → committed → Haiku (stream, per-sentence) → TTS → speaker
state machine: LISTENING → THINKING → SPEAKING, with barge-in back to LISTENING.

# Voice Support — portable VAD → STT → LLM → TTS

Goal: a **portable, interruptible voice agent** we can drop into any product
(support line, in-app assistant). The user talks; we transcribe, think, and
speak back — and the user can **barge in at any time**, at which point we stop
talking and know **exactly how much of our answer was already heard**.

```
   mic ──▶ VAD ──▶ STT ──▶ LLM(sysmsg, transcript) ──▶ TTS ──▶ speaker
           │                                            ▲
           └────────── barge-in: VAD speech-start ──────┘  (stop TTS + abort LLM)
```

## The pipeline

1. **VAD** (local Silero, in-browser) — the control plane. Decides when speech
   starts/ends. Gates the STT stream (no cloud cost during silence) **and**
   fires barge-in while the bot is speaking.
2. **STT** (ElevenLabs Scribe v2 realtime) — speech → text. `commit_strategy=manual`;
   the local VAD's `onSpeechEnd` sends `commit:true` to close a turn.
3. **LLM** (Anthropic Haiku) — `(systemPrompt, committedTranscript)` → reply,
   **streamed** (SSE) so TTS can start on the first sentence.
4. **TTS** (Piper — local, free, has HU; ElevenLabs optional) — text → audio.
   **Streamed sentence-by-sentence:** sentence 1 plays the moment its boundary
   arrives (low first-audio latency); each later sentence is synthesized **one
   ahead** while the current one plays, so the next clip is ready when the
   current ends (no growing gap on long answers) and playback is strictly
   one-at-a-time (never two voices at once).

## Interruption (the hard requirement)

The user must be able to cut in mid-answer. The local VAD gives us this for free:

- While **SPEAKING**, a VAD `onSpeechStart` = barge-in →
  - **stop TTS playback** immediately,
  - **abort the LLM stream**,
  - go back to **LISTENING** with the new input winning.
- **Track read progress (proportional estimate):** TTS is played per
  sentence/chunk. On barge-in we estimate the heard prefix from playback time:
  `spokenChars ≈ (playedMs / totalDurationMs) * text.length`, applied per chunk
  so the error stays within the in-flight sentence (whole earlier sentences are
  known-complete). `playedMs` must come from actual speaker output, not generated
  audio. We feed this `spokenSoFar` (heard prefix) into the next LLM turn — the
  user reacts to what they *heard*, not the full generated answer. (Piper returns
  a whole WAV with no char timings, so estimate; ElevenLabs could give exact
  char-level alignment if we ever need it.)
- **Echo:** `getUserMedia({ echoCancellation: true })` keeps our own TTS out of
  the mic so it doesn't trigger a false barge-in / get re-transcribed.

State machine:

```
LISTENING  ── committed transcript ──▶  THINKING
THINKING   ── first sentence ready  ──▶  SPEAKING (TTS starts)
SPEAKING   ── VAD speech-start (barge-in) ──▶ stop TTS + abort LLM ─▶ LISTENING
SPEAKING   ── TTS finished ──▶ LISTENING
```

## Why local VAD (validated here)

`scribe-realtime.html` runs two Scribe channels on the same mic to prove the
approach before building the full agent:

- **A — full stream:** sends everything (baseline accuracy + cost).
- **B — VAD-gated:** sends only while local VAD detects speech; commits manually
  on speech-end. Shows transcript equality vs A, and bytes sent (cost proxy).

Findings so far:

- **Billing is by streamed wall-clock time, including silence** (~0.6 credit/s,
  measured). VAD-gating cuts cost to the speech fraction of a call. See `NOTES.md`.
- **Accuracy stays equal** to the full stream once tuned — the gate must not clip
  word starts, so B keeps a **pre-roll buffer** (~1s) and flushes it when speech
  begins. Remaining mismatches were the Scribe model mis-hearing English/technical
  terms under `language_code=hu` (a biasing problem, not a VAD problem).
- **Gating implies reconnect handling.** A silent channel lets the WS idle-close;
  B **lazily reopens on the next speech** and buffers audio + the commit in an
  `outbox` until connected — so nothing (including the turn-closing commit) is lost.

## Portability

Framework-free vanilla module; endpoints (`llmUrl`, `sttTokenUrl`, `sttUrl`, …) are
constructor options. LLM and TTS are already pluggable (pass `llm` / `tts`); VAD+STT
(Silero+Scribe) are still wired into the orchestrator — extracting them behind an
interface is the remaining portability step:

```
VAD : onSpeechStart / onSpeechEnd / isSpeaking      (Silero)
STT : start() / sendAudio() / commit() / onCommit   (Scribe)
LLM : stream(system, text) → tokens / abort()       (Haiku, pluggable)
TTS : speak(text) / stop() / spokenSoFar() → string (Piper, pluggable)
```

## Files

- `index.html` — the voice-agent demo (full loop with barge-in).
- `voice-agent.js` — the portable agent module (4 stages + barge-in state machine).
- `scribe-realtime.html` — A/B VAD-gating harness (validated cost ↓ = equal accuracy).
- `NOTES.md` — Scribe auth, measured billing, tuning notes.

## Run the demo

Keys stay server-side: the demo gets the LLM + STT token from the **backend**
(`/api/v1/llm`, `/api/v1/stt/token`). Set `tfa_api_key` in localStorage for `X-API-Key`,
then serve these static files with any web server (e.g. `npx serve`) and open it — the
backend must be up on `:4000` (override via the `baseUrl` constructor option).

Endpoints are constructor options (`baseUrl`, `llmUrl`, `sttTokenUrl`, `sttUrl`, …) so the module drops
into any product without editing source. LLM: CLIProxyAPI **Haiku 4.5** (free via
subscription, ~1.2s to first token); the backend route poses as `claude-cli` to skip the
proxy's persona-cloaking so our system prompt reaches Anthropic unchanged.

## Status

- [x] STT realtime in browser (Scribe v2, token auth)
- [x] Local Silero VAD in browser, gating the stream
- [x] Manual commit driven by VAD speech-end
- [x] Pre-roll buffer + lazy reconnect (no clipped starts, survives long silences)
- [x] A/B harness proving cost ↓ with equal accuracy
- [x] LLM stage (CLIProxyAPI Haiku 4.5, via backend `/api/v1/llm`, `claude-cli` UA to skip cloaking)
- [x] Barge-in: stop TTS + abort LLM on VAD speech-start, record heard prefix
- [x] Voice-agent demo wiring the 4 stages (`index.html` + `voice-agent.js`)
- [x] TTS stage: Piper (local, HU) via WASM; whole-WAV playback, proportional `spokenSoFar`
- [x] Stream the LLM (SSE) and speak sentence-by-sentence (synth one ahead, strict one-at-a-time playback — minimal gaps)
- [x] Endpoints as constructor options (`llmUrl`/`sttTokenUrl`/`sttUrl`/…); STT token via backend `/api/v1/stt/token`
- [x] Built-in `create_task` tool (T/1 → command to high-level AI); host supplies only `onTask`
- [ ] Exact `spokenSoFar` (char-level alignment) — needs ElevenLabs TTS; Piper gives only proportional estimate
- [ ] Pluggable STT interface (VAD+Scribe still wired into the orchestrator; LLM/TTS already swappable)

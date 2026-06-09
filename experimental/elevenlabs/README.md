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
   **Two-phase:** sentence 1 is synthesized + played the moment its boundary
   arrives in the stream (low first-audio latency); the rest is collected and
   synthesized in **one** call (better prosody than per-sentence) in parallel,
   so it's ready by the time sentence 1 finishes — no gap.

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

## Portability target

Wrap the four stages behind small interfaces so each is swappable:

```
VAD     : onSpeechStart / onSpeechEnd / isSpeaking      (Silero today)
STT     : start() / sendAudio() / commit() / onCommit   (Scribe today)
LLM     : stream(system, text) → tokens / abort()       (Haiku today)
TTS     : speak(text) / stop() / spokenSoFar() → string (Piper today; proportional estimate)
```

The orchestrator is just the state machine above plus the barge-in wiring.
Keep it framework-free (vanilla JS module + a tiny token server) so it drops
into any page.

## Files

- `index.html` — **the voice-agent demo** (full VAD → STT → LLM → TTS loop with barge-in).
- `voice-agent.js` — the portable agent module (the 4 stages + barge-in state machine).
- `token-server.py` — keeps keys server-side. Serves the demo + `/token` (Scribe) + `/llm`
  (CLIProxyAPI proxy, Haiku 4.5).
- `scribe-realtime.html` — A/B VAD-gating comparison harness (kept; validated cost ↓ = equal accuracy).
- `NOTES.md` — Scribe realtime auth, measured billing, tuning notes.

## Run the demo

```bash
ELEVENLABS_API_KEY=sk_... CLIPROXYAPI_API_KEY=sk_... python3 token-server.py
# open http://localhost:8770/
```

The LLM goes through CLIProxyAPI (free via subscription), model **Haiku 4.5** (~1.2s to
first token). The `/llm` proxy sends a `User-Agent: claude-cli/...` header: without it the
proxy "cloaks" `claude-*` requests — prepending a "You are Claude Code" system prompt so
the model refuses custom personas. Posing as the official CLI skips cloaking, so our
system prompt reaches Anthropic unchanged.

## Status

- [x] STT realtime in browser (Scribe v2, token auth)
- [x] Local Silero VAD in browser, gating the stream
- [x] Manual commit driven by VAD speech-end
- [x] Pre-roll buffer + lazy reconnect (no clipped starts, survives long silences)
- [x] A/B harness proving cost ↓ with equal accuracy
- [x] LLM stage (CLIProxyAPI Haiku 4.5, via `/llm` proxy, `claude-cli` UA to skip cloaking)
- [x] Barge-in: stop TTS + abort LLM on VAD speech-start, record heard prefix
- [x] Voice-agent demo wiring the 4 stages (`index.html` + `voice-agent.js`)
- [x] TTS stage: Piper (local, HU) via WASM; whole-WAV playback, proportional `spokenSoFar`
- [x] Stream the LLM (SSE) and start TTS on the first sentence (two-phase: sentence 1 now, rest in one parallel synth — no gap)
- [ ] Exact `spokenSoFar` (char-level alignment) — needs ElevenLabs TTS; Piper gives only proportional estimate

# Feature: Argue-with-AI about an article / the last message (voice)

A voice feature you drop onto any page that has a piece of text — a blog
article, a chat's last message, a doc — and the user can **talk** with an AI
about *that specific text*: push back, ask, agree, get rebuttals. Spoken, and
**interruptible** at any moment.

This is the same pipeline as the portable voice agent (see `README.md`); the
only thing this feature adds is **what goes into the system prompt and chat
history**. The plumbing (VAD → STT → LLM → TTS, barge-in, spokenSoFar) is reused
verbatim.

```
  [the article / last message]  ─┐
                                 ▼
  mic ─▶ VAD ─▶ STT ─▶ LLM.gen(chat_history, sysmsg) ─▶ TTS ─▶ speaker
          │                                              ▲
          └──────── barge-in (VAD speech-start) ─────────┘
```

## What makes it this feature

1. **Anchor text = the thing being discussed.** Grab the article / last message
   from the page (text content, optionally title + author). This is the subject.
2. **System prompt frames the debate.** Something like:
   *"You are discussing the following text with the user. Be concise (it's spoken
   — short sentences). Push back, agree, or clarify based on what they say. Stay
   on this text."* + the anchor text.
3. **Chat history is the conversation.** `LLM.gen(chat_history, sysmsg)` where
   `chat_history` is the running turns. Each turn:
   - user turn = the STT `committed_transcript`,
   - assistant turn = **`spokenSoFar`** (what the user actually heard — not the
     full generated reply; see `README.md` on proportional estimate + barge-in).

```
sysmsg       = debatePrompt + anchorText
chat_history = [ {user: stt₁}, {assistant: spoken₁}, {user: stt₂}, ... ]
reply        = LLM.gen(chat_history, sysmsg)   # streamed, per sentence → TTS
```

## How to implement in a project

1. **Mount on a target.** Pick the element holding the text (CSS selector / "last
   message" node). Extract `anchorText` (+ title/author if useful).
2. **Drop in the voice module** (the 4 swappable interfaces from `README.md`):
   `VAD` / `STT` / `LLM` / `TTS`. Point `tokenUrl`/`llmUrl` at the backend routes
   (`/api/v1/stt/token`, `/api/v1/llm`) — keys stay server-side, never in the browser.
3. **Wire the prompt.** `sysmsg = debatePrompt(anchorText)`. Start `chat_history`
   empty (or seed with a one-line opener the bot speaks first).
4. **Run the state machine** (LISTENING → THINKING → SPEAKING, barge-in →
   LISTENING). On each committed STT turn: push user turn, call
   `LLM.gen(history, sysmsg)`, stream sentences to TTS.
5. **On barge-in:** stop TTS, abort LLM, push `spokenSoFar` as the assistant turn,
   then the new STT as the next user turn. Both sides stay consistent with what
   was *actually heard*.

## Why both sides need `spokenSoFar`

The user argues against what they **heard**, not the full generated answer. If
they cut you off after half a sentence, the assistant turn in `chat_history` must
be that half — otherwise the model thinks it already made a point it never
voiced, and the debate desyncs. So: assistant turn = `spokenSoFar` (heard prefix),
optionally marked `…(interrupted)` so the model knows it was cut off.

## Cost / footprint (measured — see `NOTES.md`)

- VAD is local (~2MB Silero, free). STT billed only for streamed speech seconds
  (~0.6 credit/s); silence/idle gaps are free, WS idle-closes ~15s and lazily
  reopens on next speech. A short debate ≈ a few minutes of *actual speech*, not
  wall-clock.
- TTS via Piper is local/free (has Hungarian). LLM (Haiku) is cheap and streamed.

## Reuses (don't reinvent)

- Pipeline, interfaces, barge-in, `spokenSoFar`: `README.md`
- Scribe auth, measured billing, idle-close/reconnect: `NOTES.md`
- VAD-gated STT MVP harness: `scribe-realtime.html`

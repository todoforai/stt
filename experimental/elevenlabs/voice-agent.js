// voice-agent.js — VAD → STT → LLM → TTS, interruptible. One small module.
//
//   const agent = new VoiceAgent({ sysmsg, llm, tts });
//   agent.start();
//
// The whole loop is `_onUserTurn` below:
//   text = vad+stt  →  history.push(user:text)  →  llm STREAMS the answer
//   →  tts.speak(stream)   (barge-in stops tts + aborts llm, and records what was heard)
//
// First-sentence latency: `tts.speak` takes an async iterator of text deltas. It
// synthesizes the FIRST sentence as soon as its boundary arrives and starts playing,
// while the REST is collected from the stream and synthesized in ONE call (better
// prosody than per-sentence, and ready by the time sentence 1 finishes playing).
//
// `llm` and `tts` are pluggable. Defaults: llm → /llm proxy (Haiku, SSE), tts → Piper (local WASM, HU).

const CDN = 'https://cdn.jsdelivr.net/npm/';
const PREROLL_CHUNKS = 4;   // ~1s kept before VAD fires, so word starts aren't clipped

// Built-in `create_task` tool: forward a command to a high-level AI. Generic/portable, so it
// lives here — the host only supplies the `run` via the `onTask` option (no need to restate
// the description/params per product). Merged with (and overridable by) any custom `tools`.
const CREATE_TASK_TOOL = {
  description: 'Issue a command to the high-level AI. Call this whenever the user speaks in the FIRST-PERSON PLURAL ("we"/"let\'s" — e.g. Hungarian "csináljuk meg", "nézzük meg", "kérjük le", "indítsuk el"): that is NOT chatting, it is a command to be carried out. Also call it whenever the user otherwise asks you to act on something (create a task / to-do / next step). Forward the command staying close to the user\'s own wording, preserving every important detail they gave — don\'t over-paraphrase or strip specifics. You can suppose the high-level AI already knows everything from our conversation/sysmsg. A plain question or remark that just seeks information or keeps the discussion going ("what\'s the status?", "can you explain that?", "got it, thanks") stays normal conversation — don\'t call this. After calling, confirm it out loud in one short sentence.',
  params: { content: { type: 'string', description: 'The task to do, in the user\'s own words with all the details they provided.' } },
};

export class VoiceAgent {
  // `sttLang` is the STT language_code only (TTS voice is fixed by the chosen `tts`).
  // `tools`: { name: { description, params, run(args) } } — exposed to the LLM and run
  // after the spoken answer. `params` is the JSON-schema `properties` for the arguments.
  // `onTask(args)`: shorthand to enable the built-in `create_task` tool (forward a command
  // to the high-level AI) without redefining it per host — its `run` is your callback.
  // `keyterms`: domain words to bias STT towards (names, jargon it would otherwise
  // mishear). Max 50, ≤20 chars each; adds a 20% premium to the transcription cost.
  constructor({ sysmsg = '', llm, llmApiKey = '', tts = defaultTTS, sttLang = 'hu', tools = {}, onTask, keyterms = [], onEvent = () => {} } = {}) {
    this.sysmsg = sysmsg; this.tts = tts; this.sttLang = sttLang; this.onEvent = onEvent;
    this.tools = { ...(onTask ? { create_task: { ...CREATE_TASK_TOOL, run: onTask } } : {}), ...tools };
    this.keyterms = keyterms.filter(k => k && k.length <= 20).slice(0, 50);
    const defs = Object.entries(this.tools).map(([name, t]) =>
      ({ name, description: t.description || '', input_schema: { type: 'object', properties: t.params || {} } }));
    this.llm = llm || makeLLM(defs, llmApiKey);
    this.history = [];
    this.state = 'idle';           // idle | listening | thinking | speaking
    this._abort = null;            // aborts the in-flight LLM
    this._ws = null; this._preroll = []; this._wasSpeaking = false; this._outbox = []; this._closed = false;
  }

  async start() {
    this._closed = false;
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true } });
    this._ctx = new AudioContext({ sampleRate: 16000 });
    const src = this._ctx.createMediaStreamSource(this._stream);
    this._node = this._ctx.createScriptProcessor(4096, 1, 1);
    this._node.onaudioprocess = e => this._feed(e.inputBuffer.getChannelData(0));
    src.connect(this._node); this._node.connect(this._ctx.destination);
    await this._startVad();
    // Don't talk to a backgrounded tab: stop TTS when hidden (the next turn speaks again
    // when it's foreground). STT keeps running, so the user isn't cut off mid-sentence.
    this._onHidden = () => { if (document.hidden) this.tts.stop?.(); };
    document.addEventListener('visibilitychange', this._onHidden);
    this._set('listening');
  }

  stop() {
    this._closed = true; this._abort?.abort(); this.tts.stop?.();
    document.removeEventListener('visibilitychange', this._onHidden);
    this._vad?.pause(); this._ws?.close();
    this._node?.disconnect(); this._ctx?.close(); this._stream?.getTracks().forEach(t => t.stop());
    this._set('idle');
  }

  _set(s) { this.state = s; this.onEvent({ type: 'state', state: s }); }

  // List the TTS voices for a picker (if the active TTS supports it): [{ id, name, language }].
  listVoices() { return this.tts.voices ? this.tts.voices() : Promise.resolve([]); }
  // Switch TTS voice (takes effect on the next utterance).
  setVoice(voiceId) { this.tts.setVoice?.(voiceId); }

  // ── the loop ──────────────────────────────────────────────────────────
  // State drives everything the UI needs: listening = VAD/STT, thinking = LLM,
  // speaking = TTS. No separate 'stage'/'user' events — derive the pipeline from state.
  async _onUserTurn(text) {
    if (!text.trim()) return;
    // Serialize turns: abort any in-flight LLM + stop any playing TTS before starting a new
    // one. Without this, a second `committed` transcript spawns a parallel turn → overlapping
    // (double/triple) TTS and chat bubbles.
    this._abort?.abort();
    this.tts.stop?.();
    this.history.push({ role: 'user', content: text });

    this._set('thinking');                          // LLM stage
    this._abort = new AbortController();
    const signal = this._abort.signal;

    // Tap the LLM stream once: yield speech text to TTS, set aside any tool call for
    // after we finish speaking. The first delta flips us to 'speaking', and we accumulate
    // the full answer to report heard-vs-unheard on barge-in.
    // NOTE: `this.llm(...)` returns a lazy async generator — its fetch fires only when the
    // consumer (tts.speak) first pulls from `tapped`, by which point any prior turn's signal
    // is already aborted above. This is what prevents parallel LLM streams; keep llm lazy.
    let answer = '', call = null;
    const tapped = (async function* (self, src) {
      for await (const item of src) {
        if (item.tool) { call = item; continue; }              // tool_use → run it after speaking
        if (self.state === 'thinking') self._set('speaking');  // first token → TTS stage
        answer += item.text;
        self.onEvent({ type: 'assistant', text: answer, final: false });   // streaming draft (mirrors stt)
        yield item.text;
      }
    })(this, this.llm(this.history, this.sysmsg, signal));

    let spoken;
    try {
      spoken = await this.tts.speak(tapped, signal);   // resolves with text actually heard
    } catch (e) {
      if (e.name !== 'AbortError') {                   // real failure (not barge-in) → surface it
        this.onEvent({ type: 'error', error: e.message });
        if (this.state !== 'idle') this._set('listening');
        return;
      }
      // barge-in: TTS/LLM aborted. Still run the tool if the model already chose one —
      // a decided tool call must never be dropped just because we stopped speaking.
      await this._runTool(call);
      return;
    }

    if (signal.aborted) return;                                        // a newer turn superseded us — don't touch state

    if (spoken) {
      this.history.push({ role: 'assistant', content: spoken });       // ← heard prefix, not full answer
      this.onEvent({ type: 'assistant', text: spoken, full: answer, final: true });
    }

    await this._runTool(call);                                         // model asked for a tool
    if (this.state !== 'idle') this._set('listening');                 // back to listening (also if tool-only, no speech)
  }

  // Run a chosen tool call (if any) and report it. Idempotent-guarded so barge-in and
  // normal completion can't double-run it.
  async _runTool(call) {
    if (!call || call.ran) return;
    call.ran = true;
    const result = await this.tools[call.tool]?.run?.(call.args);
    this.onEvent({ type: 'tool', name: call.tool, args: call.args, result });
  }

  _onBargeIn() {                       // user spoke while we were talking
    this._abort?.abort();              // stop LLM
    this.tts.stop?.();                 // stop TTS (tts.speak resolves with the heard prefix)
    this._set('listening');
  }

  // ── VAD ───────────────────────────────────────────────────────────────
  async _startVad() {
    const d = window.vad;
    this._vad = await d.MicVAD.new({
      model: 'v5',
      onnxWASMBasePath: `${CDN}onnxruntime-web@1.22.0/dist/`,
      baseAssetPath:    `${CDN}@ricky0123/vad-web@0.0.29/dist/`,
      positiveSpeechThreshold: 0.5, negativeSpeechThreshold: 0.25,
      minSpeechFrames: 4, redemptionFrames: 24, preSpeechPadFrames: 10,
      onSpeechStart: () => { this._speaking = true; this.onEvent({ type: 'vad', active: true }); if (this.state === 'speaking') this._onBargeIn(); },
      onSpeechEnd:   () => { this._speaking = false; this.onEvent({ type: 'vad', active: false }); this._commit(); },
    });
    this._vad.start();
  }

  // ── STT (Scribe, VAD-gated, manual commit, lazy reconnect) ──────────────
  async _openWs() {
    const { token } = await (await fetch('/token', { method: 'POST' })).json();
    const p = new URLSearchParams({ model_id: 'scribe_v2_realtime', commit_strategy: 'manual', audio_format: 'pcm_16000', token });
    if (this.sttLang) p.set('language_code', this.sttLang);
    for (const k of this.keyterms) p.append('keyterms', k);   // repeated param, not comma-joined — bias STT towards domain terms
    this._sttStart = performance.now();                            // for elapsed timing of this STT turn
    this._ws = new WebSocket(`wss://api.elevenlabs.io/v1/speech-to-text/realtime?${p}`);
    this._ws.onopen = () => { const q = this._outbox; this._outbox = []; for (const it of q) it === 'commit' ? this._sendCommit() : this._sendAudio(it); };
    this._ws.onerror = ev => { this.onEvent({ type: 'error', error: 'STT socket error' }); console.error('STT ws error', ev); };
    this._ws.onclose  = ev => { if (ev.code !== 1000) { this.onEvent({ type: 'error', error: `STT closed ${ev.code}: ${ev.reason || 'no reason'}` }); console.error('STT ws closed', ev.code, ev.reason, 'clean:', ev.wasClean); } };
    this._ws.onmessage = ev => {
      const m = JSON.parse(ev.data);
      const ms = Math.round(performance.now() - this._sttStart);
      if (m.message_type?.endsWith('_error')) { this.onEvent({ type: 'error', error: `${m.message_type}: ${m.error || m.reason || m.message || ''}` }); console.error('STT error msg', m); return; }
      if (m.message_type === 'partial_transcript') {
        this.onEvent({ type: 'stt', final: false, text: m.text || '', ms });     // live interim transcript
      } else if (m.message_type?.startsWith('committed') && m.text) {
        this.onEvent({ type: 'stt', final: true, text: m.text, ms });            // committed turn (≈ the user event) + latency
        this._onUserTurn(m.text);
      }
    };
  }

  _feed(f32) {
    const i16 = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) i16[i] = Math.max(-1, Math.min(1, f32[i])) * 32767;
    if (!this._speaking) { this._preroll.push(i16); if (this._preroll.length > PREROLL_CHUNKS) this._preroll.shift(); this._wasSpeaking = false; return; }
    if (!this._wasSpeaking) {
      if (!this._closed && (!this._ws || this._ws.readyState >= 2)) this._openWs();
      for (const c of this._preroll) this._sendAudio(c); this._preroll = []; this._wasSpeaking = true;
    }
    this._sendAudio(i16);
  }
  _sendAudio(i16) {
    if (!this._ws || this._ws.readyState !== 1) { this._outbox.push(i16); return; }
    let bin = ''; const b = new Uint8Array(i16.buffer); for (let i = 0; i < b.length; i++) bin += String.fromCharCode(b[i]);
    this._ws.send(JSON.stringify({ message_type: 'input_audio_chunk', audio_base_64: btoa(bin) }));
  }
  _sendCommit() { this._ws.send(JSON.stringify({ message_type: 'input_audio_chunk', audio_base_64: '', commit: true })); }
  _commit() { if (this._ws?.readyState === 1) this._sendCommit(); else this._outbox.push('commit'); }
}

// ── default LLM: Haiku straight from CLIProxyAPI (no Python /llm proxy), SSE-streamed ──
//    Posts the native Anthropic Messages request and parses the native SSE: text_delta →
//    { text }, tool_use content_block → { tool, args } (input_json_delta accumulated).
//    The `claude-cli` User-Agent skips the proxy's cloaking so our system prompt + native
//    tools reach the model unchanged (that's what gives real tool_use, not XML text).
//    Throws on a non-OK response so the agent surfaces the error instead of speaking junk.
const LLM_URL = 'http://localhost:4000/api/v1/llm';      // backend SSE route (CLIProxyAPI key stays server-side)
function makeLLM(tools = [], apiKey = '') {
  return async function* (history, system, signal) {
    const r = await fetch(LLM_URL, {
      method: 'POST', signal,
      headers: { 'Content-Type': 'application/json', 'X-API-Key': apiKey },
      body: JSON.stringify({ system, messages: history, tools }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.error || r.statusText);

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '', tool = null, args = null;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const m = JSON.parse(line.slice(5).trim());      // backend chunks: { text } | { tool, args } | { error }
        if (m.error) throw new Error(m.error);
        if (m.text != null) yield { text: m.text };
        else if (m.tool != null) { tool = m.tool; args = m.args; }   // tool call (args already parsed)
      }
    }
    if (tool) yield { tool, args: args ?? {} };
  };
}

// First sentence boundary in `s`, or -1. Treats . ! ? … followed by space/end as a break.
// Earliest natural break to end the FIRST spoken chunk (low first-audio latency).
// Sentence enders (. ! ? …) are best; a dash/colon is a fine clause break to start on
// too. Requires trailing space/end so we don't split inside "3.14" or "pl.".
function firstSentenceEnd(s) {
  const m = /[.!?…]+(?=\s|$)|\s[–—-](?=\s)|:(?=\s)/.exec(s);
  return m ? m.index + m[0].trimEnd().length : -1;
}

// First sentence ender (. ! ? …, or a trailing-space colon) in `s`, or -1 — used for every
// sentence after the first, where we prefer whole sentences (fewer cuts, better prosody) over
// the earliest break. A colon ends a clause often enough to be a good, natural TTS break.
function sentenceEnd(s) {
  const m = /[.!?…]+(?=\s|$)|:(?=\s)/.exec(s);
  return m ? m.index + m[0].trimEnd().length : -1;
}

// ── default TTS: Piper (local WASM, free, has Hungarian) via @mintplex-labs/piper-tts-web.
//    Takes an async iterator of text deltas. Streams sentence-by-sentence for low latency
//    AND small inter-sentence gaps on long answers:
//      • re-chunk the deltas into sentences (sentence 1 on the earliest natural break for
//        fast first audio; later ones on whole-sentence enders for better prosody).
//      • synth ONE sentence ahead while the current one plays, so the next WAV is ready (or
//        nearly) the moment the current finishes — no growing gap as the answer gets longer.
//      • play STRICTLY in order, one clip at a time (never two voices at once).
//    Whole WAVs (no char timings) → on barge-in the heard prefix is estimated proportionally
//    within the playing sentence. Voice: hu_HU-anna-medium.
const CDN_PIPER = `${CDN}@mintplex-labs/piper-tts-web@1.0.4/dist/piper-tts-web.js`;
export class PiperTTS {
  constructor(voiceId = 'hu_HU-anna-medium') { this.voiceId = voiceId; this._ctx = null; this._node = null; }
  async _tts() { return this._lib ??= await import(/* @vite-ignore */ CDN_PIPER); }

  // List installable Piper voices: [{ id, name, language }]. `id` is what you pass as
  // `voiceId` (lib calls it `key`). Lets the UI build a voice picker; switch with setVoice.
  async voices() {
    const list = await (await this._tts()).voices();              // [{ key, name, language: { code, … }, … }]
    return list.map(v => ({ id: v.key, name: v.name || v.key, language: v.language?.code || v.language || '' }));
  }
  setVoice(voiceId) { this.voiceId = voiceId; }

  // Synthesize text → decoded AudioBuffer (the slow part; run it ahead of playback).
  async _synth(text, signal) {
    if (signal?.aborted || !text) return null;
    const wav = await (await this._tts()).predict({ text, voiceId: this.voiceId });   // Blob
    if (signal?.aborted) return null;
    this._ctx ??= new AudioContext();
    return this._ctx.decodeAudioData(await wav.arrayBuffer());
  }

  // Play a pre-synthesized buffer; resolve with the chars of `text` actually heard.
  _playBuf(buf, text, signal) {
    if (signal?.aborted || !buf) return Promise.resolve('');
    const total = buf.duration;
    const startedAt = this._ctx.currentTime;
    const src = this._ctx.createBufferSource();
    src.buffer = buf; src.connect(this._ctx.destination);
    this._node = src;
    return new Promise(resolve => {
      const heard = () => text.slice(0, Math.round(text.length * Math.min(this._ctx.currentTime - startedAt, total) / total));
      const onAbort = () => { try { src.stop(); } catch {} resolve(heard()); };   // barge-in → heard prefix
      src.onended = () => { signal?.removeEventListener('abort', onAbort); resolve(text); };  // finished → all heard
      signal?.addEventListener('abort', onAbort);
      src.start();
    });
  }

  // Re-chunk an async iterable of text deltas into sentences. The FIRST sentence breaks on
  // the earliest natural boundary (fast first audio); the rest on whole-sentence enders
  // (better prosody). Flushes whatever's left when the stream ends.
  async *_sentences(src, signal) {
    const it = src[Symbol.asyncIterator]();
    const next = async () => { try { return await it.next(); } catch (e) { if (e.name === 'AbortError') return { done: true }; throw e; } };
    let acc = '', first = true;
    for (let r = await next(); !r.done; r = await next()) {
      if (signal?.aborted) return;
      acc += r.value;
      let end;
      while ((end = (first ? firstSentenceEnd : sentenceEnd)(acc)) >= 0) {
        const s = acc.slice(0, end).trim();
        acc = acc.slice(end).trimStart();
        if (s) { yield s; first = false; }
      }
    }
    const tail = acc.trim();
    if (tail) yield tail;
  }

  // `input` is an async iterable of text deltas (or a plain string).
  // Pipeline: synthesize ONE sentence ahead, play strictly in order one-at-a-time. Keeping a
  // single synth in flight means the next clip is ready (or close) when the current ends —
  // minimal gap — without ever overlapping playback.
  async speak(input, signal) {
    if (signal?.aborted) return '';
    const src = typeof input === 'string' ? (async function* () { yield input; })() : input;

    const spokenParts = [];
    let nextBufP = null, nextText = '';                 // the one sentence synthesized ahead
    for await (const sentence of this._sentences(src, signal)) {
      // Kick off this sentence's synth, then play whatever was queued from the previous round.
      const bufP = this._synth(sentence, signal);
      if (nextBufP) {
        const heard = await this._playBuf(await nextBufP, nextText, signal);
        spokenParts.push(heard);
        if (heard.length < nextText.length || signal?.aborted) { await bufP; return spokenParts.join(' '); }  // barge-in mid-sentence
      }
      nextBufP = bufP; nextText = sentence;
    }
    // Drain the last queued sentence.
    if (nextBufP && !signal?.aborted) {
      spokenParts.push(await this._playBuf(await nextBufP, nextText, signal));
    }
    return spokenParts.join(' ').trim();                // joined with spaces at splits (model may omit them)
  }
  stop() { try { this._node?.stop(); } catch {} }
}
const defaultTTS = new PiperTTS();

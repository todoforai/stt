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

export class VoiceAgent {
  // `sttLang` is the STT language_code only (TTS voice is fixed by the chosen `tts`).
  // `tools`: { name: { description, params, run(args) } } — exposed to the LLM and run
  // after the spoken answer. `params` is the JSON-schema `properties` for the arguments.
  constructor({ sysmsg = '', llm, tts = defaultTTS, sttLang = 'hu', tools = {}, onEvent = () => {} } = {}) {
    this.sysmsg = sysmsg; this.tts = tts; this.sttLang = sttLang; this.tools = tools; this.onEvent = onEvent;
    const defs = Object.entries(tools).map(([name, t]) =>
      ({ name, description: t.description || '', input_schema: { type: 'object', properties: t.params || {} } }));
    this.llm = llm || makeLLM(defs);
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
    this._set('listening');
  }

  stop() {
    this._closed = true; this._abort?.abort(); this.tts.stop?.();
    this._vad?.pause(); this._ws?.close();
    this._node?.disconnect(); this._ctx?.close(); this._stream?.getTracks().forEach(t => t.stop());
    this._set('idle');
  }

  _set(s) { this.state = s; this.onEvent({ type: 'state', state: s }); }

  // ── the loop ──────────────────────────────────────────────────────────
  // State drives everything the UI needs: listening = VAD/STT, thinking = LLM,
  // speaking = TTS. No separate 'stage'/'user' events — derive the pipeline from state.
  async _onUserTurn(text) {
    if (!text.trim()) return;
    this.history.push({ role: 'user', content: text });

    this._set('thinking');                          // LLM stage
    this._abort = new AbortController();
    const signal = this._abort.signal;

    // Tap the LLM stream once: yield speech text to TTS, set aside any tool call for
    // after we finish speaking. The first delta flips us to 'speaking', and we accumulate
    // the full answer to report heard-vs-unheard on barge-in.
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
    } catch (e) { if (e.name === 'AbortError') return; throw e; }

    if (spoken) {
      this.history.push({ role: 'assistant', content: spoken });       // ← heard prefix, not full answer
      this.onEvent({ type: 'assistant', text: spoken, full: answer, final: true });
    }

    if (call) {                                                        // model asked for a tool
      const result = await this.tools[call.tool]?.run?.(call.args);
      this.onEvent({ type: 'tool', name: call.tool, args: call.args, result });
    }
    if (this.state !== 'idle') this._set('listening');                 // back to listening (also if tool-only, no speech)
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
    this._sttStart = performance.now();                            // for elapsed timing of this STT turn
    this._ws = new WebSocket(`wss://api.elevenlabs.io/v1/speech-to-text/realtime?${p}`);
    this._ws.onopen = () => { const q = this._outbox; this._outbox = []; for (const it of q) it === 'commit' ? this._sendCommit() : this._sendAudio(it); };
    this._ws.onmessage = ev => {
      const m = JSON.parse(ev.data);
      const ms = Math.round(performance.now() - this._sttStart);
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

// ── default LLM: Haiku via /llm proxy (key stays server-side), SSE-streamed ──
//    Yields { text } for speech deltas and { tool, args } when the model calls a tool
//    (args is the JSON string, streamed). Throws on a non-OK response so the agent
//    surfaces "LLM service down" instead of speaking undefined.
function makeLLM(tools = []) {
  return async function* (history, system, signal) {
    const r = await fetch('/llm', {
      method: 'POST', signal, headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ system, messages: history, tools }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({})))?.error || r.statusText);

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = '', tool = null, args = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line.startsWith('data:')) continue;
        const m = JSON.parse(line.slice(5).trim());
        if (m.text != null) yield { text: m.text };
        else if (m.tool != null) tool = m.tool;          // tool_use started; args stream next
        else if (m.args != null) args += m.args;         // accumulate the JSON arguments
      }
    }
    if (tool) yield { tool, args: args ? JSON.parse(args) : {} };
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

// ── default TTS: Piper (local WASM, free, has Hungarian) via @mintplex-labs/piper-tts-web.
//    Takes an async iterator of text deltas. Two-phase for low first-audio latency:
//      1. as soon as sentence 1's boundary arrives → synth + PLAY it (short → fast).
//      2. drain the rest of the stream, synth it in ONE call (good prosody), play after.
//    Both synth calls run while audio plays, so there's no gap. Whole WAVs (no char
//    timings) → on barge-in the heard prefix is estimated proportionally per chunk;
//    sentence 1 is known-complete once chunk 2 starts. Voice: hu_HU-anna-medium.
const CDN_PIPER = `${CDN}@mintplex-labs/piper-tts-web@1.0.4/dist/piper-tts-web.js`;
class PiperTTS {
  constructor(voiceId = 'hu_HU-anna-medium') { this.voiceId = voiceId; this._ctx = null; this._node = null; }
  async _tts() { return this._lib ??= await import(/* @vite-ignore */ CDN_PIPER); }

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

  // `input` is an async iterable of text deltas (or a plain string).
  async speak(input, signal) {
    if (signal?.aborted) return '';
    const src = typeof input === 'string' ? (async function* () { yield input; })() : input;
    // One manual iterator, driven across both phases — a `for await … break` would
    // call return() and CLOSE the generator, losing everything after sentence 1.
    const it = src[Symbol.asyncIterator]();
    const next = async () => { try { return await it.next(); } catch (e) { if (e.name === 'AbortError') return { done: true }; throw e; } };

    // Phase 1: accumulate until sentence 1's boundary, then split there.
    let acc = '', first = '', rest = '';
    for (let r = await next(); !r.done; r = await next()) {
      acc += r.value;
      const end = firstSentenceEnd(acc);
      if (end >= 0) { first = acc.slice(0, end); rest = acc.slice(end); break; }
    }
    if (!first) { first = acc; rest = ''; }   // stream ended with no sentence break
    if (signal?.aborted && !first) return '';

    // Synthesize + start sentence 1 playing; meanwhile drain the rest of the stream.
    const firstHeardP = this._synth(first, signal).then(b => this._playBuf(b, first, signal));
    for (let r = await next(); !r.done; r = await next()) rest += r.value;
    rest = rest.trimStart();

    // Synthesize the rest NOW (in parallel, while sentence 1 is still playing).
    const restBufP = this._synth(rest, signal);

    // If user barged in during sentence 1, stop here with its heard prefix.
    const heardFirst = await firstHeardP;
    if (heardFirst.length < first.length || signal?.aborted) return heardFirst;

    // Sentence 1 done; the rest buffer is ready (or nearly) → play with no gap.
    const heardRest = await this._playBuf(await restBufP, rest, signal);
    return heardRest ? `${first} ${heardRest}` : first;   // space at the split (model may omit it)
  }
  stop() { try { this._node?.stop(); } catch {} }
}
const defaultTTS = new PiperTTS();

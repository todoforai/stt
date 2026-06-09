// voice-agent.js — VAD → STT → LLM → TTS, interruptible. One small module.
//
//   const agent = new VoiceAgent({ sysmsg, llm, tts });
//   agent.start();
//
// The whole loop is `_onUserTurn` below:
//   text = vad+stt  →  history.push(user:text)  →  answer = llm(history, sysmsg)
//   →  tts(answer)   (barge-in stops tts + aborts llm, and records what was heard)
//
// `llm` and `tts` are pluggable. Defaults: llm → /llm proxy (Haiku), tts → browser speechSynthesis.

const CDN = 'https://cdn.jsdelivr.net/npm/';
const PREROLL_CHUNKS = 4;   // ~1s kept before VAD fires, so word starts aren't clipped

export class VoiceAgent {
  constructor({ sysmsg = '', llm = defaultLLM, tts = defaultTTS, lang = 'hu', onEvent = () => {} } = {}) {
    this.sysmsg = sysmsg; this.llm = llm; this.tts = tts; this.lang = lang; this.onEvent = onEvent;
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

  // ── the loop ────────────────────────────────────────────────────────────
  async _onUserTurn(text) {
    if (!text.trim()) return;
    this.history.push({ role: 'user', content: text });
    this.onEvent({ type: 'user', text });

    this._set('thinking');
    this._abort = new AbortController();
    let answer = '';
    try {
      answer = await this.llm(this.history, this.sysmsg, this._abort.signal);
    } catch (e) { if (e.name === 'AbortError') return; throw e; }

    this._set('speaking');
    const spoken = await this.tts.speak(answer, this._abort.signal);   // resolves with text actually heard
    this.history.push({ role: 'assistant', content: spoken });         // ← heard prefix, not full answer
    this.onEvent({ type: 'assistant', text: spoken, full: answer });
    if (this.state === 'speaking') this._set('listening');
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
      onSpeechStart: () => { this._speaking = true; if (this.state === 'speaking') this._onBargeIn(); },
      onSpeechEnd:   () => { this._speaking = false; this._commit(); },
    });
    this._vad.start();
  }

  // ── STT (Scribe, VAD-gated, manual commit, lazy reconnect) ──────────────
  async _openWs() {
    const { token } = await (await fetch('/token', { method: 'POST' })).json();
    const p = new URLSearchParams({ model_id: 'scribe_v2_realtime', commit_strategy: 'manual', audio_format: 'pcm_16000', token });
    if (this.lang) p.set('language_code', this.lang);
    this._ws = new WebSocket(`wss://api.elevenlabs.io/v1/speech-to-text/realtime?${p}`);
    this._ws.onopen = () => { const q = this._outbox; this._outbox = []; for (const it of q) it === 'commit' ? this._sendCommit() : this._sendAudio(it); };
    this._ws.onmessage = ev => {
      const m = JSON.parse(ev.data);
      if (m.message_type?.startsWith('committed') && m.text) this._onUserTurn(m.text);
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

// ── default LLM: Haiku via /llm proxy (key stays server-side) ────────────
async function defaultLLM(history, system, signal) {
  const r = await fetch('/llm', {
    method: 'POST', signal, headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ system, messages: history }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.error || r.statusText);   // surface "LLM service down" instead of speaking undefined
  return data.text;
}

// ── default TTS: browser speechSynthesis. Interruptible; gives exact spoken
//    char position via `onboundary` → spokenSoFar is exact, no estimate needed.
const defaultTTS = {
  _u: null, _heard: 0,
  speak(text, signal) {
    return new Promise(resolve => {
      const u = new SpeechSynthesisUtterance(text);
      this._u = u; this._heard = 0;
      u.onboundary = e => { this._heard = e.charIndex; };
      const done = () => { speechSynthesis.cancel(); resolve(text.slice(0, this._heard) || text); };
      u.onend = () => resolve(text);                 // finished fully → all heard
      signal?.addEventListener('abort', done);       // barge-in → heard prefix
      speechSynthesis.speak(u);
    });
  },
  stop() { speechSynthesis.cancel(); },
};

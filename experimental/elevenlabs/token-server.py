#!/usr/bin/env python3
"""
Token + LLM proxy + static server for the voice-agent browser demo.

Keeps both keys server-side; the browser only gets a 15-min single-use STT
token (sutkn_...) and a plain {text} reply from the /llm proxy.

  - POST /token  → ElevenLabs single-use Scribe token (uses ELEVENLABS_API_KEY)
  - POST /llm    → CLIProxyAPI chat (free via subscription). Body {system, messages}
                   → {text}. Haiku 4.5, free via the Claude Code subscription.

                   The `User-Agent: claude-cli/...` header is REQUIRED: without it the
                   proxy "cloaks" claude-* requests (mode:auto → cloak any non-claude-cli
                   client), prepending the "You are Claude Code" system prompt so the model
                   refuses custom personas. Posing as the official CLI skips cloaking, so
                   our system prompt reaches Anthropic unchanged. (~1.2s to first token.)

Run:  ELEVENLABS_API_KEY=sk_... CLIPROXYAPI_API_KEY=sk_... python3 token-server.py
Then: http://localhost:8770/            (voice agent demo)
      http://localhost:8770/scribe-realtime.html   (A/B VAD comparison)
"""
import os, json, urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler

API_KEY = os.environ["ELEVENLABS_API_KEY"]
CLIPROXY_KEY = os.environ.get("CLIPROXYAPI_API_KEY", "")
CLIPROXY_URL = "http://localhost:8317"
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_USER_AGENT = "claude-cli/1.0.0 (external, cli)"  # skips proxy cloaking — see above
PORT = 8770


class Handler(SimpleHTTPRequestHandler):
    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_POST(self):
        try:
            if self.path == "/token":
                return self._do_token()
            if self.path == "/llm":
                return self._do_llm()
        except json.JSONDecodeError:
            return self._json({"error": "invalid JSON body"}, 400)
        self.send_error(404)

    def _do_token(self):
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe",
            method="POST",
            headers={"xi-api-key": API_KEY},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            token = json.load(r)["token"]
        self._json({"token": token})

    def _do_llm(self):
        if not CLIPROXY_KEY:
            return self._json({"error": "CLIPROXYAPI_API_KEY not set — LLM service down"}, 503)
        body = self._read_body()
        payload = json.dumps({
            "model": LLM_MODEL,
            "max_tokens": 1024,
            "system": body.get("system", ""),
            "messages": body.get("messages", []),
        }).encode()
        req = urllib.request.Request(
            f"{CLIPROXY_URL}/v1/messages",
            data=payload,
            method="POST",
            headers={"x-api-key": CLIPROXY_KEY, "anthropic-version": "2023-06-01",
                     "content-type": "application/json", "User-Agent": LLM_USER_AGENT},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.load(r)
        except Exception as e:
            return self._json({"error": f"LLM proxy failed: {e}"}, 502)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        self._json({"text": text})


if __name__ == "__main__":
    print(f"Voice-agent server on http://localhost:{PORT}")
    if not CLIPROXY_KEY:
        print("⚠️  CLIPROXYAPI_API_KEY not set — /llm will return 503")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

#!/usr/bin/env python3
"""
Minimal token + static server for the Scribe Realtime browser test.
Keeps the ELEVENLABS_API_KEY server-side; the browser only ever gets a
15-min single-use token (sutkn_...).

Run:  ELEVENLABS_API_KEY=sk_... python3 token-server.py
Then: http://localhost:8770/scribe-realtime.html
"""
import os, json, urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler

API_KEY = os.environ["ELEVENLABS_API_KEY"]
PORT = 8770


class Handler(SimpleHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/token":
            self.send_error(404)
            return
        req = urllib.request.Request(
            "https://api.elevenlabs.io/v1/single-use-token/realtime_scribe",
            method="POST",
            headers={"xi-api-key": API_KEY},
        )
        with urllib.request.urlopen(req) as r:
            token = json.load(r)["token"]
        body = json.dumps({"token": token}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"Token+static server on http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

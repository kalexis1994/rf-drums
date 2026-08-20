#!/usr/bin/env python3
"""RF-Drums development server.

Serves the dev frontend and the freshly built component on one port:

    python3 tools/dev-server.py            # http://localhost:8090
    python3 tools/dev-server.py 9000       # another port

Routes:
    /               -> web-dev/index.html
    /worklet.js     -> web-dev/worklet.js
    /rf_drums.wasm  -> target/wasm32-unknown-unknown/release/rf_drums.wasm
    POST /build     -> runs cargo build --release --target wasm32-unknown-unknown
                       and answers {"ok": bool, "output": str}, so the page's
                       rebuild button gives an edit-listen loop without leaving
                       the browser.

Everything is served with no-store: the whole point is hearing the build you
just made, never a cached one.
"""

import json
import os
import subprocess
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WASM = os.path.join(
    ROOT, "target", "wasm32-unknown-unknown", "release", "rf_drums.wasm"
)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090

# cargo lives in the proot at /root/.cargo/bin and is not on the default PATH.
CARGO_ENV = dict(os.environ)
CARGO_ENV["PATH"] = "/root/.cargo/bin:" + CARGO_ENV.get("PATH", "")


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        # The page cache-busts the worklet with ?v=...; routes match on the
        # bare path.
        self.path = self.path.split("?", 1)[0]
        if self.path in ("/", "/index.html"):
            return self._file(os.path.join(ROOT, "web-dev", "index.html"), "text/html")
        if self.path == "/worklet.js":
            return self._file(
                os.path.join(ROOT, "web-dev", "worklet.js"), "text/javascript"
            )
        if self.path == "/rf_drums.wasm":
            if not os.path.exists(WASM):
                return self._error(404, "component not built — POST /build or run cargo")
            return self._file(WASM, "application/wasm")
        return self._error(404, "not found")

    def do_POST(self):
        if self.path != "/build":
            return self._error(404, "not found")
        result = subprocess.run(
            ["cargo", "build", "--release", "--target", "wasm32-unknown-unknown"],
            cwd=ROOT,
            env=CARGO_ENV,
            capture_output=True,
            text=True,
            timeout=600,
        )
        body = json.dumps(
            {"ok": result.returncode == 0, "output": result.stdout + result.stderr}
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path, content_type):
        try:
            with open(path, "rb") as handle:
                body = handle.read()
        except OSError:
            return self._error(404, "missing " + path)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code, message):
        body = message.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - stdlib signature
        sys.stderr.write("%s %s\n" % (self.log_date_time_string(), format % args))


if __name__ == "__main__":
    print(f"RF-Drums dev: http://localhost:{PORT}")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

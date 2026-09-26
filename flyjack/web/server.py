"""Fly Lab: a local web interface to test and watch the Blackjack-playing fly.

    python -m flyjack.web                # http://127.0.0.1:8000
    python -m flyjack.web --port 8080 --no-browser
"""

import argparse
import json
import threading
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

STATIC = Path(__file__).parent / "static"
FIGURES = Path(__file__).resolve().parents[2] / "figures"
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}


def _default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def make_handler(lab):
    get_routes = {
        "/api/meta": lambda q: lab.meta(),
        "/api/policy": lambda q: lab.policy(q.get("model", "trained")),
        "/api/spiking": lambda q: {"status": lab.spiking_status},
        "/api/train/status": lambda q: lab.training_status(),
    }
    post_routes = {
        "/api/sense": lambda b: lab.sense(b["obs"], b.get("mode", "readout"), b.get("trial"),
                                          b.get("model", "trained")),
        "/api/probe": lambda b: lab.probe(b["obs"], b.get("mode", "readout"),
                                          min(int(b.get("trials", 16)), 64), b.get("model", "trained")),
        "/api/hand/new": lambda b: lab.new_hand(b.get("player"), b.get("dealer"), b.get("seed")),
        "/api/hand/act": lambda b: lab.act(b["id"], b["action"]),
        "/api/tournament": lambda b: lab.tournament(min(int(b.get("hands", 2000)), 200_000),
                                                    b.get("seed"), b.get("model", "trained")),
        "/api/spiking/load": lambda b: {"status": lab.load_spiking()},
        "/api/train/start": lambda b: lab.start_training(
            min(int(b.get("hands", 300_000)), 2_000_000), float(b.get("eta", 0.05)),
            float(b.get("epsilon", 0.3)), int(b.get("seed", 0))),
        "/api/train/stop": lambda b: lab.stop_training(),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):          # keep the terminal quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body, default=_default).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _api(self, fn, arg):
            try:
                self._send(200, fn(arg))
            except (KeyError, ValueError, RuntimeError) as e:
                self._send(400, {"error": str(e)})
            except Exception as e:
                traceback.print_exc()
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def do_GET(self):
            path, _, qs = self.path.partition("?")
            q = dict(p.split("=", 1) for p in qs.split("&") if "=" in p)
            if path in get_routes:
                return self._api(get_routes[path], q)
            root = FIGURES if path.startswith("/figures/") else STATIC
            rel = path.removeprefix("/figures/") if root is FIGURES else path.lstrip("/")
            f = (root / (rel or "index.html")).resolve()
            if f.is_file() and root.resolve() in f.parents:
                return self._send(200, f.read_bytes(), TYPES.get(f.suffix, "application/octet-stream"))
            self._send(404, {"error": "not found"})

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": "invalid JSON"})
            if self.path in post_routes:
                return self._api(post_routes[self.path], body)
            self._send(404, {"error": "not found"})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--spiking", action="store_true", help="load the GPU spiking brain at startup")
    a = ap.parse_args()

    from .lab import FlyLab
    print("loading the fly (recorded brain responses + trained mushroom body) ...")
    lab = FlyLab()
    if a.spiking:
        lab.load_spiking()
    server = ThreadingHTTPServer((a.host, a.port), make_handler(lab))
    url = f"http://{a.host}:{a.port}"
    print(f"Fly Lab running at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

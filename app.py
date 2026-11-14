"""Local UI.  Run:  python app.py   then the browser opens http://localhost:8000

Only needs opencv-python and numpy. The web server is Python's standard library.
"""
import json
import mimetypes
import os
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
from measure import measure            # noqa: E402
from run import detect_pass            # noqa: E402
from autocorners import auto_corners   # noqa: E402

CLIP = ROOT / "data" / "pass_0216.mp4"
PORT = int(os.environ.get("PORT", 8000))
_cache, _lock = {}, threading.Lock()
STATIC = {"/": ROOT / "web" / "index.html"}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path):
        if not path.is_file():
            return self._json({"error": "not found"}, 404)
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):          # the <video> tag needs range requests
            if m.group(1):
                start = int(m.group(1)); end = int(m.group(2)) if m.group(2) else end
            else:
                start = size - int(m.group(2))
            end = min(end, size - 1)
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", mimetypes.guess_type(str(path))[0] or "application/octet-stream")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 16, left))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                left -= len(chunk)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in STATIC:
            return self._file(STATIC[p])
        if p == "/api/annotations":
            return self._json(json.load(open(ROOT / "annotations" / "corners.json")))
        if p == "/api/detect":
            with _lock:
                if "detect" not in _cache:
                    _cache["detect"] = detect_pass(str(CLIP))[0]
            return self._json(_cache["detect"])
        if p == "/api/auto":
            try:
                with _lock:
                    if "auto" not in _cache:
                        ann = json.load(open(ROOT / "annotations" / "corners.json"))
                        _cache["auto"] = auto_corners(str(CLIP), sorted(int(k) for k in ann if k.isdigit()))
                return self._json(_cache["auto"])
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
        if p.startswith("/data/") or p.startswith("/out/"):
            target = (ROOT / p.lstrip("/")).resolve()
            if ROOT.resolve() in target.parents:
                return self._file(target)
        return self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path.split("?")[0] != "/api/fit":
            return self._json({"error": "not found"}, 404)
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            res = measure(req["frames"], tuple(req.get("size", (1920, 1080))), n_mc=int(req.get("mc", 0)))
            return self._json(res)
        except Exception as e:                       # show the reason in the UI instead of a blank page
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)


if __name__ == "__main__":
    if not CLIP.is_file():
        sys.exit(f"Missing {CLIP}. See README, section 'The clip'.")
    url = f"http://localhost:{PORT}"
    print(f"Forklift load volume UI running at {url}  (Ctrl+C to stop)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

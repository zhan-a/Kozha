#!/usr/bin/env python3
"""Dependency-free local server for eyeballing SiGML in the CWASA avatar.

Serves ``public/`` at ``/`` and ``data/`` at ``/data/`` (mirrors the routing the
FastAPI app uses) WITHOUT importing the heavy translate stack (spaCy/argos). Use
it to review hand-authored ASL signs:

    python3 scripts/serve_local.py            # -> http://localhost:8000/asl_review.html
    python3 scripts/serve_local.py --port 9000

Ctrl-C to stop.
"""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PUBLIC = REPO / "public"
DATA = REPO / "data"


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/data/"):
            return str(DATA / clean[len("/data/"):])
        if clean in ("", "/"):
            return str(PUBLIC / "asl_review.html")
        return str(PUBLIC / clean.lstrip("/"))

    def end_headers(self) -> None:
        # No caching — we re-emit the curated SiGML between review rounds.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, *args) -> None:  # quieter console
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler))
    url = f"http://localhost:{args.port}/asl_review.html"
    print(f"Serving public/ + /data on {url}\n(Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()

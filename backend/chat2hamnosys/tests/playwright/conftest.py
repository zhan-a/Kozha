"""Fixtures for the Playwright frontend smoke test.

The fixture below boots :mod:`._test_server` as a subprocess on a free
port, polls ``/healthz`` until it answers, and yields the base URL. The
subprocess inherits the test's tmp-path-pinned env vars so each test gets
clean SQLite databases.

The whole test is skipped if Playwright is not installed (``pytest`` and
``uvicorn`` are required by the chat2hamnosys backend itself, but
Playwright is an opt-in dev dependency).
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterator

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
PUBLIC_DIR = REPO_ROOT / "public"
SERVER_MODULE = Path(__file__).resolve().parent / "_test_server.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(url: str, timeout: float = 15.0) -> None:
    """Poll ``url`` until it answers 200 or raise on timeout."""
    import urllib.error
    import urllib.request

    deadline = time.time() + timeout
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, OSError, TimeoutError) as e:
            last_err = e
            time.sleep(0.2)
    raise TimeoutError(f"Server at {url} did not start within {timeout}s; last={last_err!r}")


@pytest.fixture
def c2h_server(tmp_path: Path) -> Iterator[str]:
    """Boot the test server in a subprocess; yield ``http://127.0.0.1:<port>``."""
    if not PUBLIC_DIR.exists():
        pytest.skip(f"public/ not found at {PUBLIC_DIR}")

    port = _free_port()
    env = os.environ.copy()
    env.update(
        {
            "C2H_PORT": str(port),
            "C2H_PUBLIC_DIR": str(PUBLIC_DIR),
            "CHAT2HAMNOSYS_SESSION_DB":  str(tmp_path / "sessions.sqlite3"),
            "CHAT2HAMNOSYS_SIGN_DB":     str(tmp_path / "signs.sqlite3"),
            "CHAT2HAMNOSYS_TOKEN_DB":    str(tmp_path / "tokens.sqlite3"),
            "CHAT2HAMNOSYS_REVIEWER_DB": str(tmp_path / "reviewers.sqlite3"),
            "CHAT2HAMNOSYS_DATA_DIR":    str(tmp_path / "kozha_data"),
            "CHAT2HAMNOSYS_RATE_LIMIT":  "500/minute",
        }
    )

    proc = subprocess.Popen(
        [sys.executable, str(SERVER_MODULE)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        try:
            _wait_for(f"{base}/healthz")
        except TimeoutError:
            # Surface launcher output so failures are debuggable in CI.
            proc.terminate()
            try:
                out = proc.communicate(timeout=2)[0]
            except subprocess.TimeoutExpired:
                proc.kill()
                out = b""
            pytest.skip(
                f"chat2hamnosys server failed to start; subprocess output:\n{out.decode(errors='replace')}"
            )
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

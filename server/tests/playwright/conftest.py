"""Boot the minimal translator test server for Playwright scenarios.

Follows the same shape as ``backend/chat2hamnosys/tests/playwright/conftest.py``
so the two suites stay parallel.
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


REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_DIR = REPO_ROOT / "public"
SERVER_MODULE = Path(__file__).resolve().parent / "_test_server.py"


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(url: str, timeout: float = 15.0) -> None:
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
    raise TimeoutError(f"server {url} did not start within {timeout}s; last={last_err!r}")


@pytest.fixture
def kozha_server() -> Iterator[str]:
    """Boot the stub translator server; yield its base URL."""
    if not PUBLIC_DIR.exists():
        pytest.skip(f"public/ not found at {PUBLIC_DIR}")

    port = _free_port()
    env = os.environ.copy()
    env["KOZHA_TEST_PORT"] = str(port)
    proc = subprocess.Popen(
        [sys.executable, str(SERVER_MODULE)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        try:
            _wait_for(f"{base}/app.html")
        except TimeoutError:
            proc.terminate()
            try:
                out = proc.communicate(timeout=2)[0]
            except subprocess.TimeoutExpired:
                proc.kill()
                out = b""
            pytest.skip(
                f"kozha translator test server failed to start; subprocess output:\n"
                f"{out.decode(errors='replace')}"
            )
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()

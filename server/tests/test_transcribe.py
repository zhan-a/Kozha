"""Regression tests for the /api/transcribe Gladia fallback.

This route exists so the in-browser Whisper pipeline (transformers.js
+ onnxruntime-web in public/app.html and public/index.html) has
somewhere to fall back to when WebAssembly init fails. The actual
Gladia network call is mocked here — the contract under test is:

  * Missing TRANSCRIBE_KEY → 503 (the client must surface the local
    Whisper error, not a server-side mystery).
  * Empty body → 400.
  * Oversize body → 413.
  * Happy path → 200 with {"text": ..., "provider": "gladia"}.
  * Gladia upstream failure → 502 with a stable, non-leaky message.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "server") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "server"))

from server import app  # noqa: E402
import server as _server_mod  # noqa: E402


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    # Default to "configured": individual tests override.
    monkeypatch.setenv("TRANSCRIBE_KEY", "test-gladia-key")
    return TestClient(app)


def test_transcribe_returns_503_when_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRANSCRIBE_KEY", raising=False)
    c = TestClient(app)
    r = c.post("/api/transcribe", content=b"\x00" * 16, headers={"content-type": "audio/webm"})
    assert r.status_code == 503
    body = r.json()
    assert "configured" in (body.get("detail") or "").lower()


def test_transcribe_rejects_empty_body(client: TestClient) -> None:
    r = client.post("/api/transcribe", content=b"", headers={"content-type": "audio/webm"})
    assert r.status_code == 400


def test_transcribe_rejects_oversize_body(client: TestClient) -> None:
    # Just over the 25 MB cap.
    big = b"\x00" * (25 * 1024 * 1024 + 1)
    r = client.post("/api/transcribe", content=big, headers={"content-type": "audio/webm"})
    assert r.status_code == 413


def test_transcribe_happy_path(client: TestClient) -> None:
    # Stub the two Gladia legs so no network egress occurs in the test.
    with patch.object(_server_mod, "_gladia_upload_audio", return_value="https://api.gladia.io/file/v2/abc"), \
         patch.object(_server_mod, "_gladia_transcribe", return_value="hello world"):
        r = client.post(
            "/api/transcribe",
            content=b"\x52\x49\x46\x46" + b"\x00" * 16,
            headers={"content-type": "audio/wav", "x-input-language": "en"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hello world"
    assert body["provider"] == "gladia"


def test_transcribe_returns_502_when_gladia_fails(client: TestClient) -> None:
    with patch.object(
        _server_mod,
        "_gladia_upload_audio",
        side_effect=RuntimeError("gladia HTTP 401: bad key"),
    ):
        r = client.post(
            "/api/transcribe",
            content=b"\x00" * 32,
            headers={"content-type": "audio/webm"},
        )
    assert r.status_code == 502
    # The internal error message must not leak verbatim — only the
    # generic "Transcription service failed" copy.
    body = r.json()
    assert "401" not in (body.get("detail") or "")
    assert "bad key" not in (body.get("detail") or "")

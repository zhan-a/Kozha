"""Unit tests for ``GET /hamnosys/symbols``.

The endpoint exposes the full HamNoSys 4.0 symbol inventory plus
plain-English class labels so the contribute page's notation panel can
drive its hover/tap legend without shipping a second copy of
``hamnosys/symbols.py`` into JS. The table is immutable for the life of
the process, so the response carries a stable ETag and a
``Cache-Control: public, max-age=31536000, immutable`` header —
browsers revalidate once, then never again.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def _build_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_DB", str(tmp_path / "sessions.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_SIGN_DB", str(tmp_path / "signs.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_TOKEN_DB", str(tmp_path / "tokens.sqlite3"))
    monkeypatch.setenv("CHAT2HAMNOSYS_DATA_DIR", str(tmp_path / "kozha_data"))
    monkeypatch.setenv("CHAT2HAMNOSYS_RATE_LIMIT", "1000/minute")
    monkeypatch.setenv("CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT", "1000/minute")

    from api.dependencies import reset_stores

    reset_stores()

    from api.router import limiter as _module_limiter

    _module_limiter.reset()

    from api import create_app

    return TestClient(create_app())


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    c = _build_client(tmp_path, monkeypatch)
    try:
        yield c
    finally:
        c.close()


def test_symbols_endpoint_returns_200_with_payload(client: TestClient) -> None:
    resp = client.get("/hamnosys/symbols")
    assert resp.status_code == 200
    body = resp.json()
    assert body["schema_version"] == 1
    assert isinstance(body["count"], int) and body["count"] > 0
    assert isinstance(body["symbols"], list)
    assert len(body["symbols"]) == body["count"]


def test_symbols_entries_have_expected_shape(client: TestClient) -> None:
    body = client.get("/hamnosys/symbols").json()
    required_keys = {
        "codepoint",
        "hex",
        "char",
        "short_name",
        "latex_command",
        "class",
        "class_label",
        "slots",
    }
    for entry in body["symbols"]:
        assert required_keys.issubset(entry.keys()), (
            f"missing keys in {entry!r}: {required_keys - set(entry.keys())}"
        )
        assert isinstance(entry["codepoint"], int)
        assert entry["hex"].startswith("U+") and len(entry["hex"]) >= 6
        assert isinstance(entry["char"], str) and len(entry["char"]) == 1
        assert isinstance(entry["slots"], list)


def test_symbols_contains_known_handshape_and_label(client: TestClient) -> None:
    """``hamflathand`` (U+E001) is a well-known anchor in the BSL set."""
    body = client.get("/hamnosys/symbols").json()
    by_name = {s["short_name"]: s for s in body["symbols"]}
    assert "hamflathand" in by_name
    flat = by_name["hamflathand"]
    assert flat["codepoint"] == 0xE001
    assert flat["hex"] == "U+E001"
    assert flat["class"] == "handshape_base"
    # Plain-English label, not the enum slug.
    assert flat["class_label"] == "Handshape"
    assert "handshape" in flat["slots"]


def test_symbols_response_is_cacheable_immutably(client: TestClient) -> None:
    resp = client.get("/hamnosys/symbols")
    cache_control = resp.headers.get("cache-control", "")
    assert "immutable" in cache_control
    assert "public" in cache_control
    assert "max-age=31536000" in cache_control
    assert resp.headers.get("etag"), "expected an ETag so browsers can revalidate"


def test_symbols_endpoint_honours_if_none_match(client: TestClient) -> None:
    first = client.get("/hamnosys/symbols")
    etag = first.headers["etag"]
    assert etag  # sanity

    second = client.get("/hamnosys/symbols", headers={"If-None-Match": etag})
    assert second.status_code == 304
    # Revalidated 304s must still carry the cache headers so intermediaries
    # don't drop them on their way back through.
    assert second.headers.get("etag") == etag
    assert "immutable" in second.headers.get("cache-control", "")


def test_symbols_endpoint_ignores_stale_etag(client: TestClient) -> None:
    resp = client.get(
        "/hamnosys/symbols", headers={"If-None-Match": '"hamnosys-stale-0"'}
    )
    assert resp.status_code == 200
    assert resp.json()["schema_version"] == 1

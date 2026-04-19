"""PII handling: hashing, stripping, config + telemetry invariants."""

from __future__ import annotations

import re

import pytest

from security.config import DEFAULT_SIGNER_SALT, load_security_config
from security.pii import hash_signer_id, strip_signer_identifiers


# ---------------------------------------------------------------------------
# hash_signer_id
# ---------------------------------------------------------------------------


def test_hash_is_deterministic_for_same_salt() -> None:
    h1 = hash_signer_id("alice-001", salt="my-salt")
    h2 = hash_signer_id("alice-001", salt="my-salt")
    assert h1 == h2


def test_hash_changes_with_salt() -> None:
    h1 = hash_signer_id("alice-001", salt="salt-A")
    h2 = hash_signer_id("alice-001", salt="salt-B")
    assert h1 != h2


def test_hash_has_prefix_and_fixed_length() -> None:
    h = hash_signer_id("alice-001", salt="salt")
    assert h.startswith("h:")
    assert len(h) == 2 + 32
    assert re.fullmatch(r"h:[0-9a-f]{32}", h) is not None


def test_hash_is_idempotent_on_already_hashed_value() -> None:
    h1 = hash_signer_id("alice-001", salt="salt")
    h2 = hash_signer_id(h1, salt="salt")
    assert h1 == h2


def test_hash_rejects_empty_inputs() -> None:
    with pytest.raises(ValueError):
        hash_signer_id("", salt="salt")
    with pytest.raises(ValueError):
        hash_signer_id("alice", salt="")


# ---------------------------------------------------------------------------
# strip_signer_identifiers
# ---------------------------------------------------------------------------


def test_strip_removes_signer_id_and_display_name() -> None:
    payload = {
        "id": "abc",
        "gloss": "ABROAD",
        "author": {
            "signer_id": "alice-001",
            "display_name": "Alice",
            "is_deaf_native": True,
        },
    }
    stripped = strip_signer_identifiers(payload)
    assert stripped["author"] == {"is_deaf_native": True}
    # Original untouched (deep copy semantics).
    assert payload["author"]["signer_id"] == "alice-001"


def test_strip_handles_missing_author() -> None:
    payload = {"id": "abc", "gloss": "X"}
    assert strip_signer_identifiers(payload) == payload


def test_strip_recurses_into_nested_sessions() -> None:
    payload = {
        "entries": [
            {"author": {"signer_id": "alice", "display_name": "Alice"}},
            {"author": {"signer_id": "bob", "is_deaf_native": False}},
        ]
    }
    stripped = strip_signer_identifiers(payload)
    for entry in stripped["entries"]:
        assert "signer_id" not in entry["author"]
        assert "display_name" not in entry["author"]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def test_default_config_uses_hashed_pii_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CHAT2HAMNOSYS_PII_POLICY", raising=False)
    cfg = load_security_config()
    assert cfg.pii_policy == "hashed"


def test_plaintext_policy_requires_explicit_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT2HAMNOSYS_PII_POLICY", "plaintext")
    cfg = load_security_config()
    assert cfg.pii_policy == "plaintext"


def test_invalid_policy_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHAT2HAMNOSYS_PII_POLICY", "cleartext")
    with pytest.raises(ValueError):
        load_security_config()


def test_default_salt_is_signaled_on_startup(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("CHAT2HAMNOSYS_SIGNER_ID_SALT", raising=False)
    import logging

    with caplog.at_level(logging.WARNING, logger="security.config"):
        load_security_config()
    assert any("SIGNER_ID_SALT" in r.message for r in caplog.records)


def test_config_env_overrides() -> None:
    import os

    os.environ["CHAT2HAMNOSYS_MAX_INPUT_LEN"] = "500"
    os.environ["CHAT2HAMNOSYS_PER_IP_DAILY_CAP_USD"] = "25"
    os.environ["CHAT2HAMNOSYS_GLOBAL_DAILY_CAP_USD"] = "100"
    try:
        cfg = load_security_config()
        assert cfg.max_input_len == 500
        assert cfg.per_ip_daily_cap_usd == 25.0
        assert cfg.global_daily_cap_usd == 100.0
    finally:
        os.environ.pop("CHAT2HAMNOSYS_MAX_INPUT_LEN", None)
        os.environ.pop("CHAT2HAMNOSYS_PER_IP_DAILY_CAP_USD", None)
        os.environ.pop("CHAT2HAMNOSYS_GLOBAL_DAILY_CAP_USD", None)


# ---------------------------------------------------------------------------
# Telemetry PII invariant — content must be off by default
# ---------------------------------------------------------------------------


def test_telemetry_logger_does_not_log_content_by_default(tmp_path) -> None:
    """Regression guard: Prompt 4's invariant that prompt content is
    not logged unless the operator explicitly opts in.

    Prompt 17 §6 asks us to re-verify this after the hardening pass —
    the check is cheap and the guarantee is load-bearing.
    """
    import json

    from llm.telemetry import TelemetryLogger

    logger = TelemetryLogger(log_dir=tmp_path / "telemetry")
    assert logger.log_content is False

    path = logger.log_call(
        request_id="req-1",
        model="gpt-4o",
        prompt_tokens=50,
        completion_tokens=10,
        latency_ms=100,
        cost_usd=0.001,
        temperature=0.1,
        success=True,
        prompt_content=[{"role": "system", "content": "SECRET PROMPT"}],
        completion_content="SECRET COMPLETION",
    )
    line = path.read_text(encoding="utf-8").strip()
    record = json.loads(line)
    assert "prompt" not in record
    assert "completion" not in record
    # Sanity: the strings never reach disk.
    assert "SECRET PROMPT" not in line
    assert "SECRET COMPLETION" not in line

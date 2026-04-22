"""Regression tests for the progress snapshot pipeline (polish-13 §1).

These tests drive the exported ``build_snapshot`` and
``append_progress_log`` functions against the live ``data/`` layout —
the snapshot is read by the public progress dashboard, so a
malformation here would silently break the dashboard on deploy.

Guarded invariants:

1. ``build_snapshot`` returns a dict with the expected top-level keys.
2. Every per-language entry carries the keys the frontend reads.
3. Counts are non-negative; ``total >= reviewed`` (we can't review more
   signs than we've kept).
4. The fingerspelling alphabet summary uses bounded string values
   (``full`` / ``partial`` / ``none`` / ``unknown``) — the dashboard's
   state chip reads this literal.
5. ``append_progress_log`` writes a valid JSONL line and is
   idempotent within a UTC date.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "server") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "server"))

import progress_snapshot as sp  # noqa: E402 — flat import, matches sibling tests
from progress_snapshot import (  # noqa: E402
    append_progress_log,
    build_snapshot,
    write_snapshot,
)


EXPECTED_TOP_KEYS = {
    "generated_at",
    "generator",
    "schema_version",
    "totals",
    "languages",
    "progress_series",
    "recent_activity",
    "coverage_gaps",
}

EXPECTED_LANG_KEYS = {
    "code",
    "name",
    "native_name",
    "source",
    "source_kind",
    "total",
    "reviewed",
    "awaiting_review",
    "community_pending",
    "alphabet",
    "alphabet_present",
    "alphabet_expected",
    "top500_covered",
    "top500_total",
    "top500_coverage_pct",
    "last_updated",
    "partial_data",
}

VALID_ALPHABET_STATES = {"full", "partial", "none", "unknown"}


def test_build_snapshot_has_expected_top_level_keys() -> None:
    snapshot = build_snapshot()
    assert EXPECTED_TOP_KEYS.issubset(snapshot.keys()), (
        f"snapshot is missing keys: {EXPECTED_TOP_KEYS - snapshot.keys()}"
    )


def test_build_snapshot_languages_have_expected_shape() -> None:
    snapshot = build_snapshot()
    languages = snapshot["languages"]
    assert languages, "snapshot produced no language rows"
    for lang in languages:
        missing = EXPECTED_LANG_KEYS - lang.keys()
        assert not missing, f"{lang.get('code', '?')} missing keys: {missing}"


def test_build_snapshot_counts_are_non_negative() -> None:
    snapshot = build_snapshot()
    for lang in snapshot["languages"]:
        for field in ("total", "reviewed", "awaiting_review", "community_pending"):
            value = lang.get(field)
            if value is None:
                continue
            assert value >= 0, (
                f"{lang['code']}.{field} is negative: {value}"
            )


def test_build_snapshot_reviewed_does_not_exceed_total() -> None:
    snapshot = build_snapshot()
    for lang in snapshot["languages"]:
        total = lang.get("total")
        reviewed = lang.get("reviewed")
        if total is None or reviewed is None:
            continue
        # ``partial_data`` rows are permitted to mismatch (the tooltip
        # on the frontend explains). Skip those.
        if lang.get("partial_data"):
            continue
        assert reviewed <= total, (
            f"{lang['code']} reports reviewed={reviewed} > total={total}"
        )


def test_build_snapshot_alphabet_uses_bounded_states() -> None:
    snapshot = build_snapshot()
    for lang in snapshot["languages"]:
        state = lang.get("alphabet")
        assert state in VALID_ALPHABET_STATES, (
            f"{lang['code']} alphabet state {state!r} not in {VALID_ALPHABET_STATES}"
        )


def test_build_snapshot_totals_match_language_sum() -> None:
    """Top-line ``totals.signs`` is the sum of per-language totals
    unless every language is partial-data (then totals go to ``None``).
    """
    snapshot = build_snapshot()
    totals = snapshot["totals"]
    languages = snapshot["languages"]
    if all(e["partial_data"] for e in languages):
        assert totals["signs"] is None
        return
    expected = sum((e["total"] or 0) for e in languages)
    assert totals["signs"] == expected


def test_append_progress_log_writes_valid_jsonl(tmp_path: Path) -> None:
    snapshot = build_snapshot()
    log_path = tmp_path / "progress_log.jsonl"
    row = append_progress_log(snapshot, log_path=log_path)
    assert row is not None
    assert log_path.exists()
    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["date"] == datetime.now(timezone.utc).strftime("%Y-%m-%d")
    assert "signs" in parsed
    assert "reviewed" in parsed
    assert "by_language" in parsed and isinstance(parsed["by_language"], dict)


def test_append_progress_log_dedups_within_one_day(tmp_path: Path) -> None:
    """Running the snapshot twice in a day must not create a duplicate
    row — the dashboard's chart would over-count a busy deploy day."""
    snapshot = build_snapshot()
    log_path = tmp_path / "progress_log.jsonl"
    first = append_progress_log(snapshot, log_path=log_path)
    second = append_progress_log(snapshot, log_path=log_path)
    assert first is not None
    assert second is None, "second append in same UTC day should be a no-op"
    lines = [l for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1


def test_write_snapshot_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End-to-end: ``write_snapshot`` writes a file the dashboard can
    consume without a re-build step."""
    out = tmp_path / "snapshot.json"
    monkeypatch.setattr(sp, "PROGRESS_LOG_PATH", tmp_path / "progress_log.jsonl")
    payload = sp.write_snapshot(out_path=out)
    assert out.exists()
    reparsed = json.loads(out.read_text(encoding="utf-8"))
    assert reparsed["schema_version"] == payload["schema_version"]
    assert EXPECTED_TOP_KEYS.issubset(reparsed.keys())

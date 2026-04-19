"""Tests for the structured event logger (ring buffer, rotation, retention)."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from obs import events as evs
from obs.logger import (
    EventLogger,
    UnknownEventError,
    emit_event,
    get_logger,
    hash_user_id,
    iter_log_files,
    read_jsonl,
    reset_logger,
    set_logger,
)


@pytest.fixture
def logger(tmp_path: Path) -> EventLogger:
    lg = EventLogger(log_dir=tmp_path, sink="file", retention_days=30)
    set_logger(lg)
    yield lg
    reset_logger()


def test_emit_writes_to_daily_file_and_ring(logger: EventLogger, tmp_path: Path) -> None:
    record = logger.emit(
        evs.SESSION_CREATED, session_id="s1", sign_language="bsl"
    )
    assert record.event == evs.SESSION_CREATED
    assert record.session_id == "s1"
    files = list(iter_log_files(tmp_path))
    assert len(files) == 1
    rows = read_jsonl(files[0])
    assert len(rows) == 1
    assert rows[0]["event"] == evs.SESSION_CREATED
    assert rows[0]["session_id"] == "s1"
    assert rows[0]["sign_language"] == "bsl"
    # Ring
    ring = logger.recent()
    assert len(ring) == 1
    assert ring[0].event == evs.SESSION_CREATED


def test_unknown_event_rejected(logger: EventLogger) -> None:
    with pytest.raises(UnknownEventError):
        logger.emit("session.not_a_real_event")


def test_hash_user_id_stable_and_short() -> None:
    assert hash_user_id(None) is None
    a = hash_user_id("user-42")
    b = hash_user_id("user-42")
    c = hash_user_id("user-43")
    assert a == b and a != c
    assert len(a) == 16


def test_top_level_emit_event_uses_singleton(tmp_path: Path) -> None:
    reset_logger()
    lg = EventLogger(log_dir=tmp_path, sink="file")
    set_logger(lg)
    try:
        emit_event(evs.LLM_CALL_STARTED, request_id="r1", model="gpt-4o")
        assert len(lg.recent()) == 1
    finally:
        reset_logger()


def test_recent_for_session_filters(logger: EventLogger) -> None:
    logger.emit(evs.SESSION_CREATED, session_id="a")
    logger.emit(evs.SESSION_CREATED, session_id="b")
    logger.emit(evs.SESSION_ACCEPTED, session_id="a")
    a_rows = logger.recent_for_session("a")
    assert [e.event for e in a_rows] == [
        evs.SESSION_CREATED,
        evs.SESSION_ACCEPTED,
    ]


def test_retention_purges_old_files(tmp_path: Path) -> None:
    cutoff_old = (date.today() - timedelta(days=45)).isoformat()
    (tmp_path / f"{cutoff_old}.jsonl").write_text('{"event":"x"}\n')
    recent = (date.today() - timedelta(days=5)).isoformat()
    (tmp_path / f"{recent}.jsonl").write_text('{"event":"x"}\n')
    lg = EventLogger(log_dir=tmp_path, sink="file", retention_days=30)
    lg.emit(evs.SESSION_CREATED, session_id="s")
    # Old file purged; recent one kept; today's new one exists.
    names = {p.stem for p in tmp_path.iterdir() if p.suffix == ".jsonl"}
    assert cutoff_old not in names
    assert recent in names
    reset_logger()


def test_get_logger_singleton() -> None:
    reset_logger()
    a = get_logger()
    b = get_logger()
    assert a is b
    reset_logger()

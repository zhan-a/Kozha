"""Tests for :class:`session.storage.SessionStore`.

Covers CRUD round-trips, the ``resume`` staleness rule that is the
load-bearing half of "sessions time out after 24 hours of inactivity",
the retention cleanup (``delete_older_than``), and a simulated
process-restart roundtrip (fresh SessionStore on the same DB file).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from session.state import (
    INACTIVITY_TIMEOUT,
    RETENTION_WINDOW,
    AbandonedEvent,
    AuthoringSession,
    DescribedEvent,
    SessionState,
    SignEntryDraft,
)
from session.storage import SessionStore
from session import resume_session


# ---------------------------------------------------------------------------
# Construction / schema
# ---------------------------------------------------------------------------


def test_store_constructor_creates_db_file(tmp_path: Path):
    db = tmp_path / "sessions.sqlite3"
    assert not db.exists()
    SessionStore(db_path=db)
    assert db.exists()


def test_store_constructor_creates_parent_dirs(tmp_path: Path):
    db = tmp_path / "nested" / "deeper" / "sessions.sqlite3"
    SessionStore(db_path=db)
    assert db.exists()


def test_store_constructor_is_idempotent(tmp_path: Path):
    db = tmp_path / "sessions.sqlite3"
    SessionStore(db_path=db)
    SessionStore(db_path=db)  # reopening existing DB is safe
    SessionStore(db_path=db)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


def test_save_then_get_round_trip(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    s = AuthoringSession(
        draft=SignEntryDraft(gloss="TEMPLE", author_signer_id="alice"),
    )
    s = s.append_event(DescribedEvent(prose="hello", gaps_found=0))
    store.save(s)

    fetched = store.get(s.id)
    assert fetched is not None
    assert fetched.id == s.id
    assert fetched.state == s.state
    assert fetched.draft.gloss == "TEMPLE"
    assert len(fetched.history) == 1
    assert fetched.history[0].type == "described"


def test_get_missing_returns_none(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    assert store.get(uuid4()) is None


def test_save_is_upsert(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    s = AuthoringSession()
    store.save(s)
    # Update the state and save again — count stays at 1, row reflects latest.
    s = s.with_state(SessionState.CLARIFYING)
    store.save(s)
    assert store.count() == 1
    assert store.get(s.id).state == SessionState.CLARIFYING


def test_delete_removes_row(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    s = AuthoringSession()
    store.save(s)
    assert store.delete(s.id) is True
    assert store.get(s.id) is None
    # Second delete returns False.
    assert store.delete(s.id) is False


def test_list_orders_by_last_activity_desc(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    now = datetime.now(timezone.utc)
    a = AuthoringSession(last_activity_at=now - timedelta(minutes=30))
    b = AuthoringSession(last_activity_at=now - timedelta(minutes=5))
    c = AuthoringSession(last_activity_at=now - timedelta(hours=2))
    for sess in (a, b, c):
        store.save(sess)
    ordered = store.list()
    assert [x.id for x in ordered] == [b.id, a.id, c.id]


def test_count_reflects_inserts_and_deletes(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    assert store.count() == 0
    s1 = AuthoringSession()
    s2 = AuthoringSession()
    store.save(s1)
    store.save(s2)
    assert store.count() == 2
    store.delete(s1.id)
    assert store.count() == 1


# ---------------------------------------------------------------------------
# Resumption: fresh, stale, terminal, missing
# ---------------------------------------------------------------------------


def test_resume_returns_fresh_session_unchanged(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    s = AuthoringSession()
    store.save(s)
    resumed = store.resume(s.id)
    assert resumed is not None
    assert resumed.id == s.id
    assert resumed.state == SessionState.AWAITING_DESCRIPTION


def test_resume_missing_returns_none(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    assert store.resume(uuid4()) is None


def test_resume_marks_stale_session_abandoned(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    ancient = datetime.now(timezone.utc) - INACTIVITY_TIMEOUT - timedelta(seconds=1)
    s = AuthoringSession(
        state=SessionState.CLARIFYING,
        last_activity_at=ancient,
    )
    store.save(s)
    resumed = store.resume(s.id)
    assert resumed is not None
    assert resumed.state == SessionState.ABANDONED
    # ABANDONED event appended with the expected reason.
    assert any(
        isinstance(e, AbandonedEvent) and e.reason == "inactivity timeout"
        for e in resumed.history
    )
    # Persisted: re-fetching reflects the abandoned state.
    assert store.get(s.id).state == SessionState.ABANDONED


def test_resume_is_idempotent_on_already_abandoned(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    ancient = datetime.now(timezone.utc) - INACTIVITY_TIMEOUT - timedelta(seconds=1)
    s = AuthoringSession(state=SessionState.ABANDONED, last_activity_at=ancient)
    store.save(s)
    # First resume is a no-op (already terminal); history stays empty.
    resumed = store.resume(s.id)
    assert resumed is not None
    assert resumed.state == SessionState.ABANDONED
    assert resumed.history == []


def test_resume_is_idempotent_on_finalized_session(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    ancient = datetime.now(timezone.utc) - INACTIVITY_TIMEOUT - timedelta(seconds=1)
    s = AuthoringSession(state=SessionState.FINALIZED, last_activity_at=ancient)
    store.save(s)
    resumed = store.resume(s.id)
    assert resumed.state == SessionState.FINALIZED


def test_resume_custom_timeout_and_now(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    base = datetime.now(timezone.utc)
    s = AuthoringSession(
        state=SessionState.CLARIFYING,
        last_activity_at=base - timedelta(minutes=10),
    )
    store.save(s)
    # With a 5-minute timeout and the current instant, the session is stale.
    resumed = store.resume(s.id, now=base, timeout=timedelta(minutes=5))
    assert resumed.state == SessionState.ABANDONED


def test_resume_via_package_function(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    s = AuthoringSession()
    store.save(s)
    # The package-level helper delegates to the store.
    resumed = resume_session(s.id, store=store)
    assert resumed is not None
    assert resumed.id == s.id


# ---------------------------------------------------------------------------
# Retention cleanup
# ---------------------------------------------------------------------------


def test_delete_older_than_removes_only_expired_rows(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    now = datetime.now(timezone.utc)
    fresh = AuthoringSession(last_activity_at=now - timedelta(days=1))
    borderline = AuthoringSession(
        last_activity_at=now - RETENTION_WINDOW + timedelta(seconds=30)
    )
    expired = AuthoringSession(last_activity_at=now - RETENTION_WINDOW - timedelta(hours=1))
    very_old = AuthoringSession(last_activity_at=now - timedelta(days=90))
    for sess in (fresh, borderline, expired, very_old):
        store.save(sess)

    removed = store.delete_older_than(now=now)
    assert removed == 2  # expired + very_old
    remaining_ids = {x.id for x in store.list()}
    assert remaining_ids == {fresh.id, borderline.id}


def test_delete_older_than_with_custom_retention(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    now = datetime.now(timezone.utc)
    a = AuthoringSession(last_activity_at=now - timedelta(hours=2))
    b = AuthoringSession(last_activity_at=now - timedelta(hours=1))
    store.save(a)
    store.save(b)
    # Retention = 90 minutes: a should be deleted, b kept.
    removed = store.delete_older_than(now=now, retention=timedelta(minutes=90))
    assert removed == 1
    assert {x.id for x in store.list()} == {b.id}


def test_delete_older_than_noop_when_all_fresh(tmp_path: Path):
    store = SessionStore(db_path=tmp_path / "sessions.sqlite3")
    s = AuthoringSession()
    store.save(s)
    assert store.delete_older_than() == 0
    assert store.count() == 1


# ---------------------------------------------------------------------------
# Simulated process restart: new SessionStore on the same DB file
# ---------------------------------------------------------------------------


def test_session_survives_process_restart(tmp_path: Path):
    db = tmp_path / "sessions.sqlite3"
    store_a = SessionStore(db_path=db)
    s = AuthoringSession(
        draft=SignEntryDraft(gloss="TEMPLE", author_signer_id="alice"),
    )
    s = s.append_event(DescribedEvent(prose="hello", gaps_found=0))
    store_a.save(s)
    # Simulate process restart: forget store_a, open a fresh one.
    del store_a
    store_b = SessionStore(db_path=db)
    restored = store_b.get(s.id)
    assert restored is not None
    assert restored.id == s.id
    assert restored.draft.gloss == "TEMPLE"
    assert [e.type for e in restored.history] == ["described"]


def test_stale_session_across_restart_is_abandoned_on_resume(tmp_path: Path):
    db = tmp_path / "sessions.sqlite3"
    store_a = SessionStore(db_path=db)
    ancient = datetime.now(timezone.utc) - INACTIVITY_TIMEOUT - timedelta(seconds=1)
    s = AuthoringSession(
        state=SessionState.CLARIFYING,
        last_activity_at=ancient,
    )
    store_a.save(s)
    del store_a
    store_b = SessionStore(db_path=db)
    resumed = store_b.resume(s.id)
    assert resumed is not None
    assert resumed.state == SessionState.ABANDONED
    # And the persisted row reflects the abandonment.
    assert store_b.get(s.id).state == SessionState.ABANDONED

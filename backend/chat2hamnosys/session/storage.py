"""SQLite-backed persistence for :class:`AuthoringSession`.

Sessions are Pydantic-serialisable end to end, so the store is a thin
layer on top of SQLite: one row per session, payload as a JSON blob,
with the few indexed columns (state, last_activity_at) denormalised
for the resumption / timeout / cleanup queries.

Lifecycle:

- :meth:`SessionStore.save` — insert or replace; bumps the activity
  timestamp indirectly (we write whatever the session model carries).
- :meth:`SessionStore.get` — fetch by UUID; returns ``None`` if absent.
- :meth:`SessionStore.resume` — fetch + mark ABANDONED if the session
  has exceeded :data:`state.INACTIVITY_TIMEOUT`. Implements the
  "sessions time out after 24 hours of inactivity" rule.
- :meth:`SessionStore.delete_older_than` — hard-delete sessions whose
  last activity predates ``cutoff``. Used by the 30-day retention
  cleanup.
- :meth:`SessionStore.list` — iterate all stored sessions (ordered by
  last activity desc).

The store is process-local and uses ``sqlite3``'s default thread
affinity; multi-process access is not a goal of Prompt 9.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from .state import (
    INACTIVITY_TIMEOUT,
    RETENTION_WINDOW,
    AbandonedEvent,
    AuthoringSession,
    SessionState,
    TERMINAL_STATES,
    _utcnow,
)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                TEXT PRIMARY KEY,
    state             TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    last_activity_at  TEXT NOT NULL,
    payload           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_state ON sessions(state);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity_at);
"""


class SessionStore:
    """SQLite-backed authoring-session repository.

    Constructor opens (and creates if needed) the DB file; the schema
    migration is idempotent (``IF NOT EXISTS``) so re-opening an
    existing store is safe.
    """

    def __init__(self, *, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- CRUD --------------------------------------------------------------

    def save(self, session: AuthoringSession) -> None:
        """Insert or replace ``session``."""
        payload = session.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions
                    (id, state, created_at, last_activity_at, payload)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state=excluded.state,
                    created_at=excluded.created_at,
                    last_activity_at=excluded.last_activity_at,
                    payload=excluded.payload
                """,
                (
                    str(session.id),
                    session.state.value,
                    session.created_at.isoformat(),
                    session.last_activity_at.isoformat(),
                    payload,
                ),
            )

    def get(self, session_id: UUID) -> Optional[AuthoringSession]:
        """Fetch a session by id. Returns ``None`` if not present."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM sessions WHERE id = ?", (str(session_id),)
            ).fetchone()
        if row is None:
            return None
        return AuthoringSession.model_validate_json(row["payload"])

    def delete(self, session_id: UUID) -> bool:
        """Delete a session by id; returns ``True`` if a row was removed."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE id = ?", (str(session_id),)
            )
            return cur.rowcount > 0

    def list(self) -> list[AuthoringSession]:
        """Return all sessions, newest activity first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM sessions ORDER BY last_activity_at DESC"
            ).fetchall()
        return [AuthoringSession.model_validate_json(r["payload"]) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()
        return int(row["c"])

    # -- Resumption --------------------------------------------------------

    def resume(
        self,
        session_id: UUID,
        *,
        now: Optional[datetime] = None,
        timeout=INACTIVITY_TIMEOUT,
    ) -> Optional[AuthoringSession]:
        """Fetch ``session_id`` and mark it ABANDONED if stale.

        Returns the (possibly updated) session, or ``None`` if the id
        is not stored. A stale session is persisted in its new
        ABANDONED state — callers can distinguish live resumption by
        checking ``resumed.state == SessionState.ABANDONED``.

        Idempotent: calling ``resume`` on an already-ABANDONED session
        does nothing and returns it unchanged.
        """
        session = self.get(session_id)
        if session is None:
            return None
        if session.state in TERMINAL_STATES:
            return session
        if not session.is_stale(now=now, timeout=timeout):
            return session
        session = session.append_event(
            AbandonedEvent(reason="inactivity timeout")
        ).with_state(SessionState.ABANDONED)
        self.save(session)
        return session

    # -- Retention ---------------------------------------------------------

    def delete_older_than(
        self,
        *,
        now: Optional[datetime] = None,
        retention=RETENTION_WINDOW,
    ) -> int:
        """Hard-delete sessions whose last activity predates ``now - retention``.

        Returns the number of rows removed. Typically wired to a daily
        cleanup job.
        """
        cutoff = (now if now is not None else _utcnow()) - retention
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE last_activity_at < ?",
                (cutoff.isoformat(),),
            )
            return cur.rowcount


__all__ = ["SessionStore"]

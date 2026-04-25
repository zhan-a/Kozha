"""Session-creation tokens — prevents random session hijacking.

Every session created via ``POST /sessions`` gets a fresh
``secrets.token_urlsafe`` token; the client stores it and echoes it
back in the ``X-Session-Token`` header on subsequent requests. The
server compares via :func:`secrets.compare_digest` to avoid timing
leaks.

The binding lives in a tiny SQLite table next to the session-state DB
(a plain ``session_id TEXT PRIMARY KEY → token TEXT`` pair) so that
restarting the FastAPI process keeps existing sessions reachable by
their holder. No expiry is enforced at this layer — retention mirrors
the session itself, so when the session is abandoned / expired by
:class:`session.storage.SessionStore`, the token row can be cleared
alongside.
"""

from __future__ import annotations

import secrets
import sqlite3
from pathlib import Path
from typing import Optional
from uuid import UUID


_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_tokens (
    session_id TEXT PRIMARY KEY,
    token      TEXT NOT NULL
);
"""


class TokenStore:
    """Persist session → token bindings in a dedicated SQLite table.

    Construction migrates the schema idempotently; a reopened store over
    an existing DB file keeps its rows.
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

    @staticmethod
    def new_token() -> str:
        """Generate a fresh URL-safe token (~32 random bytes base64'd)."""
        return secrets.token_urlsafe(24)

    def put(self, session_id: UUID, token: str) -> None:
        """Insert or overwrite the token for ``session_id``."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO session_tokens (session_id, token) VALUES (?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET token=excluded.token",
                (str(session_id), token),
            )

    def get(self, session_id: UUID) -> Optional[str]:
        """Return the stored token for ``session_id`` or ``None``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT token FROM session_tokens WHERE session_id = ?",
                (str(session_id),),
            ).fetchone()
        return row["token"] if row else None

    def find_session_by_token(self, token: Optional[str]) -> Optional[UUID]:
        """Return the session id this token belongs to, or ``None``.

        Reverse lookup used by the stateless ``GET /signs/by-token/<token>``
        endpoint. The token has ~192 bits of entropy so a linear scan
        with case-sensitive equality is unguessable in practice; we still
        verify with :func:`secrets.compare_digest` to keep the timing
        behavior consistent with :meth:`verify`.
        """
        if not token:
            return None
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, token FROM session_tokens"
            ).fetchall()
        for row in rows:
            if secrets.compare_digest(row["token"], token):
                try:
                    return UUID(row["session_id"])
                except ValueError:
                    return None
        return None

    def delete(self, session_id: UUID) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM session_tokens WHERE session_id = ?",
                (str(session_id),),
            )
            return cur.rowcount > 0

    def verify(self, session_id: UUID, supplied_token: Optional[str]) -> bool:
        """Constant-time compare of ``supplied_token`` to the stored token.

        Returns ``False`` if no token exists for the session or the
        supplied token is missing — callers should treat that as
        authorization failed.
        """
        stored = self.get(session_id)
        if stored is None or supplied_token is None:
            return False
        return secrets.compare_digest(stored, supplied_token)


__all__ = ["TokenStore"]

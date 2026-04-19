"""Reviewer storage and the tamper-evident export-audit log.

Two persistent stores live here:

- :class:`ReviewerStore` — SQLite table holding registered reviewers.
  Tokens are stored as SHA-256 hex digests; the raw token is shown to
  the operator once at creation and never re-derivable.
- :class:`ExportAuditLog` — append-only JSONL file recording every sign
  exported into the Kozha library. Each record carries the SHA-256 of
  the (HamNoSys, SiGML) pair at export time, so silent post-hoc edits
  to the canonical SiGML file are detectable by recomputing and
  comparing.

Both stores are intentionally minimal — no cross-table joins, no
schema migrations beyond ``IF NOT EXISTS``. The audit log writes a
``record_hash`` chained from the previous row so an intruder must
rewrite the entire suffix to forge history; we still recommend
periodic external archiving of the file.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from uuid import UUID

from .models import Reviewer


logger = logging.getLogger(__name__)


_REVIEWER_SCHEMA = """
CREATE TABLE IF NOT EXISTS reviewers (
    id                  TEXT PRIMARY KEY,
    display_name        TEXT NOT NULL,
    is_deaf_native      INTEGER NOT NULL,
    is_board            INTEGER NOT NULL,
    signs               TEXT NOT NULL,         -- JSON array
    regional_background TEXT,
    token_hash          TEXT NOT NULL UNIQUE,
    active              INTEGER NOT NULL DEFAULT 1,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_reviewers_token_hash ON reviewers(token_hash);
CREATE INDEX IF NOT EXISTS idx_reviewers_active     ON reviewers(active);
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def hash_token(token: str) -> str:
    """SHA-256 hex digest of ``token`` — comparable in constant time."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def new_reviewer_token() -> str:
    """Mint a fresh URL-safe bearer token (~32 random bytes base64'd)."""
    return secrets.token_urlsafe(32)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReviewerNotFoundError(KeyError):
    """Requested reviewer id is not in the store."""


class ReviewerAuthError(PermissionError):
    """Supplied bearer token did not match any active reviewer."""


# ---------------------------------------------------------------------------
# Reviewer store
# ---------------------------------------------------------------------------


class ReviewerStore:
    """SQLite-backed reviewer registry.

    Construction migrates the schema idempotently so re-opening the same
    DB file is safe. Tokens are never stored in raw form — the
    :func:`new_reviewer` helper returns the token to the caller, which
    must show it to the operator and forget it.
    """

    def __init__(self, *, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_REVIEWER_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @staticmethod
    def _row_to_reviewer(row: sqlite3.Row) -> Reviewer:
        return Reviewer(
            id=UUID(row["id"]),
            display_name=row["display_name"],
            is_deaf_native=bool(row["is_deaf_native"]),
            is_board=bool(row["is_board"]),
            signs=json.loads(row["signs"]),
            regional_background=row["regional_background"],
            token_hash=row["token_hash"],
            active=bool(row["active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def add(self, reviewer: Reviewer) -> None:
        """Insert ``reviewer``. Caller already owns the id and token hash."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reviewers
                    (id, display_name, is_deaf_native, is_board,
                     signs, regional_background, token_hash, active, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(reviewer.id),
                    reviewer.display_name,
                    int(reviewer.is_deaf_native),
                    int(reviewer.is_board),
                    json.dumps(list(reviewer.signs)),
                    reviewer.regional_background,
                    reviewer.token_hash,
                    int(reviewer.active),
                    reviewer.created_at.isoformat(),
                ),
            )

    def create(
        self,
        *,
        display_name: str,
        is_deaf_native: bool = False,
        is_board: bool = False,
        signs: Iterable[str] = (),
        regional_background: Optional[str] = None,
    ) -> tuple[Reviewer, str]:
        """Mint a new reviewer + raw token. Returns (reviewer, raw_token).

        The raw token is only returned here — it is not persisted in the
        DB. If the operator loses it, the reviewer must be deleted and
        re-added.
        """
        token = new_reviewer_token()
        reviewer = Reviewer(
            display_name=display_name,
            is_deaf_native=is_deaf_native,
            is_board=is_board,
            signs=list(signs),
            regional_background=regional_background,
            token_hash=hash_token(token),
        )
        self.add(reviewer)
        return reviewer, token

    def get(self, reviewer_id: UUID) -> Reviewer:
        """Fetch by id. Raises :class:`ReviewerNotFoundError` if absent."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reviewers WHERE id = ?", (str(reviewer_id),)
            ).fetchone()
        if row is None:
            raise ReviewerNotFoundError(str(reviewer_id))
        return self._row_to_reviewer(row)

    def list(self, *, only_active: bool = True) -> list[Reviewer]:
        with self._connect() as conn:
            sql = "SELECT * FROM reviewers"
            args: tuple = ()
            if only_active:
                sql += " WHERE active = 1"
            sql += " ORDER BY created_at ASC"
            rows = conn.execute(sql, args).fetchall()
        return [self._row_to_reviewer(r) for r in rows]

    def deactivate(self, reviewer_id: UUID) -> bool:
        """Mark a reviewer inactive. Returns ``True`` if a row was changed."""
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE reviewers SET active = 0 WHERE id = ?",
                (str(reviewer_id),),
            )
            return cur.rowcount > 0

    def authenticate(self, token: Optional[str]) -> Reviewer:
        """Return the active reviewer matching ``token`` or raise.

        Constant-time comparison: we look up by token_hash then verify
        with :func:`secrets.compare_digest` to keep timing flat against
        an attacker probing for valid tokens.
        """
        if not token:
            raise ReviewerAuthError("missing reviewer token")
        digest = hash_token(token)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM reviewers WHERE token_hash = ? AND active = 1",
                (digest,),
            ).fetchone()
        if row is None:
            # Still spend a compare to keep the failure path branch-time-stable.
            secrets.compare_digest(digest, "0" * len(digest))
            raise ReviewerAuthError("invalid reviewer token")
        if not secrets.compare_digest(digest, row["token_hash"]):
            raise ReviewerAuthError("invalid reviewer token")
        return self._row_to_reviewer(row)


# ---------------------------------------------------------------------------
# Export audit log
# ---------------------------------------------------------------------------


class ExportAuditLog:
    """Append-only JSONL log of every export to the Kozha library.

    Each row carries:

    - ``sign_id``       — the sign exported.
    - ``exported_at``   — ISO-8601 UTC timestamp.
    - ``reviewer_ids``  — the reviewers whose approvals justified the export.
    - ``payload_hash``  — SHA-256 of (``hamnosys`` + "\\x1f" + ``sigml``)
      taken at export time. Recomputing later and comparing detects
      silent edits to the SiGML file.
    - ``prev_hash``     — SHA-256 of the previous row's serialised JSON,
      chaining the log so deleting a record breaks the chain.
    - ``record_hash``   — SHA-256 of this row's serialised JSON
      (excluding the ``record_hash`` field itself), so external
      tooling can verify a single row in isolation.

    The chain root hash is the literal string ``"GENESIS"`` to make the
    first row's prev_hash deterministic.
    """

    GENESIS = "GENESIS"

    def __init__(self, *, log_path: Path | str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def payload_hash(hamnosys: str, sigml: Optional[str]) -> str:
        """SHA-256 of (hamnosys + 0x1F + sigml). 0x1F is the unit separator."""
        material = hamnosys + "\x1f" + (sigml or "")
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def _last_record_hash(self) -> str:
        if not self.log_path.exists():
            return self.GENESIS
        last_line = ""
        # Walk to the final non-blank line — small log files are read in
        # full because the cost is bounded and the simplicity is worth it.
        with self.log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line:
                    last_line = line
        if not last_line:
            return self.GENESIS
        try:
            obj = json.loads(last_line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"audit log {self.log_path} is corrupt: cannot parse last row"
            ) from exc
        return str(obj.get("record_hash", self.GENESIS))

    @staticmethod
    def _hash_record_obj(obj: dict) -> str:
        """SHA-256 over the canonical JSON of ``obj`` minus ``record_hash``."""
        clone = {k: v for k, v in obj.items() if k != "record_hash"}
        canonical = json.dumps(clone, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def append(
        self,
        *,
        sign_id: UUID,
        reviewer_ids: Iterable[UUID | str],
        hamnosys: str,
        sigml: Optional[str],
        sign_language: str,
        gloss: str,
    ) -> dict:
        """Append one export record. Returns the serialised dict."""
        prev = self._last_record_hash()
        record: dict = {
            "sign_id": str(sign_id),
            "sign_language": sign_language,
            "gloss": gloss,
            "exported_at": _utcnow().isoformat(),
            "reviewer_ids": [str(r) for r in reviewer_ids],
            "payload_hash": self.payload_hash(hamnosys, sigml),
            "prev_hash": prev,
        }
        record["record_hash"] = self._hash_record_obj(record)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
        # Append-with-fsync — we're writing one line at a time so the
        # cost is small and we want the row durable before returning.
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        return record

    def read_all(self) -> list[dict]:
        """Return every row as a list of dicts (oldest first)."""
        if not self.log_path.exists():
            return []
        out: list[dict] = []
        with self.log_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"audit log {self.log_path} corrupt at line {i}: {exc}"
                    ) from exc
        return out

    def verify(self) -> tuple[bool, list[str]]:
        """Walk the log and verify the hash chain.

        Returns ``(ok, errors)``. ``errors`` is a list of human-readable
        descriptions of every mismatch found; empty when ``ok`` is True.
        Each row's ``record_hash`` must match a recompute, and each
        ``prev_hash`` must equal the previous row's ``record_hash``
        (with :data:`GENESIS` for the first row).
        """
        rows = self.read_all()
        errors: list[str] = []
        prev = self.GENESIS
        for i, row in enumerate(rows, 1):
            stored_hash = row.get("record_hash")
            recomputed = self._hash_record_obj(row)
            if stored_hash != recomputed:
                errors.append(
                    f"row {i} record_hash mismatch (sign_id={row.get('sign_id')!r})"
                )
            if row.get("prev_hash") != prev:
                errors.append(
                    f"row {i} prev_hash chain broken "
                    f"(expected {prev!r}, got {row.get('prev_hash')!r})"
                )
            prev = str(stored_hash)
        return (len(errors) == 0, errors)


__all__ = [
    "ExportAuditLog",
    "ReviewerAuthError",
    "ReviewerNotFoundError",
    "ReviewerStore",
    "hash_token",
    "new_reviewer_token",
]

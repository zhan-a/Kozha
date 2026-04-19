"""FastAPI dependency providers for the review API.

Mirrors the pattern in :mod:`api.dependencies`: each store is a
process-wide singleton built lazily on first request, configurable via
environment variables, resettable from tests via :func:`reset_review_stores`.

Environment variables
---------------------
- ``CHAT2HAMNOSYS_REVIEWER_DB``  — :class:`ReviewerStore` SQLite path.
- ``CHAT2HAMNOSYS_EXPORT_AUDIT`` — :class:`ExportAuditLog` JSONL path.

Plus the policy variables read by :meth:`ReviewPolicy.from_env`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import Header

from .models import Reviewer
from .policy import ReviewPolicy
from .storage import (
    ExportAuditLog,
    ReviewerAuthError,
    ReviewerStore,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REVIEWER_DB = _REPO_ROOT / "data" / "chat2hamnosys" / "reviewers.sqlite3"
DEFAULT_EXPORT_AUDIT = _REPO_ROOT / "data" / "chat2hamnosys" / "exports.jsonl"


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var, "").strip()
    return Path(raw) if raw else default


_reviewer_store: Optional[ReviewerStore] = None
_audit_log: Optional[ExportAuditLog] = None


def get_reviewer_store() -> ReviewerStore:
    global _reviewer_store
    if _reviewer_store is None:
        _reviewer_store = ReviewerStore(
            db_path=_env_path("CHAT2HAMNOSYS_REVIEWER_DB", DEFAULT_REVIEWER_DB)
        )
    return _reviewer_store


def get_audit_log() -> ExportAuditLog:
    global _audit_log
    if _audit_log is None:
        _audit_log = ExportAuditLog(
            log_path=_env_path("CHAT2HAMNOSYS_EXPORT_AUDIT", DEFAULT_EXPORT_AUDIT)
        )
    return _audit_log


def get_review_policy() -> ReviewPolicy:
    """Return the active review policy. Re-read on every call so tests can
    override env vars without a process restart."""
    return ReviewPolicy.from_env()


def reset_review_stores() -> None:
    """Clear singleton caches — used by tests."""
    global _reviewer_store, _audit_log
    _reviewer_store = None
    _audit_log = None


def reviewer_from_token(
    x_reviewer_token: Optional[str] = Header(default=None),
    store: ReviewerStore = None,  # injected via Depends below
) -> Reviewer:
    """Resolve the X-Reviewer-Token header to an active :class:`Reviewer`.

    This is a *plain* helper — it raises :class:`ReviewerAuthError` on
    invalid tokens. The router wraps it in a Depends() that swaps the
    error for an ApiError with the proper HTTP status.
    """
    if store is None:
        store = get_reviewer_store()
    return store.authenticate(x_reviewer_token)


__all__ = [
    "DEFAULT_EXPORT_AUDIT",
    "DEFAULT_REVIEWER_DB",
    "get_audit_log",
    "get_review_policy",
    "get_reviewer_store",
    "reset_review_stores",
    "reviewer_from_token",
]

"""Deaf reviewer workflow and export gate.

Public surface::

    from review import (
        ReviewPolicy,         # env-driven policy dataclass
        ReviewerStore,         # SQLite-backed reviewer registry
        ExportAuditLog,        # tamper-evident JSONL of exports
        approve, reject,       # action functions
        request_revision, flag,
        clear_quarantine,
        qualifying_approval_count,
    )

The FastAPI router for the review endpoints lives in
:mod:`review.router`; mount it under ``/review`` inside the main
chat2hamnosys sub-app (see :mod:`api.app`).
"""

from .actions import (
    CannotActOnTerminalEntry,
    NonNativeApprovalForbidden,
    ReviewActionError,
    ReviewerNotCompetent,
    approve,
    clear_quarantine,
    flag,
    qualifying_approval_count,
    qualifying_approval_ids,
    reject,
    request_revision,
)
from .models import (
    ApproveRequest,
    ClearQuarantineRequest,
    FlagRequest,
    RejectRequest,
    RequestRevisionRequest,
    Reviewer,
    ReviewerCreateRequest,
    ReviewerCredentials,
    ReviewerPublic,
)
from .policy import ReviewPolicy
from .storage import (
    ExportAuditLog,
    ReviewerAuthError,
    ReviewerNotFoundError,
    ReviewerStore,
    hash_token,
    new_reviewer_token,
)


__all__ = [
    "ApproveRequest",
    "CannotActOnTerminalEntry",
    "ClearQuarantineRequest",
    "ExportAuditLog",
    "FlagRequest",
    "NonNativeApprovalForbidden",
    "RejectRequest",
    "RequestRevisionRequest",
    "ReviewActionError",
    "ReviewPolicy",
    "Reviewer",
    "ReviewerAuthError",
    "ReviewerCreateRequest",
    "ReviewerCredentials",
    "ReviewerNotCompetent",
    "ReviewerNotFoundError",
    "ReviewerPublic",
    "ReviewerStore",
    "approve",
    "clear_quarantine",
    "flag",
    "hash_token",
    "new_reviewer_token",
    "qualifying_approval_count",
    "qualifying_approval_ids",
    "reject",
    "request_revision",
]

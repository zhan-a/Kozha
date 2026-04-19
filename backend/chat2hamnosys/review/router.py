"""HTTP surface for the Deaf-reviewer workflow.

Exposes the review queue, sign-detail view, four review actions
(approve / reject / request_revision / flag), the governance dashboard,
and a board-only quarantine-clearance endpoint. Mounted under
``/review`` of the chat2hamnosys sub-app.

Authentication
--------------
Every endpoint (except the public dashboard summary) requires an
``X-Reviewer-Token`` header. Tokens are minted by the admin CLI and
stored hashed; verification is constant-time. Unknown or revoked
tokens return 403 ``reviewer_forbidden``.

This is **prototype-grade** auth — see :mod:`review.storage` for the
caveats. The header name is deliberately distinct from
``X-Session-Token`` so the two surfaces can't be confused.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Query

from api.errors import ApiError
from models import SignEntry, SignStatus
from storage import (
    ExportNotAllowedError,
    InsufficientApprovalsError,
    SQLiteSignStore,
    SignNotFoundError,
)

from . import actions
from .dependencies import (
    get_audit_log,
    get_review_policy,
    get_reviewer_store,
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
    ReviewerStore,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Auth dependency — converts ReviewerAuthError to ApiError with 403.
# ---------------------------------------------------------------------------


def require_reviewer(
    x_reviewer_token: Optional[str] = Header(default=None),
    store: ReviewerStore = Depends(get_reviewer_store),
) -> Reviewer:
    try:
        return store.authenticate(x_reviewer_token)
    except ReviewerAuthError as exc:
        raise ApiError(
            str(exc), status_code=403, code="reviewer_forbidden"
        )


def require_board(reviewer: Reviewer = Depends(require_reviewer)) -> Reviewer:
    if not reviewer.is_board:
        raise ApiError(
            "this endpoint is restricted to governance-board reviewers",
            status_code=403,
            code="board_only",
        )
    return reviewer


# ---------------------------------------------------------------------------
# Sign-store dependency
#
# We intentionally re-import api.dependencies inside the providers below
# rather than at module import time — avoids a circular import: the api
# package mounts this router during create_app(), and the api.dependencies
# module pulls in storage which (transitively, via its export gate) pulls
# in this package. Lazy resolution sidesteps it.
# ---------------------------------------------------------------------------


def _get_sign_store() -> SQLiteSignStore:
    from api.dependencies import get_sign_store
    return get_sign_store()


# ---------------------------------------------------------------------------
# Serializers
# ---------------------------------------------------------------------------


def _entry_to_summary(entry: SignEntry, policy: ReviewPolicy) -> dict[str, Any]:
    """Compact view used in the queue."""
    return {
        "id": str(entry.id),
        "gloss": entry.gloss,
        "sign_language": entry.sign_language,
        "regional_variant": entry.regional_variant,
        "domain": entry.domain,
        "status": entry.status,
        "qualifying_approvals": actions.qualifying_approval_count(entry, policy),
        "min_approvals_required": policy.effective_min_approvals(),
        "review_count": len(entry.reviewers),
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
    }


def _entry_to_detail(entry: SignEntry, policy: ReviewPolicy) -> dict[str, Any]:
    """Full view used by the sign-detail endpoint."""
    return {
        "id": str(entry.id),
        "gloss": entry.gloss,
        "sign_language": entry.sign_language,
        "regional_variant": entry.regional_variant,
        "domain": entry.domain,
        "status": entry.status,
        "hamnosys": entry.hamnosys,
        "sigml": entry.sigml,
        "description_prose": entry.description_prose,
        "clarifications": [c.model_dump(mode="json") for c in entry.clarifications],
        "parameters": entry.parameters.model_dump(mode="json"),
        "reviewers": [r.model_dump(mode="json") for r in entry.reviewers],
        "qualifying_approvals": actions.qualifying_approval_count(entry, policy),
        "min_approvals_required": policy.effective_min_approvals(),
        "policy": {
            "min_approvals": policy.min_approvals,
            "require_native_deaf": policy.require_native_deaf,
            "allow_single_approval": policy.allow_single_approval,
            "require_region_match": policy.require_region_match,
        },
        "created_at": entry.created_at.isoformat(),
        "updated_at": entry.updated_at.isoformat(),
    }


def _action_error_to_api(exc: actions.ReviewActionError) -> ApiError:
    return ApiError(str(exc), status_code=exc.status_code, code=exc.code)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(tags=["review"])


# -- Reviewer self-info ------------------------------------------------------


@router.get("/me", summary="Return the authenticated reviewer's record")
def get_me(reviewer: Reviewer = Depends(require_reviewer)) -> dict[str, Any]:
    return ReviewerPublic.from_reviewer(reviewer).model_dump(mode="json")


# -- Queue + entry detail ----------------------------------------------------


@router.get("/queue", summary="List signs awaiting review")
def get_queue(
    sign_language: Optional[str] = Query(default=None),
    regional_variant: Optional[str] = Query(default=None),
    include_quarantined: bool = Query(default=False),
    reviewer: Reviewer = Depends(require_reviewer),
    store: SQLiteSignStore = Depends(_get_sign_store),
    policy: ReviewPolicy = Depends(get_review_policy),
) -> dict[str, Any]:
    statuses: tuple[SignStatus, ...]
    if include_quarantined:
        statuses = ("pending_review", "quarantined")
    else:
        statuses = ("pending_review",)

    items: list[SignEntry] = []
    for e in store.list():
        if e.status not in statuses:
            continue
        if sign_language and e.sign_language.lower() != sign_language.lower():
            continue
        if regional_variant and (e.regional_variant or "").lower() != regional_variant.lower():
            continue
        # Per-reviewer competence filter — only show signs the reviewer
        # is registered for. Board members see everything regardless.
        if not reviewer.is_board and e.sign_language.lower() not in reviewer.signs:
            continue
        items.append(e)

    # Oldest-first (by created_at), so signs don't starve in the queue.
    items.sort(key=lambda e: e.created_at)

    return {
        "count": len(items),
        "items": [_entry_to_summary(e, policy) for e in items],
        "filters": {
            "sign_language": sign_language,
            "regional_variant": regional_variant,
            "include_quarantined": include_quarantined,
        },
    }


@router.get("/entries/{sign_id}", summary="Full review-detail view")
def get_entry(
    sign_id: UUID,
    reviewer: Reviewer = Depends(require_reviewer),
    store: SQLiteSignStore = Depends(_get_sign_store),
    policy: ReviewPolicy = Depends(get_review_policy),
) -> dict[str, Any]:
    try:
        entry = store.get(sign_id)
    except SignNotFoundError as exc:
        raise ApiError(
            str(exc), status_code=404, code="sign_not_found"
        )
    return _entry_to_detail(entry, policy)


# -- Actions -----------------------------------------------------------------


@router.post(
    "/entries/{sign_id}/approve", summary="Approve a sign for validation"
)
def post_approve(
    sign_id: UUID,
    body: ApproveRequest,
    reviewer: Reviewer = Depends(require_reviewer),
    store: SQLiteSignStore = Depends(_get_sign_store),
    policy: ReviewPolicy = Depends(get_review_policy),
) -> dict[str, Any]:
    try:
        entry = actions.approve(
            sign_id=sign_id,
            reviewer=reviewer,
            store=store,
            policy=policy,
            comment=body.comment or "",
            allow_non_native=body.allow_non_native,
            justification=body.justification or "",
        )
    except SignNotFoundError as exc:
        raise ApiError(str(exc), status_code=404, code="sign_not_found")
    except actions.ReviewActionError as exc:
        raise _action_error_to_api(exc)
    return _entry_to_detail(entry, policy)


@router.post(
    "/entries/{sign_id}/reject", summary="Reject a sign"
)
def post_reject(
    sign_id: UUID,
    body: RejectRequest,
    reviewer: Reviewer = Depends(require_reviewer),
    store: SQLiteSignStore = Depends(_get_sign_store),
    policy: ReviewPolicy = Depends(get_review_policy),
) -> dict[str, Any]:
    try:
        entry = actions.reject(
            sign_id=sign_id,
            reviewer=reviewer,
            store=store,
            policy=policy,
            reason=body.reason,
            category=body.category,
        )
    except SignNotFoundError as exc:
        raise ApiError(str(exc), status_code=404, code="sign_not_found")
    except actions.ReviewActionError as exc:
        raise _action_error_to_api(exc)
    return _entry_to_detail(entry, policy)


@router.post(
    "/entries/{sign_id}/request_revision",
    summary="Send a sign back to its author for revision",
)
def post_request_revision(
    sign_id: UUID,
    body: RequestRevisionRequest,
    reviewer: Reviewer = Depends(require_reviewer),
    store: SQLiteSignStore = Depends(_get_sign_store),
    policy: ReviewPolicy = Depends(get_review_policy),
) -> dict[str, Any]:
    try:
        entry = actions.request_revision(
            sign_id=sign_id,
            reviewer=reviewer,
            store=store,
            policy=policy,
            comment=body.comment,
            fields_to_revise=body.fields_to_revise,
        )
    except SignNotFoundError as exc:
        raise ApiError(str(exc), status_code=404, code="sign_not_found")
    except actions.ReviewActionError as exc:
        raise _action_error_to_api(exc)
    return _entry_to_detail(entry, policy)


@router.post(
    "/entries/{sign_id}/flag",
    summary="Emergency flag — quarantine the sign regardless of current status",
)
def post_flag(
    sign_id: UUID,
    body: FlagRequest,
    reviewer: Reviewer = Depends(require_reviewer),
    store: SQLiteSignStore = Depends(_get_sign_store),
    policy: ReviewPolicy = Depends(get_review_policy),
) -> dict[str, Any]:
    try:
        entry = actions.flag(
            sign_id=sign_id, reviewer=reviewer, store=store, reason=body.reason
        )
    except SignNotFoundError as exc:
        raise ApiError(str(exc), status_code=404, code="sign_not_found")
    except actions.ReviewActionError as exc:
        raise _action_error_to_api(exc)
    return _entry_to_detail(entry, policy)


@router.post(
    "/entries/{sign_id}/clear_quarantine",
    summary="Board only — lift a quarantine and re-route the sign",
)
def post_clear_quarantine(
    sign_id: UUID,
    body: ClearQuarantineRequest,
    reviewer: Reviewer = Depends(require_board),
    store: SQLiteSignStore = Depends(_get_sign_store),
    policy: ReviewPolicy = Depends(get_review_policy),
) -> dict[str, Any]:
    try:
        entry = actions.clear_quarantine(
            sign_id=sign_id,
            reviewer=reviewer,
            store=store,
            target_status=body.target_status,
            comment=body.comment,
        )
    except SignNotFoundError as exc:
        raise ApiError(str(exc), status_code=404, code="sign_not_found")
    except actions.ReviewActionError as exc:
        raise _action_error_to_api(exc)
    return _entry_to_detail(entry, policy)


# -- Export gate -------------------------------------------------------------


@router.post(
    "/entries/{sign_id}/export",
    summary="Export a validated sign to the Kozha library (board only)",
)
def post_export(
    sign_id: UUID,
    reviewer: Reviewer = Depends(require_board),
    store: SQLiteSignStore = Depends(_get_sign_store),
    policy: ReviewPolicy = Depends(get_review_policy),
    audit: ExportAuditLog = Depends(get_audit_log),
) -> dict[str, Any]:
    try:
        store.export_to_kozha_library(sign_id, policy=policy, audit_log=audit)
    except SignNotFoundError as exc:
        raise ApiError(str(exc), status_code=404, code="sign_not_found")
    except ExportNotAllowedError as exc:
        raise ApiError(
            str(exc), status_code=409, code="export_not_allowed"
        )
    except InsufficientApprovalsError as exc:
        raise ApiError(
            str(exc), status_code=409, code="insufficient_approvals",
        )
    entry = store.get(sign_id)
    return {
        "exported": True,
        "sign_id": str(sign_id),
        "sign_language": entry.sign_language,
        "gloss": entry.gloss,
    }


# -- Governance dashboard ----------------------------------------------------


def _bucket_by(reviewers: list[Reviewer]) -> list[dict[str, Any]]:
    """Reviewer summary — name + signs + flags only (no token data)."""
    out = []
    for r in reviewers:
        out.append(ReviewerPublic.from_reviewer(r).model_dump(mode="json"))
    return out


@router.get(
    "/dashboard",
    summary="Read-only governance dashboard for the Deaf advisory board",
)
def get_dashboard(
    reviewer: Reviewer = Depends(require_reviewer),
    store: SQLiteSignStore = Depends(_get_sign_store),
    policy: ReviewPolicy = Depends(get_review_policy),
    rstore: ReviewerStore = Depends(get_reviewer_store),
    audit: ExportAuditLog = Depends(get_audit_log),
) -> dict[str, Any]:
    entries = store.list()
    by_status: Counter = Counter()
    rejection_categories: Counter = Counter()
    flag_count = 0
    review_count_by_reviewer: Counter = Counter()
    time_in_queue_seconds: list[float] = []
    now = datetime.now(timezone.utc)

    for e in entries:
        by_status[e.status] += 1
        if e.status == "pending_review":
            time_in_queue_seconds.append((now - e.created_at).total_seconds())
        for r in e.reviewers:
            review_count_by_reviewer[r.reviewer_id] += 1
            if r.verdict == "rejected" and r.category:
                rejection_categories[r.category] += 1
            if r.verdict == "flagged":
                flag_count += 1

    mean_time_in_queue = (
        sum(time_in_queue_seconds) / len(time_in_queue_seconds)
        if time_in_queue_seconds
        else 0.0
    )

    audit_rows = audit.read_all()
    audit_ok, audit_errors = audit.verify()

    quarantined: list[dict[str, Any]] = []
    for e in entries:
        if e.status != "quarantined":
            continue
        flag_records = [
            r.model_dump(mode="json") for r in e.reviewers if r.verdict == "flagged"
        ]
        quarantined.append(
            {
                "id": str(e.id),
                "gloss": e.gloss,
                "sign_language": e.sign_language,
                "flagged_by": flag_records,
                "updated_at": e.updated_at.isoformat(),
            }
        )

    return {
        "policy": {
            "min_approvals": policy.min_approvals,
            "effective_min_approvals": policy.effective_min_approvals(),
            "require_native_deaf": policy.require_native_deaf,
            "allow_single_approval": policy.allow_single_approval,
            "require_region_match": policy.require_region_match,
            "fatigue_threshold": policy.fatigue_threshold,
        },
        "counts_by_status": dict(by_status),
        "flag_count": flag_count,
        "rejection_reasons": dict(rejection_categories),
        "mean_time_in_queue_seconds": mean_time_in_queue,
        "reviewers": _bucket_by(rstore.list(only_active=False)),
        "reviewer_activity": dict(review_count_by_reviewer),
        "quarantined": quarantined,
        "exports": {
            "total": len(audit_rows),
            "audit_chain_ok": audit_ok,
            "audit_errors": audit_errors,
            "recent": audit_rows[-10:],
        },
    }


# -- Reviewer admin (board only) --------------------------------------------


@router.post(
    "/reviewers",
    summary="Create a new reviewer (board only) — returns the bearer token once",
    status_code=201,
)
def post_create_reviewer(
    body: ReviewerCreateRequest,
    _board: Reviewer = Depends(require_board),
    rstore: ReviewerStore = Depends(get_reviewer_store),
) -> ReviewerCredentials:
    reviewer, token = rstore.create(
        display_name=body.display_name,
        is_deaf_native=body.is_deaf_native,
        is_board=body.is_board,
        signs=[s.lower() for s in body.signs],
        regional_background=body.regional_background,
    )
    return ReviewerCredentials(
        reviewer=ReviewerPublic.from_reviewer(reviewer),
        token=token,
    )


__all__ = ["require_board", "require_reviewer", "router"]

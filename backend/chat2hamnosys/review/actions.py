"""Reviewer action service — the state-transition core.

Each function here is a small pure-ish operation: it takes a sign id, a
reviewer, the action payload, and the stores it needs, and returns the
updated :class:`SignEntry`. Persistence is the only side effect; the
status-transition logic is concentrated here so the rest of the system
can stay dumb about the workflow.

Append-only history
-------------------
Every action appends a :class:`ReviewRecord` to ``SignEntry.reviewers``
— we never delete prior records. That history is what the governance
dashboard reads and what the export-audit log references.

Two-reviewer rule
-----------------
:func:`approve` checks the policy after appending the new record: if
the count of qualifying approvals (after this one) reaches the policy
threshold, the entry is promoted to ``"validated"``. Otherwise it
stays in ``"pending_review"``. The qualification rules live in
:func:`_qualifying_approvals` so tests can hit them directly.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional
from uuid import UUID

from models import (
    RejectionCategory,
    ReviewRecord,
    SignEntry,
    SignStatus,
)
from obs import events as _evs
from obs import metrics as _metrics
from obs.logger import emit_event
from storage import SignStore

from .models import Reviewer
from .policy import ReviewPolicy


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ReviewActionError(RuntimeError):
    """Base class for review-action failures with an HTTP-friendly code."""

    code: str = "review_action_failed"
    status_code: int = 422

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if status_code is not None:
            self.status_code = status_code


class ReviewerNotCompetent(ReviewActionError):
    """Reviewer's signs/region don't match the entry."""

    code = "reviewer_not_competent"
    status_code = 403


class NonNativeApprovalForbidden(ReviewActionError):
    """A non-native reviewer tried to approve without an explicit override."""

    code = "non_native_approval_forbidden"
    status_code = 403


class CannotActOnTerminalEntry(ReviewActionError):
    """Entry is in a status where this action is not allowed."""

    code = "invalid_review_state"
    status_code = 409


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _push_to_pending(entry: SignEntry) -> SignEntry:
    """If a draft is acted on, promote it to pending_review on first touch."""
    if entry.status == "draft":
        entry.status = "pending_review"
    return entry


def _update_queue_depth(store: SignStore) -> None:
    """Refresh the ``pending_reviews_queue_depth`` gauge after a transition."""
    try:
        depth = sum(1 for e in store.list() if e.status == "pending_review")
    except Exception:
        return
    _metrics.pending_reviews_queue_depth.set(value=float(depth))


def _competent_or_raise(
    reviewer: Reviewer, entry: SignEntry, policy: ReviewPolicy
) -> None:
    """Raise if the reviewer can't review this entry's language/region."""
    if not reviewer.can_review(entry.sign_language, entry.regional_variant):
        raise ReviewerNotCompetent(
            f"reviewer {reviewer.id} not registered for "
            f"sign_language={entry.sign_language!r}"
        )


def _qualifies_as_approval(
    record: ReviewRecord, entry: SignEntry, policy: ReviewPolicy
) -> bool:
    """Does ``record`` count toward the validation threshold?

    An approval qualifies when:

    - verdict == "approved"
    - reviewer is native Deaf, OR the record carries an explicit
      ``allow_non_native=True`` override and the policy allows it
      (the policy always allows the override; the requirement is on
      the reviewer to set the flag and supply justification).
    - if the policy requires region match: either the entry has no
      ``regional_variant`` set, or the reviewer's ``regional_background``
      equals it (case-insensitive).
    """
    if record.verdict != "approved":
        return False
    if policy.require_native_deaf:
        if not (record.is_deaf_native or record.allow_non_native):
            return False
    if policy.require_region_match and entry.regional_variant:
        rb = (record.regional_background or "").strip().lower()
        rv = entry.regional_variant.strip().lower()
        if rb and rb != rv:
            return False
    return True


def _qualifying_approvals(
    entry: SignEntry, policy: ReviewPolicy
) -> list[ReviewRecord]:
    """Return only the approvals that count toward validation.

    De-duplicates by ``reviewer_id`` — a single reviewer cannot satisfy
    the two-reviewer rule by approving twice.
    """
    seen: set[str] = set()
    out: list[ReviewRecord] = []
    for r in entry.reviewers:
        if not _qualifies_as_approval(r, entry, policy):
            continue
        if r.reviewer_id in seen:
            continue
        seen.add(r.reviewer_id)
        out.append(r)
    return out


def qualifying_approval_count(entry: SignEntry, policy: ReviewPolicy) -> int:
    """Public helper for the export gate and the dashboard."""
    return len(_qualifying_approvals(entry, policy))


def qualifying_approval_ids(
    entry: SignEntry, policy: ReviewPolicy
) -> list[str]:
    """Public helper — reviewer ids of qualifying approvals (insertion order)."""
    return [r.reviewer_id for r in _qualifying_approvals(entry, policy)]


def _make_record(
    reviewer: Reviewer,
    *,
    verdict: str,
    notes: str = "",
    comment: str = "",
    category: Optional[RejectionCategory] = None,
    fields_to_revise: Optional[Iterable[str]] = None,
    allow_non_native: Optional[bool] = None,
    entry_sign_language: Optional[str] = None,
) -> ReviewRecord:
    lang_match: Optional[bool] = None
    if entry_sign_language is not None:
        lang_match = entry_sign_language.strip().lower() in {
            s.strip().lower() for s in (reviewer.signs or [])
        }
    return ReviewRecord(
        reviewer_id=str(reviewer.id),
        is_deaf_native=reviewer.is_deaf_native,
        verdict=verdict,
        notes=notes,
        comment=comment,
        category=category,
        regional_background=reviewer.regional_background,
        signs=list(reviewer.signs),
        allow_non_native=allow_non_native,
        fields_to_revise=list(fields_to_revise or []),
        reviewer_language_match=lang_match,
    )


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------


def approve(
    *,
    sign_id: UUID,
    reviewer: Reviewer,
    store: SignStore,
    policy: ReviewPolicy,
    comment: str = "",
    allow_non_native: bool = False,
    justification: str = "",
) -> SignEntry:
    """Reviewer approves a sign.

    Behavior:

    - The new :class:`ReviewRecord` is appended unconditionally — the
      history is permanent.
    - Status transitions only when the qualifying-approval count
      reaches :meth:`ReviewPolicy.effective_min_approvals`.
    - Non-native approvals require ``allow_non_native=True`` and a
      non-empty ``justification`` when the policy requires native Deaf.

    Cannot act on entries in ``"quarantined"``, ``"validated"``, or
    ``"rejected"`` status.
    """
    entry = store.get(sign_id)
    if entry.status in ("quarantined", "validated", "rejected"):
        raise CannotActOnTerminalEntry(
            f"cannot approve a sign with status {entry.status!r}"
        )
    _competent_or_raise(reviewer, entry, policy)

    if (
        policy.require_native_deaf
        and not reviewer.is_deaf_native
        and not allow_non_native
    ):
        raise NonNativeApprovalForbidden(
            "policy requires a native-Deaf approver; either pair with a "
            "native reviewer or pass allow_non_native=true with a justification"
        )
    if not reviewer.is_deaf_native and allow_non_native and not justification.strip():
        raise ReviewActionError(
            "non-native approval requires a written justification",
            code="justification_required",
            status_code=422,
        )

    notes = ""
    if not reviewer.is_deaf_native and allow_non_native:
        notes = f"non-native override; justification: {justification.strip()}"

    record = _make_record(
        reviewer,
        verdict="approved",
        notes=notes,
        comment=comment.strip(),
        allow_non_native=(allow_non_native if not reviewer.is_deaf_native else None),
        entry_sign_language=entry.sign_language,
    )
    entry = _push_to_pending(entry)
    entry.reviewers = list(entry.reviewers) + [record]

    threshold = policy.effective_min_approvals()
    if qualifying_approval_count(entry, policy) >= threshold:
        if policy.allow_single_approval and threshold == 1:
            policy.warn_if_bootstrap()
        entry.status = "validated"
    elif entry.status != "pending_review":
        entry.status = "pending_review"

    store.put(entry)
    emit_event(
        _evs.REVIEW_APPROVED,
        sign_id=str(entry.id),
        reviewer_id=str(reviewer.id),
        status=str(entry.status),
        qualifying_approvals=qualifying_approval_count(entry, policy),
        threshold=threshold,
    )
    _metrics.reviews_approved_total.inc()
    _update_queue_depth(store)
    return entry


def reject(
    *,
    sign_id: UUID,
    reviewer: Reviewer,
    store: SignStore,
    policy: ReviewPolicy,
    reason: str,
    category: RejectionCategory,
) -> SignEntry:
    """Reviewer rejects a sign — terminal state ``"rejected"``."""
    entry = store.get(sign_id)
    if entry.status in ("quarantined", "validated", "rejected"):
        raise CannotActOnTerminalEntry(
            f"cannot reject a sign with status {entry.status!r}"
        )
    _competent_or_raise(reviewer, entry, policy)
    if not reason.strip():
        raise ReviewActionError(
            "reject requires a non-empty reason",
            code="reason_required",
            status_code=422,
        )

    record = _make_record(
        reviewer,
        verdict="rejected",
        notes=reason.strip(),
        comment=reason.strip(),
        category=category,
        entry_sign_language=entry.sign_language,
    )
    entry = _push_to_pending(entry)
    entry.reviewers = list(entry.reviewers) + [record]
    entry.status = "rejected"
    store.put(entry)
    emit_event(
        _evs.REVIEW_REJECTED,
        sign_id=str(entry.id),
        reviewer_id=str(reviewer.id),
        category=str(category),
    )
    _metrics.reviews_rejected_total.inc()
    _update_queue_depth(store)
    return entry


def request_revision(
    *,
    sign_id: UUID,
    reviewer: Reviewer,
    store: SignStore,
    policy: ReviewPolicy,
    comment: str,
    fields_to_revise: Iterable[str],
) -> SignEntry:
    """Send the entry back to the author for revision.

    The status returns to ``"draft"`` so the original session can resume
    and address the reviewer's comment. The comment + field list is
    appended as a :class:`ReviewRecord` so the history survives.
    """
    entry = store.get(sign_id)
    if entry.status in ("quarantined", "validated", "rejected"):
        raise CannotActOnTerminalEntry(
            f"cannot request revision on a sign with status {entry.status!r}"
        )
    _competent_or_raise(reviewer, entry, policy)
    if not comment.strip():
        raise ReviewActionError(
            "request_revision requires a non-empty comment",
            code="comment_required",
            status_code=422,
        )

    record = _make_record(
        reviewer,
        verdict="changes_requested",
        notes=comment.strip(),
        comment=comment.strip(),
        fields_to_revise=fields_to_revise,
        entry_sign_language=entry.sign_language,
    )
    entry.reviewers = list(entry.reviewers) + [record]
    entry.status = "draft"
    store.put(entry)
    emit_event(
        _evs.REVIEW_REVISION_REQUESTED,
        sign_id=str(entry.id),
        reviewer_id=str(reviewer.id),
        fields_to_revise=list(fields_to_revise),
    )
    _update_queue_depth(store)
    return entry


def flag(
    *,
    sign_id: UUID,
    reviewer: Reviewer,
    store: SignStore,
    reason: str,
) -> SignEntry:
    """Emergency quarantine — any reviewer, from any status.

    Even validated signs can be flagged: a discovered cultural problem
    after publication should not require waiting for a board meeting
    before getting the sign out of the export pipeline. The board
    clears the quarantine separately.
    """
    if not reason.strip():
        raise ReviewActionError(
            "flag requires a non-empty reason",
            code="reason_required",
            status_code=422,
        )
    entry = store.get(sign_id)
    record = _make_record(
        reviewer,
        verdict="flagged",
        notes=reason.strip(),
        comment=reason.strip(),
        entry_sign_language=entry.sign_language,
    )
    entry.reviewers = list(entry.reviewers) + [record]
    entry.status = "quarantined"
    store.put(entry)
    emit_event(
        _evs.REVIEW_FLAGGED,
        sign_id=str(entry.id),
        reviewer_id=str(reviewer.id),
        reason=reason.strip()[:200],
    )
    _update_queue_depth(store)
    return entry


def clear_quarantine(
    *,
    sign_id: UUID,
    reviewer: Reviewer,
    store: SignStore,
    target_status: SignStatus,
    comment: str,
) -> SignEntry:
    """Lift a quarantine — board members only.

    The board member chooses what status to send the sign to (typically
    ``"pending_review"`` for a fresh review pass, or ``"rejected"`` if
    the flag is upheld). The reasoning is appended as a
    :class:`ReviewRecord`.
    """
    if not reviewer.is_board:
        raise ReviewActionError(
            "only board members may clear quarantines",
            code="board_only",
            status_code=403,
        )
    entry = store.get(sign_id)
    if entry.status != "quarantined":
        raise CannotActOnTerminalEntry(
            f"cannot clear quarantine on a sign with status {entry.status!r}"
        )
    if target_status not in ("draft", "pending_review", "rejected"):
        raise ReviewActionError(
            f"target_status {target_status!r} not allowed for clear_quarantine",
            code="invalid_target_status",
            status_code=422,
        )
    if not comment.strip():
        raise ReviewActionError(
            "clear_quarantine requires a non-empty comment",
            code="comment_required",
            status_code=422,
        )
    record = _make_record(
        reviewer,
        verdict="changes_requested",
        notes=f"quarantine cleared → {target_status}: {comment.strip()}",
        comment=comment.strip(),
        entry_sign_language=entry.sign_language,
    )
    entry.reviewers = list(entry.reviewers) + [record]
    entry.status = target_status
    store.put(entry)
    return entry


__all__ = [
    "CannotActOnTerminalEntry",
    "NonNativeApprovalForbidden",
    "ReviewActionError",
    "ReviewerNotCompetent",
    "approve",
    "clear_quarantine",
    "flag",
    "qualifying_approval_count",
    "qualifying_approval_ids",
    "reject",
    "request_revision",
]

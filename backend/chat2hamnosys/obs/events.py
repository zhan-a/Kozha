"""Event taxonomy for chat2hamnosys observability.

Each constant is the canonical event name used by structured logs and
metric labels. Keeping them in one module turns accidental typos into a
static-analysis problem and lets the dashboard discover the full
vocabulary at a glance.

Naming convention is ``<surface>.<noun>.<verb>`` so logs and dashboard
filters compose cleanly:

- ``session.*`` — high-level authoring lifecycle
- ``llm.call.*`` — every wrapped OpenAI call
- ``parse.*`` / ``clarify.*`` / ``generate.*`` / ``render.*``
                 — pipeline stages
- ``correct.*`` — correction interpreter
- ``review.*`` — Deaf-reviewer workflow
- ``export.*`` — export gate
- ``security.*`` — security tripwires (injection, budget, rate-limit)

Severity hints in :data:`EVENT_LEVEL` are read by the dashboard's
recent-events feed to pick a colour. Anything not in the map renders as
``info``.
"""

from __future__ import annotations

from typing import Final


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

SESSION_CREATED: Final = "session.created"
SESSION_STATE_CHANGED: Final = "session.state_changed"
SESSION_ACCEPTED: Final = "session.accepted"
SESSION_REJECTED: Final = "session.rejected"
SESSION_ABANDONED: Final = "session.abandoned"

# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

LLM_CALL_STARTED: Final = "llm.call.started"
LLM_CALL_SUCCEEDED: Final = "llm.call.succeeded"
LLM_CALL_RETRIED: Final = "llm.call.retried"
LLM_CALL_FAILED: Final = "llm.call.failed"
LLM_CALL_FALLBACK_USED: Final = "llm.call.fallback_used"

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

PARSE_DESCRIPTION_COMPLETED: Final = "parse.description.completed"
PARSE_DESCRIPTION_GAPS_FOUND: Final = "parse.description.gaps_found"

# ---------------------------------------------------------------------------
# Clarifier
# ---------------------------------------------------------------------------

CLARIFY_QUESTION_ASKED: Final = "clarify.question_asked"
CLARIFY_ANSWER_RECEIVED: Final = "clarify.answer_received"

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

GENERATE_HAMNOSYS_ATTEMPTED: Final = "generate.hamnosys.attempted"
GENERATE_HAMNOSYS_VALIDATED: Final = "generate.hamnosys.validated"
GENERATE_HAMNOSYS_INVALID_RETRY: Final = "generate.hamnosys.invalid_retry"
GENERATE_HAMNOSYS_GAVE_UP: Final = "generate.hamnosys.gave_up"

# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

RENDER_PREVIEW_STARTED: Final = "render.preview.started"
RENDER_PREVIEW_SUCCEEDED: Final = "render.preview.succeeded"
RENDER_PREVIEW_FAILED: Final = "render.preview.failed"
RENDER_CACHE_HIT: Final = "render.cache.hit"

# ---------------------------------------------------------------------------
# Correction interpreter
# ---------------------------------------------------------------------------

CORRECT_SUBMITTED: Final = "correct.submitted"
CORRECT_APPLIED: Final = "correct.applied"
CORRECT_AMBIGUOUS: Final = "correct.ambiguous"

# ---------------------------------------------------------------------------
# Reviewer workflow
# ---------------------------------------------------------------------------

REVIEW_QUEUED: Final = "review.queued"
REVIEW_APPROVED: Final = "review.approved"
REVIEW_REJECTED: Final = "review.rejected"
REVIEW_REVISION_REQUESTED: Final = "review.revision_requested"
REVIEW_FLAGGED: Final = "review.flagged"

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

EXPORT_ATTEMPTED: Final = "export.attempted"
EXPORT_BLOCKED: Final = "export.blocked"
EXPORT_SUCCEEDED: Final = "export.succeeded"

# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------

SECURITY_INJECTION_DETECTED: Final = "security.injection_detected"
SECURITY_BUDGET_EXCEEDED: Final = "security.budget_exceeded"
SECURITY_RATE_LIMITED: Final = "security.rate_limited"


# ---------------------------------------------------------------------------
# Allow-list — every emitted event must appear here. Keeps the taxonomy
# closed so a typo at a call site is caught by the logger rather than
# silently appearing as a new event in the feed.
# ---------------------------------------------------------------------------

ALL_EVENTS: frozenset[str] = frozenset(
    {
        SESSION_CREATED,
        SESSION_STATE_CHANGED,
        SESSION_ACCEPTED,
        SESSION_REJECTED,
        SESSION_ABANDONED,
        LLM_CALL_STARTED,
        LLM_CALL_SUCCEEDED,
        LLM_CALL_RETRIED,
        LLM_CALL_FAILED,
        LLM_CALL_FALLBACK_USED,
        PARSE_DESCRIPTION_COMPLETED,
        PARSE_DESCRIPTION_GAPS_FOUND,
        CLARIFY_QUESTION_ASKED,
        CLARIFY_ANSWER_RECEIVED,
        GENERATE_HAMNOSYS_ATTEMPTED,
        GENERATE_HAMNOSYS_VALIDATED,
        GENERATE_HAMNOSYS_INVALID_RETRY,
        GENERATE_HAMNOSYS_GAVE_UP,
        RENDER_PREVIEW_STARTED,
        RENDER_PREVIEW_SUCCEEDED,
        RENDER_PREVIEW_FAILED,
        RENDER_CACHE_HIT,
        CORRECT_SUBMITTED,
        CORRECT_APPLIED,
        CORRECT_AMBIGUOUS,
        REVIEW_QUEUED,
        REVIEW_APPROVED,
        REVIEW_REJECTED,
        REVIEW_REVISION_REQUESTED,
        REVIEW_FLAGGED,
        EXPORT_ATTEMPTED,
        EXPORT_BLOCKED,
        EXPORT_SUCCEEDED,
        SECURITY_INJECTION_DETECTED,
        SECURITY_BUDGET_EXCEEDED,
        SECURITY_RATE_LIMITED,
    }
)


# ---------------------------------------------------------------------------
# Severity hints — the dashboard recent-events feed uses these for colour.
# Missing entries default to ``info``.
# ---------------------------------------------------------------------------

LEVEL_INFO = "info"
LEVEL_WARNING = "warning"
LEVEL_ERROR = "error"

EVENT_LEVEL: dict[str, str] = {
    LLM_CALL_RETRIED: LEVEL_WARNING,
    LLM_CALL_FALLBACK_USED: LEVEL_WARNING,
    LLM_CALL_FAILED: LEVEL_ERROR,
    GENERATE_HAMNOSYS_INVALID_RETRY: LEVEL_WARNING,
    GENERATE_HAMNOSYS_GAVE_UP: LEVEL_ERROR,
    RENDER_PREVIEW_FAILED: LEVEL_ERROR,
    CORRECT_AMBIGUOUS: LEVEL_WARNING,
    REVIEW_REJECTED: LEVEL_WARNING,
    REVIEW_FLAGGED: LEVEL_WARNING,
    EXPORT_BLOCKED: LEVEL_WARNING,
    SESSION_ABANDONED: LEVEL_WARNING,
    SECURITY_INJECTION_DETECTED: LEVEL_ERROR,
    SECURITY_BUDGET_EXCEEDED: LEVEL_ERROR,
    SECURITY_RATE_LIMITED: LEVEL_WARNING,
}


def level_for(event: str) -> str:
    """Return the severity bucket for ``event``; defaults to ``info``."""
    return EVENT_LEVEL.get(event, LEVEL_INFO)


__all__ = [
    "ALL_EVENTS",
    "CLARIFY_ANSWER_RECEIVED",
    "CLARIFY_QUESTION_ASKED",
    "CORRECT_AMBIGUOUS",
    "CORRECT_APPLIED",
    "CORRECT_SUBMITTED",
    "EVENT_LEVEL",
    "EXPORT_ATTEMPTED",
    "EXPORT_BLOCKED",
    "EXPORT_SUCCEEDED",
    "GENERATE_HAMNOSYS_ATTEMPTED",
    "GENERATE_HAMNOSYS_GAVE_UP",
    "GENERATE_HAMNOSYS_INVALID_RETRY",
    "GENERATE_HAMNOSYS_VALIDATED",
    "LEVEL_ERROR",
    "LEVEL_INFO",
    "LEVEL_WARNING",
    "LLM_CALL_FAILED",
    "LLM_CALL_FALLBACK_USED",
    "LLM_CALL_RETRIED",
    "LLM_CALL_STARTED",
    "LLM_CALL_SUCCEEDED",
    "PARSE_DESCRIPTION_COMPLETED",
    "PARSE_DESCRIPTION_GAPS_FOUND",
    "RENDER_CACHE_HIT",
    "RENDER_PREVIEW_FAILED",
    "RENDER_PREVIEW_STARTED",
    "RENDER_PREVIEW_SUCCEEDED",
    "REVIEW_APPROVED",
    "REVIEW_FLAGGED",
    "REVIEW_QUEUED",
    "REVIEW_REJECTED",
    "REVIEW_REVISION_REQUESTED",
    "SECURITY_BUDGET_EXCEEDED",
    "SECURITY_INJECTION_DETECTED",
    "SECURITY_RATE_LIMITED",
    "SESSION_ABANDONED",
    "SESSION_ACCEPTED",
    "SESSION_CREATED",
    "SESSION_REJECTED",
    "SESSION_STATE_CHANGED",
    "level_for",
]

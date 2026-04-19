"""Authoring-session state machine.

The authoring flow is explicitly modelled as a state machine:

- :class:`SessionState` enumerates the phases (AWAITING_DESCRIPTION,
  CLARIFYING, GENERATING, RENDERED, AWAITING_CORRECTION, FINALIZED,
  ABANDONED …).
- :class:`AuthoringSession` carries the current state, a progressively
  populated :class:`SignEntryDraft`, and an append-only event log.
- The ``on_*`` functions in :mod:`.orchestrator` transition the
  session; each is a pure function that returns a new session.
- :class:`SessionStore` persists sessions to SQLite so
  :func:`resume_session` can rehydrate one after a process restart.

The HTTP shell for this machine lives in Prompt 12; this module is the
transport-neutral core.
"""

from .orchestrator import (
    Correction,
    InvalidTransitionError,
    apply_correction_diff,
    check_timeout,
    on_accept,
    on_clarification_answer,
    on_correction,
    on_description,
    on_reject,
    run_generation,
    start_session,
)
from .state import (
    AbandonedEvent,
    AcceptedEvent,
    AuthoringSession,
    ClarificationAnsweredEvent,
    ClarificationAskedEvent,
    CorrectionAppliedEvent,
    CorrectionRequestedEvent,
    DescribedEvent,
    DraftFinalizationError,
    GeneratedEvent,
    INACTIVITY_TIMEOUT,
    RETENTION_WINDOW,
    RejectedEvent,
    SessionEvent,
    SessionState,
    SignEntryDraft,
    TERMINAL_STATES,
)
from .storage import SessionStore


def resume_session(
    session_id,
    *,
    store: SessionStore,
) -> AuthoringSession | None:
    """Rehydrate a session by id, marking it ABANDONED if stale.

    Delegates to :meth:`SessionStore.resume` — surfaced at the package
    level because the prompt spec names ``resume_session`` explicitly.
    """
    return store.resume(session_id)


__all__ = [
    "AbandonedEvent",
    "AcceptedEvent",
    "AuthoringSession",
    "ClarificationAnsweredEvent",
    "ClarificationAskedEvent",
    "CorrectionAppliedEvent",
    "CorrectionRequestedEvent",
    "Correction",
    "DescribedEvent",
    "DraftFinalizationError",
    "GeneratedEvent",
    "INACTIVITY_TIMEOUT",
    "InvalidTransitionError",
    "RETENTION_WINDOW",
    "RejectedEvent",
    "SessionEvent",
    "SessionState",
    "SessionStore",
    "SignEntryDraft",
    "TERMINAL_STATES",
    "apply_correction_diff",
    "check_timeout",
    "on_accept",
    "on_clarification_answer",
    "on_correction",
    "on_description",
    "on_reject",
    "resume_session",
    "run_generation",
    "start_session",
]

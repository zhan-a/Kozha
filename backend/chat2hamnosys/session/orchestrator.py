"""Authoring session transition functions.

Each public ``on_*`` function is a pure transition on
:class:`AuthoringSession`: it takes a session, any inputs, and a few
callables for side-effecting services, and returns a new session. The
old session is never mutated. This keeps transitions trivially
unit-testable — a test can hand in stub callables and assert the
resulting session's state and event log.

The effectful services are injected as callables rather than imported
objects:

- ``parse_fn`` — prose → :class:`ParseResult`
- ``question_fn`` — :class:`ParseResult` → list of :class:`Question`
- ``apply_fn`` — (params, question, answer) → updated params
- ``generate_fn`` — params → :class:`GenerateResult`
- ``to_sigml_fn`` — HamNoSys → SiGML XML
- ``render_fn`` — SiGML → :class:`PreviewResult`

Transition matrix (summary):

::

    AWAITING_DESCRIPTION --[on_description]--> CLARIFYING | GENERATING*
    CLARIFYING           --[on_clarification_answer]--> CLARIFYING | GENERATING*
    GENERATING*          --[internal: on_description / on_answer tail]
                            runs generator and transitions to RENDERED
    RENDERED             --[on_correction]--> APPLYING_CORRECTION
    APPLYING_CORRECTION  --[on_correction_applied]--> RENDERED
    RENDERED             --[on_accept]--> FINALIZED
    (any non-terminal)   --[on_reject]--> ABANDONED
    (any non-terminal)   --[check_timeout]--> ABANDONED (if stale)

*GENERATING is a transient label: ``on_description`` and
``on_clarification_answer`` internally set it only while the generator
is running, and the same call transitions through to RENDERED on
success. Callers that want an explicit two-step (set GENERATING, then
generate) can call :func:`run_generation` on a session whose state is
already GENERATING.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from clarify import AnswerParseError, Question, apply_answer, generate_questions
from generator import GenerateResult, VOCAB, generate
from generator.params_to_hamnosys import _compose_pieces
from models import ClarificationTurn
from obs import events as _evs
from obs import metrics as _metrics
from obs.logger import emit_event
from parser import Gap, ParseResult, PartialSignParameters, parse_description
from rendering.hamnosys_to_sigml import to_sigml
from rendering.preview import PreviewResult, PreviewStatus, render_preview
from storage import SignStore

from .state import (
    AbandonedEvent,
    AcceptedEvent,
    AuthoringSession,
    ClarificationAnsweredEvent,
    ClarificationAskedEvent,
    CorrectionAppliedEvent,
    CorrectionRequestedEvent,
    DescribedEvent,
    GeneratedEvent,
    INACTIVITY_TIMEOUT,
    RejectedEvent,
    SessionState,
    SignEntryDraft,
    TERMINAL_STATES,
    _utcnow,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors and typed helpers
# ---------------------------------------------------------------------------


class InvalidTransitionError(RuntimeError):
    """Raised when a transition is attempted from a state that forbids it."""

    def __init__(
        self, transition: str, current: SessionState, expected: tuple[SessionState, ...]
    ) -> None:
        allowed = ", ".join(s.value for s in expected)
        super().__init__(
            f"cannot {transition} from state {current.value!r}; "
            f"expected one of: {allowed}"
        )
        self.transition = transition
        self.current = current
        self.expected = expected


@dataclass
class Correction:
    """Minimal correction shape — mirrors the Prompt-10 interpreter input.

    Concretely typed here so the state machine has a stable contract
    without taking a compile-time dependency on the interpreter.
    """

    raw_text: str
    target_time_ms: Optional[int] = None
    target_region: Optional[str] = None


# ---------------------------------------------------------------------------
# Transition preconditions
# ---------------------------------------------------------------------------


def _require_state(
    session: AuthoringSession,
    transition: str,
    allowed: tuple[SessionState, ...],
) -> None:
    if session.state not in allowed:
        raise InvalidTransitionError(transition, session.state, allowed)


# ---------------------------------------------------------------------------
# Session construction
# ---------------------------------------------------------------------------


def start_session(
    *,
    signer_id: str,
    gloss: Optional[str] = None,
    sign_language: str = "bsl",
    is_deaf_native: Optional[bool] = None,
    display_name: Optional[str] = None,
    domain: Optional[str] = None,
) -> AuthoringSession:
    """Create a fresh AWAITING_DESCRIPTION session.

    ``signer_id`` is required up front so finalization later can build
    a valid :class:`AuthorInfo`. The other fields are optional and can
    be set at any time before acceptance.
    """
    if not signer_id:
        raise ValueError("signer_id must be a non-empty string")
    draft = SignEntryDraft(
        gloss=gloss,
        sign_language=sign_language,  # type: ignore[arg-type]
        domain=domain,
        author_signer_id=signer_id,
        author_is_deaf_native=is_deaf_native,
        author_display_name=display_name,
    )
    session = AuthoringSession(draft=draft)
    emit_event(
        _evs.SESSION_CREATED,
        session_id=str(session.id),
        sign_language=sign_language,
        gloss=gloss,
        is_deaf_native=is_deaf_native,
    )
    _metrics.sessions_started_total.inc()
    _metrics.active_sessions.inc()
    return session


# ---------------------------------------------------------------------------
# Helpers shared between transitions
# ---------------------------------------------------------------------------


# Maps parser-gap field paths to the vocab slot + draft-key under which
# we record a resolved codepoint. Two paths ("handshape_dominant" vs
# "handshape_nondominant") share the same vocab slot.
_PARTIAL_TO_VOCAB_SLOT: dict[str, str] = {
    "handshape_dominant": "handshape",
    "handshape_nondominant": "handshape",
    "orientation_extended_finger": "orientation_ext_finger",
    "orientation_palm": "orientation_palm",
    "location": "location",
    "contact": "contact",
}


def _author_turn(prose: str) -> ClarificationTurn:
    return ClarificationTurn(role="author", text=prose)


def _assistant_turn(questions: list[Question]) -> ClarificationTurn:
    text = "\n".join(q.text for q in questions) or "(no questions)"
    return ClarificationTurn(role="assistant", text=text)


def _slot_codepoints_from_partial(
    partial: PartialSignParameters,
) -> dict[str, str]:
    """Re-run the composer over ``partial`` to extract per-slot PUA chunks.

    Called at generate time so the draft carries enough data to build a
    validated :class:`SignParameters` at finalization. Unresolved or
    LLM-only slots leave entries empty; :func:`_build_sign_parameters`
    raises at finalization if that proves insufficient.
    """
    pieces = _compose_pieces(partial)
    out: dict[str, str] = {}
    for piece in pieces:
        if piece.field_name == "symmetry":
            continue
        if piece.chunk:
            out[piece.field_name] = piece.chunk
    return out


def _generate_and_ask(
    session: AuthoringSession,
    parse_result: ParseResult,
    *,
    question_fn: Callable[..., list[Question]],
    is_deaf_native: Optional[bool],
) -> AuthoringSession:
    """Common tail for ``on_description``: emit questions or proceed."""
    prior_turns = list(session.draft.clarifications)
    questions = question_fn(
        parse_result,
        prior_turns,
        is_deaf_native=is_deaf_native,
    )
    if not questions:
        # No parser gaps or clarifier returned nothing — move to GENERATING.
        return session.with_state(SessionState.GENERATING)
    # Ask one clarification per turn, even if the generator surfaced
    # multiple gaps — presenting two at once fragments the chat and
    # makes free-form answers ambiguous about which field they address.
    # Remaining gaps will be asked on the next turn after the first
    # answer lands.
    questions = questions[:1]
    for q in questions:
        emit_event(
            _evs.CLARIFY_QUESTION_ASKED,
            session_id=str(session.id),
            field=q.field,
        )
    assistant_turn = _assistant_turn(questions)
    session = session.append_event(ClarificationAskedEvent(questions=questions))
    session = session.with_draft(
        pending_questions=questions,
        clarifications=list(session.draft.clarifications) + [assistant_turn],
    )
    return session.with_state(SessionState.CLARIFYING)


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


def on_description(
    session: AuthoringSession,
    prose: str,
    *,
    parse_fn: Callable[[str], ParseResult] = parse_description,
    question_fn: Callable[..., list[Question]] = generate_questions,
) -> AuthoringSession:
    """Apply a prose description, run the parser, branch to next phase.

    The incoming session must be in :data:`SessionState.AWAITING_DESCRIPTION`
    or — for a revision — :data:`SessionState.CLARIFYING` /
    :data:`SessionState.AWAITING_CORRECTION`.
    """
    _require_state(
        session,
        "on_description",
        (
            SessionState.AWAITING_DESCRIPTION,
            SessionState.CLARIFYING,
            SessionState.AWAITING_CORRECTION,
        ),
    )
    if not isinstance(prose, str) or not prose.strip():
        raise ValueError("prose must be a non-empty string")

    prose = prose.strip()
    # Run the parser (effectful — the parse_fn may call an LLM).
    parse_result = parse_fn(prose)

    emit_event(
        _evs.PARSE_DESCRIPTION_COMPLETED,
        session_id=str(session.id),
        gaps_found=len(parse_result.gaps),
        prose_length=len(prose),
    )
    if parse_result.gaps:
        emit_event(
            _evs.PARSE_DESCRIPTION_GAPS_FOUND,
            session_id=str(session.id),
            gaps=[g.field for g in parse_result.gaps],
        )

    # Record what the author said. The clarifier tail uses this to
    # ground the next batch of questions.
    author_turn = _author_turn(prose)
    session = session.append_event(
        DescribedEvent(prose=prose, gaps_found=len(parse_result.gaps))
    )
    session = session.with_draft(
        description_prose=prose,
        parameters_partial=parse_result.parameters,
        gaps=list(parse_result.gaps),
        pending_questions=[],
        clarifications=list(session.draft.clarifications) + [author_turn],
    )

    try:
        return _generate_and_ask(
            session,
            parse_result,
            question_fn=question_fn,
            is_deaf_native=session.draft.author_is_deaf_native,
        )
    except Exception as exc:
        # If the clarifier LLM fails on the very first turn (budget,
        # network, provider outage), skip clarification entirely and
        # try to generate from the parsed params. The user can still
        # submit-as-is or issue corrections afterwards.
        logger.warning(
            "initial clarification failed (%s); moving to GENERATING",
            type(exc).__name__,
        )
        return session.with_state(SessionState.GENERATING)


def on_clarification_answer(
    session: AuthoringSession,
    question_field: str,
    answer: str,
    *,
    apply_fn: Callable[..., PartialSignParameters] = apply_answer,
    question_fn: Callable[..., list[Question]] = generate_questions,
) -> AuthoringSession:
    """Apply one answer back into the partial parameters.

    Resolves the next state as follows:
    - If more pending questions remain in the current turn → CLARIFYING.
    - Else if gaps still exist → ask the clarifier for another batch;
      transition to CLARIFYING if questions come back, GENERATING if
      they don't.
    - Else → GENERATING.
    """
    _require_state(session, "on_clarification_answer", (SessionState.CLARIFYING,))
    if not question_field:
        raise ValueError("question_field must be a non-empty string")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer must be a non-empty string")
    question = next(
        (q for q in session.draft.pending_questions if q.field == question_field),
        None,
    )
    if question is None:
        raise ValueError(
            f"no pending question for field {question_field!r}; "
            f"pending: {[q.field for q in session.draft.pending_questions]}"
        )
    if session.draft.parameters_partial is None:
        raise RuntimeError(
            "session has no partial parameters; on_description must run first"
        )

    try:
        new_params = apply_fn(session.draft.parameters_partial, question, answer)
    except AnswerParseError:
        # An LLM-generated question with ``allow_freeform=False`` would
        # otherwise 500 the session on any unusual phrasing. Re-apply
        # with a lenient copy of the question so the user's answer lands
        # in the slot verbatim instead of being rejected.
        lenient_q = question.model_copy(update={"allow_freeform": True})
        new_params = apply_fn(
            session.draft.parameters_partial, lenient_q, answer
        )
    resolved_value = _resolve_field_from_partial(new_params, question_field)

    emit_event(
        _evs.CLARIFY_ANSWER_RECEIVED,
        session_id=str(session.id),
        field=question_field,
        resolved_value=resolved_value,
    )

    remaining_pending = [
        q for q in session.draft.pending_questions if q.field != question_field
    ]
    remaining_gaps = [
        g for g in session.draft.gaps if g.field != question_field
    ]
    author_turn = ClarificationTurn(role="author", text=answer.strip())
    session = session.append_event(
        ClarificationAnsweredEvent(
            question_field=question_field,
            answer=answer.strip(),
            resolved_value=resolved_value,
        )
    )
    session = session.with_draft(
        parameters_partial=new_params,
        gaps=remaining_gaps,
        pending_questions=remaining_pending,
        clarifications=list(session.draft.clarifications) + [author_turn],
    )

    if remaining_pending:
        # Stay in CLARIFYING — more questions in this turn await.
        return session
    if not remaining_gaps:
        return session.with_state(SessionState.GENERATING)

    # More gaps — ask another batch. If the follow-up LLM call fails
    # for any reason (budget, network, provider outage), degrade to
    # GENERATING: the contributor has already answered once, and a
    # best-effort draft is better than a 500 they can't recover from.
    stub_parse = ParseResult(
        parameters=new_params,
        gaps=remaining_gaps,
        raw_response="",
    )
    try:
        return _generate_and_ask(
            session,
            stub_parse,
            question_fn=question_fn,
            is_deaf_native=session.draft.author_is_deaf_native,
        )
    except Exception as exc:
        logger.warning(
            "follow-up clarification failed (%s); moving to GENERATING",
            type(exc).__name__,
        )
        return session.with_state(SessionState.GENERATING)


def _resolve_field_from_partial(
    params: PartialSignParameters, field: str
) -> str:
    """Read back the value the clarifier wrote for ``field``.

    Mirrors the leaf paths used by :func:`clarify.apply_answer._set_field`.
    """
    if field in {
        "handshape_dominant",
        "handshape_nondominant",
        "orientation_extended_finger",
        "orientation_palm",
        "location",
        "contact",
    }:
        return str(getattr(params, field) or "")
    if field == "movement":
        if params.movement:
            return str(params.movement[0].path or "")
        return ""
    if field.startswith("non_manual."):
        leaf = field.split(".", 1)[1]
        if params.non_manual is None:
            return ""
        return str(getattr(params.non_manual, leaf) or "")
    return ""


def run_generation(
    session: AuthoringSession,
    *,
    generate_fn: Callable[..., GenerateResult] = generate,
    to_sigml_fn: Callable[..., str] = to_sigml,
    render_fn: Optional[Callable[..., PreviewResult]] = render_preview,
) -> AuthoringSession:
    """Run generator → SiGML → preview on a session in GENERATING or APPLYING_CORRECTION.

    The session stays in GENERATING / APPLYING_CORRECTION on failure
    (with errors recorded on the draft and a :class:`GeneratedEvent`
    logged) so the caller can decide whether to route the user back to
    clarification or surface the error. On success the state moves to
    :data:`SessionState.RENDERED`.
    """
    _require_state(
        session,
        "run_generation",
        (SessionState.GENERATING, SessionState.APPLYING_CORRECTION),
    )
    if session.draft.parameters_partial is None:
        raise RuntimeError("run_generation requires a populated parameters_partial")

    emit_event(
        _evs.GENERATE_HAMNOSYS_ATTEMPTED,
        session_id=str(session.id),
    )
    gen_result = generate_fn(session.draft.parameters_partial)

    if not gen_result.hamnosys:
        emit_event(
            _evs.GENERATE_HAMNOSYS_GAVE_UP,
            session_id=str(session.id),
            errors=list(gen_result.errors)[:5],
            used_llm_fallback=gen_result.used_llm_fallback,
        )
        session = session.append_event(
            GeneratedEvent(
                success=False,
                confidence=0.0,
                used_llm_fallback=gen_result.used_llm_fallback,
                errors=list(gen_result.errors)
                or [gen_result.validation.summary() if gen_result.validation else "generation failed"],
            )
        )
        session = session.with_draft(
            generation_errors=list(gen_result.errors),
            generation_confidence=0.0,
        )
        return session  # stays in GENERATING / APPLYING_CORRECTION

    emit_event(
        _evs.GENERATE_HAMNOSYS_VALIDATED,
        session_id=str(session.id),
        confidence=gen_result.confidence,
        used_llm_fallback=gen_result.used_llm_fallback,
    )
    _metrics.validator_failures_before_success.observe(
        value=float(len(gen_result.errors))
    )

    hamnosys = gen_result.hamnosys
    gloss = session.draft.gloss or "SIGN"
    # SiGML conversion — use the populated non-manual features for the
    # mouthpicture tag when present.
    non_manual = None
    pnm = session.draft.parameters_partial.non_manual
    if pnm is not None and pnm.mouth_picture:
        from models import NonManualFeatures  # local import to avoid cycles
        non_manual = NonManualFeatures(mouth_picture=pnm.mouth_picture)
    sigml = to_sigml_fn(hamnosys, gloss=gloss, non_manual=non_manual)

    preview: PreviewResult | None = None
    if render_fn is not None:
        emit_event(
            _evs.RENDER_PREVIEW_STARTED,
            session_id=str(session.id),
            gloss=gloss,
        )
        try:
            preview = render_fn(sigml, gloss=gloss)
        except Exception as exc:
            logger.warning("render_preview raised %s; continuing without video", exc)
            emit_event(
                _evs.RENDER_PREVIEW_FAILED,
                session_id=str(session.id),
                error_class=type(exc).__name__,
            )
            preview = None
        else:
            status_val = preview.status.value if preview else "unknown"
            if preview and preview.status == PreviewStatus.OK:
                emit_event(
                    _evs.RENDER_PREVIEW_SUCCEEDED,
                    session_id=str(session.id),
                    video_path=str(preview.video_path) if preview.video_path else None,
                )
            elif preview and preview.status == PreviewStatus.CACHED:
                emit_event(
                    _evs.RENDER_CACHE_HIT,
                    session_id=str(session.id),
                    video_path=str(preview.video_path) if preview.video_path else None,
                )
                emit_event(
                    _evs.RENDER_PREVIEW_SUCCEEDED,
                    session_id=str(session.id),
                    cached=True,
                )
            else:
                emit_event(
                    _evs.RENDER_PREVIEW_FAILED,
                    session_id=str(session.id),
                    status=status_val,
                    message=preview.message if preview else "no preview",
                )

    slot_codepoints = _slot_codepoints_from_partial(session.draft.parameters_partial)

    session = session.append_event(
        GeneratedEvent(
            success=True,
            hamnosys=hamnosys,
            sigml=sigml,
            confidence=gen_result.confidence,
            used_llm_fallback=gen_result.used_llm_fallback,
        )
    )
    session = session.with_draft(
        hamnosys=hamnosys,
        sigml=sigml,
        slot_codepoints=slot_codepoints,
        video_path=str(preview.video_path) if preview and preview.video_path else None,
        preview_status=preview.status.value if preview else PreviewStatus.RENDERER_NOT_AVAILABLE.value,
        preview_message=preview.message if preview else "renderer not invoked",
        generation_confidence=gen_result.confidence,
        generation_errors=list(gen_result.errors),
    )
    return session.with_state(SessionState.RENDERED)


def on_correction(
    session: AuthoringSession,
    correction: Correction,
) -> AuthoringSession:
    """Acknowledge a user correction; move to APPLYING_CORRECTION.

    Prompt 9 only records the request and transitions state. Prompt 10
    adds the interpreter that produces the parameter diff; the diff
    apply + regenerate happens via :func:`apply_correction_diff` +
    :func:`run_generation` thereafter.
    """
    _require_state(
        session,
        "on_correction",
        (SessionState.RENDERED, SessionState.AWAITING_CORRECTION),
    )
    if not isinstance(correction, Correction):
        raise TypeError("correction must be a Correction instance")
    if not correction.raw_text or not correction.raw_text.strip():
        raise ValueError("correction.raw_text must be a non-empty string")

    emit_event(
        _evs.CORRECT_SUBMITTED,
        session_id=str(session.id),
        target_region=correction.target_region,
        target_time_ms=correction.target_time_ms,
    )
    _metrics.corrections_submitted_total.inc()
    session = session.append_event(
        CorrectionRequestedEvent(
            raw_text=correction.raw_text.strip(),
            target_time_ms=correction.target_time_ms,
            target_region=correction.target_region,
        )
    )
    session = session.with_draft(
        corrections_count=session.draft.corrections_count + 1,
    )
    return session.with_state(SessionState.APPLYING_CORRECTION)


def apply_correction_diff(
    session: AuthoringSession,
    *,
    field_changes: list[dict],
    summary: str = "",
    updated_params: Optional[PartialSignParameters] = None,
) -> AuthoringSession:
    """Record a correction diff and prepare for regeneration.

    Called from :mod:`backend.chat2hamnosys.correct` (Prompt 10) with
    the interpreter's output. The new partial parameters, if supplied,
    are written onto the draft so a subsequent :func:`run_generation`
    call can re-run the generator + renderer chain.
    """
    _require_state(
        session, "apply_correction_diff", (SessionState.APPLYING_CORRECTION,)
    )
    emit_event(
        _evs.CORRECT_APPLIED,
        session_id=str(session.id),
        summary=summary,
        field_changes_count=len(field_changes),
    )
    session = session.append_event(
        CorrectionAppliedEvent(summary=summary, field_changes=list(field_changes))
    )
    if updated_params is not None:
        session = session.with_draft(parameters_partial=updated_params)
    return session.with_state(SessionState.GENERATING)


def on_accept(
    session: AuthoringSession,
    *,
    store: SignStore,
) -> AuthoringSession:
    """Finalize the session: promote the draft to a SignEntry and persist it."""
    _require_state(session, "on_accept", (SessionState.RENDERED,))
    entry = session.draft.to_sign_entry()
    store.put(entry)
    emit_event(
        _evs.SESSION_ACCEPTED,
        session_id=str(session.id),
        sign_entry_id=str(entry.id),
        entry_status=str(entry.status),
        corrections_count=session.draft.corrections_count,
    )
    _metrics.sessions_accepted_total.inc()
    _metrics.active_sessions.dec()
    _metrics.session_duration_seconds.observe(
        value=(datetime.now(timezone.utc) - session.created_at).total_seconds()
    )
    _metrics.corrections_per_session.observe(
        value=float(session.draft.corrections_count)
    )
    session = session.append_event(
        AcceptedEvent(sign_entry_id=entry.id, status=entry.status)
    )
    return session.with_state(SessionState.FINALIZED)


def on_reject(
    session: AuthoringSession,
    reason: str = "",
) -> AuthoringSession:
    """Cancel a session — emits a :class:`RejectedEvent`, moves to ABANDONED.

    Valid from any non-terminal state; no-op (raises) if the session is
    already FINALIZED or ABANDONED.
    """
    if session.state in TERMINAL_STATES:
        raise InvalidTransitionError(
            "on_reject",
            session.state,
            tuple(s for s in SessionState if s not in TERMINAL_STATES),
        )
    emit_event(
        _evs.SESSION_REJECTED,
        session_id=str(session.id),
        reason=reason or "",
    )
    _metrics.sessions_abandoned_total.inc()
    _metrics.active_sessions.dec()
    _metrics.session_duration_seconds.observe(
        value=(datetime.now(timezone.utc) - session.created_at).total_seconds()
    )
    session = session.append_event(RejectedEvent(reason=reason or ""))
    return session.with_state(SessionState.ABANDONED)


def check_timeout(
    session: AuthoringSession,
    *,
    now: Optional[datetime] = None,
    timeout=INACTIVITY_TIMEOUT,
) -> AuthoringSession:
    """Transition the session to ABANDONED if it's been inactive long enough.

    Idempotent: already-terminal sessions and still-active sessions are
    returned unchanged.
    """
    if not session.is_stale(now=now, timeout=timeout):
        return session
    emit_event(
        _evs.SESSION_ABANDONED,
        session_id=str(session.id),
        reason="inactivity timeout",
    )
    _metrics.sessions_abandoned_total.inc()
    _metrics.active_sessions.dec()
    _metrics.session_duration_seconds.observe(
        value=(datetime.now(timezone.utc) - session.created_at).total_seconds()
    )
    session = session.append_event(AbandonedEvent(reason="inactivity timeout"))
    return session.with_state(SessionState.ABANDONED)


__all__ = [
    "Correction",
    "InvalidTransitionError",
    "apply_correction_diff",
    "check_timeout",
    "on_accept",
    "on_clarification_answer",
    "on_correction",
    "on_description",
    "on_reject",
    "run_generation",
    "start_session",
]

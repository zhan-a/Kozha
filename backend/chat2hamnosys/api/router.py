"""HTTP surface for the chat2hamnosys authoring state machine.

Exposes the orchestrator (see :mod:`session.orchestrator`) as a small
REST + SSE API under ``/api/chat2hamnosys``. The router is transport-
side only — every stateful operation delegates to the orchestrator's
pure transitions, persists the new session via :class:`SessionStore`,
and returns a typed :class:`SessionEnvelope`.

Endpoints
---------
- ``POST   /sessions``                       → create
- ``GET    /sessions/{id}``                  → fetch
- ``POST   /sessions/{id}/describe``         → initial prose + parse
- ``POST   /sessions/{id}/answer``           → resolve one clarification
- ``POST   /sessions/{id}/generate``         → force a generation attempt
- ``GET    /sessions/{id}/preview``          → latest preview metadata
- ``GET    /sessions/{id}/preview/video``    → mp4 binary (if rendered)
- ``POST   /sessions/{id}/correct``          → submit correction
- ``POST   /sessions/{id}/accept``           → finalize (store SignEntry)
- ``POST   /sessions/{id}/reject``           → abandon
- ``GET    /sessions/{id}/events``           → SSE stream of history events

Authentication
--------------
Every session issues a ``session_token`` on creation. Subsequent
requests must send the same value in ``X-Session-Token``; mismatches
return 403 ``session_forbidden``.

Rate limiting
-------------
Per-IP, ``30/minute`` by default. The :class:`Limiter` instance lives
on ``app.state.limiter`` and is created by :func:`build_api`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import FileResponse, StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from clarify import Question, apply_answer, generate_questions
from correct import (
    Correction as _InterpCorrection,
    apply_correction,
    interpret_correction,
)
from generator import GenerateResult, generate
from models import SignEntry
from parser import ParseResult, parse_description
from rendering.hamnosys_to_sigml import to_sigml
from rendering.preview import PreviewResult, render_preview
from session import (
    AuthoringSession,
    SessionState,
    SessionStore,
    TERMINAL_STATES,
)
from session.orchestrator import (
    Correction,
    InvalidTransitionError,
    on_accept,
    on_clarification_answer,
    on_correction,
    on_description,
    on_reject,
    run_generation,
    start_session,
)
from storage import SQLiteSignStore

from .dependencies import (
    get_apply_fn,
    get_generate_fn,
    get_parse_fn,
    get_question_fn,
    get_render_fn,
    get_session_store,
    get_sign_store,
    get_to_sigml_fn,
    get_token_store,
)
from .errors import ApiError, InvalidTransition, SessionForbidden, SessionNotFound
from .models import (
    AcceptResponse,
    AnswerRequest,
    CorrectRequest,
    CreateSessionRequest,
    CreateSessionResponse,
    DescribeRequest,
    GapOut,
    NextAction,
    OptionOut,
    PreviewOut,
    QuestionOut,
    RejectRequest,
    SessionEnvelope,
    SignEntryOut,
)
from .token_store import TokenStore


logger = logging.getLogger(__name__)


def _default_rate_limit() -> str:
    """Read the current rate-limit config from the environment.

    Re-read on every call so :func:`api.app.create_app` picks up tests'
    ``monkeypatch.setenv`` without needing a module reload.
    """
    return os.environ.get("CHAT2HAMNOSYS_RATE_LIMIT", "30/minute")


# Back-compat alias — some importers and the router's own ``__all__``
# still reference ``DEFAULT_RATE_LIMIT`` as a string. Resolved once at
# import time; :func:`api.app.create_app` uses :func:`_default_rate_limit`
# so per-test env overrides take effect.
DEFAULT_RATE_LIMIT = _default_rate_limit()

# SSE poll interval — the events endpoint checks for newly-appended
# history every ``EVENTS_POLL_INTERVAL`` seconds. Short enough to feel
# live in the browser; long enough that a sleepy session is not
# gratuitously hammering SQLite.
EVENTS_POLL_INTERVAL = 1.0
# How long to hold an SSE stream open before closing idle — caps
# orphaned connections if the client disappears.
EVENTS_MAX_DURATION = 300.0


# ---------------------------------------------------------------------------
# Limiter — a module-level placeholder so ``@limiter.limit(...)`` can be
# written as a decorator; :func:`api.app.create_app` replaces ``app.state
# .limiter`` with a fresh instance carrying the currently-configured
# default limit. Rate enforcement flows through :class:`SlowAPIMiddleware`,
# which reads ``request.app.state.limiter`` — so the instance swapped in
# by ``create_app`` is the one that actually limits traffic.
# ---------------------------------------------------------------------------


limiter = Limiter(key_func=get_remote_address, default_limits=[DEFAULT_RATE_LIMIT])


# ---------------------------------------------------------------------------
# Helpers: session loading + token verification
# ---------------------------------------------------------------------------


def _load_session(
    session_id: UUID,
    *,
    store: SessionStore,
    x_session_token: Optional[str],
    token_store: TokenStore,
) -> AuthoringSession:
    """Fetch + verify ownership of ``session_id``.

    Raises :class:`SessionNotFound` (404) or :class:`SessionForbidden`
    (403). Token verification short-circuits: if no token is stored,
    the session is treated as forbidden (shouldn't happen in practice
    because ``POST /sessions`` always mints one).
    """
    session = store.get(session_id)
    if session is None:
        raise SessionNotFound(f"session {session_id} not found")
    if not token_store.verify(session_id, x_session_token):
        raise SessionForbidden("X-Session-Token does not match session")
    return session


def _persist(store: SessionStore, session: AuthoringSession) -> AuthoringSession:
    store.save(session)
    return session


# ---------------------------------------------------------------------------
# Serializers — from AuthoringSession to the HTTP envelope.
# ---------------------------------------------------------------------------


def _question_out(q: Question) -> QuestionOut:
    options = None
    if q.options:
        options = [OptionOut(label=o.label, value=o.value) for o in q.options]
    return QuestionOut(
        question_id=q.field,
        field=q.field,
        text=q.text,
        options=options,
        allow_freeform=q.allow_freeform,
        rationale=q.rationale,
    )


def _gap_out(g: Any) -> GapOut:
    return GapOut(field=g.field, reason=g.reason, suggested_question=g.suggested_question)


def _preview_for(session: AuthoringSession, request: Request) -> PreviewOut:
    draft = session.draft
    video_url: Optional[str] = None
    if draft.video_path:
        # Always serve via the API route so clients don't need to know
        # on-disk layout. ``request.url_for`` generates the path with
        # the router's mount prefix included.
        try:
            video_url = str(
                request.url_for(
                    "get_preview_video", session_id=str(session.id)
                )
            )
        except Exception:
            video_url = None
    return PreviewOut(
        status=draft.preview_status,
        message=draft.preview_message,
        video_url=video_url,
        sigml=draft.sigml,
        hamnosys=draft.hamnosys,
    )


def _next_action(
    session: AuthoringSession, request: Request
) -> NextAction:
    """Pick a single next-action hint for the frontend."""
    state = session.state
    if state == SessionState.AWAITING_DESCRIPTION:
        return NextAction(kind="await_description")
    if state == SessionState.CLARIFYING:
        return NextAction(
            kind="answer_questions",
            questions=[_question_out(q) for q in session.draft.pending_questions],
        )
    if state == SessionState.GENERATING:
        return NextAction(kind="await_generation")
    if state == SessionState.RENDERED:
        return NextAction(
            kind="preview_ready", preview=_preview_for(session, request)
        )
    if state == SessionState.AWAITING_CORRECTION:
        return NextAction(kind="await_correction")
    if state == SessionState.APPLYING_CORRECTION:
        return NextAction(kind="await_generation")
    if state == SessionState.FINALIZED:
        return NextAction(kind="finalized")
    if state == SessionState.ABANDONED:
        return NextAction(kind="abandoned")
    return NextAction(kind="await_description")


def _envelope(session: AuthoringSession, request: Request) -> SessionEnvelope:
    draft = session.draft
    params_dump: Optional[dict[str, Any]] = None
    if draft.parameters_partial is not None:
        params_dump = draft.parameters_partial.model_dump(mode="json")
    history = [e.model_dump(mode="json") for e in session.history]
    clarifications = [c.model_dump(mode="json") for c in draft.clarifications]
    return SessionEnvelope(
        session_id=session.id,
        state=session.state.value,
        gloss=draft.gloss,
        sign_language=draft.sign_language,
        domain=draft.domain,
        regional_variant=draft.regional_variant,
        description_prose=draft.description_prose,
        parameters=params_dump,
        gaps=[_gap_out(g) for g in draft.gaps],
        pending_questions=[_question_out(q) for q in draft.pending_questions],
        clarifications=clarifications,
        hamnosys=draft.hamnosys,
        sigml=draft.sigml,
        preview=_preview_for(session, request),
        generation_confidence=draft.generation_confidence,
        generation_errors=list(draft.generation_errors),
        corrections_count=draft.corrections_count,
        history=history,
        created_at=session.created_at,
        last_activity_at=session.last_activity_at,
        next_action=_next_action(session, request),
    )


def _sign_entry_out(entry: SignEntry) -> SignEntryOut:
    return SignEntryOut(
        id=entry.id,
        gloss=entry.gloss,
        sign_language=entry.sign_language,
        domain=entry.domain,
        hamnosys=entry.hamnosys,
        sigml=entry.sigml,
        status=entry.status,
        parameters=entry.parameters.model_dump(mode="json"),
        regional_variant=entry.regional_variant,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


# ---------------------------------------------------------------------------
# Router — one APIRouter, mounted at /api/chat2hamnosys by the app layer.
# ---------------------------------------------------------------------------


router = APIRouter(tags=["chat2hamnosys"])


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=201,
    summary="Create a new authoring session",
)
def create_session(
    request: Request,
    body: Optional[CreateSessionRequest] = None,
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
) -> CreateSessionResponse:
    body = body or CreateSessionRequest()
    signer_id = (body.signer_id or "anonymous").strip() or "anonymous"
    session = start_session(
        signer_id=signer_id,
        display_name=body.display_name,
        is_deaf_native=body.author_is_deaf_native,
        sign_language=body.sign_language,
        domain=body.domain,
        gloss=body.gloss,
    )
    if body.regional_variant is not None:
        session = session.with_draft(regional_variant=body.regional_variant)
    session_store.save(session)
    token = TokenStore.new_token()
    token_store.put(session.id, token)
    return CreateSessionResponse(
        session_id=session.id,
        state=session.state.value,
        session_token=token,
        session=_envelope(session, request),
    )


@router.get(
    "/sessions/{session_id}",
    response_model=SessionEnvelope,
    summary="Fetch the full state of a session",
)
def get_session(
    request: Request,
    session_id: UUID,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
) -> SessionEnvelope:
    session = _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    return _envelope(session, request)


@router.post(
    "/sessions/{session_id}/describe",
    response_model=SessionEnvelope,
    summary="Submit the initial prose description",
)
def post_describe(
    request: Request,
    session_id: UUID,
    body: DescribeRequest,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
    parse_fn=Depends(get_parse_fn),
    question_fn=Depends(get_question_fn),
    generate_fn=Depends(get_generate_fn),
    to_sigml_fn=Depends(get_to_sigml_fn),
    render_fn=Depends(get_render_fn),
) -> SessionEnvelope:
    session = _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    if body.gloss:
        session = session.with_draft(gloss=body.gloss)
    session = on_description(
        session,
        body.prose,
        parse_fn=parse_fn,
        question_fn=question_fn,
    )
    if session.state == SessionState.GENERATING:
        session = run_generation(
            session,
            generate_fn=generate_fn,
            to_sigml_fn=to_sigml_fn,
            render_fn=render_fn,
        )
    session_store.save(session)
    return _envelope(session, request)


@router.post(
    "/sessions/{session_id}/answer",
    response_model=SessionEnvelope,
    summary="Answer one clarification question",
)
def post_answer(
    request: Request,
    session_id: UUID,
    body: AnswerRequest,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
    apply_fn=Depends(get_apply_fn),
    question_fn=Depends(get_question_fn),
    generate_fn=Depends(get_generate_fn),
    to_sigml_fn=Depends(get_to_sigml_fn),
    render_fn=Depends(get_render_fn),
) -> SessionEnvelope:
    session = _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    answer_text = str(body.answer).strip()
    if not answer_text:
        raise ApiError("answer must be a non-empty string or integer",
                       status_code=422, code="validation_error")
    session = on_clarification_answer(
        session,
        body.question_id,
        answer_text,
        apply_fn=apply_fn,
        question_fn=question_fn,
    )
    if session.state == SessionState.GENERATING:
        session = run_generation(
            session,
            generate_fn=generate_fn,
            to_sigml_fn=to_sigml_fn,
            render_fn=render_fn,
        )
    session_store.save(session)
    return _envelope(session, request)


@router.post(
    "/sessions/{session_id}/generate",
    response_model=SessionEnvelope,
    summary="Force a generation attempt",
)
def post_generate(
    request: Request,
    session_id: UUID,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
    generate_fn=Depends(get_generate_fn),
    to_sigml_fn=Depends(get_to_sigml_fn),
    render_fn=Depends(get_render_fn),
) -> SessionEnvelope:
    session = _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    if session.state == SessionState.CLARIFYING:
        if session.draft.parameters_partial is None:
            raise InvalidTransition(
                "no parameters populated yet; describe first",
                details={"state": session.state.value},
            )
        session = session.with_state(SessionState.GENERATING)
    if session.state not in (SessionState.GENERATING, SessionState.APPLYING_CORRECTION):
        raise InvalidTransition(
            f"cannot force generation from state {session.state.value!r}",
            details={"state": session.state.value},
        )
    session = run_generation(
        session,
        generate_fn=generate_fn,
        to_sigml_fn=to_sigml_fn,
        render_fn=render_fn,
    )
    session_store.save(session)
    return _envelope(session, request)


@router.get(
    "/sessions/{session_id}/preview",
    response_model=PreviewOut,
    summary="Return the latest preview metadata",
)
def get_preview(
    request: Request,
    session_id: UUID,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
) -> PreviewOut:
    session = _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    return _preview_for(session, request)


@router.get(
    "/sessions/{session_id}/preview/video",
    summary="Stream the rendered preview video (if available)",
    name="get_preview_video",
)
def get_preview_video(
    request: Request,
    session_id: UUID,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
):
    session = _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    video_path = session.draft.video_path
    if not video_path or not Path(video_path).exists():
        raise ApiError(
            "no rendered preview available for this session",
            status_code=404,
            code="preview_not_available",
        )
    return FileResponse(video_path, media_type="video/mp4")


@router.post(
    "/sessions/{session_id}/correct",
    response_model=SessionEnvelope,
    summary="Submit a correction against the rendered preview",
)
def post_correct(
    request: Request,
    session_id: UUID,
    body: CorrectRequest,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
    generate_fn=Depends(get_generate_fn),
    to_sigml_fn=Depends(get_to_sigml_fn),
    render_fn=Depends(get_render_fn),
) -> SessionEnvelope:
    session = _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    correction = Correction(
        raw_text=body.raw_text,
        target_time_ms=body.target_time_ms,
        target_region=body.target_region,
    )
    session = on_correction(session, correction)
    plan = interpret_correction(session, correction)
    outcome = apply_correction(
        session,
        plan,
        generate_fn=generate_fn,
        to_sigml_fn=to_sigml_fn,
        render_fn=render_fn,
    )
    session = outcome.session
    session_store.save(session)
    return _envelope(session, request)


@router.post(
    "/sessions/{session_id}/accept",
    response_model=AcceptResponse,
    summary="Finalize the session and persist the sign entry",
)
def post_accept(
    request: Request,
    session_id: UUID,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
    sign_store: SQLiteSignStore = Depends(get_sign_store),
) -> AcceptResponse:
    session = _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    session = on_accept(session, store=sign_store)
    session_store.save(session)
    from session.state import AcceptedEvent
    accepted = [e for e in session.history if isinstance(e, AcceptedEvent)]
    if not accepted:
        raise ApiError(
            "accept did not produce an accepted event",
            status_code=500,
            code="internal_error",
        )
    entry = sign_store.get(accepted[-1].sign_entry_id)
    return AcceptResponse(
        sign_entry=_sign_entry_out(entry),
        session=_envelope(session, request),
    )


@router.post(
    "/sessions/{session_id}/reject",
    response_model=SessionEnvelope,
    summary="Abandon the session",
)
def post_reject(
    request: Request,
    session_id: UUID,
    body: Optional[RejectRequest] = None,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
) -> SessionEnvelope:
    session = _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    reason = (body.reason if body else "") or ""
    session = on_reject(session, reason=reason)
    session_store.save(session)
    return _envelope(session, request)


# ---------------------------------------------------------------------------
# Server-sent events stream
# ---------------------------------------------------------------------------


async def _events_generator(
    *,
    session_id: UUID,
    session_store: SessionStore,
    poll_interval: float = EVENTS_POLL_INTERVAL,
    max_duration: float = EVENTS_MAX_DURATION,
) -> AsyncIterator[bytes]:
    """Yield SSE ``data:`` frames for session-history events as they appear.

    Emits a replay of the current history on connect, then polls for
    newly-appended events until the session reaches a terminal state
    or ``max_duration`` elapses.
    """
    seen = 0
    loop = asyncio.get_event_loop()
    deadline = loop.time() + max_duration
    terminal_values = {s.value for s in TERMINAL_STATES}
    while loop.time() < deadline:
        session = session_store.get(session_id)
        if session is None:
            payload = json.dumps(
                {"type": "error", "code": "session_not_found"},
                ensure_ascii=False,
            )
            yield f"event: error\ndata: {payload}\n\n".encode("utf-8")
            return
        history = session.history
        while seen < len(history):
            event = history[seen]
            seen += 1
            data = event.model_dump(mode="json")
            payload = json.dumps(data, ensure_ascii=False, default=str)
            yield f"event: {data.get('type', 'event')}\ndata: {payload}\n\n".encode("utf-8")
        if session.state.value in terminal_values:
            state_payload = json.dumps(
                {"type": "state", "state": session.state.value},
                ensure_ascii=False,
            )
            yield f"event: closed\ndata: {state_payload}\n\n".encode("utf-8")
            return
        # Lightweight keep-alive comment to prevent proxy timeouts.
        yield b": keep-alive\n\n"
        await asyncio.sleep(poll_interval)
    # Timed out — tell the client to reconnect if they still care.
    yield b"event: timeout\ndata: {\"type\":\"timeout\"}\n\n"


@router.get(
    "/sessions/{session_id}/events",
    summary="Server-sent event stream for live session updates",
)
async def get_events(
    request: Request,
    session_id: UUID,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
):
    _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    return StreamingResponse(
        _events_generator(session_id=session_id, session_store=session_store),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["DEFAULT_RATE_LIMIT", "limiter", "router"]

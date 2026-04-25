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

from fastapi import APIRouter, BackgroundTasks, Depends, Header, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from clarify import Question, apply_answer, generate_questions
from llm.client import (
    _REQUEST_OPENAI_API_KEY,
    reset_request_openai_api_key,
    set_request_openai_api_key,
)
from hamnosys.symbols import SYMBOLS, SymClass
from correct import (
    Correction as _InterpCorrection,
    apply_correction,
    apply_swap_to_session,
    interpret_correction,
    match_deterministic_swap,
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
from .contributors import require_contributor
from .errors import ApiError, InvalidTransition, SessionForbidden, SessionNotFound
from .security import (
    sanitize_user_input,
    screen_user_description,
)
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
    ReviewerCommentOut,
    SessionEnvelope,
    SignEntryOut,
    StatusResponse,
)
from .token_store import TokenStore
from review.dependencies import get_reviewer_store
from review.storage import ReviewerStore


logger = logging.getLogger(__name__)


def _default_rate_limit() -> str:
    """Read the current rate-limit config from the environment.

    Re-read on every call so :func:`api.app.create_app` picks up tests'
    ``monkeypatch.setenv`` without needing a module reload.
    """
    return os.environ.get("CHAT2HAMNOSYS_RATE_LIMIT", "30/minute")


def _session_create_rate_limit() -> str:
    """Per-IP cap on session creation.

    Cheap for an attacker to open-and-drop sessions, so we clamp this
    an order of magnitude below the default that covers
    describe/answer/correct. Override with
    ``CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT`` if a fixture needs to
    spin up many sessions in the same test window.
    """
    return os.environ.get(
        "CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT", "2/minute"
    )


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
        generation_path=list(draft.generation_path),
        candidate_hamnosys=draft.candidate_hamnosys,
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
# HamNoSys symbol table — plain-English legend for the contribute UI.
#
# The frontend's notation panel (see public/contribute-notation.js) shows
# the class_label next to whichever glyph the user is hovering. Labels are
# kept deliberately literal so a contributor who doesn't know HamNoSys can
# still tell what kind of thing a symbol is.
# ---------------------------------------------------------------------------


_CLASS_LABELS: dict[SymClass, str] = {
    SymClass.SYMMETRY:        "Symmetry marker",
    SymClass.NONDOM:          "Non-dominant hand marker",
    SymClass.HANDSHAPE_BASE:  "Handshape",
    SymClass.THUMB_MOD:       "Thumb modifier",
    SymClass.FINGER_MOD:      "Finger modifier",
    SymClass.EXT_FINGER_DIR:  "Extended-finger direction",
    SymClass.ORI_RELATIVE:    "Orientation (relative)",
    SymClass.PALM_DIR:        "Palm direction",
    SymClass.LOC_HEAD:        "Location (head)",
    SymClass.LOC_TORSO:       "Location (torso)",
    SymClass.LOC_NEUTRAL:     "Location (neutral space)",
    SymClass.LOC_ARM:         "Location (arm)",
    SymClass.LOC_HAND_ZONE:   "Hand zone",
    SymClass.LOC_FINGER:      "Finger part",
    SymClass.COMBINER:        "Two-hand combiner",
    SymClass.COREF:           "Coreference",
    SymClass.MOVE_STRAIGHT:   "Movement (straight)",
    SymClass.MOVE_CIRCLE:     "Movement (circular)",
    SymClass.MOVE_ACTION:     "Action",
    SymClass.MOVE_CLOCK:      "Clock position",
    SymClass.MOVE_ARC:        "Movement (arc)",
    SymClass.MOVE_WAVY:       "Movement (wavy)",
    SymClass.MOVE_ELLIPSE:    "Movement (elliptical)",
    SymClass.SIZE_MOD:        "Size modifier",
    SymClass.SPEED_MOD:       "Speed modifier",
    SymClass.TIMING:          "Timing",
    SymClass.CONTACT:         "Contact",
    SymClass.REPEAT:          "Repetition",
    SymClass.SEQ_BEGIN:       "Sequence begin",
    SymClass.SEQ_END:         "Sequence end",
    SymClass.PAR_BEGIN:       "Parallel begin",
    SymClass.PAR_END:         "Parallel end",
    SymClass.FUSION_BEGIN:    "Fusion begin",
    SymClass.FUSION_END:      "Fusion end",
    SymClass.JOINER:          "Joiner",
    SymClass.MIME:            "Mime marker",
    SymClass.VERSION:         "Version marker",
    SymClass.ALT_BEGIN:       "Alternative begin",
    SymClass.ALT_END:         "Alternative end",
    SymClass.META_ALT:        "Alternative separator",
    SymClass.PUNCT:           "Punctuation",
    SymClass.OBSOLETE:        "Obsolete symbol",
}


def _build_symbols_payload() -> dict[str, Any]:
    """Serialize the full HamNoSys 4.0 symbol table for the frontend.

    Precomputed at import time because the table is immutable for the
    process lifetime; the response carries ``Cache-Control: immutable``
    so browsers never revalidate.
    """
    entries: list[dict[str, Any]] = []
    for cp, sym in sorted(SYMBOLS.items()):
        entries.append(
            {
                "codepoint":     cp,
                "hex":           f"U+{cp:04X}",
                "char":          chr(cp),
                "short_name":    sym.short_name,
                "latex_command": sym.latex_command,
                "class":         sym.sym_class.value,
                "class_label":   _CLASS_LABELS.get(sym.sym_class, sym.sym_class.value),
                "slots":         sorted(s.value for s in sym.slots),
            }
        )
    return {"schema_version": 1, "count": len(entries), "symbols": entries}


_SYMBOLS_PAYLOAD: dict[str, Any] = _build_symbols_payload()
_SYMBOLS_ETAG: str = f'"hamnosys-4-0-{_SYMBOLS_PAYLOAD["count"]}"'


def _build_sigml_reference_payload() -> dict[str, Any]:
    """Cached SiGML reference catalog for the contribute UI's annotated
    editor + LLM-prompt audit. See ``generator.sigml_reference``.
    """
    from generator.sigml_reference import get_catalog
    payload = get_catalog()
    payload["schema_version"] = 1
    return payload


_SIGML_REFERENCE_PAYLOAD: dict[str, Any] = _build_sigml_reference_payload()
_SIGML_REFERENCE_ETAG: str = (
    f'"sigml-ref-1-{len(_SIGML_REFERENCE_PAYLOAD["entries"])}"'
)


# ---------------------------------------------------------------------------
# Router — one APIRouter, mounted at /api/chat2hamnosys by the app layer.
# ---------------------------------------------------------------------------


router = APIRouter(tags=["chat2hamnosys"])


# ---------------------------------------------------------------------------
# Background-task helpers for long-running LLM work
# ---------------------------------------------------------------------------
#
# ``/correct`` and — on the GENERATING branch — ``/answer`` previously
# ran ``run_generation`` synchronously inside the request handler. With
# gpt-5.4 reasoning that can take several minutes, which reliably blew
# through nginx's ``proxy_read_timeout`` (60s by default) and surfaced
# to the contributor as ``504 Gateway Time-out``. The fix: do the
# *fast* bits inline (apply the diff / apply the answer, move the
# session into APPLYING_CORRECTION or GENERATING, save), then schedule
# ``run_generation`` as a FastAPI BackgroundTask and return the
# envelope immediately. The SSE stream at ``/events`` picks up the new
# history entries (``GeneratedEvent``, ``CorrectionAppliedEvent``) as
# the background task saves the session, and the frontend lights up
# the preview when they land. No client-visible timeout path.


def _run_correction_async(
    *,
    session_id: UUID,
    session_store: SessionStore,
    generate_fn,
    to_sigml_fn,
    render_fn,
    correction: "_InterpCorrection",
    byo_key: str,
) -> None:
    """Background task: interpret the correction + apply + regenerate.

    Loads the session fresh from the store (the in-memory session from
    the request handler is stale by the time this runs). Threads the
    contributor's BYO OpenAI key through the ``_REQUEST_OPENAI_API_KEY``
    contextvar for the duration of the task so ``LLMClient`` picks it
    up the same way the request handler would. Catches every
    exception and writes a ``generation_errors`` entry on the draft —
    the task should never crash the worker.

    Always appends a history event (``GeneratedEvent`` on failure, or
    whatever ``apply_correction`` wrote on success) so the SSE stream
    at ``/events`` has something to deliver. The frontend subscribes
    to both ``generated`` and ``correction_applied`` event types and
    refetches the session on either.
    """
    from session.state import GeneratedEvent
    token = set_request_openai_api_key(byo_key or "")
    try:
        session = session_store.get(session_id)
        if session is None:
            logger.warning("background correct: session %s vanished", session_id)
            return
        try:
            # Deterministic SiGML-rewrite fast-path. Catches structured
            # chip-swap payloads (target_region="swap:from:to") and
            # common natural-language directional changes ("palm should
            # face down") so the contributor's correction lands as a
            # SiGML rewrite + GeneratedEvent without an LLM round-trip.
            # Falls through to the LLM-backed interpreter on a miss.
            swap = match_deterministic_swap(correction, session.draft.sigml)
            if swap is not None:
                session = apply_swap_to_session(session, swap)
                session_store.save(session)
                return
            plan = interpret_correction(session, correction)
            outcome = apply_correction(
                session,
                plan,
                generate_fn=generate_fn,
                to_sigml_fn=to_sigml_fn,
                render_fn=render_fn,
            )
            session = outcome.session
        except Exception as exc:  # noqa: BLE001 — never crash the worker
            logger.exception(
                "background correction failed for session %s: %s",
                session_id,
                exc,
            )
            err = f"background correction crashed: {type(exc).__name__}: {exc}"
            # 1. Park the draft in a recoverable state with the error
            #    surfaced on ``generation_errors`` so the chat panel's
            #    maybeSurfaceGenerationError path picks it up.
            session = session.with_draft(
                generation_errors=[err],
            ).with_state(SessionState.AWAITING_CORRECTION)
            # 2. Emit a GeneratedEvent so SSE delivers a ``generated``
            #    frame — the frontend SSE subscriber refetches the
            #    session on that event and sees the new state + error.
            session = session.append_event(
                GeneratedEvent(success=False, errors=[err])
            )
        session_store.save(session)
    finally:
        reset_request_openai_api_key(token)


def _run_generation_async(
    *,
    session_id: UUID,
    session_store: SessionStore,
    generate_fn,
    to_sigml_fn,
    render_fn,
    byo_key: str,
) -> None:
    """Background task: run generation for a session already moved to
    GENERATING. Used by ``/answer`` when the last clarification answer
    exhausts the gaps.

    ``run_generation`` already appends a ``GeneratedEvent`` to session
    history on both success and internal failure. We add a second
    GeneratedEvent on hard exceptions so the SSE stream delivers
    something even when ``run_generation`` itself crashed.
    """
    from session.state import GeneratedEvent
    token = set_request_openai_api_key(byo_key or "")
    try:
        session = session_store.get(session_id)
        if session is None:
            logger.warning("background generate: session %s vanished", session_id)
            return
        try:
            session = run_generation(
                session,
                generate_fn=generate_fn,
                to_sigml_fn=to_sigml_fn,
                render_fn=render_fn,
            )
        except Exception as exc:  # noqa: BLE001 — never crash the worker
            logger.exception(
                "background generation failed for session %s: %s",
                session_id,
                exc,
            )
            err = f"background generation crashed: {type(exc).__name__}: {exc}"
            session = session.with_draft(
                generation_errors=[err],
            ).with_state(SessionState.AWAITING_DESCRIPTION)
            session = session.append_event(
                GeneratedEvent(success=False, errors=[err])
            )
        session_store.save(session)
    finally:
        reset_request_openai_api_key(token)


@router.get(
    "/hamnosys/symbols",
    summary="Full HamNoSys 4.0 symbol table with plain-English class labels",
)
def get_hamnosys_symbols(
    if_none_match: Optional[str] = Header(default=None, alias="If-None-Match"),
) -> JSONResponse:
    """Return the immutable HamNoSys 4.0 symbol inventory.

    The table is static for the life of the process (and in practice for
    the life of the build), so responses carry an immutable cache header
    and a stable ETag — browsers revalidate once, then never again.
    """
    headers = {
        "Cache-Control": "public, max-age=31536000, immutable",
        "ETag":          _SYMBOLS_ETAG,
    }
    if if_none_match and if_none_match.strip() == _SYMBOLS_ETAG:
        return JSONResponse(content=None, status_code=304, headers=headers)
    return JSONResponse(content=_SYMBOLS_PAYLOAD, headers=headers)


@router.get(
    "/reference/sigml",
    summary="Full SiGML tag catalog with per-tag semantic roles",
)
def get_sigml_reference(
    if_none_match: Optional[str] = Header(default=None, alias="If-None-Match"),
) -> JSONResponse:
    """Return the SiGML reference catalog the contribute UI uses for the
    interactive annotated editor.

    Each entry carries the canonical tag name, PUA codepoint, slot
    category (handshape, palm direction, etc.), and a plain-English
    role description. The frontend uses ``by_category`` to populate
    the click-to-swap picker that appears next to each tag in a
    sign's SiGML, so contributors can browse alternatives without
    knowing the HamNoSys notation system.

    Static for the life of the build — same caching as
    ``/hamnosys/symbols``.
    """
    headers = {
        "Cache-Control": "public, max-age=31536000, immutable",
        "ETag":          _SIGML_REFERENCE_ETAG,
    }
    if if_none_match and if_none_match.strip() == _SIGML_REFERENCE_ETAG:
        return JSONResponse(content=None, status_code=304, headers=headers)
    return JSONResponse(content=_SIGML_REFERENCE_PAYLOAD, headers=headers)


@router.post(
    "/sessions",
    response_model=CreateSessionResponse,
    status_code=201,
    summary="Create a new authoring session",
)
@limiter.limit(_session_create_rate_limit)
def create_session(
    request: Request,
    body: Optional[CreateSessionRequest] = None,
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
    contributor_id: Optional[str] = Depends(require_contributor),
) -> CreateSessionResponse:
    from security import hash_signer_id
    from .security import get_security_config

    body = body or CreateSessionRequest()
    if contributor_id:
        # Authenticated contributor — use their registry id as the
        # signer_id and skip the legacy hashing step (the id is already
        # an opaque uuid4, not PII).
        signer_id = contributor_id
    else:
        raw_signer = (body.signer_id or "anonymous").strip() or "anonymous"
        cfg = get_security_config()
        if cfg.pii_policy == "hashed" and raw_signer != "anonymous":
            signer_id = hash_signer_id(raw_signer, salt=cfg.signer_id_salt)
        else:
            signer_id = raw_signer
    # Reject a client-supplied session_id that collides with an existing
    # row — otherwise ``start_session`` would silently overwrite a draft
    # the previous owner is mid-edit on. The contribute UI uses
    # ``crypto.randomUUID()`` so a collision here is effectively zero,
    # but a malicious client could try one and we should not honour it.
    if body.session_id is not None and session_store.get(body.session_id) is not None:
        raise ApiError(
            f"session id {body.session_id} already exists",
            status_code=409,
            code="session_id_conflict",
        )
    session = start_session(
        signer_id=signer_id,
        display_name=body.display_name,
        is_deaf_native=body.author_is_deaf_native,
        sign_language=body.sign_language,
        domain=body.domain,
        gloss=body.gloss,
        session_id=body.session_id,
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
    prose = sanitize_user_input(body.prose, field_name="prose")
    screen_user_description(prose, request=request, field_name="prose")
    if body.gloss:
        session = session.with_draft(gloss=sanitize_user_input(body.gloss, field_name="gloss", max_len=200))
    session = on_description(
        session,
        prose,
        parse_fn=parse_fn,
        question_fn=question_fn,
    )
    if session.state == SessionState.GENERATING:
        # ``run_generation`` already converts internal generator failures
        # (empty hamnosys, SiGML conversion error) into a failed
        # ``GeneratedEvent`` and a recovery state. A hard exception out
        # of the helper itself (e.g. an unexpected RuntimeError, or a
        # bug in a stub) would otherwise propagate past
        # ``session_store.save`` below and leave the session at
        # GENERATING with no SSE frame for the client. Catch it here
        # and append a failed event so the SSE channel always delivers
        # a ``generated`` frame the chat panel can render.
        try:
            session = run_generation(
                session,
                generate_fn=generate_fn,
                to_sigml_fn=to_sigml_fn,
                render_fn=render_fn,
            )
        except Exception as exc:  # noqa: BLE001 — never strand the session in GENERATING
            from session.state import GeneratedEvent
            logger.exception(
                "post_describe: run_generation raised for session %s",
                session.id,
            )
            err = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
            session = session.append_event(
                GeneratedEvent(success=False, errors=[err])
            )
            session = session.with_draft(
                generation_errors=[err],
                preview_status="generation_failed",
                preview_message=err[:500],
            ).with_state(SessionState.AWAITING_DESCRIPTION)
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
    background_tasks: BackgroundTasks,
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
    # Sanitize but do not injection-screen — answers are short, often
    # single-word (e.g. "up", "flat") and the screen's false-positive
    # cost outweighs the detection yield on those inputs.
    answer_text = sanitize_user_input(answer_text, field_name="answer", max_len=500)
    question_id = sanitize_user_input(body.question_id, field_name="question_id", max_len=128)
    session = on_clarification_answer(
        session,
        question_id,
        answer_text,
        apply_fn=apply_fn,
        question_fn=question_fn,
    )
    # If the orchestrator transitioned us into GENERATING (the last
    # clarification has been answered), move the long-running
    # generator call into a background task. See ``_run_generation_async``
    # for rationale; same motivation as ``/correct``: never block the
    # HTTP response on a multi-minute reasoning call, let SSE deliver
    # the result.
    if session.state == SessionState.GENERATING:
        session_store.save(session)
        byo_key = _REQUEST_OPENAI_API_KEY.get("") or ""
        background_tasks.add_task(
            _run_generation_async,
            session_id=session.id,
            session_store=session_store,
            generate_fn=generate_fn,
            to_sigml_fn=to_sigml_fn,
            render_fn=render_fn,
            byo_key=byo_key,
        )
        return _envelope(session, request)
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
    # Accept CLARIFYING and the two recovery states (AWAITING_DESCRIPTION
    # and AWAITING_CORRECTION) when the session already has populated
    # parameters. run_generation lands the session in AWAITING_DESCRIPTION
    # on a first-pass failure and AWAITING_CORRECTION on a correction-apply
    # failure; both are "retry points" that the frontend's auto-retry
    # leans on after a transient LLM stumble. Without this relaxation
    # /generate 409s and the contributor only gets the sign back by
    # reloading the page, which is exactly what the auto-retry was
    # introduced to avoid.
    retry_entry_states = (
        SessionState.CLARIFYING,
        SessionState.AWAITING_DESCRIPTION,
        SessionState.AWAITING_CORRECTION,
    )
    if session.state in retry_entry_states:
        if session.draft.parameters_partial is None:
            raise InvalidTransition(
                "no parameters populated yet; describe first",
                details={"state": session.state.value},
            )
        next_state = (
            SessionState.APPLYING_CORRECTION
            if session.state == SessionState.AWAITING_CORRECTION
            else SessionState.GENERATING
        )
        session = session.with_state(next_state)
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
    background_tasks: BackgroundTasks,
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
    raw_text = sanitize_user_input(body.raw_text, field_name="raw_text")
    screen_user_description(raw_text, request=request, field_name="raw_text")
    target_region = (
        sanitize_user_input(body.target_region, field_name="target_region", max_len=64)
        if body.target_region
        else None
    )
    # Structured chip-swap: prefer the typed ``swap`` field (sent by the
    # annotated SiGML editor) over a target_region encoding, but accept
    # either. Encoding into target_region preserves backward-compat for
    # any client that hasn't been updated to the new payload shape.
    if body.swap is not None:
        target_region = (
            f"swap:{body.swap.from_tag}:{body.swap.to_tag}"
            if body.swap.index is None
            else f"swap:{body.swap.from_tag}:{body.swap.to_tag}:{body.swap.index}"
        )
    correction = Correction(
        raw_text=raw_text,
        target_time_ms=body.target_time_ms,
        target_region=target_region,
    )
    # Move the session into APPLYING_CORRECTION inline so the envelope
    # the contributor receives immediately shows "applying correction"
    # state. The interpret + generate work runs in the background —
    # the response returns the moment the state transition is saved,
    # which guarantees we don't burn through nginx's 60s proxy ceiling
    # while the LLM thinks.
    session = on_correction(session, correction)
    session_store.save(session)

    # Deterministic SiGML rewrite fast-path: chip swaps and common
    # natural-language directional changes don't need an LLM. Apply
    # them inline so the response envelope already carries the new
    # SiGML and the SSE stream gets the ``generated`` frame in the
    # same request cycle. Falls through to the background interpret
    # path on a miss.
    swap_decision = match_deterministic_swap(correction, session.draft.sigml)
    if swap_decision is not None:
        try:
            session = apply_swap_to_session(session, swap_decision)
            session_store.save(session)
            return _envelope(session, request)
        except Exception:  # noqa: BLE001 — fall through to LLM path on rewrite failure
            logger.exception(
                "deterministic SiGML swap failed for session %s; falling "
                "back to LLM interpreter",
                session.id,
            )
            # Roll the session back to APPLYING_CORRECTION so the
            # background path picks up where we left off.
            session = session_store.get(session.id) or session

    byo_key = _REQUEST_OPENAI_API_KEY.get("") or ""
    background_tasks.add_task(
        _run_correction_async,
        session_id=session.id,
        session_store=session_store,
        generate_fn=generate_fn,
        to_sigml_fn=to_sigml_fn,
        render_fn=render_fn,
        correction=correction,
        byo_key=byo_key,
    )
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
    force: bool = Query(
        default=False,
        description=(
            "Submit-as-is escape hatch: when true, accept the draft even "
            "if generation failed. Draft is saved with status=\"draft\" "
            "and a placeholder HamNoSys so the reviewer can complete it "
            "manually from the description prose."
        ),
    ),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
    sign_store: SQLiteSignStore = Depends(get_sign_store),
    reviewer_store: ReviewerStore = Depends(get_reviewer_store),
) -> AcceptResponse:
    session = _load_session(
        session_id,
        store=session_store,
        x_session_token=x_session_token,
        token_store=token_store,
    )
    if force:
        # Submit-as-is path: relax the RENDERED requirement and patch
        # the draft with a placeholder HamNoSys if generation failed.
        # The contributor explicitly chose to hand this to the
        # reviewer incomplete — we persist whatever we've got so the
        # reviewer can complete it from the description prose, rather
        # than losing the draft entirely.
        if session.state in (SessionState.FINALIZED, SessionState.ABANDONED):
            raise InvalidTransition(
                "cannot submit an already-closed session",
                details={"state": session.state.value},
            )
        if not session.draft.gloss:
            raise InvalidTransition(
                "submit-as-is requires a gloss on the draft",
                details={"state": session.state.value},
            )
        if not session.draft.hamnosys:
            # Minimal placeholder so ``to_sign_entry`` succeeds.
            # hamflathand (U+E001) + hamextfingeru (U+E020) +
            # hampalmd (U+E03C) + hamneutralspace (U+E05F) — four
            # mandatory slots, canonical order. Reviewers see a
            # "generation failed; please complete" note in the
            # description_prose and the bland flat-hand placeholder.
            placeholder_hs = chr(0xE001)
            placeholder_ext = chr(0xE020)
            placeholder_palm = chr(0xE03C)
            placeholder_loc = chr(0xE05F)
            placeholder = placeholder_hs + placeholder_ext + placeholder_palm + placeholder_loc
            # ``_build_sign_parameters`` also requires slot_codepoints
            # for each of the four mandatory slots; populate them
            # directly so ``to_sign_entry`` doesn't raise.
            existing_slots = dict(session.draft.slot_codepoints or {})
            existing_slots.setdefault("handshape_dominant", placeholder_hs)
            existing_slots.setdefault("orientation_extended_finger", placeholder_ext)
            existing_slots.setdefault("orientation_palm", placeholder_palm)
            existing_slots.setdefault("location", placeholder_loc)
            # Clear any movement segments in the partial so
            # ``_build_sign_parameters`` doesn't demand per-segment
            # path codepoints we don't have.
            from parser.models import PartialSignParameters as _PSP
            cleaned_partial = (
                session.draft.parameters_partial.model_copy(update={"movement": []})
                if session.draft.parameters_partial is not None
                else _PSP()
            )
            session = session.with_draft(
                hamnosys=placeholder,
                sigml=session.draft.sigml or "",
                slot_codepoints=existing_slots,
                parameters_partial=cleaned_partial,
                preview_status="submitted_incomplete",
                preview_message=(
                    "Submitted without generator output; reviewer to "
                    "complete from description prose."
                ),
            )
        session = session.with_state(SessionState.RENDERED)
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
    # Promote draft → pending_review when a reviewer competent for this
    # sign's language is on the roster. See
    # docs/contribute-redesign/10-backend-gaps.md for the rationale.
    # Defensive: if the reviewer DB is unavailable the accept still
    # succeeds as a plain draft.
    if entry.status == "draft":
        try:
            for reviewer in reviewer_store.list(only_active=True):
                if reviewer.can_review(entry.sign_language, entry.regional_variant):
                    entry.status = "pending_review"
                    sign_store.put(entry)
                    break
        except Exception:
            logger.warning(
                "reviewer-roster check failed during accept; leaving entry as draft",
                exc_info=True,
            )
    return AcceptResponse(
        sign_entry=_sign_entry_out(entry),
        session=_envelope(session, request),
    )


def _status_response(
    *,
    session: AuthoringSession,
    sign_store: SQLiteSignStore,
    has_token: bool,
) -> StatusResponse:
    """Build the StatusResponse for a session that has produced a SignEntry.

    Shared between :func:`get_status` (session-id path) and
    :func:`get_signs_by_token` (token path) so the public/private
    field-gating policy stays in one place. The caller is responsible
    for the 404s on "session missing" and "session not yet submitted".
    """
    from session.state import AcceptedEvent

    accepted = [e for e in session.history if isinstance(e, AcceptedEvent)]
    if not accepted:
        raise ApiError(
            "session has not been submitted",
            status_code=404,
            code="not_submitted",
        )
    entry = sign_store.get(accepted[-1].sign_entry_id)
    is_validated = entry.status == "validated"
    # The contributor can always see the sign they submitted — token or
    # not — because they already pasted the URL somewhere. Before
    # validation, only the contributor (with the token) should see the
    # draft notation. After validation the notation is public anyway.
    show_notation = is_validated or has_token

    rejection_category: Optional[str] = None
    if entry.status == "rejected":
        for r in reversed(list(entry.reviewers)):
            if r.verdict == "rejected" and r.category:
                rejection_category = str(r.category)
                break

    reviewer_comments: list[ReviewerCommentOut] = []
    description_prose: Optional[str] = None
    if has_token:
        description_prose = entry.description_prose or ""
        for r in entry.reviewers:
            comment_text = (r.comment or r.notes or "").strip()
            if not comment_text:
                continue
            reviewer_comments.append(
                ReviewerCommentOut(
                    verdict=str(r.verdict),
                    category=str(r.category) if r.category else None,
                    comment=comment_text,
                    reviewed_at=r.reviewed_at,
                )
            )

    return StatusResponse(
        session_id=session.id,
        sign_id=entry.id,
        gloss=entry.gloss,
        sign_language=entry.sign_language,
        regional_variant=entry.regional_variant,
        status=str(entry.status),
        hamnosys=entry.hamnosys if show_notation else None,
        sigml=entry.sigml if show_notation else None,
        rejection_category=rejection_category,
        description_prose=description_prose,
        reviewer_comments=reviewer_comments,
        corrections_count=session.draft.corrections_count,
        has_token=has_token,
        created_at=entry.created_at,
        updated_at=entry.updated_at,
    )


@router.get(
    "/sessions/{session_id}/status",
    response_model=StatusResponse,
    summary="Token-optional submission status for the contributor-facing URL",
)
def get_status(
    session_id: UUID,
    x_session_token: Optional[str] = Header(default=None),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
    sign_store: SQLiteSignStore = Depends(get_sign_store),
) -> StatusResponse:
    """Public read path for ``/contribute/status/<session_id>``.

    Unlike :func:`get_session`, this endpoint does not 403 on a missing
    token — the status URL is designed to be shareable. With a valid
    ``X-Session-Token`` the response folds in the contributor's prose
    description and reviewer comments; without one, only the public
    envelope is returned.
    """
    session = session_store.get(session_id)
    if session is None:
        raise ApiError(
            f"session {session_id} not found",
            status_code=404,
            code="session_not_found",
        )
    has_token = bool(x_session_token) and token_store.verify(
        session_id, x_session_token
    )
    return _status_response(
        session=session, sign_store=sign_store, has_token=has_token
    )


@router.get(
    "/signs/by-token/{token}",
    response_model=StatusResponse,
    summary="Stateless lookup of the sign entry a session token last produced",
)
def get_signs_by_token(
    token: str,
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
    sign_store: SQLiteSignStore = Depends(get_sign_store),
) -> StatusResponse:
    """Resolve a session token directly to the SignEntry it produced.

    Per ``docs/contrib-fix/01-audit.md`` § 6 (Option A): the contribute
    UI persists ``{ uuid, token, lastUpdated }`` to localStorage so a
    contributor's status / dashboard pages can fetch their submission
    without remembering the server-side session id. Token presence
    is itself the authorization grant — ``has_token`` is always
    ``True`` on the response, and the ``description_prose`` and
    reviewer comments come back populated. 404 on either an unknown
    token or a session that has not yet emitted an ``AcceptedEvent``.
    The 404 envelope matches the existing ``error.code`` shape
    (``token_not_found`` / ``not_submitted``) so the client renderer
    in ``contribute-me.js`` and ``contribute-status.js`` can branch
    on it.
    """
    session_id = token_store.find_session_by_token(token)
    if session_id is None:
        raise ApiError(
            "no session is bound to this token",
            status_code=404,
            code="token_not_found",
        )
    session = session_store.get(session_id)
    if session is None:
        raise ApiError(
            f"session {session_id} not found",
            status_code=404,
            code="session_not_found",
        )
    return _status_response(
        session=session, sign_store=sign_store, has_token=True
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
    start_after: int = 0,
    poll_interval: float = EVENTS_POLL_INTERVAL,
    max_duration: float = EVENTS_MAX_DURATION,
) -> AsyncIterator[bytes]:
    """Yield SSE ``data:`` frames for session-history events as they appear.

    Emits a replay of the current history on connect, then polls for
    newly-appended events until the session reaches a terminal state
    or ``max_duration`` elapses.

    Each frame carries an ``id:`` line whose value is the zero-based
    history index. Clients can resume via ``Last-Event-ID`` so a
    reconnect after a network blip does not cause duplicate deliveries.
    """
    seen = max(0, start_after)
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
            event_id = seen
            seen += 1
            data = event.model_dump(mode="json")
            payload = json.dumps(data, ensure_ascii=False, default=str)
            yield (
                f"id: {event_id}\n"
                f"event: {data.get('type', 'event')}\n"
                f"data: {payload}\n\n"
            ).encode("utf-8")
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


def _parse_last_event_id(raw: Optional[str]) -> int:
    """Interpret a ``Last-Event-ID`` header as a non-negative int.

    Malformed values fall back to ``0`` so a broken client replays the
    full history rather than getting stuck.
    """
    if not raw:
        return 0
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return 0
    return value + 1 if value >= 0 else 0


@router.get(
    "/sessions/{session_id}/events",
    summary="Server-sent event stream for live session updates",
)
async def get_events(
    request: Request,
    session_id: UUID,
    x_session_token: Optional[str] = Header(default=None),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
    session_store: SessionStore = Depends(get_session_store),
    token_store: TokenStore = Depends(get_token_store),
):
    # EventSource can't send custom headers, so accept the session
    # token via a query-param fallback too. Header still wins when
    # both are present (matches every other authenticated route).
    token = x_session_token or request.query_params.get("token")
    _load_session(
        session_id,
        store=session_store,
        x_session_token=token,
        token_store=token_store,
    )
    start_after = _parse_last_event_id(last_event_id)
    return StreamingResponse(
        _events_generator(
            session_id=session_id,
            session_store=session_store,
            start_after=start_after,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


__all__ = ["DEFAULT_RATE_LIMIT", "limiter", "router"]

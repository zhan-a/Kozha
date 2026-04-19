"""Admin, metrics, and health endpoints for chat2hamnosys.

Mounted by :func:`api.app.create_app` alongside the authoring router.

Routes
------
- ``GET /metrics`` — Prometheus exposition (no auth; internal by convention).
- ``GET /admin/dashboard`` — HTML operations overview.
- ``GET /admin/sessions/{session_id}`` — per-session trace HTML.
- ``GET /admin/cost`` — 30-day cost breakdown HTML.
- ``GET /health`` — trivial liveness probe; always 200 if the process is up.
- ``GET /health/ready`` — readiness probe; pokes dependencies.

Auth
----
``/metrics`` is unauthenticated. Deployments scrape it from an internal
network and any external exposure should be fronted by a firewall rule
or an ingress ACL. The ``/admin/*`` pages require the same
``X-Reviewer-Token`` header as the Deaf-reviewer router, restricted
further to board reviewers. Health endpoints require no auth.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, Header
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from obs import dashboard as dashboard_mod
from obs import metrics as metrics_mod
from obs.logger import DEFAULT_LOG_DIR, EventLogger, get_logger

from review.models import Reviewer
from review.storage import ReviewerAuthError, ReviewerStore

from .dependencies import get_sign_store, get_session_store, get_token_store
from .errors import ApiError


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def get_event_logger() -> EventLogger:
    """Return the process-wide :class:`EventLogger`."""
    return get_logger()


def get_log_dir() -> Path:
    """Resolve the log directory the admin pages should read."""
    raw = os.environ.get("CHAT2HAMNOSYS_LOG_DIR", "").strip()
    return Path(raw) if raw else DEFAULT_LOG_DIR


def require_board_reviewer(
    x_reviewer_token: Optional[str] = Header(default=None),
) -> Reviewer:
    """Enforce the ``is_board`` flag on admin pages.

    Lazy-imports :func:`review.dependencies.get_reviewer_store` so this
    module is safe to import before the review dependency singletons
    have been primed.
    """
    from review.dependencies import get_reviewer_store

    store: ReviewerStore = get_reviewer_store()
    try:
        reviewer = store.authenticate(x_reviewer_token)
    except ReviewerAuthError as exc:
        raise ApiError(
            str(exc), status_code=403, code="reviewer_forbidden"
        )
    if not reviewer.is_board:
        raise ApiError(
            "admin pages are restricted to governance-board reviewers",
            status_code=403,
            code="board_only",
        )
    return reviewer


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(tags=["admin"])


# ---- Metrics ----------------------------------------------------------------


@router.get(
    "/metrics",
    response_class=PlainTextResponse,
    summary="Prometheus exposition for internal scrapers",
)
def get_metrics() -> PlainTextResponse:
    body = metrics_mod.render_text()
    return PlainTextResponse(
        content=body,
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ---- Dashboard --------------------------------------------------------------


@router.get(
    "/admin/dashboard",
    response_class=HTMLResponse,
    summary="Operations dashboard (board reviewers only)",
)
def get_admin_dashboard(
    _board: Reviewer = Depends(require_board_reviewer),
    event_logger: EventLogger = Depends(get_event_logger),
    log_dir: Path = Depends(get_log_dir),
) -> HTMLResponse:
    html = dashboard_mod.render_dashboard(logger=event_logger, log_dir=log_dir)
    return HTMLResponse(content=html)


@router.get(
    "/admin/sessions/{session_id}",
    response_class=HTMLResponse,
    summary="Per-session event trace (board reviewers only)",
)
def get_admin_session_trace(
    session_id: UUID,
    _board: Reviewer = Depends(require_board_reviewer),
    event_logger: EventLogger = Depends(get_event_logger),
    log_dir: Path = Depends(get_log_dir),
) -> HTMLResponse:
    html = dashboard_mod.render_session_trace(
        str(session_id), logger=event_logger, log_dir=log_dir
    )
    return HTMLResponse(content=html)


@router.get(
    "/admin/cost",
    response_class=HTMLResponse,
    summary="30-day cost breakdown + projection (board reviewers only)",
)
def get_admin_cost(
    _board: Reviewer = Depends(require_board_reviewer),
    log_dir: Path = Depends(get_log_dir),
) -> HTMLResponse:
    html = dashboard_mod.render_cost_report(log_dir=log_dir)
    return HTMLResponse(content=html)


# ---- Health ----------------------------------------------------------------


@router.get(
    "/health",
    summary="Liveness probe — 200 while the app process is up",
)
def get_health() -> JSONResponse:
    body: dict[str, str] = {"status": "ok"}
    sha = os.environ.get("BUILD_SHA", "").strip()
    if sha:
        body["build_sha"] = sha
    return JSONResponse(body)


@router.get(
    "/health/ready",
    summary="Readiness probe — verifies DBs and LLM configuration",
)
def get_ready() -> JSONResponse:
    """Returns 200 only if every hard dependency can be reached.

    - session store: open a connection and run a trivial query.
    - sign store: same.
    - LLM client: check the OPENAI_API_KEY env var is set; no network call.
    """
    checks: dict[str, dict[str, object]] = {}
    all_ok = True

    # Session store
    try:
        store = get_session_store()
        store.count()
        checks["session_store"] = {"ok": True}
    except Exception as exc:
        all_ok = False
        checks["session_store"] = {"ok": False, "error": type(exc).__name__}

    # Sign store
    try:
        sign_store = get_sign_store()
        list(sign_store.list())
        checks["sign_store"] = {"ok": True}
    except Exception as exc:
        all_ok = False
        checks["sign_store"] = {"ok": False, "error": type(exc).__name__}

    # Token store
    try:
        token_store = get_token_store()
        token_store.stats()
        checks["token_store"] = {"ok": True}
    except Exception as exc:
        all_ok = False
        checks["token_store"] = {"ok": False, "error": type(exc).__name__}

    # LLM client — check for api-key configuration without contacting OpenAI.
    has_key = bool(os.environ.get("OPENAI_API_KEY", "").strip())
    checks["llm_config"] = {"ok": has_key}
    if not has_key:
        all_ok = False

    status = 200 if all_ok else 503
    return JSONResponse(
        {"status": "ok" if all_ok else "degraded", "checks": checks},
        status_code=status,
    )


__all__ = ["get_event_logger", "get_log_dir", "require_board_reviewer", "router"]

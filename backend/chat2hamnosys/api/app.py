"""FastAPI app factory for the chat2hamnosys HTTP surface.

:func:`create_app` produces a self-contained :class:`FastAPI` instance
that exposes the authoring orchestrator under the configured prefix
(``"" `` by default — the main Kozha server mounts this sub-app at
``/api/chat2hamnosys``).

Responsibilities
----------------
- Wire CORS from ``CHAT2HAMNOSYS_CORS_ORIGINS`` (comma-separated list;
  ``*`` means permissive — the dev default).
- Attach the :mod:`slowapi` :class:`Limiter` to ``app.state`` and its
  middleware so the ``@limiter.limit(...)`` decorators in
  :mod:`.router` are actually enforced.
- Register the error handlers from :mod:`.errors` so every route
  returns the canonical ``{"error": {code, message, details}}`` shape.
- Include the router at the supplied ``api_prefix`` (empty by default).
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

from llm.client import reset_request_openai_api_key, set_request_openai_api_key
from review.router import router as review_router

from .admin import router as admin_router
from .contributors import router as contributors_router
from .errors import register_error_handlers
from .router import _default_rate_limit, router


CORS_ENV_VAR = "CHAT2HAMNOSYS_CORS_ORIGINS"


class _ByoOpenAIKeyMiddleware(BaseHTTPMiddleware):
    """Thread the ``X-OpenAI-Api-Key`` header into an async-local contextvar.

    Pre-launch scaffolding: until the project's ``OPENAI_API_KEY`` secret
    is provisioned on the host, contributors can paste a personal key
    on ``/contribute.html``. The browser sends it as a header on every
    authoring call; this middleware parks it in a contextvar that
    :class:`llm.client.LLMClient` consults before falling back to the
    env var. Each request runs in its own asyncio task, so there is no
    cross-request leakage. The key is never logged.
    """

    async def dispatch(self, request, call_next):
        key = request.headers.get("x-openai-api-key", "")
        token = set_request_openai_api_key(key)
        try:
            return await call_next(request)
        finally:
            reset_request_openai_api_key(token)


def _parse_origins(raw: Optional[str]) -> list[str]:
    """Split a comma-delimited origins list; ``*`` stays as a single entry."""
    if raw is None:
        return ["*"]
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def create_app(
    *,
    api_prefix: str = "",
    cors_origins: Optional[Iterable[str]] = None,
    rate_limit: Optional[str] = None,
    title: str = "chat2hamnosys",
    description: str = "Authoring HTTP surface for the chat2hamnosys pipeline.",
) -> FastAPI:
    """Build and return a fully-wired :class:`FastAPI` sub-app.

    Parameters
    ----------
    api_prefix:
        Path prefix for every route. Use ``""`` when mounting the
        returned app under a parent mount point (e.g.
        ``app.mount("/api/chat2hamnosys", sub)``) so the final URL is
        not double-prefixed.
    cors_origins:
        Explicit override for allowed origins. ``None`` (the default)
        reads ``CHAT2HAMNOSYS_CORS_ORIGINS`` and falls back to ``["*"]``.
    rate_limit:
        slowapi-style rate string (e.g. ``"30/minute"``). ``None``
        reads ``CHAT2HAMNOSYS_RATE_LIMIT`` every call so tests can
        ``monkeypatch.setenv`` before :func:`create_app`.
    """
    app = FastAPI(title=title, description=description)

    # --- CORS ---------------------------------------------------------
    if cors_origins is None:
        origins = _parse_origins(os.environ.get(CORS_ENV_VAR))
    else:
        origins = list(cors_origins) or ["*"]
    allow_credentials = "*" not in origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=[
            "Content-Type",
            "X-Session-Token",
            "X-Contributor-Token",
            "X-OpenAI-Api-Key",
        ],
    )
    app.add_middleware(_ByoOpenAIKeyMiddleware)

    # --- Rate limiting ------------------------------------------------
    # Build a fresh :class:`Limiter` per app so the limit string is
    # honored without reimporting the router module; :class:`SlowAPIMiddleware`
    # reads ``request.app.state.limiter`` and enforces ``default_limits``
    # against every route automatically (no per-route decorators needed).
    effective_limit = rate_limit or _default_rate_limit()
    app_limiter = Limiter(
        key_func=get_remote_address, default_limits=[effective_limit]
    )
    app.state.limiter = app_limiter
    app.add_middleware(SlowAPIMiddleware)

    # --- Error envelopes ---------------------------------------------
    register_error_handlers(app)

    # --- Routes -------------------------------------------------------
    app.include_router(router, prefix=api_prefix)
    # Review workflow endpoints — mounted under ``/review`` so the
    # public URL becomes ``/api/chat2hamnosys/review/...`` once the
    # parent server mounts this sub-app at ``/api/chat2hamnosys``.
    app.include_router(review_router, prefix=f"{api_prefix}/review")
    # Contributor on-ramp: captcha + registration. Public URL becomes
    # ``/api/chat2hamnosys/contribute/captcha`` and ``.../register``.
    app.include_router(contributors_router, prefix=api_prefix)
    # Observability surface: /metrics, /admin/*, /health at the same
    # mount prefix so the parent server exposes them at
    # ``/api/chat2hamnosys/metrics`` and ``/api/chat2hamnosys/health``.
    app.include_router(admin_router, prefix=api_prefix)

    return app


__all__ = ["CORS_ENV_VAR", "create_app"]

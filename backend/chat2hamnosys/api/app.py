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

from .errors import register_error_handlers
from .router import _default_rate_limit, router


CORS_ENV_VAR = "CHAT2HAMNOSYS_CORS_ORIGINS"


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
        allow_headers=["Content-Type", "X-Session-Token"],
    )

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

    return app


__all__ = ["CORS_ENV_VAR", "create_app"]

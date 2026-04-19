"""Error envelope and exception handlers for the chat2hamnosys API.

Every failure response carries the same shape::

    {"error": {"code": str, "message": str, "details": dict | null}}

:class:`ApiError` is the internal contract — handlers wrap known
exception types into one. The mapping table :data:`STATUS_BY_CODE`
documents which code turns into which HTTP status.

Known exception translations
----------------------------
- :class:`LLMConfigError` → 500 (server misconfigured)
- :class:`BudgetExceeded`  → 429 (per-session spend cap hit)
- :class:`ValueError` / pydantic :class:`ValidationError` → 422
- :class:`SessionNotFoundError` → 404
- :class:`InvalidTransitionError` → 409
- :class:`slowapi.RateLimitExceeded` → 429
- Anything else → 500 ``internal_error``
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from llm import BudgetExceeded, LLMConfigError
from session.orchestrator import InvalidTransitionError


logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    """One error — the envelope's inner object."""

    code: str
    message: str
    details: Optional[dict[str, Any]] = None


class ErrorResponse(BaseModel):
    """Canonical failure-response envelope."""

    error: ErrorDetail


# ---------------------------------------------------------------------------
# Internal exception types
# ---------------------------------------------------------------------------


class ApiError(Exception):
    """Base class for chat2hamnosys API errors with an HTTP mapping.

    Direct ``raise ApiError(...)`` in a route body is the escape hatch
    for ad-hoc failures; the typed subclasses below cover the common
    cases so handlers can be exhaustive in tests.
    """

    status_code: int = 500
    code: str = "internal_error"

    def __init__(
        self,
        message: str,
        *,
        details: Optional[dict[str, Any]] = None,
        status_code: Optional[int] = None,
        code: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class SessionNotFound(ApiError):
    """Requested session id is not in the store."""

    status_code = 404
    code = "session_not_found"


class SessionForbidden(ApiError):
    """Client supplied a session-token that did not match the stored one."""

    status_code = 403
    code = "session_forbidden"


class InvalidTransition(ApiError):
    """Session is in a state that does not permit the requested transition."""

    status_code = 409
    code = "invalid_transition"


# ---------------------------------------------------------------------------
# Response builder
# ---------------------------------------------------------------------------


def _envelope(code: str, message: str, details: Optional[dict[str, Any]] = None) -> dict:
    payload: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        payload["details"] = details
    else:
        payload["details"] = None
    return {"error": payload}


def _response(status: int, code: str, message: str, details: Optional[dict[str, Any]] = None) -> JSONResponse:
    return JSONResponse(status_code=status, content=_envelope(code, message, details))


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    return _response(exc.status_code, exc.code, exc.message, exc.details)


async def handle_llm_config_error(request: Request, exc: LLMConfigError) -> JSONResponse:
    logger.error("llm_config_error on %s: %s", request.url.path, exc)
    return _response(500, "llm_config_error", str(exc))


async def handle_budget_exceeded(request: Request, exc: BudgetExceeded) -> JSONResponse:
    details = {
        "spent": getattr(exc, "spent", None),
        "would_add": getattr(exc, "would_add", None),
        "cap": getattr(exc, "cap", None),
    }
    return _response(429, "budget_exceeded", str(exc), details)


async def handle_invalid_transition(
    request: Request, exc: InvalidTransitionError
) -> JSONResponse:
    details = {
        "transition": getattr(exc, "transition", None),
        "current": getattr(exc, "current").value if getattr(exc, "current", None) else None,
        "expected": [s.value for s in getattr(exc, "expected", ())],
    }
    return _response(409, "invalid_transition", str(exc), details)


async def handle_validation_error(
    request: Request, exc: ValidationError
) -> JSONResponse:
    return _response(422, "validation_error", "request body failed validation",
                     {"errors": exc.errors()})


async def handle_request_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    return _response(422, "validation_error", "request body failed validation",
                     {"errors": exc.errors()})


async def handle_value_error(request: Request, exc: ValueError) -> JSONResponse:
    return _response(422, "validation_error", str(exc))


async def handle_uncaught(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("uncaught exception on %s: %s", request.url.path, exc)
    return _response(500, "internal_error", "internal server error")


# ---------------------------------------------------------------------------
# Rate-limit handler — wraps slowapi's so the envelope matches.
# ---------------------------------------------------------------------------


def rate_limit_handler(request: Request, exc: Exception) -> JSONResponse:
    detail = getattr(exc, "detail", None) or "rate limit exceeded"
    return _response(429, "rate_limited", str(detail))


def register_error_handlers(app: Any) -> None:
    """Attach every handler declared above to ``app``."""
    from slowapi.errors import RateLimitExceeded

    app.add_exception_handler(ApiError, handle_api_error)
    app.add_exception_handler(LLMConfigError, handle_llm_config_error)
    app.add_exception_handler(BudgetExceeded, handle_budget_exceeded)
    app.add_exception_handler(InvalidTransitionError, handle_invalid_transition)
    app.add_exception_handler(ValidationError, handle_validation_error)
    app.add_exception_handler(RequestValidationError, handle_request_validation_error)
    app.add_exception_handler(ValueError, handle_value_error)
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)
    app.add_exception_handler(Exception, handle_uncaught)


__all__ = [
    "ApiError",
    "ErrorDetail",
    "ErrorResponse",
    "InvalidTransition",
    "SessionForbidden",
    "SessionNotFound",
    "register_error_handlers",
]

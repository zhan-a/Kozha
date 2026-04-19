"""Request-level security wiring for the chat2hamnosys API.

Thin layer that the router calls into for every untrusted input:

- :func:`sanitize_user_input` — NFC-normalize, strip control chars, and
  enforce the configured length cap. Raises :class:`InputTooLong` which
  the error handler translates into ``400 input_too_long``.
- :func:`screen_user_description` — run the regex + optional LLM
  injection screen on prose / correction bodies. Raises
  :class:`InjectionRejected` on a non-``DESCRIPTION`` verdict.
- :func:`moderate_user_facing_text` — moderate a model-generated string
  that will be shown to the user; returns the text or a safe fallback.
- :data:`per_ip_tracker` / :data:`global_budget` — process-wide cost
  caps from :mod:`security.rate_limit`, created lazily from the
  current :class:`SecurityConfig`.

Everything is wired through FastAPI ``Depends`` so tests can override
with stubs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional
from uuid import uuid4

from fastapi import Request

from obs import events as _evs
from obs import metrics as _metrics
from obs.logger import emit_event
from security import (
    CostCapTracker,
    GlobalDailyBudget,
    InjectionResult,
    InjectionVerdict,
    InputTooLong,
    RegexInjectionScreen,
    SecurityConfig,
    load_security_config,
    sanitize_for_prompt,
    screen_description,
)

from .errors import ApiError


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InputTooLongApi(ApiError):
    """Wrapper so FastAPI returns a 400 envelope for oversized input."""

    status_code = 400
    code = "input_too_long"


class InjectionRejected(ApiError):
    """User input appears to be an instruction to the model, not a description."""

    status_code = 400
    code = "injection_rejected"


# ---------------------------------------------------------------------------
# Process-wide singletons — lazily constructed from the env config.
# ---------------------------------------------------------------------------


_config: Optional[SecurityConfig] = None
_per_ip_tracker: Optional[CostCapTracker] = None
_global_budget: Optional[GlobalDailyBudget] = None
_regex_screen: Optional[RegexInjectionScreen] = None


def get_security_config() -> SecurityConfig:
    global _config
    if _config is None:
        _config = load_security_config()
    return _config


def get_per_ip_tracker() -> CostCapTracker:
    global _per_ip_tracker
    if _per_ip_tracker is None:
        cfg = get_security_config()
        _per_ip_tracker = CostCapTracker(
            cap_usd=cfg.per_ip_daily_cap_usd, scope="per-ip"
        )
    return _per_ip_tracker


def get_global_budget() -> GlobalDailyBudget:
    global _global_budget
    if _global_budget is None:
        cfg = get_security_config()
        _global_budget = GlobalDailyBudget(cap_usd=cfg.global_daily_cap_usd)
    return _global_budget


def get_regex_screen() -> RegexInjectionScreen:
    global _regex_screen
    if _regex_screen is None:
        _regex_screen = RegexInjectionScreen()
    return _regex_screen


def reset_security_singletons() -> None:
    """Test helper — clears the lazily-constructed singletons."""
    global _config, _per_ip_tracker, _global_budget, _regex_screen
    _config = None
    _per_ip_tracker = None
    _global_budget = None
    _regex_screen = None


# ---------------------------------------------------------------------------
# Input-side helpers
# ---------------------------------------------------------------------------


def sanitize_user_input(
    text: str,
    *,
    field_name: str,
    max_len: Optional[int] = None,
) -> str:
    """Sanitize ``text`` for prompt use; surface :class:`InputTooLongApi`.

    ``field_name`` identifies the body field in the error envelope so
    the frontend can attach the message to the right input.
    """
    cap = max_len if max_len is not None else get_security_config().max_input_len
    try:
        return sanitize_for_prompt(text, max_len=cap)
    except InputTooLong as exc:
        raise InputTooLongApi(
            f"'{field_name}' is {exc.length} characters, exceeding the "
            f"{exc.max_len}-character maximum",
            details={
                "field": field_name,
                "length": exc.length,
                "max_len": exc.max_len,
            },
        ) from exc


@dataclass
class ScreenOutcome:
    """Result of :func:`screen_user_description`.

    Exposed separately from :class:`InjectionResult` so the router can
    log the incident with the client IP without re-deriving it.
    """

    result: InjectionResult
    client_ip: str


def screen_user_description(
    text: str,
    *,
    request: Request,
    field_name: str,
    classifier=None,
) -> ScreenOutcome:
    """Run the layered injection screen; reject with 400 on non-DESCRIPTION.

    The caller supplies the optional LLM classifier (``None`` = regex
    only). The classifier is typically disabled for short fields like
    ``/answer`` and enabled for ``/describe`` and ``/correct``.
    """
    cfg = get_security_config()
    result = screen_description(
        text,
        regex_screen=get_regex_screen(),
        classifier=classifier if cfg.enable_injection_classifier else None,
        request_id=f"inject-{uuid4().hex[:12]}",
    )
    client_ip = getattr(getattr(request, "client", None), "host", "") or "unknown"

    if result.verdict != InjectionVerdict.DESCRIPTION:
        logger.warning(
            "injection screen rejected input on field=%s client=%s reason=%s",
            field_name,
            client_ip,
            result.reason,
        )
        emit_event(
            _evs.SECURITY_INJECTION_DETECTED,
            field=field_name,
            verdict=result.verdict.value,
            client_ip=client_ip,
            reason=result.reason,
        )
        _metrics.injection_detections_total.inc()
        raise InjectionRejected(
            "We didn't interpret this as a sign description. Please describe "
            "only the sign itself.",
            details={
                "field": field_name,
                "verdict": result.verdict.value,
            },
        )
    return ScreenOutcome(result=result, client_ip=client_ip)


# ---------------------------------------------------------------------------
# Output-side helper
# ---------------------------------------------------------------------------


def moderate_user_facing_text(
    text: str,
    *,
    moderator: Optional[Callable[[str], bool]] = None,
) -> str:
    """Return ``text`` unchanged, or a safe fallback if moderation blocks it.

    ``moderator`` is a callable taking the candidate string and returning
    ``True`` when the text should be blocked. ``None`` means
    moderation is disabled for this call (the usual case in tests and
    when ``SecurityConfig.enable_output_moderation`` is off).
    """
    if not text or moderator is None:
        return text
    try:
        blocked = bool(moderator(text))
    except Exception as exc:
        logger.warning("moderator raised, ignoring: %s", exc)
        return text
    if blocked:
        logger.warning("moderation blocked user-facing text (len=%d)", len(text))
        return "Sorry — I couldn't generate a suitable response. Please try again."
    return text


__all__ = [
    "InjectionRejected",
    "InputTooLongApi",
    "ScreenOutcome",
    "get_global_budget",
    "get_per_ip_tracker",
    "get_regex_screen",
    "get_security_config",
    "moderate_user_facing_text",
    "reset_security_singletons",
    "sanitize_user_input",
    "screen_user_description",
]

"""FastAPI dependency providers for the chat2hamnosys API.

Resource-lifecycle helpers so the router file stays focused on request
handling. Each provider reads a paired env var so tests (and the run
script) can point the API at a tmp directory without touching the live
``data/`` folder:

- ``CHAT2HAMNOSYS_SESSION_DB`` — :class:`SessionStore` SQLite path.
- ``CHAT2HAMNOSYS_SIGN_DB``    — :class:`SQLiteSignStore` SQLite path.
- ``CHAT2HAMNOSYS_TOKEN_DB``   — :class:`TokenStore` SQLite path.
- ``CHAT2HAMNOSYS_DATA_DIR``   — Kozha data dir for exports (optional).

Sub-function callables (parser, clarifier, generator, renderer, …) are
exposed as dependencies too so tests can override them via
``app.dependency_overrides`` without monkeypatching module globals.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from clarify import Question, apply_answer, generate_questions
from generator import GenerateResult, generate
from parser import ParseResult, parse_description
from rendering.hamnosys_to_sigml import to_sigml
from rendering.preview import PreviewResult, render_preview
from session import SessionStore
from storage import SQLiteSignStore

from .token_store import TokenStore


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = _REPO_ROOT / "data"
DEFAULT_SESSION_DB = DEFAULT_DATA_DIR / "chat2hamnosys" / "sessions.sqlite3"
DEFAULT_SIGN_DB = DEFAULT_DATA_DIR / "authored_signs.sqlite3"
DEFAULT_TOKEN_DB = DEFAULT_DATA_DIR / "chat2hamnosys" / "session_tokens.sqlite3"


def _env_path(var: str, default: Path) -> Path:
    raw = os.environ.get(var, "").strip()
    return Path(raw) if raw else default


# ---------------------------------------------------------------------------
# Singleton caches — rebuilt on process start; tests override via
# app.dependency_overrides rather than rebinding these globals.
# ---------------------------------------------------------------------------


_session_store: Optional[SessionStore] = None
_sign_store: Optional[SQLiteSignStore] = None
_token_store: Optional[TokenStore] = None


def get_session_store() -> SessionStore:
    """Return a process-wide :class:`SessionStore` at the configured path."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore(
            db_path=_env_path("CHAT2HAMNOSYS_SESSION_DB", DEFAULT_SESSION_DB)
        )
    return _session_store


def get_sign_store() -> SQLiteSignStore:
    """Return a process-wide :class:`SQLiteSignStore` at the configured path."""
    global _sign_store
    if _sign_store is None:
        _sign_store = SQLiteSignStore(
            db_path=_env_path("CHAT2HAMNOSYS_SIGN_DB", DEFAULT_SIGN_DB),
            kozha_data_dir=_env_path("CHAT2HAMNOSYS_DATA_DIR", DEFAULT_DATA_DIR),
        )
    return _sign_store


def get_token_store() -> TokenStore:
    """Return a process-wide :class:`TokenStore` at the configured path."""
    global _token_store
    if _token_store is None:
        _token_store = TokenStore(
            db_path=_env_path("CHAT2HAMNOSYS_TOKEN_DB", DEFAULT_TOKEN_DB)
        )
    return _token_store


def reset_stores() -> None:
    """Clear the singleton caches — used by tests between runs."""
    global _session_store, _sign_store, _token_store
    _session_store = None
    _sign_store = None
    _token_store = None


# ---------------------------------------------------------------------------
# Orchestration callables — dependency-injected so tests can stub.
# ---------------------------------------------------------------------------


def get_parse_fn() -> Callable[[str], ParseResult]:
    return parse_description


def get_question_fn() -> Callable[..., list[Question]]:
    return generate_questions


def get_apply_fn() -> Callable[..., object]:
    return apply_answer


def get_generate_fn() -> Callable[..., GenerateResult]:
    return generate


def get_to_sigml_fn() -> Callable[..., str]:
    return to_sigml


def get_render_fn() -> Optional[Callable[..., PreviewResult]]:
    """Return the preview renderer.

    Default: the real :func:`render_preview`. Tests pass a lambda that
    returns ``None`` via dependency_overrides when they want to skip
    the video-cache subprocess entirely.
    """
    return render_preview


__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_SESSION_DB",
    "DEFAULT_SIGN_DB",
    "DEFAULT_TOKEN_DB",
    "get_apply_fn",
    "get_generate_fn",
    "get_parse_fn",
    "get_question_fn",
    "get_render_fn",
    "get_session_store",
    "get_sign_store",
    "get_to_sigml_fn",
    "get_token_store",
    "reset_stores",
]

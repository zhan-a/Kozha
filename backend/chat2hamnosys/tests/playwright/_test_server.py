"""Tiny FastAPI launcher for the Playwright smoke test.

Runs as a subprocess (so Playwright can hit a real port) and:

1. Creates the chat2hamnosys sub-app with stubbed parser / clarifier so
   the test never calls the LLM.
2. Mounts it under ``/api/chat2hamnosys``.
3. Mounts the repository's ``public/`` directory so the static frontend
   (``/chat2hamnosys/index.html``) is served from the same origin.

Configured purely via env vars set by the conftest fixture:

- ``C2H_PORT`` — port to bind (required).
- ``CHAT2HAMNOSYS_SESSION_DB`` / ``..._SIGN_DB`` / ``..._TOKEN_DB`` —
  per-test SQLite paths (set by the fixture; see conftest.py).
- ``C2H_PUBLIC_DIR`` — absolute path to the repo ``public/`` directory.

The script does not import ``server.server`` so this stays decoupled from
the Kozha translator app — the only surface under test is chat2hamnosys.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _ensure_path() -> None:
    """Make ``backend/chat2hamnosys`` importable as a package root."""
    here = Path(__file__).resolve()
    pkg_root = here.parents[2]  # …/backend/chat2hamnosys
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))


_ensure_path()


from fastapi import FastAPI                                  # noqa: E402
from fastapi.staticfiles import StaticFiles                  # noqa: E402
from fastapi.responses import RedirectResponse               # noqa: E402

from api import (                                            # noqa: E402
    create_app,
    get_apply_fn,
    get_parse_fn,
    get_question_fn,
    get_render_fn,
)
from api.dependencies import reset_stores                    # noqa: E402
from review.dependencies import reset_review_stores          # noqa: E402
from clarify import Option, Question                         # noqa: E402
from parser import Gap, ParseResult                          # noqa: E402
from parser.models import (                                  # noqa: E402
    PartialMovementSegment,
    PartialSignParameters,
)


# ---------------------------------------------------------------------------
# Stubs — mirror tests/test_api_integration.py, kept tiny so the smoke test
# exercises only routing + frontend wiring, not the real LLM stack.
# ---------------------------------------------------------------------------


def _initial_partial() -> PartialSignParameters:
    return PartialSignParameters(
        handshape_dominant="flat",
        orientation_palm="down",
        location="temple",
        movement=[PartialMovementSegment(path="down")],
    )


_PARSE_RESULT = ParseResult(
    parameters=_initial_partial(),
    gaps=[
        Gap(
            field="orientation_extended_finger",
            reason="flat hand finger orientation not stated",
            suggested_question="Which way do the fingers point?",
        )
    ],
    raw_response="{}",
)


def _stub_parser(prose: str) -> ParseResult:
    return _PARSE_RESULT


class _StubQuestioner:
    """Returns one question on the first call, nothing thereafter."""

    def __init__(self) -> None:
        self._served = False

    def __call__(self, parse_result, prior_turns, *, is_deaf_native):
        if self._served:
            return []
        self._served = True
        if not any(g.field == "orientation_extended_finger" for g in parse_result.gaps):
            return []
        return [
            Question(
                field="orientation_extended_finger",
                text="Which way do the fingers point?",
                options=[Option(label="Up", value="up")],
                allow_freeform=True,
                rationale="stub",
            )
        ]


def _stub_apply(params, question, answer):
    if question.field == "orientation_extended_finger":
        return params.model_copy(update={"orientation_extended_finger": str(answer).strip()})
    return params


# ---------------------------------------------------------------------------
# Build the app
# ---------------------------------------------------------------------------


def build_app() -> FastAPI:
    reset_stores()
    reset_review_stores()

    sub = create_app()
    sub.dependency_overrides[get_parse_fn] = lambda: _stub_parser
    sub.dependency_overrides[get_question_fn] = lambda: _StubQuestioner()
    sub.dependency_overrides[get_apply_fn] = lambda: _stub_apply
    # Skip avatar render — VOCAB composer still produces real HamNoSys+SiGML.
    sub.dependency_overrides[get_render_fn] = lambda: None

    parent = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    parent.mount("/api/chat2hamnosys", sub)

    public_dir = Path(os.environ.get("C2H_PUBLIC_DIR") or "").resolve()
    if not public_dir.exists():
        raise RuntimeError(f"C2H_PUBLIC_DIR not found: {public_dir}")

    @parent.get("/", include_in_schema=False)
    def _root():
        return RedirectResponse(url="/chat2hamnosys/")

    @parent.get("/healthz", include_in_schema=False)
    def _healthz():
        return {"ok": True}

    # Prompt 10: mirror the production server's status-page route so
    # /contribute/status/<id> serves the static HTML shell even though the
    # StaticFiles mount below can't match the path parameter.
    from fastapi.responses import FileResponse  # noqa: WPS433

    @parent.get("/contribute/status/{session_id}", include_in_schema=False)
    @parent.get("/contribute/status/{session_id}/", include_in_schema=False)
    def _status_page(session_id: str):  # noqa: ARG001 — id read by client JS
        return FileResponse(public_dir / "contribute-status.html")

    parent.mount("/", StaticFiles(directory=str(public_dir), html=True), name="public")
    return parent


app = build_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("C2H_PORT", "0"))
    if not port:
        raise SystemExit("C2H_PORT must be set")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

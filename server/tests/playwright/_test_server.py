"""Minimal launcher for the translator Playwright tests (prompt 4).

Serves ``public/`` + ``data/`` statically, stubs ``/api/translate-text`` and
``/api/plan`` so the tests run without argostranslate or spaCy installed.
The stubs honor the prompt-4 payload shape including the new
``target_sign_lang`` field and echo an ``error`` for unknown languages so
the 4th test scenario can be exercised purely client-side.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_DIR = REPO_ROOT / "public"
DATA_DIR = REPO_ROOT / "data"


_SUPPORTED_BASE_LANGS = {"en", "fr", "de", "es", "pl", "nl", "el", "ru", "ar"}
_SUPPORTED_SIGN_LANGS = {
    "bsl", "asl", "dgs", "lsf", "lse", "pjm", "gsl", "rsl",
    "algerian", "bangla", "ngt", "fsl", "isl", "kurdish", "vsl",
}


class TranslateTextReq(BaseModel):
    text: Optional[str] = None
    source_text: Optional[str] = None
    source_lang: str
    target_lang: str
    target_sign_lang: Optional[str] = None


class PlanReq(BaseModel):
    text: str
    language: str = "en"
    sign_language: str = "bsl"


def build_app() -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/translate-text")
    def translate(req: TranslateTextReq):
        text = (req.text or req.source_text or "").strip()
        if not text or req.source_lang == req.target_lang:
            return {"translated": text}
        if req.target_sign_lang and req.target_sign_lang not in _SUPPORTED_SIGN_LANGS:
            return {
                "translated": text,
                "error": f"unknown target sign language: {req.target_sign_lang}",
            }
        if (
            req.source_lang not in _SUPPORTED_BASE_LANGS
            or req.target_lang not in _SUPPORTED_BASE_LANGS
        ):
            return {
                "translated": text,
                "error": f"no translation route from {req.source_lang} to {req.target_lang}",
            }
        return {"translated": text}

    @app.post("/api/plan")
    def plan(req: PlanReq):
        return {
            "final": (req.text or "").strip() + " .",
            "raw": req.text,
            "language": req.language,
            "sign_language": req.sign_language,
            "allowed": [],
        }

    if DATA_DIR.exists():
        app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")
    app.mount("/", StaticFiles(directory=str(PUBLIC_DIR), html=True), name="public")
    return app


app = build_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("KOZHA_TEST_PORT", "0"))
    if not port:
        raise SystemExit("KOZHA_TEST_PORT must be set")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")

"""Static source checks for the preview → chat handoff (prompt 9).

These assertions read the JS files as text rather than executing them —
there's no JSDOM/Puppeteer harness wired into this repo yet, so the
contracts we care about are validated by looking for the exact symbols
and string literals the three modules agree on.

Why not just a single end-to-end test? The preview pane depends on
CWASA, which in turn needs WebGL; running that in CI means pulling in
Puppeteer + a headless Chrome binary. Until that harness exists, these
source-level checks are the cheap-but-real guard: if any of them fire,
the preview → chat glue will be broken at runtime too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PUBLIC_DIR = REPO_ROOT / "public"


def _build_app() -> FastAPI:
    app = FastAPI()
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_app())


def test_chat_module_exposes_correction_target_setter(client: TestClient) -> None:
    js = client.get("/contribute-chat.js").text
    assert 'setCorrectionTarget:' in js
    assert 'clearCorrectionTarget:' in js
    # Pill DOM references the controller uses.
    assert "document.getElementById('chatTargetPill')" in js
    assert "document.getElementById('chatTargetPillText')" in js
    assert "document.getElementById('chatTargetPillClear')" in js


def test_chat_sendCorrection_forwards_time_and_region(client: TestClient) -> None:
    js = client.get("/contribute-chat.js").text
    # Both targeting fields must thread through the /correct call.
    assert 'targetTimeMs' in js
    assert 'targetRegion' in js
    # And the pill must be cleared after submission so the next
    # correction starts fresh unless the user re-targets.
    assert 'clearCorrectionTarget()' in js


def test_preview_controller_calls_chat_setter(client: TestClient) -> None:
    js = client.get("/contribute-preview.js").text
    # The preview pane hands region + time over via the chat module's
    # public surface — not via a shared global object.
    assert 'KOZHA_CONTRIB_CHAT' in js
    assert 'setCorrectionTarget' in js
    # Clicking a body region captures both the region name and the
    # scrubber's current time.
    assert 'data-region' in js or "getAttribute('data-region')" in js
    # Scrubber time is formatted the same way in the pill and the
    # time readout so the numbers match exactly.
    assert 'formatSeconds' in js


def test_chat_pill_uses_bullet_separator(client: TestClient) -> None:
    """Pill text format per prompt 9 §4: 'At 0.82s • hand (dominant)'."""
    js = client.get("/contribute-chat.js").text
    assert '•' in js


def test_preview_controller_probes_webgl_and_cwasa(client: TestClient) -> None:
    """If either probe is skipped, the fallback path can't fire — which
    ends with a blank canvas instead of the explanatory text."""
    js = client.get("/contribute-preview.js").text
    assert 'probeWebGL' in js
    assert 'waitForCWASA' in js
    assert 'markRenderFailed' in js
    # The fallback copy itself lives in the HTML, but the trigger
    # reasons must be distinguishable in logs.
    assert 'webgl_unavailable' in js
    assert 'cwasa_missing' in js


def test_context_store_accepts_correction_target_opts(client: TestClient) -> None:
    """Prompt 9's pill forwards `targetTimeMs` + `targetRegion` through
    context.correct(raw, opts). If the store stops reading these, the
    pill becomes a silent no-op."""
    js = client.get("/contribute-context.js").text
    assert 'target_time_ms' in js
    assert 'target_region' in js
    assert 'targetTimeMs' in js
    assert 'targetRegion' in js

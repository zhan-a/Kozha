"""Static-markup checks for the contribute page's avatar preview pane.

Prompt 9 adds the avatar preview pane between the chat panel and the
notation panel. These assertions catch the case where any of the
structural IDs the preview controller depends on get accidentally
renamed — no JSDOM required, just a TestClient pointed at ``public/``.

Mirrors the ``_build_app`` pattern from the other static-markup tests
so each file stays self-contained (dropping one shouldn't break the
others).
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


def test_preview_pane_structural_ids_present(client: TestClient) -> None:
    body = client.get("/contribute.html").text
    # Root pane + the CWASA mount points the renderer writes its
    # canvas into. Without these the controller silently bails.
    assert 'id="avatarPreview"' in body
    assert 'id="avatarStage"' in body
    assert 'id="avatarCanvas"' in body
    assert 'CWASAAvatar' in body
    assert 'CWASAGUI' in body
    # Body-region overlay used by the correction-targeting flow.
    assert 'id="avatarRegions"' in body
    assert 'data-region="hand_dominant"' in body
    assert 'data-region="hand_nondominant"' in body
    assert 'data-region="face"' in body
    # Generation UI + fallback.
    assert 'id="avatarPulse"' in body
    assert 'id="avatarFallback"' in body
    assert 'Preview unavailable in this environment' in body
    # Status strip carries aria-live so screen-readers announce state
    # transitions the hearing UI conveys with small text.
    assert 'id="avatarStatus"' in body
    assert 'aria-live="polite"' in body


def test_preview_control_bar_has_all_named_controls(client: TestClient) -> None:
    body = client.get("/contribute.html").text
    # Play + loop + speed buttons at 0.5/1/2. The scrubber-slider was
    # intentionally removed (see commit dca6209) — CWASA has no seek API
    # and the control misled testers. Play / loop / speed remain.
    assert 'id="avatarPlayBtn"' in body
    assert 'id="avatarLoopInput"' in body
    assert 'data-speed="0.5"' in body
    assert 'data-speed="1"' in body
    assert 'data-speed="2"' in body


def test_preview_pane_loads_cwasa_assets_and_controller(client: TestClient) -> None:
    body = client.get("/contribute.html").text
    # CWASA bundle + stylesheet must be referenced from the head so
    # window.CWASA is populated before the preview controller boots.
    assert '/cwa/cwasa.css' in body
    assert '/cwa/allcsa.js' in body
    # Preview controller is loaded after the context + chat modules so
    # their globals are available when it initialises.
    assert '/contribute-preview.js' in body
    ctx_idx  = body.index('/contribute-context.js')
    chat_idx = body.index('/contribute-chat.js')
    prv_idx  = body.index('/contribute-preview.js')
    assert ctx_idx < prv_idx
    assert chat_idx < prv_idx


def test_preview_captions_strip_holds_gloss_and_description(client: TestClient) -> None:
    body = client.get("/contribute.html").text
    assert 'id="avatarCaptionGloss"' in body
    assert 'id="avatarCaptionDesc"' in body


def test_chat_correction_target_pill_markup_present(client: TestClient) -> None:
    """Pill rendered above the chat textarea when the contributor clicks
    a body region on the preview (prompt 9 → prompt 6 hand-off)."""
    body = client.get("/contribute.html").text
    assert 'id="chatTargetPill"' in body
    assert 'id="chatTargetPillText"' in body
    assert 'id="chatTargetPillClear"' in body
    # Dismiss button carries an aria-label for screen readers — the
    # visible × glyph is not a pronounceable control.
    assert 'aria-label="Clear correction target"' in body


def test_preview_pane_sits_between_chat_and_notation(client: TestClient) -> None:
    """Layout contract: the preview pane lives above the notation panel
    and below the chat panel, so the contributor reviews the rendered
    avatar before they read the generated notation."""
    body = client.get("/contribute.html").text
    chat_idx     = body.index('id="chatPanel"')
    preview_idx  = body.index('id="avatarPreview"')
    notation_idx = body.index('id="notationPanel"')
    assert chat_idx < preview_idx < notation_idx


def test_preview_css_declares_stage_aspect_ratio_and_backdrop(client: TestClient) -> None:
    """The stage is always 16:9 with a light-gray backdrop. If these
    rules regress, the avatar renders on an off-brand dark background
    that hurts legibility for Deaf users (Kipp et al. 2011)."""
    css = client.get("/contribute.css").text
    assert '.avatar-stage' in css
    assert 'aspect-ratio: 16 / 9' in css
    assert '.avatar-backdrop' in css
    # Backdrop color must be a light tone (not the CWASA default blue).
    assert '#e9e6df' in css
    # Fallback + pulse styles the controller toggles via hidden.
    assert '.avatar-fallback' in css
    assert '.avatar-pulse' in css


def test_preview_css_respects_reduced_motion(client: TestClient) -> None:
    """The backdrop pulse must not animate for users who opted out of
    motion — a quiet surface is a design-principle requirement, not a
    polish item."""
    css = client.get("/contribute.css").text
    assert 'prefers-reduced-motion' in css
    # The rule must target the pulse specifically.
    pulse_block = css[css.index('.avatar-pulse'):]
    assert 'animation' in pulse_block


def test_preview_controller_script_is_served(client: TestClient) -> None:
    resp = client.get("/contribute-preview.js")
    assert resp.status_code == 200
    js = resp.text
    # Public surface the preview controller exposes (consumed by the
    # chat pill setter + by tests).
    assert 'KOZHA_CONTRIB_PREVIEW' in js
    # Keyboard controls per prompt 9 §9.
    assert 'Spacebar' in js or 'ev.key' in js
    # Cache badge and generation/rendering status strings must land
    # verbatim so screen-reader users get consistent announcements.
    assert 'From cache' in js
    assert 'Generating HamNoSys' in js
    assert 'Rendering avatar' in js

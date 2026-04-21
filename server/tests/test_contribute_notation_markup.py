"""Static-markup checks for the contribute page's notation panel.

Prompt 7 adds a full notation panel (HamNoSys glyph display + SiGML
XML source) below the avatar preview. These assertions catch the case
where the structural IDs a vanilla-JS controller depends on get
accidentally renamed — no JSDOM required, just a TestClient pointed at
``public/``.

This file intentionally duplicates the static ``_build_app`` pattern
from ``test_contribute_page_loads.py`` rather than importing it; the
original file uses a module-scoped client fixture, and mirroring the
helper here keeps each test file self-contained (you can drop either
one without breaking the other).
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


def test_notation_panel_structural_ids_present(client: TestClient) -> None:
    body = client.get("/contribute.html").text
    # Root panel + the two button-tabs the controller wires click/keydown to.
    assert 'id="notationPanel"' in body
    assert 'id="notationTabHamnosys"' in body
    assert 'id="notationTabSigml"' in body
    # HamNoSys tab: glyph display, legend, four breakdown slots, copy button.
    assert 'id="hamnosysDisplay"' in body
    assert 'id="notationLegend"' in body
    assert 'id="breakdownHandshape"' in body
    assert 'id="breakdownOrientation"' in body
    assert 'id="breakdownLocation"' in body
    assert 'id="breakdownMovement"' in body
    assert 'id="copyHamnosysBtn"' in body
    # SiGML tab: the <pre> has role="region" + aria-label so assistive
    # tech announces the source block as a landmark (per prompt 7 §7).
    assert 'id="sigmlDisplay"' in body
    assert 'id="sigmlCode"' in body
    assert 'role="region"' in body
    assert 'aria-label="SiGML source code"' in body
    # Copy + download actions the controller enables once sigml lands.
    assert 'id="copySigmlBtn"' in body
    assert 'id="downloadSigmlBtn"' in body


def test_notation_panel_uses_role_tablist_and_tabs(client: TestClient) -> None:
    """Tabs must carry the right ARIA roles — screen readers announce
    the HamNoSys/SiGML switch as a tablist, not two unrelated buttons."""
    body = client.get("/contribute.html").text
    assert 'role="tablist"' in body
    # Two role="tab" occurrences, one for each button.
    assert body.count('role="tab"') >= 2
    assert body.count('role="tabpanel"') >= 2
    # Hamnosys tab is selected by default; SiGML tab is not.
    assert 'id="notationTabHamnosys"' in body
    assert 'aria-selected="true"' in body
    assert 'aria-selected="false"' in body


def test_notation_panel_fallback_copy_is_present(client: TestClient) -> None:
    """The two-retries fallback message is baked into the markup so the
    JS can flip a ``hidden`` attribute without string-building from JS.
    """
    body = client.get("/contribute.html").text
    assert "could not be validated" in body
    assert "A reviewer will finish it" in body


def test_notation_panel_loads_controller_script(client: TestClient) -> None:
    body = client.get("/contribute.html").text
    # Controller + its context dependency.
    assert "/contribute-notation.js" in body
    assert "/contribute-context.js" in body


def test_notation_css_declares_local_hamnosys_font(client: TestClient) -> None:
    """Per the contribute-redesign plan, the Hamburg HamNoSys font loads
    from ``/fonts/`` — never from a CDN — so we ship it alongside the
    page and the page works offline."""
    resp = client.get("/contribute.css")
    assert resp.status_code == 200
    css = resp.text
    assert "@font-face" in css
    assert "bgHamNoSysUnicode" in css
    assert "/fonts/bgHamNoSysUnicode.ttf" in css


def test_notation_css_paints_sigml_line_numbers(client: TestClient) -> None:
    """Line numbers are a CSS-counter effect on ``.sigml-line``; if the
    counter rule goes missing the XML block silently drops them."""
    css = client.get("/contribute.css").text
    assert "sigml-line" in css
    assert "counter-reset" in css
    assert "counter-increment" in css


def test_sigml_font_license_file_is_shipped(client: TestClient) -> None:
    """The IDGS license text must be reachable so redistribution
    provenance doesn't get silently stripped from the deploy."""
    resp = client.get("/fonts/bgHamNoSysUnicode.LICENSE.txt")
    assert resp.status_code == 200
    body = resp.text
    assert "HamNoSys" in body
    assert "IDGS" in body or "Institute of German Sign Language" in body

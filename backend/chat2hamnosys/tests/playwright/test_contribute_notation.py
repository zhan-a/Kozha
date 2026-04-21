"""Playwright test for prompt 7 — the notation panel.

Drives the page through a full RENDERED envelope and asserts:

1. ``#notationPanel`` becomes visible once ``/describe`` delivers
   ``state=rendered`` with a ``hamnosys`` string in the envelope.
2. ``#hamnosysDisplay`` renders one ``<span class="notation-glyph">`` per
   codepoint, each tagged with the right ``data-hex`` attribute.
3. The phonological breakdown (Handshape / Orientation / Location /
   Movement) reads the partial-parameters bundle and writes plain
   English into the four ``<dd>`` slots.
4. The legend reacts to hovering a glyph — class_label + short_name
   update after the symbols table has been fetched.
5. Clicking "Copy HamNoSys" copies the raw HamNoSys string to the
   clipboard and flashes the confirmation chip.
6. Activating the SiGML tab hides the HamNoSys tabpanel and shows
   ``#sigmlDisplay`` populated with per-line highlighted spans.
7. Clicking "Copy SiGML" copies the XML source.
8. The "Download .sigml" button becomes enabled and firing it produces
   a download named ``<gloss>_<lang>.sigml``.

The Kozha backend's real router (mounted by ``_test_server.py``) serves
the live ``/hamnosys/symbols`` endpoint so the legend lookup exercises
real data — only ``POST /sessions`` and ``POST /describe`` are stubbed
via ``page.route()`` to keep the LLM stack out of the picture.

Skipped if Playwright is not installed.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest


pw = pytest.importorskip(
    "playwright.sync_api",
    reason="install with `pip install playwright && playwright install chromium` to run",
)

from playwright.sync_api import Route, sync_playwright, expect  # noqa: E402


# A realistic BSL-style HamNoSys: flat hand, finger up, palm down, at the
# head, with a downward movement. Five codepoints → five glyph spans.
HAMNOSYS = ""
HAMNOSYS_CODEPOINTS = [
    ("U+E001", ""),
    ("U+E020", ""),
    ("U+E03C", ""),
    ("U+E040", ""),
    ("U+E089", ""),
]

SIGML_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<sigml>\n'
    '  <hns_sign gloss="TEMPLE">\n'
    '    <hamnosys_manual>\n'
    '      <hamflathand/>\n'
    '      <hamextfingeru/>\n'
    '      <hampalmd/>\n'
    '      <hamhead/>\n'
    '      <hammoved/>\n'
    '    </hamnosys_manual>\n'
    '  </hns_sign>\n'
    '</sigml>\n'
)

PARAMETERS = {
    "handshape_dominant": "flat",
    "handshape_nondominant": None,
    "orientation_extended_finger": "up",
    "orientation_palm": "down",
    "location": "temple",
    "movement": [{"path": "down", "size_mod": None, "speed_mod": None, "repeat": None}],
}

GLOSS = "TEMPLE"
DESCRIPTION = "flat hand, palm down, near the temple, moves straight down"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _envelope(session_id: str, state: str, **extra: Any) -> dict[str, Any]:
    base = {
        "session_id": session_id,
        "state": state,
        "gloss": GLOSS,
        "sign_language": "bsl",
        "description_prose": DESCRIPTION,
        "pending_questions": [],
        "clarifications": [],
        "created_at": _now(),
        "last_activity_at": _now(),
    }
    base.update(extra)
    return base


def _pick_bsl(page) -> None:
    expect(page.locator("#languagePicker")).to_be_visible()
    page.locator('.picker-option[data-code="bsl"]').click()
    expect(page.locator("#langMasthead")).to_be_visible(timeout=3000)


def test_notation_panel_renders_glyphs_breakdown_and_copy(c2h_server: str) -> None:
    base = c2h_server
    session_id = str(uuid.uuid4())
    session_token = "stub-notation-token"

    def handle(route: Route) -> None:
        req = route.request
        path = req.url.split("?", 1)[0].split(base, 1)[-1]
        method = req.method

        if method == "POST" and path == "/api/chat2hamnosys/sessions":
            payload = {
                "session_id":    session_id,
                "state":         "awaiting_description",
                "session_token": session_token,
                "session":       _envelope(session_id, "awaiting_description"),
            }
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )
            return

        if method == "POST" and path.endswith("/describe"):
            payload = _envelope(
                session_id,
                "rendered",
                hamnosys=HAMNOSYS,
                sigml=SIGML_XML,
                parameters=PARAMETERS,
                generation_errors=[],
            )
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )
            return

        # SSE /events is served by the real backend — our fake session
        # id isn't registered there, so it returns an auth error and the
        # controller no-ops (that's the intended behaviour). Everything
        # else (symbols endpoint, static assets) also passes through.
        route.continue_()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context()
            # The Copy buttons write to the system clipboard. Granting
            # read+write here means navigator.clipboard.readText() inside
            # page.evaluate can read back what the button wrote.
            context.grant_permissions(
                ["clipboard-read", "clipboard-write"],
                origin=base,
            )
            page = context.new_page()
            page.route("**/api/chat2hamnosys/**", handle)

            page.goto(f"{base}/contribute.html", wait_until="domcontentloaded")
            _pick_bsl(page)

            page.locator("#glossInput").fill(GLOSS)
            page.locator("#descriptionInput").fill(DESCRIPTION)
            expect(page.locator("#startAuthoringBtn")).to_be_enabled()
            page.locator("#startAuthoringBtn").click()

            # Panel appears once the rendered envelope lands.
            expect(page.locator("#notationPanel")).to_be_visible(timeout=5000)

            # The crossfade runs for 200ms — wait for the final opaque
            # state before asserting per-glyph DOM.
            page.wait_for_function(
                "() => {"
                "  const d = document.getElementById('hamnosysDisplay');"
                "  return d && d.querySelectorAll('.notation-glyph').length > 0"
                "         && !d.classList.contains('is-fading');"
                "}",
                timeout=5000,
            )

            # (2) One span per codepoint, tagged with the correct hex.
            glyphs = page.locator("#hamnosysDisplay .notation-glyph")
            expect(glyphs).to_have_count(len(HAMNOSYS_CODEPOINTS))
            for idx, (hex_code, ch) in enumerate(HAMNOSYS_CODEPOINTS):
                g = glyphs.nth(idx)
                expect(g).to_have_attribute("data-hex", hex_code)
                expect(g).to_have_text(ch)

            # (3) Phonological breakdown reads the partial parameters.
            expect(page.locator("#breakdownHandshape")).to_have_text("flat")
            expect(page.locator("#breakdownOrientation")).to_have_text(
                "palm down, fingers up"
            )
            expect(page.locator("#breakdownLocation")).to_have_text("temple")
            expect(page.locator("#breakdownMovement")).to_have_text("down")

            # The aria-label carries English — not raw PUA codepoints —
            # so screen readers announce something intelligible (§7).
            aria_label = page.locator("#hamnosysDisplay").get_attribute("aria-label")
            assert aria_label and "Handshape: flat" in aria_label, aria_label
            for codepoint_char, _ in HAMNOSYS_CODEPOINTS:
                # The raw PUA chars should NOT appear in the aria-label.
                assert codepoint_char not in aria_label

            # (4) Legend reacts to glyph activation once the symbols
            # table has been fetched. Click (not hover — more reliable
            # under headless) the first glyph, then wait for the real
            # /hamnosys/symbols payload to land and the legend to flip
            # from its "Loading…" placeholder to the real label.
            glyphs.nth(0).click()
            expect(page.locator("#legendCode")).to_have_text("U+E001")
            expect(page.locator("#legendName")).to_have_text(
                "hamflathand", timeout=5000
            )
            # class_label is the plain-English version from the API.
            expect(page.locator("#legendClass")).to_have_text("Handshape")

            # (5) Copy HamNoSys → clipboard contains the raw string,
            # confirmation chip appears.
            page.locator("#copyHamnosysBtn").click()
            expect(page.locator("#copyHamnosysConfirm")).to_be_visible()
            clipboard_value = page.evaluate("() => navigator.clipboard.readText()")
            assert clipboard_value == HAMNOSYS, (
                f"expected clipboard to hold HAMNOSYS; got {clipboard_value!r}"
            )

            # (6) Switch to the SiGML tab via its button.
            page.locator("#notationTabSigml").click()
            expect(page.locator("#notationPanelSigml")).to_be_visible()
            expect(page.locator("#notationPanelHamnosys")).to_be_hidden()
            expect(page.locator("#notationTabSigml")).to_have_attribute(
                "aria-selected", "true"
            )
            expect(page.locator("#notationTabHamnosys")).to_have_attribute(
                "aria-selected", "false"
            )

            # (6b) SiGML block has per-line spans so the CSS counter can
            # paint line numbers, plus at least one highlighted tag span.
            expect(page.locator("#sigmlCode .sigml-line").first).to_be_visible()
            assert page.locator("#sigmlCode .sigml-line").count() >= 5, (
                "expected the SiGML XML to be split into lines"
            )
            assert page.locator("#sigmlCode .sigml-tag").count() >= 1, (
                "expected at least one highlighted tag span"
            )

            # (7) Copy SiGML → clipboard now holds the XML.
            page.locator("#copySigmlBtn").click()
            expect(page.locator("#copySigmlConfirm")).to_be_visible()
            sigml_clipboard = page.evaluate("() => navigator.clipboard.readText()")
            assert sigml_clipboard == SIGML_XML, (
                f"expected SiGML XML in clipboard; got {sigml_clipboard!r}"
            )

            # (8) Download .sigml emits a file named <gloss>_<lang>.sigml.
            expect(page.locator("#downloadSigmlBtn")).to_be_enabled()
            with page.expect_download() as dl_info:
                page.locator("#downloadSigmlBtn").click()
            download = dl_info.value
            assert download.suggested_filename == "temple_bsl.sigml", (
                f"unexpected filename: {download.suggested_filename!r}"
            )
        finally:
            browser.close()


def test_notation_panel_renderfortest_drives_panel_directly(c2h_server: str) -> None:
    """Narrower unit-style check that doesn't round-trip through the
    session flow.

    Uses the ``window.KOZHA_CONTRIB_NOTATION.renderForTest`` hook the
    controller exposes to tests so the RENDERED view can be asserted in
    isolation — handy for regressing the glyph/breakdown wiring even if
    the session or chat layers break.
    """
    base = c2h_server

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(f"{base}/contribute.html", wait_until="domcontentloaded")
            # Wait for the controller to mount before poking at it.
            page.wait_for_function(
                "() => !!window.KOZHA_CONTRIB_NOTATION"
            )
            page.evaluate(
                """(opts) => window.KOZHA_CONTRIB_NOTATION.renderForTest(opts)""",
                {
                    "hamnosys":   HAMNOSYS,
                    "sigml":      SIGML_XML,
                    "parameters": PARAMETERS,
                    "generationErrors": [],
                },
            )

            expect(page.locator("#notationPanel")).to_be_visible()
            glyphs = page.locator("#hamnosysDisplay .notation-glyph")
            expect(glyphs).to_have_count(len(HAMNOSYS_CODEPOINTS))
            expect(page.locator("#breakdownHandshape")).to_have_text("flat")
            expect(page.locator("#breakdownLocation")).to_have_text("temple")
            expect(page.locator("#copyHamnosysBtn")).to_be_enabled()
            expect(page.locator("#copySigmlBtn")).to_be_enabled()
            expect(page.locator("#downloadSigmlBtn")).to_be_enabled()
            # Fallback + error row start hidden when no errors come in.
            expect(page.locator("#notationErrors")).to_be_hidden()
            expect(page.locator("#notationFallback")).to_be_hidden()
        finally:
            browser.close()


def test_notation_panel_shows_fallback_after_two_retries(c2h_server: str) -> None:
    """After MAX_FAILED_ATTEMPTS consecutive validation errors the
    bullet list is replaced with the "A reviewer will finish it"
    fallback — the panel stays submittable, the contributor is not
    blocked by a faulty validator."""
    base = c2h_server

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.goto(f"{base}/contribute.html", wait_until="domcontentloaded")
            page.wait_for_function("() => !!window.KOZHA_CONTRIB_NOTATION")

            for i in range(3):
                page.evaluate(
                    """(opts) => window.KOZHA_CONTRIB_NOTATION.renderForTest(opts)""",
                    {
                        "hamnosys":   HAMNOSYS,
                        "sigml":      SIGML_XML,
                        "parameters": PARAMETERS,
                        # Distinct error text per attempt so the
                        # controller's "new error key" heuristic
                        # increments failedAttempts on each call.
                        "generationErrors": [f"attempt {i} failed: unknown glyph"],
                    },
                )

            # Bullet list disappears, fallback message takes its place.
            expect(page.locator("#notationErrors")).to_be_hidden()
            expect(page.locator("#notationFallback")).to_be_visible()
            expect(page.locator("#notationFallback")).to_contain_text(
                "could not be validated"
            )
        finally:
            browser.close()

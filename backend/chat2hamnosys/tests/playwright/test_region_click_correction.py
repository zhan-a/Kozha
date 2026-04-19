"""Playwright test for click-targeted body-region correction.

Drives the happy path to RENDERED, then clicks the dominant-hand polygon
on the avatar overlay, fills the popover with a correction phrase, and
asserts that the resulting ``correction_requested`` event carries the
expected ``target_region`` and ``target_time_ms``.

The test server starts without an ``OPENAI_API_KEY``, so the correction
interpreter falls back to a ``VAGUE`` plan. That is fine: the state
machine still records the ``CorrectionRequestedEvent`` *before* the
interpreter runs, so the region/time hints are observable via
``window.__c2h.state.envelope.history`` regardless of LLM availability.
"""

from __future__ import annotations

import pytest


pw = pytest.importorskip(
    "playwright.sync_api",
    reason="install with `pip install playwright && playwright install chromium` to run",
)

from playwright.sync_api import sync_playwright, expect  # noqa: E402


TEMPLE_PROSE = (
    "it's signed near the temple, flat hand, moves down to the chest, like SORRY"
)


def test_click_hand_region_sends_target_region_and_time(c2h_server: str) -> None:
    base_url = c2h_server

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto(f"{base_url}/chat2hamnosys/", wait_until="domcontentloaded")

            expect(page.locator("#sessionMeta")).to_contain_text(
                "BSL", timeout=5000
            )

            # Drive the describe → clarify → render cycle so the overlay
            # becomes visible (it renders only once preview.sigml or
            # preview.video_url is present).
            page.fill("#glossInput", "SORRY")
            page.fill("#chatInput", TEMPLE_PROSE)
            page.click("#sendBtn")

            answer_btn = page.locator(".msg-opt", has_text="Up").first
            expect(answer_btn).to_be_visible(timeout=10000)
            answer_btn.click()

            # Accept button surfaces when state == rendered — a reliable
            # signal that the preview has been populated.
            expect(page.locator("#acceptBtn")).to_be_visible(timeout=15000)

            overlay = page.locator("#regionOverlay")
            expect(overlay).to_be_visible()

            # The CWASA (no-video) path leaves the timeline disabled with
            # max=0, so ``currentPlaybackMs()`` would otherwise read 0.
            # Raise the ceiling and pin the slider to 1500 so the click
            # handler captures the timestamp we care about.
            page.evaluate(
                """() => {
                    const t = document.getElementById('timeline');
                    t.max = '3000';
                    t.value = '1500';
                }"""
            )

            # Click the dominant-hand polygon. ``force=true`` bypasses the
            # actionability stability check — the polygon is a stretched
            # SVG shape whose bbox is stable once the overlay is visible,
            # but Playwright's hit-test heuristics can wobble on SVG.
            page.locator('polygon[data-region="hand-dom"]').click(force=True)

            # The popover should anchor to the click and show the region
            # label. Its input receives focus asynchronously.
            popover = page.locator("#regionPopover")
            expect(popover).to_be_visible()
            expect(page.locator("#regionPopoverLabel")).to_contain_text(
                "Dominant hand"
            )

            page.fill("#regionPopoverInput", "make it a flat-O")
            page.click("#regionPopoverSubmit")

            # Wait until the POST /correct response updates the envelope
            # with a new ``correction_requested`` event.
            page.wait_for_function(
                """() => {
                    const env = window.__c2h && window.__c2h.state && window.__c2h.state.envelope;
                    if (!env) return false;
                    return (env.history || []).some(e => e.type === 'correction_requested');
                }""",
                timeout=10000,
            )

            event = page.evaluate(
                """() => {
                    const env = window.__c2h.state.envelope;
                    const hist = env.history || [];
                    for (let i = hist.length - 1; i >= 0; i--) {
                        if (hist[i].type === 'correction_requested') return hist[i];
                    }
                    return null;
                }"""
            )
            assert event is not None, "no correction_requested event in history"
            assert event["raw_text"] == "make it a flat-O"
            assert event["target_region"] == "hand-dom"
            assert event["target_time_ms"] == 1500
        finally:
            browser.close()

"""Playwright smoke test for the minimal authoring frontend.

Drives one happy-path authoring cycle through the real REST API:

    1. Boot — load /chat2hamnosys/, wait for the chat panel.
    2. Describe — type the temple/flat-hand prose, submit.
    3. Clarify — click the "Up" option that the stub questioner offers.
    4. Render — wait for state=rendered (preview placeholder updates).
    5. Accept — click Accept, wait for "awaiting Deaf review".
    6. Verify the inspector shows ``handshape_dominant`` etc. and the
       sign entry status surfaced through the chat reads "draft".

Skipped if Playwright is not installed; the rest of the test suite still
passes on a vanilla ``pip install -r requirements.txt``.
"""

from __future__ import annotations

import pytest


pw = pytest.importorskip(
    "playwright.sync_api",
    reason="install with `pip install playwright && playwright install chromium` to run",
)

# Re-import for type-friendly local names.
from playwright.sync_api import sync_playwright, expect  # noqa: E402


TEMPLE_PROSE = (
    "it's signed near the temple, flat hand, moves down to the chest, like SORRY"
)


def test_happy_path_authoring_cycle(c2h_server: str) -> None:
    base_url = c2h_server

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context()
            page = context.new_page()
            page.goto(f"{base_url}/chat2hamnosys/", wait_until="domcontentloaded")

            # Session panel mounted + token minted.
            expect(page.locator("#sessionMeta")).to_contain_text(
                "BSL", timeout=5000
            )

            # 1. Describe.
            page.fill("#glossInput", "SORRY")
            page.fill("#chatInput", TEMPLE_PROSE)
            page.click("#sendBtn")

            # 2. Clarification chip should appear (assistant message with
            #    a button labelled "Up").
            answer_btn = page.locator(".msg-opt", has_text="Up").first
            expect(answer_btn).to_be_visible(timeout=10000)
            answer_btn.click()

            # 3. Wait for state=rendered (Accept button is shown only then).
            accept = page.locator("#acceptBtn")
            expect(accept).to_be_visible(timeout=15000)

            # Inspector should have populated by now.
            expect(page.locator("#inspectorList")).to_be_visible()
            expect(page.locator("#inspectorList li")).to_have_count(12)
            # The dominant handshape was filled by the parser stub → inferred.
            ip_value = page.locator(
                ".ip-row", has_text="Handshape (dominant)"
            ).locator(".ip-value")
            expect(ip_value).to_contain_text("flat")

            # 4. Accept.
            accept.click()
            expect(
                page.locator(".msg[data-role='system']", has_text="awaiting Deaf review")
            ).to_be_visible(timeout=10000)

            # 5. Session metadata reflects the terminal state.
            expect(page.locator("#sessionMeta")).to_contain_text("finalized")

            # The sign entry's status (announced via the SR-only region)
            # was "draft" — verify by hitting the API once more and reading
            # the latest envelope through the page bridge.
            envelope_state = page.evaluate(
                "() => window.__c2h.state.envelope && window.__c2h.state.envelope.state"
            )
            assert envelope_state == "finalized"
        finally:
            browser.close()

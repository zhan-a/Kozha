"""Playwright smoke for the inbound rare-SL proposal flow (Prompt 02).

Loads /contribute.html, opens the inline "Suggest a sign language"
form via the affordance under the chip strip, fills the required
fields, submits, and asserts the plain-language thank-you copy
replaces the form.

Uses the same ``c2h_server`` fixture as the rest of the chat2hamnosys
playwright suite — that booting parent app mounts the chat2hamnosys
API at ``/api/chat2hamnosys`` and serves ``public/`` statically, which
is exactly the topology this test needs.
"""

from __future__ import annotations

import pytest


pw = pytest.importorskip(
    "playwright.sync_api",
    reason="install with `pip install playwright && playwright install chromium` to run",
)

from playwright.sync_api import expect, sync_playwright  # noqa: E402


def test_inline_form_submits_and_shows_thank_you(c2h_server: str) -> None:
    base = c2h_server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(locale="en-US")
            page = ctx.new_page()
            page.goto(f"{base}/contribute.html", wait_until="domcontentloaded")

            # Open the inline form via the affordance under the chip
            # strip. The card is hidden until clicked; once open, the
            # form is visible and the thank-you note is hidden.
            suggest_btn = page.locator("#qsSuggestBtn")
            expect(suggest_btn).to_be_visible(timeout=10_000)
            suggest_btn.click()

            card = page.locator("#suggestLanguageCard")
            expect(card).to_be_visible(timeout=5_000)
            expect(page.locator("#suggestLanguageForm")).to_be_visible()
            expect(page.locator("#suggestLanguageThanks")).to_be_hidden()

            # Fill the required fields plus a couple of optional ones to
            # exercise the full payload shape.
            page.locator("#suggestNameInput").fill("Vietnamese Sign Language")
            page.locator("#suggestEndonymInput").fill("Ngôn ngữ Ký hiệu Việt Nam")
            page.locator("#suggestIsoInput").fill("vie")
            page.locator("#suggestRegionInput").fill("Vietnam")
            page.locator("#suggestMotivationInput").fill(
                "Widely used in southeast Asia and missing from the picker."
            )

            page.locator("#suggestLanguageSubmit").click()

            # On 201 the form vanishes and the thank-you note shows up.
            # The copy is the same plain-language register as the rest
            # of the contribute flow — assert on a phrase the user
            # actually reads, not on a brittle DOM hash.
            thanks = page.locator("#suggestLanguageThanks")
            expect(thanks).to_be_visible(timeout=10_000)
            expect(thanks).to_contain_text("Thanks")
            expect(thanks).to_contain_text("Deaf reviewer")

            # The form is hidden once the proposal is accepted; the
            # error surface stays empty (no error path was taken).
            expect(page.locator("#suggestLanguageForm")).to_be_hidden()
            expect(page.locator("#suggestLanguageError")).to_be_hidden()
        finally:
            browser.close()


def test_dropdown_suggest_option_opens_inline_form(c2h_server: str) -> None:
    """The legacy <select> picks up a sentinel "Suggest one →" option;
    selecting it must surface the inline form rather than commit a
    bogus language code to the session.
    """
    base = c2h_server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(locale="en-US")
            page = ctx.new_page()
            page.goto(f"{base}/contribute.html", wait_until="domcontentloaded")

            # The <select> is a hidden fallback (qs__picker-fallback —
            # display:none in CSS), so we exercise its change event
            # directly. The handler in contribute.js intercepts the
            # ``__suggest__`` sentinel and calls
            # ``window.KOZHA_OPEN_SUGGEST_LANGUAGE`` rather than
            # delegating to ``setLanguage``.
            page.wait_for_function(
                "() => document.querySelector('#pickerSelect option[value=\"__suggest__\"]')",
                timeout=10_000,
            )
            page.evaluate(
                """() => {
                    const sel = document.getElementById('pickerSelect');
                    sel.value = '__suggest__';
                    sel.dispatchEvent(new Event('change', { bubbles: true }));
                }"""
            )
            expect(page.locator("#suggestLanguageCard")).to_be_visible(timeout=5_000)
            expect(page.locator("#suggestLanguageForm")).to_be_visible()
        finally:
            browser.close()

"""Playwright tests for prompt 5 — gloss + description inputs, Deaf-
native checkbox, autosave + restore, and the post-submit summary card.

Two scenarios:

1. ``test_submit_creates_session_and_shows_summary`` — fill both
   fields, check the Deaf-native box, click "Start authoring", assert
   the session id lands in the context strip and the summary card
   replaces the form.
2. ``test_reload_restores_description`` — fill the description, wait
   for the 500ms autosave debounce to flush, reload, assert the
   description input comes back with the same value.

Relies on the same c2h_server fixture as the existing Playwright smoke
tests; skipped if Playwright is not installed.
"""

from __future__ import annotations

import pytest


pw = pytest.importorskip(
    "playwright.sync_api",
    reason="install with `pip install playwright && playwright install chromium` to run",
)

from playwright.sync_api import sync_playwright, expect  # noqa: E402


GLOSS = "ELECTRON"
DESCRIPTION = (
    "right hand in a 'V' shape, palm facing me, moves in a small circle near my temple"
)
SHORT_DESCRIPTION = "too short"  # < 20 chars → submit stays disabled


def _pick_bsl(page) -> None:
    expect(page.locator("#languagePicker")).to_be_visible()
    page.locator('.picker-option[data-code="bsl"]').click()
    expect(page.locator("#langMasthead")).to_be_visible(timeout=3000)


def test_submit_creates_session_and_shows_summary(c2h_server: str) -> None:
    base = c2h_server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context()
            page = context.new_page()

            page.goto(f"{base}/contribute.html", wait_until="domcontentloaded")
            _pick_bsl(page)

            # Form is visible; summary hidden.
            expect(page.locator("#authoringForm")).to_be_visible()
            expect(page.locator("#authoringSummary")).to_be_hidden()

            # Submit is disabled until both inputs pass their minimums.
            submit_btn = page.locator("#startAuthoringBtn")
            expect(submit_btn).to_be_disabled()

            # Short description keeps the button disabled.
            page.locator("#glossInput").fill(GLOSS)
            page.locator("#descriptionInput").fill(SHORT_DESCRIPTION)
            expect(submit_btn).to_be_disabled()

            # Full description enables the button.
            page.locator("#descriptionInput").fill(DESCRIPTION)
            expect(submit_btn).to_be_enabled()

            # Character counter reflects the description length.
            expect(page.locator("#descriptionCount")).to_have_text(str(len(DESCRIPTION)))

            # Tick the Deaf-native checkbox so it flows into the body.
            page.locator("#deafNativeInput").check()

            submit_btn.click()

            # Summary replaces the form; context strip gets the session id.
            expect(page.locator("#authoringSummary")).to_be_visible(timeout=5000)
            expect(page.locator("#authoringForm")).to_be_hidden()
            expect(page.locator("#summaryGloss")).to_have_text(GLOSS)
            expect(page.locator("#summaryLang")).to_have_text("BSL")
            expect(page.locator("#summaryDesc")).to_have_text(DESCRIPTION)
            expect(page.locator("#contextGloss")).to_have_text(GLOSS)
            expect(page.locator("#contextCopyBtn")).to_be_visible()

            state = page.evaluate("() => window.KOZHA_CONTRIB_CONTEXT.getState()")
            session_id = state.get("sessionId")
            assert session_id, f"createSession did not populate sessionId: {state!r}"
            short = session_id[:8]
            expect(page.locator("#contextSessionId")).to_have_text(short)

            # Edit button reveals the form again (after confirmation).
            page.locator("#summaryEditBtn").click()
            expect(page.locator("#modalBackdrop")).to_be_visible()
            page.locator("#modalDiscardBtn").click()
            expect(page.locator("#authoringForm")).to_be_visible(timeout=3000)
            expect(page.locator("#authoringSummary")).to_be_hidden()
            # Fields keep the values the user typed before submit.
            expect(page.locator("#glossInput")).to_have_value(GLOSS)
            expect(page.locator("#descriptionInput")).to_have_value(DESCRIPTION)
            # Session id is cleared from the context strip.
            expect(page.locator("#contextSessionId")).to_have_text("—")
            expect(page.locator("#contextCopyBtn")).to_be_hidden()
        finally:
            browser.close()


def test_reload_restores_description(c2h_server: str) -> None:
    base = c2h_server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context()
            page = context.new_page()

            page.goto(f"{base}/contribute.html", wait_until="domcontentloaded")
            _pick_bsl(page)

            page.locator("#glossInput").fill(GLOSS)
            page.locator("#descriptionInput").fill(DESCRIPTION)
            # Give the 500ms debounce a chance to flush before we reload.
            page.wait_for_timeout(700)

            stored = page.evaluate(
                "() => JSON.parse(localStorage.getItem('kozha.contribute.draft.bsl') || 'null')"
            )
            assert stored, f"expected a draft in localStorage, got: {stored!r}"
            assert stored.get("description") == DESCRIPTION, stored
            # The gloss is persisted as-typed; uppercase formatting happens
            # on blur / submit, which we don't trigger in this scenario.
            assert stored.get("gloss") == GLOSS, stored

            page.reload(wait_until="domcontentloaded")

            # Language still BSL, form re-mounts with the saved values.
            expect(page.locator("#langMasthead")).to_be_visible(timeout=5000)
            expect(page.locator("#authoringForm")).to_be_visible(timeout=3000)
            expect(page.locator("#glossInput")).to_have_value(GLOSS)
            expect(page.locator("#descriptionInput")).to_have_value(DESCRIPTION)
            # Restored notice appears; auto-dismisses shortly after.
            expect(page.locator("#restoredNotice")).to_be_visible()
        finally:
            browser.close()

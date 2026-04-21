"""Playwright smoke test for the persistent language header, context
strip, and session URL carrier built in prompt 4.

Drives the four scenarios the prompt calls out:

(a) First visit — only the language picker is rendered; neither the
    masthead nor the resume prompt is visible.
(b) After picking BSL — the language header + context strip mount in
    place of the picker; code / name / gloss / state / session-id all
    show their empty-state values.
(c) After a dummy description submission — the store receives a
    session_id + token from POST /sessions, the context strip renders
    the short id, and the URL fragment matches.
(d) Reloading the page — sessionStorage keeps the session id + token,
    the fragment is re-parsed, a GET /sessions/{id} rehydrates gloss
    and state, and the masthead reappears with the same values.

Relies on the c2h_server fixture (conftest.py) which boots the
chat2hamnosys API with stubbed parser / questioner / applier so no
LLM is hit. Skipped if Playwright is not installed.
"""

from __future__ import annotations

import pytest


pw = pytest.importorskip(
    "playwright.sync_api",
    reason="install with `pip install playwright && playwright install chromium` to run",
)

from playwright.sync_api import sync_playwright, expect  # noqa: E402


DUMMY_PROSE = "flat hand near the temple moves down"


def test_language_header_and_session_url(c2h_server: str) -> None:
    base = c2h_server
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context()
            page = context.new_page()

            # ---- (a) First visit: only picker visible ----
            page.goto(f"{base}/contribute.html", wait_until="domcontentloaded")
            expect(page.locator("#languagePicker")).to_be_visible()
            expect(page.locator("#langMasthead")).to_be_hidden()
            expect(page.locator("#tokenPrompt")).to_be_hidden()
            expect(page.locator("#pickerOptions .picker-option")).to_have_count(8)

            # ---- (b) Pick BSL: header + context strip mount ----
            page.locator('.picker-option[data-code="bsl"]').click()
            expect(page.locator("#langMasthead")).to_be_visible(timeout=3000)
            expect(page.locator("#languagePicker")).to_be_hidden()
            expect(page.locator("#languageBadgeCode")).to_have_text("BSL")
            expect(page.locator("#languageBadgeName")).to_have_text(
                "British Sign Language"
            )
            expect(page.locator("#contextGloss")).to_have_text("No sign selected")
            expect(page.locator("#contextState")).to_have_text("Draft")
            # No session yet — short id shows the placeholder em-dash, copy
            # affordance is hidden.
            expect(page.locator("#contextSessionId")).to_have_text("—")
            expect(page.locator("#contextCopyBtn")).to_be_hidden()

            # ---- (c) Dummy description submission populates session id ----
            state = page.evaluate(
                """async ({prose, gloss}) => {
                    await window.KOZHA_CONTRIB_CONTEXT.createSession({prose, gloss});
                    return window.KOZHA_CONTRIB_CONTEXT.getState();
                }""",
                {"prose": DUMMY_PROSE, "gloss": "TEST"},
            )
            session_id = state.get("sessionId")
            assert session_id, f"createSession did not populate sessionId: {state!r}"
            assert state.get("sessionToken"), "createSession did not persist token"
            short = session_id[:8]

            expect(page.locator("#contextSessionId")).to_have_text(short)
            expect(page.locator("#contextCopyBtn")).to_be_visible()
            expect(page.locator("#contextGloss")).to_have_text("TEST")

            hash_value = page.evaluate("() => window.location.hash")
            assert hash_value == f"#s/{session_id}", (
                f"expected #s/{session_id}, got {hash_value!r}"
            )

            # sessionStorage must carry the subset the prompt specifies —
            # language + sessionId + sessionToken, and nothing else load-
            # bearing (gloss / state are rehydrated from the envelope).
            persisted = page.evaluate(
                "() => JSON.parse(sessionStorage.getItem('kozha.contribute.context') || 'null')"
            )
            assert persisted == {
                "language":     "bsl",
                "sessionId":    session_id,
                "sessionToken": state["sessionToken"],
            }, persisted

            # ---- (d) Reload restores header + strip from sessionStorage ----
            page.reload(wait_until="domcontentloaded")
            expect(page.locator("#langMasthead")).to_be_visible(timeout=5000)
            expect(page.locator("#languageBadgeCode")).to_have_text("BSL")
            expect(page.locator("#contextSessionId")).to_have_text(short)
            # Gloss is not persisted directly; it comes back from the
            # resume GET /sessions/{id} envelope.
            expect(page.locator("#contextGloss")).to_have_text("TEST")
            # The fragment survives reload and still carries the same id.
            reloaded_hash = page.evaluate("() => window.location.hash")
            assert reloaded_hash == f"#s/{session_id}", reloaded_hash
        finally:
            browser.close()

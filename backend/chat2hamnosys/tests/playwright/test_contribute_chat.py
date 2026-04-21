"""Playwright test for prompt 6 — minimalist clarification chat panel.

The contribute page's chat panel is exercised end-to-end with the API
stubbed via ``page.route()`` so the test asserts pure UI behaviour
without depending on the LLM stack:

1. Pick BSL, fill the gloss + description, click "Start authoring".
2. ``POST /sessions`` is intercepted to return a session id + token.
3. The chained ``POST /sessions/<id>/describe`` returns a CLARIFYING
   envelope with two pending questions.
4. Assert the first question text + its option buttons render in
   ``#chatLog`` / ``#chatOptions``.
5. Click the first option. Assert the outgoing ``POST /answer`` carries
   ``{question_id: q1_field, answer: q1_value}`` and stub the response
   to a CLARIFYING envelope with only the second question.
6. Assert the second question replaces the first in the log, with its
   own options.
7. Click an option for the second question. Stub the response to a
   GENERATING envelope, then assert the input disables and shows the
   "Generating sign…" placeholder.

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


GLOSS = "ELECTRON"
DESCRIPTION = (
    "right hand in a 'V' shape, palm facing me, moves in a small circle near my temple"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _question(field: str, text: str, options: list[tuple[str, str]]) -> dict[str, Any]:
    return {
        "question_id": field,
        "field": field,
        "text": text,
        "options": [{"label": label, "value": value} for label, value in options],
        "allow_freeform": True,
        "rationale": "stub",
    }


def _envelope(
    session_id: str,
    state: str,
    *,
    pending: list[dict[str, Any]] | None = None,
    clarifications: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Minimal SessionEnvelope-shaped JSON for the frontend.

    The frontend only reads state / gloss / pending_questions /
    clarifications, so a partial dict is enough — we don't need to
    reproduce every Pydantic field.
    """
    return {
        "session_id": session_id,
        "state": state,
        "gloss": GLOSS,
        "sign_language": "bsl",
        "description_prose": DESCRIPTION,
        "pending_questions": pending or [],
        "clarifications": clarifications or [],
        "created_at": _now(),
        "last_activity_at": _now(),
    }


def _pick_bsl(page) -> None:
    expect(page.locator("#languagePicker")).to_be_visible()
    page.locator('.picker-option[data-code="bsl"]').click()
    expect(page.locator("#langMasthead")).to_be_visible(timeout=3000)


def test_chat_panel_renders_two_questions_then_generating(c2h_server: str) -> None:
    base = c2h_server
    session_id = str(uuid.uuid4())
    session_token = "stub-token"

    q1 = _question(
        field="orientation_extended_finger",
        text="Which way do the fingers point?",
        options=[("Up", "up"), ("Forward", "forward"), ("To the side", "side")],
    )
    q2 = _question(
        field="movement_repetition",
        text="Does the movement repeat?",
        options=[("Once", "once"), ("Twice", "twice")],
    )

    answer_calls: list[dict[str, Any]] = []
    describe_calls: list[dict[str, Any]] = []

    def handle(route: Route) -> None:
        req = route.request
        path = req.url.split("?", 1)[0].split(base, 1)[-1]
        method = req.method
        body: dict[str, Any] = {}
        if method == "POST":
            try:
                body = json.loads(req.post_data or "{}")
            except (TypeError, ValueError):
                body = {}

        if method == "POST" and path == "/api/chat2hamnosys/sessions":
            payload = {
                "session_id": session_id,
                "state": "awaiting_description",
                "session_token": session_token,
                "session": _envelope(session_id, "awaiting_description"),
            }
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )
            return

        if method == "POST" and path.endswith("/describe"):
            describe_calls.append(body)
            payload = _envelope(session_id, "clarifying", pending=[q1, q2])
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )
            return

        if method == "POST" and path.endswith("/answer"):
            answer_calls.append(body)
            if len(answer_calls) == 1:
                payload = _envelope(session_id, "clarifying", pending=[q2])
            else:
                payload = _envelope(session_id, "generating", pending=[])
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )
            return

        # Anything else (static assets, languages JSON, healthz) goes
        # through to the real test server.
        route.continue_()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context()
            page = context.new_page()
            page.route("**/api/chat2hamnosys/**", handle)

            page.goto(f"{base}/contribute.html", wait_until="domcontentloaded")
            _pick_bsl(page)

            # Fill the gloss + description and submit.
            page.locator("#glossInput").fill(GLOSS)
            page.locator("#descriptionInput").fill(DESCRIPTION)
            expect(page.locator("#startAuthoringBtn")).to_be_enabled()
            page.locator("#startAuthoringBtn").click()

            # Chat panel mounts after the session is created.
            expect(page.locator("#chatPanel")).to_be_visible(timeout=5000)

            # /describe was called with the prose payload.
            page.wait_for_function("() => document.querySelectorAll('#chatLog .chat-msg-system').length >= 1")
            assert describe_calls, "expected /describe to be called"
            assert describe_calls[0].get("prose") == DESCRIPTION, describe_calls

            # First question: text + three option buttons render.
            first_msg = page.locator("#chatLog .chat-msg-system").first
            expect(first_msg.locator(".chat-msg-text")).to_have_text(q1["text"])
            expect(first_msg.locator(".chat-msg-label")).to_have_text("Clarification:")
            options_now = page.locator("#chatOptions .chat-option-btn")
            expect(options_now).to_have_count(3)
            expect(options_now.nth(0)).to_have_text("Up")
            expect(options_now.nth(1)).to_have_text("Forward")
            expect(options_now.nth(2)).to_have_text("To the side")

            # Click the first option → outgoing /answer carries q1 field +
            # the canonical value, and the second question replaces the
            # first in the log.
            options_now.nth(0).click()

            page.wait_for_function(
                "() => document.querySelectorAll('#chatLog .chat-msg-system').length >= 2"
            )
            assert len(answer_calls) == 1, answer_calls
            assert answer_calls[0] == {
                "question_id": q1["field"],
                "answer": "up",
            }, answer_calls[0]

            # User echo appears between the two system messages.
            you_msgs = page.locator("#chatLog .chat-msg-you")
            expect(you_msgs).to_have_count(1)
            expect(you_msgs.first.locator(".chat-msg-text")).to_have_text("Up")

            # Second question + its two options now render.
            second_msg = page.locator("#chatLog .chat-msg-system").nth(1)
            expect(second_msg.locator(".chat-msg-text")).to_have_text(q2["text"])
            options_now = page.locator("#chatOptions .chat-option-btn")
            expect(options_now).to_have_count(2)
            expect(options_now.nth(0)).to_have_text("Once")
            expect(options_now.nth(1)).to_have_text("Twice")

            # Click an option for the second question → outgoing /answer
            # carries q2's field, response transitions to GENERATING.
            options_now.nth(1).click()

            page.wait_for_function(
                "() => document.getElementById('chatInput').disabled === true"
            )
            assert len(answer_calls) == 2, answer_calls
            assert answer_calls[1] == {
                "question_id": q2["field"],
                "answer": "twice",
            }, answer_calls[1]

            # GENERATING state: input is disabled, placeholder switches,
            # option buttons are hidden, and the "Preparing preview"
            # transition message is in the log.
            chat_input = page.locator("#chatInput")
            expect(chat_input).to_be_disabled()
            expect(chat_input).to_have_attribute("placeholder", "Generating sign…")
            expect(page.locator("#chatOptions")).to_be_hidden()

            system_texts = page.locator("#chatLog .chat-msg-system .chat-msg-text")
            generating_present = page.evaluate(
                """() => {
                    const nodes = document.querySelectorAll(
                        '#chatLog .chat-msg-system .chat-msg-text'
                    );
                    for (const n of nodes) {
                        if (n.textContent.indexOf('Preparing preview') !== -1) return true;
                    }
                    return false;
                }"""
            )
            assert generating_present, (
                "expected GENERATING transition message in chat log; "
                f"got {[t for t in system_texts.all_text_contents()]}"
            )
        finally:
            browser.close()

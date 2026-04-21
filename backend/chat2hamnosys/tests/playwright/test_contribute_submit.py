"""Playwright test for prompt 10 — submission flow + status URL.

Drives the page through:

1. Language pick → gloss + description → POST /sessions + /describe
   lands a RENDERED envelope so the submission panel becomes visible
   and the Submit button enables.
2. Click "Submit for review" → POST /accept returns a sign entry with
   ``status=pending_review``.  The UI flips to the confirmation view
   (heading, body copy, permanent URL populated).
3. Click "Copy link" → the clipboard gets the
   ``/contribute/status/<id>`` URL.
4. Navigate to ``/contribute/status/<id>`` directly.  With a stored
   session token the status page shows the private description and the
   reviewer comment we planted in the stub.  Clearing the token and
   reloading exercises the unauthenticated path: no description, no
   reviewer comment, and the resume-token gate is visible.

All chat2hamnosys HTTP is stubbed via ``page.route`` so the test stays
offline and independent of the reviewer store / LLM stack.

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

from playwright.sync_api import Route, expect, sync_playwright  # noqa: E402


HAMNOSYS = ""
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
    "movement": [
        {"path": "down", "size_mod": None, "speed_mod": None, "repeat": None}
    ],
}

GLOSS = "TEMPLE"
DESCRIPTION = "flat hand, palm down, near the temple, moves straight down"
REVIEWER_COMMENT = "Looks good — handshape reads cleanly."


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
        "hamnosys": None,
        "sigml": None,
        "parameters": None,
        "generation_errors": [],
        "corrections_count": 0,
        "history": [],
        "created_at": _now(),
        "last_activity_at": _now(),
        "next_action": {"kind": "await_description", "questions": [], "preview": None},
    }
    base.update(extra)
    return base


def _pick_bsl(page) -> None:
    expect(page.locator("#languagePicker")).to_be_visible()
    page.locator('.picker-option[data-code="bsl"]').click()
    expect(page.locator("#langMasthead")).to_be_visible(timeout=3000)


def test_submission_flow_and_status_url(c2h_server: str) -> None:
    base = c2h_server
    session_id = str(uuid.uuid4())
    sign_id = str(uuid.uuid4())
    session_token = "stub-submit-token"

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
            env = _envelope(
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
                body=json.dumps(env),
            )
            return

        if method == "POST" and path.endswith("/accept"):
            env = _envelope(
                session_id,
                "finalized",
                hamnosys=HAMNOSYS,
                sigml=SIGML_XML,
                parameters=PARAMETERS,
            )
            payload = {
                "sign_entry": {
                    "id":               sign_id,
                    "gloss":            GLOSS,
                    "sign_language":    "bsl",
                    "domain":           None,
                    "hamnosys":         HAMNOSYS,
                    "sigml":            SIGML_XML,
                    "status":           "pending_review",
                    "parameters":       PARAMETERS,
                    "regional_variant": None,
                    "created_at":       _now(),
                    "updated_at":       _now(),
                },
                "session": env,
            }
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(payload),
            )
            return

        if method == "GET" and path.endswith("/status"):
            has_token = req.headers.get("x-session-token") == session_token
            envelope = {
                "session_id":         session_id,
                "sign_id":            sign_id,
                "gloss":              GLOSS,
                "sign_language":      "bsl",
                "regional_variant":   None,
                "status":             "pending_review",
                "hamnosys":           HAMNOSYS if has_token else None,
                "sigml":              SIGML_XML if has_token else None,
                "rejection_category": None,
                "description_prose":  DESCRIPTION if has_token else None,
                "reviewer_comments":  (
                    [
                        {
                            "verdict":     "pending",
                            "category":    None,
                            "comment":     REVIEWER_COMMENT,
                            "reviewed_at": _now(),
                        }
                    ]
                    if has_token
                    else []
                ),
                "corrections_count":  0,
                "has_token":          has_token,
                "created_at":         _now(),
                "updated_at":         _now(),
            }
            route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(envelope),
            )
            return

        # Let everything else fall through — /events SSE, /symbols, static
        # assets. Our fake session id won't resolve against the real SSE
        # endpoint but the UI no-ops on that, which is fine.
        route.continue_()

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            context = browser.new_context()
            context.grant_permissions(
                ["clipboard-read", "clipboard-write"],
                origin=base,
            )
            page = context.new_page()
            page.route("**/api/chat2hamnosys/**", handle)

            # 1. Language pick + form submit → RENDERED envelope lands.
            page.goto(f"{base}/contribute.html", wait_until="domcontentloaded")
            _pick_bsl(page)

            page.locator("#glossInput").fill(GLOSS)
            page.locator("#descriptionInput").fill(DESCRIPTION)
            page.locator("#deafNativeInput").check()
            expect(page.locator("#startAuthoringBtn")).to_be_enabled()
            page.locator("#startAuthoringBtn").click()

            expect(page.locator("#notationPanel")).to_be_visible(timeout=5000)
            expect(page.locator("#submissionPanel")).to_be_visible()

            # 2. The "sign generated and valid" row flips to complete
            #    once the rendered envelope lands. Submit enables.
            expect(
                page.locator('.submission-item[data-key="sign"].is-complete')
            ).to_be_visible()
            expect(page.locator("#submissionSubmitBtn")).to_be_enabled()

            # Optional rows are tracked too — gloss, language, description,
            # and the deaf-native self-ID we just checked should all read
            # complete. "At least one correction" stays missing because
            # we haven't issued one.
            for key in ("gloss", "language", "description", "deafNative"):
                expect(
                    page.locator(f'.submission-item[data-key="{key}"].is-complete')
                ).to_be_visible()
            expect(
                page.locator('.submission-item[data-key="correction"].is-missing')
            ).to_be_visible()

            # 3. Submit → confirmation view replaces the authoring area.
            page.locator("#submissionSubmitBtn").click()

            expect(page.locator("#submissionConfirmation")).to_be_visible(timeout=5000)
            expect(page.locator("#authoring-root")).to_be_hidden()

            # Heading reflects the gloss + language (upper-cased code).
            expect(page.locator("#confirmationHeading")).to_have_text(
                f"Submitted: {GLOSS} in BSL"
            )
            # Body text: pending_review copy with the 3-day wording.
            expect(page.locator("#confirmationBody")).to_contain_text(
                "review queue for BSL"
            )
            expect(page.locator("#confirmationBody")).to_contain_text(
                "typical review time is 3 days"
            )

            # Permanent URL is populated with /contribute/status/<id>.
            expected_url = f"{base}/contribute/status/{session_id}"
            expect(page.locator("#confirmationUrl")).to_have_value(expected_url)

            # 4. Copy link → clipboard.
            page.locator("#confirmationCopyBtn").click()
            expect(page.locator("#confirmationCopyConfirm")).to_be_visible()
            clipboard_value = page.evaluate("() => navigator.clipboard.readText()")
            assert clipboard_value == expected_url, (
                f"expected status URL in clipboard; got {clipboard_value!r}"
            )

            # 5. Navigate to the status URL with the token in sessionStorage
            #    (the submission flow left it there). Private fields visible.
            page.goto(expected_url, wait_until="domcontentloaded")
            expect(page.locator("#statusBody")).to_be_visible(timeout=5000)
            expect(page.locator("#statusGloss")).to_have_text(GLOSS)
            expect(page.locator("#statusLanguage")).to_have_text(
                "British Sign Language"
            )
            expect(page.locator("#statusState")).to_have_text("Pending review")
            # Description prose is token-gated and should render.
            expect(page.locator("#statusPrivate")).to_be_visible()
            expect(page.locator("#statusDescription")).to_have_text(DESCRIPTION)
            # Reviewer comment section is visible with our planted comment.
            expect(page.locator("#statusComments")).to_be_visible()
            expect(page.locator("#statusCommentsList .status-comment-body")).to_have_text(
                REVIEWER_COMMENT
            )
            # Notation section also visible (we sent hamnosys in the
            # token-bearing envelope).
            expect(page.locator("#statusNotation")).to_be_visible()
            expect(page.locator("#statusHamnosys")).to_have_text(HAMNOSYS)
            # Unauthenticated token-gate should be hidden.
            expect(page.locator("#statusTokenGate")).to_be_hidden()

            # 6. Drop the stored token and reload — the public view only.
            page.evaluate("() => sessionStorage.clear()")
            page.goto(expected_url, wait_until="domcontentloaded")
            expect(page.locator("#statusBody")).to_be_visible(timeout=5000)
            expect(page.locator("#statusState")).to_have_text("Pending review")
            # Public envelope: no description, no reviewer comments,
            # no notation.
            expect(page.locator("#statusPrivate")).to_be_hidden()
            expect(page.locator("#statusComments")).to_be_hidden()
            expect(page.locator("#statusNotation")).to_be_hidden()
            # The resume-token form is offered to the unauthenticated caller.
            expect(page.locator("#statusTokenGate")).to_be_visible()
        finally:
            browser.close()

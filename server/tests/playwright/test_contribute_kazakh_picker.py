"""Playwright smoke for the rare-SL group in the contribute picker.

Loads /contribute.html, opens the "More languages" panel, confirms the
new Kazakh Sign Language chip is rendered, picks it, and verifies the
page transitions into the authoring workspace without a console error.

Skipped if Playwright is not installed. Run with::

    pip install playwright
    playwright install chromium
    pytest server/tests/playwright/test_contribute_kazakh_picker.py
"""
from __future__ import annotations

import pytest


pw = pytest.importorskip(
    "playwright.sync_api",
    reason="install with `pip install playwright && playwright install chromium` to run",
)

from playwright.sync_api import sync_playwright, expect  # noqa: E402


def test_kazakh_appears_in_more_languages_and_is_selectable(kozha_server: str) -> None:
    """KSL must render in the rare-SL chip group and clicking it activates
    the authoring workspace without raising a JS error.
    """
    console_errors: list[str] = []
    page_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(locale="en-US")
            page = ctx.new_page()
            page.on(
                "console",
                lambda msg: console_errors.append(msg.text)
                if msg.type == "error"
                else None,
            )
            page.on("pageerror", lambda exc: page_errors.append(str(exc)))

            page.goto(
                f"{kozha_server}/contribute.html",
                wait_until="domcontentloaded",
            )

            # Reveal the secondary chip row.
            more_btn = page.locator("#qsMoreBtn")
            expect(more_btn).to_be_visible(timeout=10_000)
            more_btn.click()

            # The KSL chip is rendered from /contribute-languages.json on
            # load, then bound the same way the primary chips are.
            ksl_chip = page.locator('.qs__chip[data-lang="ksl"]')
            expect(ksl_chip).to_be_visible(timeout=10_000)
            # Rare-group chips carry the qs__chip--rare class and a
            # "(seed)" tag so the contributor can tell what kind of
            # corpus they're walking into.
            expect(ksl_chip).to_have_class(
                # Allow extra whitespace/classes around the marker class.
                # Using a regex avoids brittleness against ordering.
                __import__("re").compile(r"qs__chip--rare"),
            )

            # Picking KSL must propagate to the hidden #pickerSelect
            # (which contribute.js listens on) and reveal the authoring
            # masthead just like the primary chips do.
            ksl_chip.click()
            expect(page.locator("#pickerSelect")).to_have_value(
                "ksl", timeout=5_000
            )
            expect(page.locator("#langMasthead")).to_be_visible(timeout=10_000)
            expect(page.locator("#authoring-root")).to_be_visible(timeout=10_000)

            # The reviewer-notice banner is shown for languages with no
            # Deaf reviewers — KSL falls into that bucket as a seed.
            expect(page.locator("#reviewerNotice")).to_be_visible(timeout=5_000)

            assert not page_errors, f"unexpected page errors: {page_errors}"
            # Console errors are noisier (network probes, third-party
            # scripts) — only flag a clearly fatal pattern.
            fatal = [
                e for e in console_errors
                if "Uncaught" in e or "TypeError" in e or "ReferenceError" in e
            ]
            assert not fatal, f"fatal console errors: {fatal}"
        finally:
            browser.close()

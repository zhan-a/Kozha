"""Playwright scenarios for the three-slot language selectors on /app.html.

Covers the four scenarios in prompt 4 §12:

1. Default load shows a sensible source + target.
2. Changing the target dropdown updates the route display + coverage note.
3. Typing text with a word not in the target corpus yields a visually
   distinct fingerspelled segment and bumps the miss counter.
4. Selecting an unsupported source/target pair disables the translate
   button with an explanatory message.

Skipped if Playwright is not installed. Run with::

    pip install playwright
    playwright install chromium
    pytest server/tests/playwright/test_lang_selectors.py
"""
from __future__ import annotations

import pytest


pw = pytest.importorskip(
    "playwright.sync_api",
    reason="install with `pip install playwright && playwright install chromium` to run",
)

from playwright.sync_api import sync_playwright, expect  # noqa: E402


def _wait_for_db_loaded(page) -> None:
    """Wait until switchSignLanguage() has populated glossToSign/letterToSign."""
    page.wait_for_function(
        "typeof glossToSign !== 'undefined' "
        "&& glossToSign.size > 0 "
        "&& typeof letterToSign !== 'undefined' "
        "&& letterToSign.size > 0",
        timeout=30000,
    )


def test_default_load_shows_sensible_source_and_target(kozha_server: str) -> None:
    """Prompt 4 §12.a — first-paint source + target are well-formed."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(locale="en-US")
            page = ctx.new_page()
            page.goto(f"{kozha_server}/app.html", wait_until="domcontentloaded")
            # The three-slot strip is rendered.
            expect(page.locator(".lang-strip")).to_be_visible()
            # An English browser locale resolves to en → BSL (universal
            # default). This is the English→BSL path that prompt 4's deploy
            # safety requires to still work with zero extra clicks.
            expect(page.locator("#langHint")).to_have_value("en")
            expect(page.locator("#signLangSelect")).to_have_value("bsl")
            # Route text exists and describes a direct BSL lookup.
            route = page.locator("#translationRoute")
            expect(route).to_contain_text("BSL")
            expect(route).to_contain_text("Direct lookup")
            # Translate button is enabled on the default path.
            expect(page.locator("#translateBtn")).to_be_enabled()
        finally:
            browser.close()


def test_changing_target_updates_route_and_coverage(kozha_server: str) -> None:
    """Prompt 4 §12.b — selecting a new target updates route + coverage note."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(locale="en-US")
            page = ctx.new_page()
            page.goto(f"{kozha_server}/app.html", wait_until="domcontentloaded")
            page.select_option("#signLangSelect", value="lsf")
            # Route now announces the translation step.
            route = page.locator("#translationRoute")
            expect(route).to_contain_text("LSF")
            expect(route).to_contain_text("English")
            expect(route).to_contain_text("French")
            # Coverage note under the target dropdown reflects the LSF size
            # from docs/polish/01-database-inventory.md (381 signs).
            note = page.locator("#signLangNote")
            expect(note).to_contain_text("381")
            # Switching to a limited-coverage language surfaces the
            # "limited" copy.
            page.select_option("#signLangSelect", value="ngt")
            expect(note).to_contain_text("limited")
            # And to a corpus-less language ("fsl") surfaces honest
            # no-coverage copy (FSL has neither signs nor an alphabet).
            page.select_option("#signLangSelect", value="fsl")
            expect(note).to_contain_text("No signs")
        finally:
            browser.close()


def test_fingerspelled_segment_is_visually_distinct(kozha_server: str) -> None:
    """Prompt 4 §12.c — a word not in the corpus renders as fingerspelled
    and bumps the bottom-of-output counter.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(locale="en-US")
            page = ctx.new_page()
            page.goto(f"{kozha_server}/app.html", wait_until="domcontentloaded")
            _wait_for_db_loaded(page)
            # Drive buildSigml + the UI updaters directly so the test does
            # not depend on argostranslate or the CWASA avatar being ready.
            result = page.evaluate(
                """() => {
                    const tokens = ['zzqqxxyy'];
                    const sigml = buildSigml(tokens);
                    showTokenChips(tokens, window.__lastPerToken);
                    updateCoverageCounter(window.__lastStats);
                    const chip = document.querySelector('#tokenList .token-chip');
                    return {
                        sigmlIsString: typeof sigml === 'string',
                        kind: chip ? chip.dataset.kind : null,
                        hasFingerspelledClass: chip ? chip.classList.contains('fingerspelled') : false,
                        counterText: document.getElementById('coverageCounter').textContent,
                        stats: window.__lastStats,
                    };
                }"""
            )
            assert result["sigmlIsString"], "buildSigml should return a SiGML string for fingerspellable input"
            assert result["kind"] == "fingerspelled", f"expected kind=fingerspelled, got {result['kind']!r}"
            assert result["hasFingerspelledClass"] is True
            assert "fingerspelled" in result["counterText"].lower()
            assert result["stats"]["fingerspelled"] == 1
            assert result["stats"]["total"] == 1
        finally:
            browser.close()


def test_unsupported_source_disables_translate(kozha_server: str) -> None:
    """Prompt 4 §12.d — an unsupported source language disables translate
    and explains why in the route display.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(locale="en-US")
            page = ctx.new_page()
            page.goto(f"{kozha_server}/app.html", wait_until="domcontentloaded")
            # Thai is not in ARGOS_SUPPORTED; target remains BSL (base=en),
            # so the en/th pair has no translation route.
            page.select_option("#langHint", value="th")
            btn = page.locator("#translateBtn")
            expect(btn).to_be_disabled()
            route = page.locator("#translationRoute")
            expect(route).to_contain_text("No translation path available")
            # Recovering by picking a supported source re-enables translate.
            page.select_option("#langHint", value="en")
            expect(btn).to_be_enabled()
            expect(route).not_to_contain_text("No translation path available")
        finally:
            browser.close()

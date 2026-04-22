"""Playwright coverage for the public progress dashboard (prompt 8).

Scenarios:

1. ``/progress`` loads and the four top-line numbers render as
   non-negative integers (or the honest em-dash, never a blank).
2. Clicking a sortable column header re-orders the language table in
   the direction indicated by the new sort state.
3. Clicking a "Help wanted" chip navigates to /contribute.html with
   ``?lang=<code>&gloss=<word>`` and lands with the language selected
   and the gloss pre-filled in the authoring form.

Skipped when Playwright isn't installed so the suite stays green on
minimal CI images. Run with::

    pip install playwright && playwright install chromium
    pytest server/tests/playwright/test_progress_dashboard.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
PUBLIC_DIR = REPO_ROOT / "public"

pw = pytest.importorskip(
    "playwright.sync_api",
    reason="install with `pip install playwright && playwright install chromium` to run",
)

from playwright.sync_api import expect, sync_playwright  # noqa: E402


@pytest.fixture(scope="module")
def ensure_snapshot() -> Path:
    """Guarantee public/progress_snapshot.json exists before the page loads.

    The dashboard degrades to a "snapshot unavailable" state when the
    JSON is missing, which would mask bugs in the happy-path renderer.
    Re-generating here keeps the test hermetic even if a previous run
    or a manual edit removed the file. Shells out to ``python -m
    server.progress_snapshot`` rather than importing the module because
    ``server/server.py`` shadows the ``server`` namespace-package name
    in the plain-import path and there's no clean way around that.
    """
    out = PUBLIC_DIR / "progress_snapshot.json"
    subprocess.run(
        [sys.executable, "-m", "server.progress_snapshot"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    )
    assert out.exists(), f"snapshot not written to {out}"
    return out


def test_progress_page_loads_with_top_line_numbers(
    kozha_server: str, ensure_snapshot: Path
) -> None:
    """§1 — dashboard lands, four stats render, all non-negative numbers."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(locale="en-US")
            page = ctx.new_page()
            page.goto(f"{kozha_server}/progress", wait_until="domcontentloaded")

            # Title is the first signal the route resolved to the real page.
            expect(page).to_have_title(
                "Progress — Bridgn", timeout=10_000,
            )

            # Wait for snapshot fetch to populate the stats. Until the
            # fetch resolves, the numbers read as "—".
            page.wait_for_function(
                "document.querySelector('[data-field=\"signs\"]') "
                "&& document.querySelector('[data-field=\"signs\"]').textContent.trim() !== '—' "
                "&& document.querySelector('[data-field=\"signs\"]').textContent.trim() !== '' "
                "&& document.querySelector('[data-field=\"signs\"]').textContent.trim() !== 'Loading…'",
                timeout=15_000,
            )

            for field in ("signs", "languages", "reviewed", "awaiting"):
                text = page.locator(f'[data-field="{field}"]').inner_text().strip()
                assert text, f"top-line stat {field} is empty"
                # Allowed values: a locale-formatted non-negative integer
                # (e.g. "9,679") or the honest em-dash.
                if text == "—":
                    continue
                digits = text.replace(",", "")
                assert digits.isdigit(), (
                    f"top-line stat {field} is not a non-negative integer: {text!r}"
                )
                assert int(digits) >= 0, (
                    f"top-line stat {field} is negative: {text!r}"
                )

            # Table should have at least one language row (snapshot must
            # include real data or the test environment is misconfigured).
            rows = page.locator("#progressTableBody tr:not(.progress-table-empty)")
            expect(rows.first).to_be_visible(timeout=5_000)
            assert rows.count() > 0, "language table rendered no rows"
        finally:
            browser.close()


def test_progress_table_sorts_by_column_click(
    kozha_server: str, ensure_snapshot: Path
) -> None:
    """§2 — clicking a header flips the sort, order actually changes."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(locale="en-US")
            page = ctx.new_page()
            page.goto(f"{kozha_server}/progress", wait_until="domcontentloaded")

            page.wait_for_function(
                "document.querySelectorAll('#progressTableBody tr:not(.progress-table-empty)').length > 1",
                timeout=15_000,
            )

            # Capture the pre-click order of the first column (the
            # language-code span inside each row's th).
            def _column_values() -> list[str]:
                return page.locator(
                    "#progressTableBody tr:not(.progress-table-empty) .progress-lang-code"
                ).all_inner_texts()

            before = _column_values()
            assert len(before) >= 2, (
                f"expected ≥2 language rows to test sorting; got {before}"
            )

            # Default sort is total desc. Clicking the "Language" column
            # switches to an ascending alphabetical sort. If the default
            # order already happens to match that, one more click flips
            # to descending — either way the order must change.
            name_header = page.locator('.progress-sort[data-sort="name"]')
            name_header.click()
            page.wait_for_timeout(100)
            after_first = _column_values()
            if after_first == before:
                # Already sorted; a second click must flip direction.
                name_header.click()
                page.wait_for_timeout(100)
                after_first = _column_values()
            assert after_first != before, (
                "clicking the Language header did not reorder the table"
            )

            # The header now advertises an active sort state via the
            # aria-sort attribute (ascending or descending).
            aria_sort = name_header.get_attribute("aria-sort")
            assert aria_sort in ("ascending", "descending"), (
                f"name column aria-sort is {aria_sort!r}; expected asc or desc"
            )
        finally:
            browser.close()


def test_help_wanted_link_prefills_contribute_form(
    kozha_server: str, ensure_snapshot: Path
) -> None:
    """§3 — a Help-wanted chip deep-links into contribute with prefill."""
    snapshot = json.loads((PUBLIC_DIR / "progress_snapshot.json").read_text("utf-8"))
    gaps = snapshot.get("coverage_gaps") or {}
    asl_missing = gaps.get("bsl_missing_from_asl") or []
    bsl_missing = gaps.get("asl_missing_from_bsl") or []
    if not asl_missing and not bsl_missing:
        pytest.skip(
            "snapshot reports no coverage gaps between BSL and ASL — "
            "the help-wanted section is intentionally empty and there's "
            "nothing to click on"
        )

    # Prefer the populated list. In practice the BSL corpus is larger
    # than ASL's single alphabet, so asl_missing is the interesting case.
    target_lang = "asl" if asl_missing else "bsl"
    expected_word = (asl_missing or bsl_missing)[0]

    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            ctx = browser.new_context(locale="en-US")
            page = ctx.new_page()
            page.goto(f"{kozha_server}/progress", wait_until="domcontentloaded")

            list_id = "helpAslList" if target_lang == "asl" else "helpBslList"
            page.wait_for_function(
                f"document.querySelectorAll('#{list_id} a').length > 0",
                timeout=15_000,
            )
            first_link = page.locator(f"#{list_id} a").first
            word = first_link.get_attribute("data-word")
            lang = first_link.get_attribute("data-target-lang")
            assert word == expected_word, (
                f"first help-wanted word is {word!r}; snapshot says {expected_word!r}"
            )
            assert lang == target_lang, (
                f"first help-wanted target is {lang!r}; expected {target_lang!r}"
            )

            first_link.click()
            # After click the URL must carry the prefill params and the
            # contribute page must have loaded.
            page.wait_for_url(
                f"**/contribute.html?lang={target_lang}&gloss={expected_word}",
                timeout=10_000,
            )

            # Language picker hydrates asynchronously (loads
            # /contribute-languages.json then applies the URL prefill).
            # Once a language is selected the picker section is hidden
            # and the authoring form with the gloss input takes over, so
            # the visible artefact to assert on is the glossInput value
            # rather than pickerSelect visibility.
            page.wait_for_function(
                f"document.getElementById('pickerSelect') "
                f"&& document.getElementById('pickerSelect').value === '{target_lang}'",
                timeout=15_000,
            )

            # Gloss input picks up the draft via render(). Uppercased on
            # prefill, so the test compares against an uppercased expected.
            expected_upper = expected_word.upper()
            page.wait_for_function(
                f"document.getElementById('glossInput') "
                f"&& document.getElementById('glossInput').value === '{expected_upper}'",
                timeout=15_000,
            )
            expect(page.locator("#glossInput")).to_be_visible()
        finally:
            browser.close()

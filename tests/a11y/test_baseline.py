"""A11y regression guard — prompt-polish 12.

Reads the axe-core raw output produced by ``scripts/a11y/run.mjs`` and
asserts that no scenario has *critical* axe violations. Critical is the
deploy-block bar: a genuinely broken interaction for assistive-tech
users. Serious, moderate, and minor are tracked in the baseline markdown
but do not fail this test — they go through review per prompt-12's
note that moderate violations "would block deploys for legitimate
reasons".

The test is intentionally file-driven (no subprocess to Node) so it is
fast, deterministic, and runs in the default pytest lane. Regenerating
the raw JSON is a separate step (``npm run a11y``), and the a11y.yml
workflow is what re-runs the scanner on every PR. This test is the
guard that whatever was last written to ``docs/polish/12-a11y-raw/``
is still clean.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "docs" / "polish" / "12-a11y-raw"

# The full set of scenarios the runner produces. If a scenario is added
# or renamed, update this list — the test will also fail if a scenario
# is silently missing from the raw directory so the baseline doesn't
# drift out of sync with what's recorded.
EXPECTED_SCENARIOS = [
    "landing",
    "app-fresh",
    "app-mid-translation",
    "progress",
    "credits",
    "not-found",
    "contribute-empty",
    "contribute-after-language",
    "contribute-mid-session",
    "governance",
    "status-draft",
    "status-pending_review",
    "status-under_review",
    "status-validated",
    "status-rejected",
    "status-quarantined",
]


def _axe_path(scenario: str) -> Path:
    return RAW_DIR / f"{scenario}.axe.json"


def test_a11y_raw_directory_exists():
    """The npm run a11y step must have produced the raw artefacts."""
    if not RAW_DIR.exists():
        pytest.skip(
            f"{RAW_DIR} not found; run `npm run a11y` to generate the axe baseline"
        )


@pytest.mark.parametrize("scenario", EXPECTED_SCENARIOS)
def test_no_critical_violations(scenario: str) -> None:
    """Each covered scenario must have zero axe-critical violations.

    The scenario covers an HTML page either in its fresh-load state or
    mid-interaction (seeded by the runner). A critical violation means
    the page is materially broken for an assistive-tech user — the
    deploy gate rejects it.
    """
    path = _axe_path(scenario)
    if not path.exists():
        pytest.skip(
            f"raw output missing for {scenario}; run `npm run a11y` to regenerate"
        )

    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    violations = data.get("violations") or []
    critical = [v for v in violations if v.get("impact") == "critical"]

    if critical:
        lines = [
            f"{scenario}: {len(critical)} critical axe violation(s):",
        ]
        for v in critical:
            rule = v.get("id", "(unknown)")
            help_ = v.get("help", "")
            lines.append(f"  - {rule}: {help_}")
            for node in (v.get("nodes") or [])[:3]:
                target = " > ".join(node.get("target") or [])
                lines.append(f"      selector: {target}")
        pytest.fail("\n".join(lines))


def test_all_expected_scenarios_are_recorded() -> None:
    """Every scenario the runner produces should have a raw axe JSON.

    Guards against a silent drop — for example, a refactor that skips a
    scenario in the runner without updating this test. Without this
    assertion, a missing scenario would just pass (the skip fires per
    scenario). Here we confirm all expected scenarios did emit a file.
    """
    if not RAW_DIR.exists():
        pytest.skip(f"{RAW_DIR} not found; run `npm run a11y` to generate")
    missing = [s for s in EXPECTED_SCENARIOS if not _axe_path(s).exists()]
    if missing:
        pytest.fail(
            "axe raw output missing for scenarios: "
            + ", ".join(missing)
            + ". Either run `npm run a11y` or update EXPECTED_SCENARIOS if a "
            "scenario was intentionally removed."
        )

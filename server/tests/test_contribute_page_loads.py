"""Smoke test for the rebuilt contribute page shell.

Confirms the new scaffold from
``docs/contribute-redesign/01-design-principles.md`` is what actually lands
when a browser requests ``/contribute.html`` — and that none of the old
marketing copy (hero tagline, three-card "why" block, four-step graphic)
survived the rewrite.

The test serves ``public/`` via a minimal FastAPI ``StaticFiles`` mount
that mirrors ``server/server.py`` rather than importing the full server
module, which loads spaCy / argos on import and is overkill for a static
asset check. If the static mount in ``server.py`` ever changes shape,
update the ``_build_app`` helper here too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PUBLIC_DIR = REPO_ROOT / "public"


def _build_app() -> FastAPI:
    app = FastAPI()
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
    return app


@pytest.fixture
def client() -> TestClient:
    return TestClient(_build_app())


def test_contribute_page_returns_200(client: TestClient) -> None:
    resp = client.get("/contribute.html")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")


def test_contribute_page_has_language_picker(client: TestClient) -> None:
    body = client.get("/contribute.html").text
    # Structural markers the JS depends on — if any of these disappear
    # the picker won't wire up on page load.
    assert 'id="languagePicker"' in body
    assert 'id="pickerOptions"' in body
    assert 'id="languageBadge"' in body
    assert 'id="changeLanguageBtn"' in body
    # Copy marker — the picker's prompt is visible on first visit.
    assert "Which sign language are you contributing to?" in body


def test_contribute_page_loads_new_scripts_and_styles(client: TestClient) -> None:
    body = client.get("/contribute.html").text
    assert '/contribute.css' in body
    assert '/contribute.js' in body


def test_languages_json_is_served_and_valid(client: TestClient) -> None:
    resp = client.get("/contribute-languages.json")
    assert resp.status_code == 200
    data = resp.json()
    assert "languages" in data
    codes = {lang["code"] for lang in data["languages"]}
    # The eight languages the old FAQ advertised must all be reachable
    # by the picker so the design principle of honest language state
    # can actually be enforced in a later prompt.
    assert codes >= {"bsl", "asl", "dgs", "lsf", "lse", "pjm", "ngt", "gsl"}
    # Pre-reviewer gate: nothing ships claiming review before prompt 15.
    assert all(lang["has_reviewers"] is False for lang in data["languages"])


def test_languages_json_includes_kazakh_and_rare_set(client: TestClient) -> None:
    """The rare-SL group (KSL + the curated under-served set) must be
    reachable from the contributor picker, with honest seed metadata.

    Project-name parity matters here: "Қожа" is a Kazakh word and the
    absence of KSL from the picker would be the most visible omission a
    Kazakh contributor could spot. The rest of the seed set lets the
    picker offer something other than the "majority Western SLs only"
    historical default.
    """
    data = client.get("/contribute-languages.json").json()
    by_code = {lang["code"]: lang for lang in data["languages"]}

    expected_seed_codes = {
        "ksl", "rsl", "usl", "tid", "jsl", "kvk", "csl", "arsl", "msl", "zei",
    }
    assert expected_seed_codes <= set(by_code), (
        "missing rare-SL entries: "
        f"{sorted(expected_seed_codes - set(by_code))}"
    )

    for code in expected_seed_codes:
        lang = by_code[code]
        # Cluster these under the "rare" group so the UI can render
        # them as a distinct sub-bucket.
        assert lang.get("group") == "rare", (
            f"{code}: expected group=rare, got {lang.get('group')!r}"
        )
        # Coverage starts at zero — first contributors are exactly who
        # the seed entry exists for.
        assert lang["coverage_count"] == 0, (
            f"{code}: seed languages must start with coverage_count=0"
        )
        assert lang.get("data_completeness") == "seed"
        assert lang.get("accepts_first_contributions") is True
        # Reviewer onboarding for these languages is a separate
        # governance task — the entry must not falsely claim coverage.
        assert lang["has_reviewers"] is False


def test_seed_sigml_files_exist_for_rare_languages() -> None:
    """Each rare-SL entry must have a backing SiGML + meta.json + quarantine
    file under data/, mirroring the layout of the established corpora.
    """
    from pathlib import Path

    data_dir = REPO_ROOT / "data"
    pairs = [
        ("ksl",  "Kazakh_SL_KSL"),
        ("rsl",  "Russian_SL_RSL"),
        ("usl",  "Ukrainian_SL_USL"),
        ("tid",  "Turkish_SL_TID"),
        ("jsl",  "Japanese_SL_JSL"),
        ("kvk",  "Korean_SL_KVK"),
        ("csl",  "Chinese_SL_CSL"),
        ("arsl", "Arabic_SL_ArSL"),
        ("msl",  "Mongolian_SL_MSL"),
        ("zei",  "Persian_SL_ZEI"),
    ]
    import json
    for code, slug in pairs:
        sigml = data_dir / f"{slug}.sigml"
        meta  = data_dir / f"{slug}.sigml.meta.json"
        quar  = data_dir / f"{slug}_quarantine.sigml"
        assert sigml.exists(), f"missing seed SiGML: {sigml}"
        assert meta.exists(),  f"missing seed meta: {meta}"
        assert quar.exists(),  f"missing seed quarantine: {quar}"
        payload = json.loads(meta.read_text(encoding="utf-8"))
        assert payload["language"] == code
        assert payload["source_kind"] == "seed"
        assert "unknown" in payload["license"], (
            f"{code}: license must be unknown until contributor research"
        )
        assert payload["sign_count"] == 0
        assert payload["default_review"]["deaf_native_reviewed"] is False


def test_contribute_page_has_no_old_marketing_copy(client: TestClient) -> None:
    body = client.get("/contribute.html").text
    forbidden = [
        # Old hero.
        "Help grow the",
        "Open-source · Community-built",
        # Old three-card "why" block.
        "Small effort",
        "real reach",
        "Deaf-led review",
        "Signs reach real users",
        "Under five minutes",
        # Old four-step section.
        "Four steps to a",
        "published sign",
        # Old FAQ.
        "Answers to",
        "common questions",
        # Old form section.
        "Create your",
        "contributor profile",
        "Your OpenAI API key",
        # Old misc chrome.
        "Start contributing",
        "View on GitHub",
    ]
    for phrase in forbidden:
        assert phrase not in body, (
            f"Old marketing copy leaked into the new contribute page: {phrase!r}"
        )

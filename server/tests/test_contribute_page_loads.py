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

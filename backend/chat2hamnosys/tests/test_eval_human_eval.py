"""Tests for ``eval.human_eval`` — form renderer + payload parser."""

from __future__ import annotations

import json

import pytest

from eval.human_eval import parse_ratings_payload, render_rating_form
from eval.metrics import OverallMetrics
from eval.models import EvalResult, FixtureResult, SuiteReport


def _tiny_report() -> SuiteReport:
    fr = FixtureResult(fixture_id="test-fx-1", ablation="full")
    fr.e2e_hamnosys = "\ue001\ue020"
    fr.gen_hamnosys = "\ue001\ue020"
    fr.e2e_valid = True
    fr.e2e_exact_match = True
    eval_result = EvalResult(
        suite="unit",
        ablation="full",
        run_id="rid-123",
        started_at="",
        finished_at="",
        model="stub-llm",
        prompt_versions={},
        fixture_results=[fr],
    )
    return SuiteReport(
        results_by_ablation={"full": eval_result},
        metrics_by_ablation={"full": OverallMetrics()},
    )


def test_render_rating_form_includes_fixture_card_and_sliders() -> None:
    html = render_rating_form(_tiny_report(), reviewer_id="alice")
    assert "<!DOCTYPE html>" in html
    # Fixture id appears in the card's data attribute.
    assert 'data-fixture="test-fx-1"' in html
    assert 'data-ablation="full"' in html
    # Three Huenerfauth sliders per card.
    assert 'data-kind="grammaticality"' in html
    assert 'data-kind="naturalness"' in html
    assert 'data-kind="comprehensibility"' in html
    # Pre-filled reviewer id made it into the input.
    assert 'value="alice"' in html
    # HamNoSys codepoints are present (or escaped); the string from
    # the fixture result should be reachable somewhere in the page.
    assert "\ue001" in html


def test_render_rating_form_falls_back_to_first_ablation_when_focus_missing() -> None:
    report = _tiny_report()
    # Request an ablation that isn't in the report; the form should
    # still render something rather than returning an empty page.
    html = render_rating_form(report, focus_ablation="no_clarification")
    assert "test-fx-1" in html


def test_render_rating_form_handles_empty_report() -> None:
    empty = SuiteReport(results_by_ablation={}, metrics_by_ablation={})
    html = render_rating_form(empty)
    assert "No fixture results" in html


def test_parse_ratings_payload_accepts_well_formed() -> None:
    raw = json.dumps(
        {
            "reviewer_id": "alice",
            "ratings": [
                {
                    "fixture_id": "test-fx-1",
                    "ablation": "full",
                    "reviewer_id": "alice",
                    "grammaticality": 7,
                    "naturalness": 8,
                    "comprehensibility": 6,
                    "notes": "",
                }
            ],
        }
    )
    payload = parse_ratings_payload(raw)
    assert payload["reviewer_id"] == "alice"
    assert len(payload["ratings"]) == 1
    assert payload["ratings"][0]["grammaticality"] == 7


def test_parse_ratings_payload_rejects_non_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_ratings_payload("not-json-at-all")


def test_parse_ratings_payload_rejects_missing_ratings_array() -> None:
    with pytest.raises(ValueError, match="'ratings'"):
        parse_ratings_payload(json.dumps({"reviewer_id": "alice"}))


def test_parse_ratings_payload_rejects_missing_scale_field() -> None:
    raw = json.dumps(
        {
            "reviewer_id": "alice",
            "ratings": [
                {
                    "fixture_id": "fx",
                    "grammaticality": 5,
                    "naturalness": 5,
                    # comprehensibility missing
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="comprehensibility"):
        parse_ratings_payload(raw)

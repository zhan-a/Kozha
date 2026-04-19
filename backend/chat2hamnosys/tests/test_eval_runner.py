"""Tests for ``eval.runner`` — end-to-end suite execution with the stub client.

These tests exercise the full harness glue (fixture load → per-ablation
loop → aggregation → slice computation) without calling OpenAI, using
the deterministic :class:`eval.stub_client.StubLLMClient`. They are
deliberately high-level: "does the runner produce a SuiteReport with
the expected shape, across ablations, from a real JSONL file."

We use a small inline suite file (two fixtures) in ``tmp_path`` rather
than the 50-entry golden set so tests stay fast and deterministic. The
real golden set is exercised by the CI smoke command
(``python -m eval smoke``) — not by this unit-test file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import (
    AblationConfig,
    SuiteReport,
    ablation_presets,
    run_suite,
)
from eval.runner import load_suite, resolve_suite_path
from eval.stub_client import StubLLMClient


# Pick terms we know are in ``generator/vocab_map.yaml`` so the composer
# resolves everything deterministically. These two fixtures are DGS
# one-handed signs lifted from the Prompt-7 gold set.
MINI_SUITE_ROWS = [
    {
        "id": "mini-fist-shoulders-right",
        "prose_description": (
            "Closed fist at shoulder height with palm down, moving to the right."
        ),
        "gloss": "AACHEN3",
        "sign_language": "dgs",
        "expected_parameters": {
            "handshape_dominant": "fist",
            "orientation_extended_finger": "up",
            "orientation_palm": "down",
            "location": "shoulders",
            "movement": [{"path": "right"}],
        },
        "expected_hamnosys": "\ue000\ue020\ue03c\ue051\ue082",
        "acceptable_hamnosys_variants": [],
        "source": "test-inline",
        "difficulty": "easy",
    },
    {
        "id": "mini-flat-stomach-up",
        "prose_description": (
            "Flat hand at the stomach with palm toward the signer, moving upward."
        ),
        "gloss": "TO-GROW",
        "sign_language": "dgs",
        "expected_parameters": {
            "handshape_dominant": "flat",
            "orientation_extended_finger": "up",
            "orientation_palm": "toward_signer",
            "location": "stomach",
            "movement": [{"path": "up"}],
        },
        "expected_hamnosys": "\ue001\ue020\ue03e\ue053\ue080",
        "acceptable_hamnosys_variants": [],
        "source": "test-inline",
        "difficulty": "medium",
    },
]


@pytest.fixture
def mini_suite_path(tmp_path: Path) -> Path:
    """Write a 2-row JSONL suite file and return its path."""
    suite_path = tmp_path / "mini.jsonl"
    with suite_path.open("w", encoding="utf-8") as f:
        for row in MINI_SUITE_ROWS:
            f.write(json.dumps(row) + "\n")
    return suite_path


def test_load_suite_parses_rows_into_fixtures(mini_suite_path: Path) -> None:
    fixtures = load_suite(mini_suite_path)
    assert len(fixtures) == 2
    assert fixtures[0].id == "mini-fist-shoulders-right"
    assert fixtures[1].sign_language == "dgs"
    # Accept-set always contains the expected string at minimum.
    assert fixtures[0].expected_hamnosys in fixtures[0].accept_set()


def test_load_suite_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    suite_path = tmp_path / "with-comments.jsonl"
    suite_path.write_text(
        "// comment line\n"
        "\n"
        + json.dumps(MINI_SUITE_ROWS[0])
        + "\n",
        encoding="utf-8",
    )
    fixtures = load_suite(suite_path)
    assert len(fixtures) == 1


def test_resolve_suite_path_finds_shipped_suite() -> None:
    resolved = resolve_suite_path("golden_signs")
    assert resolved.name == "golden_signs.jsonl"
    assert resolved.is_file()


def test_resolve_suite_path_raises_on_unknown_suite() -> None:
    with pytest.raises(FileNotFoundError):
        resolve_suite_path("does-not-exist")


def test_run_suite_with_stub_produces_report_shape(
    mini_suite_path: Path,
) -> None:
    report = run_suite(
        mini_suite_path,
        ablations=[AblationConfig("full")],
        make_client=lambda: StubLLMClient(),
    )
    assert isinstance(report, SuiteReport)
    assert set(report.results_by_ablation.keys()) == {"full"}
    full = report.results_by_ablation["full"]
    assert full.suite == "mini"
    assert full.ablation == "full"
    assert len(full.fixture_results) == 2
    # Full ablation populates slice buckets.
    assert "difficulty" in report.slices
    assert set(report.slices["difficulty"].keys()) == {"easy", "medium"}


def test_run_suite_runs_all_preset_ablations(mini_suite_path: Path) -> None:
    report = run_suite(
        mini_suite_path,
        ablations=ablation_presets(),
        make_client=lambda: StubLLMClient(),
    )
    assert set(report.results_by_ablation.keys()) == {
        "full",
        "no_clarification",
        "no_validator_feedback",
        "no_deterministic_map",
    }
    # Every ablation produced a metrics bundle.
    for name in report.metrics_by_ablation:
        assert report.metrics_by_ablation[name].generator.n == 2


def test_run_suite_generator_stage_hits_vocab_for_mini_suite(
    mini_suite_path: Path,
) -> None:
    """With gold params fed in, the deterministic composer must produce
    the expected HamNoSys for the two hand-picked in-vocab fixtures.

    This is the smoke check that says "the runner's generator stage is
    wired correctly" — if this regresses, the harness itself is broken,
    not the generator.
    """
    report = run_suite(
        mini_suite_path,
        ablations=[AblationConfig("full")],
        make_client=lambda: StubLLMClient(),
    )
    full = report.results_by_ablation["full"]
    for fr in full.fixture_results:
        assert fr.gen_hamnosys, f"{fr.fixture_id}: no generator output"
        assert fr.gen_valid, f"{fr.fixture_id}: generator output invalid: {fr.notes}"
        assert fr.gen_exact_match, (
            f"{fr.fixture_id}: expected exact match, got {fr.gen_hamnosys!r}"
        )


def test_run_suite_fixture_filter_narrows_selection(
    mini_suite_path: Path,
) -> None:
    report = run_suite(
        mini_suite_path,
        ablations=[AblationConfig("full")],
        make_client=lambda: StubLLMClient(),
        fixture_filter=lambda fx: "fist" in fx.id,
    )
    full = report.results_by_ablation["full"]
    assert len(full.fixture_results) == 1
    assert full.fixture_results[0].fixture_id == "mini-fist-shoulders-right"


def test_run_suite_raises_when_filter_excludes_everything(
    mini_suite_path: Path,
) -> None:
    with pytest.raises(ValueError, match="no fixtures"):
        run_suite(
            mini_suite_path,
            ablations=[AblationConfig("full")],
            make_client=lambda: StubLLMClient(),
            fixture_filter=lambda fx: False,
        )


def test_golden_suite_loads_all_50_fixtures() -> None:
    """The shipped ``golden_signs.jsonl`` must load cleanly.

    Broken JSON or missing required fields in the golden set would fail
    every CI run — this test catches it at test-collection time instead.
    """
    suite_path = resolve_suite_path("golden_signs")
    fixtures = load_suite(suite_path)
    assert len(fixtures) == 50
    # Every fixture has a language and a difficulty from the allowed sets.
    for fx in fixtures:
        assert fx.sign_language in ("bsl", "asl", "dgs")
        assert fx.difficulty in ("easy", "medium", "hard")
        assert fx.expected_hamnosys, f"{fx.id}: empty expected_hamnosys"

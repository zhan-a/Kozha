"""Regression tests for the ``/metrics`` Prometheus endpoint (polish-13 §3).

We test the endpoint instead of the registry directly because the
wiring is what breaks most often: a route handler change, a missing
middleware, or an unregistered metric would all produce green unit
tests but a broken production dashboard.

Guarded invariants:

1. ``/metrics`` serves text/plain Prometheus exposition.
2. Every expected counter / histogram / gauge name appears in the
   output with its ``# HELP`` and ``# TYPE`` preludes.
3. A translation request increments both the translation counter and
   the latency histogram.
4. The database size gauge is non-zero after the snapshot seeds it
   at startup (as long as the snapshot exists — it does in this
   repo, so the invariant is exercised for real).
5. The response includes a ``X-Request-ID`` header (contractual for
   users quoting an id back to an operator).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "server") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "server"))

from server import app  # noqa: E402

EXPECTED_METRICS = [
    "kozha_translations_total",
    "kozha_translation_latency_ms",
    "kozha_fingerspell_fallback_total",
    "kozha_unknown_word_total",
    "kozha_database_size_total",
    "kozha_reviewed_signs_total",
    "kozha_contributions_received_total",
    "kozha_contributions_validated_total",
]


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


def test_metrics_endpoint_content_type(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus text exposition format carries the 0.0.4 version tag.
    assert "text/plain" in r.headers.get("content-type", "")


def test_metrics_endpoint_emits_request_id_header(client: TestClient) -> None:
    r = client.get("/metrics")
    assert r.headers.get("x-request-id"), "every response must carry X-Request-ID"


@pytest.mark.parametrize("metric_name", EXPECTED_METRICS)
def test_metrics_names_are_registered(client: TestClient, metric_name: str) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert f"# HELP {metric_name}" in body, (
        f"{metric_name} is missing its HELP line"
    )
    assert f"# TYPE {metric_name}" in body, (
        f"{metric_name} is missing its TYPE line"
    )


def test_translation_request_increments_counter_and_histogram(client: TestClient) -> None:
    """Before/after: a POST to /api/plan must bump both the counter
    and the histogram count for the matching label set."""
    def counter_value(body: str, outcome: str) -> float:
        for line in body.splitlines():
            if line.startswith("kozha_translations_total{") and f'outcome="{outcome}"' in line:
                if 'source_lang="en"' in line and 'target_sign_lang="bsl"' in line:
                    return float(line.rsplit(" ", 1)[-1])
        return 0.0

    def histogram_count(body: str, outcome: str) -> float:
        for line in body.splitlines():
            if (
                line.startswith("kozha_translation_latency_ms_count{")
                and f'outcome="{outcome}"' in line
                and 'source_lang="en"' in line
                and 'target_sign_lang="bsl"' in line
            ):
                return float(line.rsplit(" ", 1)[-1])
        return 0.0

    before = client.get("/metrics").text
    r = client.post(
        "/api/plan",
        json={"text": "I see a cat", "language": "en", "sign_language": "bsl"},
    )
    assert r.status_code == 200
    after = client.get("/metrics").text

    assert counter_value(after, "success") == counter_value(before, "success") + 1
    assert histogram_count(after, "success") == histogram_count(before, "success") + 1


def test_database_size_gauge_is_seeded_from_snapshot(client: TestClient) -> None:
    """The snapshot exists in the repo; the server seeds gauges from
    it on import. At least one language must have a non-zero size."""
    r = client.get("/metrics")
    lines = [
        line for line in r.text.splitlines()
        if line.startswith("kozha_database_size_total{")
    ]
    assert lines, "no kozha_database_size_total samples rendered"
    values = [float(l.rsplit(" ", 1)[-1]) for l in lines]
    assert any(v > 0 for v in values), "all database_size gauges are zero"


def test_validation_error_outcome_is_labelled(client: TestClient) -> None:
    """Empty text is a validation error — the outcome label must
    reflect that so error alerts don't mis-fire on legitimate empty
    input."""
    r = client.post("/api/plan", json={"text": "", "language": "en", "sign_language": "bsl"})
    assert r.status_code == 200
    body = client.get("/metrics").text
    # The label combination might not exist yet if this is the first
    # test run; what matters is the label is never truncated to the
    # generic ``server_error``.
    matching = [
        l for l in body.splitlines()
        if l.startswith("kozha_translations_total{")
        and 'outcome="validation_error"' in l
    ]
    assert matching, "no validation_error-labelled counter found"

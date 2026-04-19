"""Tests for the metrics registry (counters, gauges, histograms, exposition)."""

from __future__ import annotations

import pytest

from obs.metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    Summary,
    llm_call_latency_ms,
    llm_calls_total,
    render_text,
    reset_registry,
    sessions_started_total,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    reset_registry()
    yield
    reset_registry()


def test_counter_inc_and_render() -> None:
    c = Counter("t_ops_total", "test ops counter")
    c.inc()
    c.inc(value=3.0)
    assert c.get() == 4.0
    lines = c.render_lines()
    assert any("# TYPE t_ops_total counter" in l for l in lines)
    assert any(l.strip().endswith(" 4") for l in lines)


def test_counter_rejects_negative() -> None:
    c = Counter("t_bad_total", "doc")
    with pytest.raises(ValueError):
        c.inc(value=-1.0)


def test_counter_with_labels() -> None:
    c = Counter("t_labelled_total", "doc", labels=("kind",))
    c.inc("a")
    c.inc("a")
    c.inc("b")
    assert c.get("a") == 2.0
    assert c.get("b") == 1.0
    text = "\n".join(c.render_lines())
    assert 't_labelled_total{kind="a"} 2' in text
    assert 't_labelled_total{kind="b"} 1' in text


def test_gauge_set_inc_dec() -> None:
    g = Gauge("t_gauge", "doc")
    g.set(value=5.0)
    g.inc(value=2.0)
    g.dec()
    assert g.get() == 6.0


def test_histogram_observes_and_renders_buckets() -> None:
    h = Histogram("t_hist_ms", "doc", buckets=(10.0, 100.0, 1000.0))
    h.observe(value=5.0)
    h.observe(value=50.0)
    h.observe(value=500.0)
    h.observe(value=2000.0)
    snap = h.snapshot()
    assert snap is not None
    assert snap.count == 4
    # 10-bucket: only 5.0 → 1
    assert snap.counts[0] == 1
    # 100-bucket: 5.0 + 50.0 → 2
    assert snap.counts[1] == 2
    # 1000-bucket: 5.0 + 50.0 + 500.0 → 3
    assert snap.counts[2] == 3
    lines = "\n".join(h.render_lines())
    assert 't_hist_ms_bucket{le="10"} 1' in lines
    assert 't_hist_ms_bucket{le="+Inf"} 4' in lines
    assert "t_hist_ms_count 4" in lines


def test_summary_window_and_totals() -> None:
    s = Summary("t_sum", "doc", window=2)
    s.observe(value=1.0)
    s.observe(value=2.0)
    s.observe(value=3.0)
    # Window is 2, so only [2,3] retained
    assert s.recent() == [2.0, 3.0]
    lines = "\n".join(s.render_lines())
    # Sum is running total of all three observations (6.0), not window sum.
    assert "t_sum_sum 6" in lines
    assert "t_sum_count 3" in lines


def test_registry_duplicate_rejected() -> None:
    reg = MetricsRegistry()
    reg.register(Counter("x_total", "doc"))
    with pytest.raises(ValueError):
        reg.register(Counter("x_total", "doc2"))


def test_render_text_has_help_and_type_lines() -> None:
    sessions_started_total.inc()
    text = render_text()
    assert "# HELP sessions_started_total" in text
    assert "# TYPE sessions_started_total counter" in text
    assert "sessions_started_total 1" in text


def test_llm_calls_total_labels() -> None:
    llm_calls_total.inc("gpt-4o", "success")
    llm_calls_total.inc("gpt-4o", "failure")
    text = render_text()
    assert 'llm_calls_total{model="gpt-4o", outcome="success"} 1' in text
    assert 'llm_calls_total{model="gpt-4o", outcome="failure"} 1' in text


def test_llm_latency_histogram_label_axis() -> None:
    llm_call_latency_ms.observe("gpt-4o", value=123.0)
    llm_call_latency_ms.observe("gpt-4o", value=4500.0)
    text = render_text()
    # One labelled series per model; +Inf bucket reports count
    assert 'llm_call_latency_ms_bucket{model="gpt-4o", le="+Inf"} 2' in text

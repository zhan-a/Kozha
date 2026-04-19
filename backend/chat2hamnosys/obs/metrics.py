"""Prometheus-format metrics registry — counters, gauges, histograms, summaries.

Implementing the exposition format directly (instead of pulling
``prometheus_client``) keeps the runtime dependency footprint tiny and
the wire format auditable. The format produced by :func:`render_text`
is the standard text-based exposition spec:

::

    # HELP <name> <doc>
    # TYPE <name> counter|gauge|histogram|summary
    <name>{label1="…", label2="…"} <value>

Every metric is labelled with a one-line docstring (the ``# HELP``
text) explaining what it means and why it matters — those lines are
the most-consulted documentation when an operator first sees a graph
they don't recognise.

Thread safety
-------------
All counter / gauge / histogram / summary mutations take an internal
lock. Mass renders snapshot under that lock so a render is always
internally consistent.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable

# Default histogram buckets for *latency in milliseconds*. Spans the
# range we care about for OpenAI calls — the 50ms floor means a fully
# cached response still lands in a non-zero bucket; the 60s ceiling
# gives the operator enough headroom that a stuck call shows up before
# it falls into ``+Inf``.
DEFAULT_LATENCY_BUCKETS_MS: tuple[float, ...] = (
    50.0, 100.0, 250.0, 500.0, 1000.0,
    2500.0, 5000.0, 10000.0, 30000.0, 60000.0,
)

# Default histogram buckets for *cost in USD*. Wide tail because
# sessions that turn into runaway loops are exactly the ones the
# operator wants on a histogram, not bucketed into ``+Inf`` instantly.
DEFAULT_COST_BUCKETS_USD: tuple[float, ...] = (
    0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0,
)

# Default histogram buckets for *session duration in seconds*. From a
# 30s "trivial" sign through a 30-minute deep clarification loop.
DEFAULT_DURATION_BUCKETS_S: tuple[float, ...] = (
    10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 900.0, 1800.0, 3600.0,
)

# Default histogram buckets for *small integer counts* (corrections per
# session, validator-failures-before-success). Using small ints keeps
# the dashboard distribution panel readable.
DEFAULT_SMALL_INT_BUCKETS: tuple[float, ...] = (
    0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _render_label_pairs(label_names: tuple[str, ...], values: tuple[str, ...]) -> str:
    if not label_names:
        return ""
    pairs = ", ".join(
        f'{name}="{_escape_label_value(str(value))}"'
        for name, value in zip(label_names, values, strict=True)
    )
    return "{" + pairs + "}"


def _format_value(value: float) -> str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "+Inf" if value > 0 else "-Inf"
    if value == int(value):
        return str(int(value))
    return repr(value)


# ---------------------------------------------------------------------------
# Base metric
# ---------------------------------------------------------------------------


@dataclass
class _Metric:
    name: str
    doc: str
    labels: tuple[str, ...] = ()
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("metric name must be non-empty")
        if not self.doc:
            raise ValueError(f"metric {self.name!r} missing doc string")

    def _key(self, label_values: tuple[str, ...]) -> tuple[str, ...]:
        if len(label_values) != len(self.labels):
            raise ValueError(
                f"metric {self.name!r} expects {len(self.labels)} label values "
                f"({list(self.labels)}); got {len(label_values)}"
            )
        return tuple(str(v) for v in label_values)

    @property
    def metric_type(self) -> str:
        raise NotImplementedError

    def render_lines(self) -> list[str]:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Counter
# ---------------------------------------------------------------------------


class Counter(_Metric):
    """Monotonically-increasing total."""

    def __init__(self, name: str, doc: str, labels: Iterable[str] = ()) -> None:
        super().__init__(name=name, doc=doc, labels=tuple(labels))
        self._values: OrderedDict[tuple[str, ...], float] = OrderedDict()
        # Ensure the un-labelled counter exists so ``render`` shows zero
        # before the first event — gives the dashboard a stable axis.
        if not self.labels:
            self._values[()] = 0.0

    @property
    def metric_type(self) -> str:
        return "counter"

    def inc(self, *label_values: str, value: float = 1.0) -> None:
        if value < 0:
            raise ValueError("counter increment must be non-negative")
        key = self._key(label_values)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def get(self, *label_values: str) -> float:
        key = self._key(label_values)
        with self._lock:
            return self._values.get(key, 0.0)

    def reset(self) -> None:
        with self._lock:
            for k in list(self._values):
                self._values[k] = 0.0

    def render_lines(self) -> list[str]:
        with self._lock:
            items = list(self._values.items())
        out = [
            f"# HELP {self.name} {self.doc}",
            f"# TYPE {self.name} counter",
        ]
        for key, value in items:
            out.append(f"{self.name}{_render_label_pairs(self.labels, key)} {_format_value(value)}")
        return out


# ---------------------------------------------------------------------------
# Gauge
# ---------------------------------------------------------------------------


class Gauge(_Metric):
    """Arbitrary point-in-time value (active sessions, queue depth, etc.)."""

    def __init__(self, name: str, doc: str, labels: Iterable[str] = ()) -> None:
        super().__init__(name=name, doc=doc, labels=tuple(labels))
        self._values: OrderedDict[tuple[str, ...], float] = OrderedDict()
        if not self.labels:
            self._values[()] = 0.0

    @property
    def metric_type(self) -> str:
        return "gauge"

    def set(self, *label_values: str, value: float) -> None:
        key = self._key(label_values)
        with self._lock:
            self._values[key] = float(value)

    def inc(self, *label_values: str, value: float = 1.0) -> None:
        key = self._key(label_values)
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def dec(self, *label_values: str, value: float = 1.0) -> None:
        self.inc(*label_values, value=-value)

    def get(self, *label_values: str) -> float:
        key = self._key(label_values)
        with self._lock:
            return self._values.get(key, 0.0)

    def render_lines(self) -> list[str]:
        with self._lock:
            items = list(self._values.items())
        out = [
            f"# HELP {self.name} {self.doc}",
            f"# TYPE {self.name} gauge",
        ]
        for key, value in items:
            out.append(f"{self.name}{_render_label_pairs(self.labels, key)} {_format_value(value)}")
        return out


# ---------------------------------------------------------------------------
# Histogram
# ---------------------------------------------------------------------------


@dataclass
class _HistogramBucket:
    counts: list[float]
    sum_value: float = 0.0
    count: int = 0


class Histogram(_Metric):
    """Bucketed distribution; renders ``_bucket`` / ``_sum`` / ``_count``."""

    def __init__(
        self,
        name: str,
        doc: str,
        buckets: Iterable[float] = DEFAULT_LATENCY_BUCKETS_MS,
        labels: Iterable[str] = (),
    ) -> None:
        super().__init__(name=name, doc=doc, labels=tuple(labels))
        sorted_buckets = tuple(sorted(set(float(b) for b in buckets)))
        if not sorted_buckets:
            raise ValueError(f"histogram {name!r} needs at least one bucket")
        self.buckets = sorted_buckets
        self._series: OrderedDict[tuple[str, ...], _HistogramBucket] = OrderedDict()
        if not self.labels:
            self._series[()] = self._fresh_bucket()

    def _fresh_bucket(self) -> _HistogramBucket:
        return _HistogramBucket(counts=[0.0] * len(self.buckets))

    @property
    def metric_type(self) -> str:
        return "histogram"

    def observe(self, *label_values: str, value: float) -> None:
        key = self._key(label_values)
        with self._lock:
            bucket = self._series.get(key)
            if bucket is None:
                bucket = self._fresh_bucket()
                self._series[key] = bucket
            for i, le in enumerate(self.buckets):
                if value <= le:
                    bucket.counts[i] += 1
            bucket.sum_value += value
            bucket.count += 1

    def snapshot(self, *label_values: str) -> _HistogramBucket | None:
        key = self._key(label_values)
        with self._lock:
            bucket = self._series.get(key)
            if bucket is None:
                return None
            return _HistogramBucket(
                counts=list(bucket.counts),
                sum_value=bucket.sum_value,
                count=bucket.count,
            )

    def render_lines(self) -> list[str]:
        with self._lock:
            items = list(self._series.items())
        out = [
            f"# HELP {self.name} {self.doc}",
            f"# TYPE {self.name} histogram",
        ]
        for key, bucket in items:
            label_render_extra = list(self.labels) + ["le"]
            for i, le in enumerate(self.buckets):
                line_label = _render_label_pairs(
                    tuple(label_render_extra), tuple(list(key) + [_format_value(le)])
                )
                out.append(
                    f"{self.name}_bucket{line_label} {_format_value(bucket.counts[i])}"
                )
            inf_label = _render_label_pairs(
                tuple(label_render_extra), tuple(list(key) + ["+Inf"])
            )
            out.append(f"{self.name}_bucket{inf_label} {_format_value(bucket.count)}")
            base_label = _render_label_pairs(self.labels, key)
            out.append(f"{self.name}_sum{base_label} {_format_value(bucket.sum_value)}")
            out.append(f"{self.name}_count{base_label} {_format_value(bucket.count)}")
        return out


# ---------------------------------------------------------------------------
# Summary — keeps a rolling window of observations for percentile views
# ---------------------------------------------------------------------------


class Summary(_Metric):
    """Rolling-window observations; renders ``_sum`` and ``_count`` lines.

    Quantile estimates aren't emitted in the wire format (the standard
    accepts ``# TYPE summary`` lines without quantiles), but the
    in-memory ring is queryable via :meth:`recent` so the dashboard can
    compute percentiles cheaply.
    """

    def __init__(
        self,
        name: str,
        doc: str,
        window: int = 256,
        labels: Iterable[str] = (),
    ) -> None:
        super().__init__(name=name, doc=doc, labels=tuple(labels))
        if window < 1:
            raise ValueError("summary window must be >= 1")
        self.window = window
        self._series: OrderedDict[tuple[str, ...], list[float]] = OrderedDict()
        self._totals: OrderedDict[tuple[str, ...], list[float]] = OrderedDict()
        if not self.labels:
            self._series[()] = []
            self._totals[()] = [0.0, 0]  # [sum, count]

    @property
    def metric_type(self) -> str:
        return "summary"

    def observe(self, *label_values: str, value: float) -> None:
        key = self._key(label_values)
        with self._lock:
            if key not in self._series:
                self._series[key] = []
                self._totals[key] = [0.0, 0]
            window = self._series[key]
            window.append(float(value))
            if len(window) > self.window:
                window.pop(0)
            self._totals[key][0] += float(value)
            self._totals[key][1] += 1

    def recent(self, *label_values: str) -> list[float]:
        key = self._key(label_values)
        with self._lock:
            return list(self._series.get(key, []))

    def render_lines(self) -> list[str]:
        with self._lock:
            items = list(self._totals.items())
        out = [
            f"# HELP {self.name} {self.doc}",
            f"# TYPE {self.name} summary",
        ]
        for key, (sum_v, count_v) in items:
            base_label = _render_label_pairs(self.labels, key)
            out.append(f"{self.name}_sum{base_label} {_format_value(sum_v)}")
            out.append(f"{self.name}_count{base_label} {_format_value(count_v)}")
        return out


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class MetricsRegistry:
    """Holds every metric in declaration order; renders the exposition text."""

    def __init__(self) -> None:
        self._metrics: list[_Metric] = []
        self._by_name: dict[str, _Metric] = {}

    def register(self, metric: _Metric) -> _Metric:
        if metric.name in self._by_name:
            raise ValueError(f"duplicate metric name {metric.name!r}")
        self._metrics.append(metric)
        self._by_name[metric.name] = metric
        return metric

    def get(self, name: str) -> _Metric:
        return self._by_name[name]

    def __contains__(self, name: str) -> bool:
        return name in self._by_name

    def metrics(self) -> list[_Metric]:
        return list(self._metrics)

    def render_text(self) -> str:
        chunks: list[str] = []
        for metric in self._metrics:
            chunks.extend(metric.render_lines())
            chunks.append("")
        return "\n".join(chunks).rstrip("\n") + "\n"


# ---------------------------------------------------------------------------
# The chat2hamnosys metric set — declared at import time.
# ---------------------------------------------------------------------------


_REGISTRY = MetricsRegistry()
_registry_lock = threading.Lock()


def _C(name: str, doc: str, labels: tuple[str, ...] = ()) -> Counter:
    return _REGISTRY.register(Counter(name, doc, labels))


def _G(name: str, doc: str, labels: tuple[str, ...] = ()) -> Gauge:
    return _REGISTRY.register(Gauge(name, doc, labels))


def _H(
    name: str,
    doc: str,
    buckets: tuple[float, ...] = DEFAULT_LATENCY_BUCKETS_MS,
    labels: tuple[str, ...] = (),
) -> Histogram:
    return _REGISTRY.register(Histogram(name, doc, buckets, labels))


def _S(
    name: str, doc: str, window: int = 256, labels: tuple[str, ...] = ()
) -> Summary:
    return _REGISTRY.register(Summary(name, doc, window, labels))


# Counters --------------------------------------------------------------------

sessions_started_total = _C(
    "sessions_started_total",
    "Total authoring sessions created — the funnel's top-of-mouth number.",
)
sessions_accepted_total = _C(
    "sessions_accepted_total",
    "Total sessions that produced an accepted SignEntry — the funnel's bottom.",
)
sessions_abandoned_total = _C(
    "sessions_abandoned_total",
    "Sessions that ended without acceptance (rejected or timed out) — drop-off signal.",
)

llm_calls_total = _C(
    "llm_calls_total",
    "OpenAI chat-completion calls dispatched, by primary model and outcome.",
    labels=("model", "outcome"),
)

corrections_submitted_total = _C(
    "corrections_submitted_total",
    "Correction requests submitted by authors — high values mean the first cut is wrong.",
)
reviews_approved_total = _C(
    "reviews_approved_total",
    "Approve actions taken by Deaf reviewers — feeds the validation throughput chart.",
)
reviews_rejected_total = _C(
    "reviews_rejected_total",
    "Reject actions — track to spot bursts of low-quality drafts.",
)
exports_succeeded_total = _C(
    "exports_succeeded_total",
    "Signs that crossed the export gate into the Kozha library.",
)
exports_blocked_total = _C(
    "exports_blocked_total",
    "Export attempts blocked by the gate, labelled by reason for triage.",
    labels=("reason",),
)
injection_detections_total = _C(
    "injection_detections_total",
    "Inputs the injection screen rejected — sustained spikes indicate abuse.",
)

# Gauges ---------------------------------------------------------------------

active_sessions = _G(
    "active_sessions",
    "Sessions currently in a non-terminal state — capacity signal.",
)
pending_reviews_queue_depth = _G(
    "pending_reviews_queue_depth",
    "Signs waiting on the reviewer queue — alert if it grows unbounded.",
)

# Histograms -----------------------------------------------------------------

llm_call_latency_ms = _H(
    "llm_call_latency_ms",
    "End-to-end OpenAI call latency in milliseconds, labelled by model.",
    buckets=DEFAULT_LATENCY_BUCKETS_MS,
    labels=("model",),
)
llm_call_cost_usd = _H(
    "llm_call_cost_usd",
    "USD cost per OpenAI call — wide tail to surface runaway prompts.",
    buckets=DEFAULT_COST_BUCKETS_USD,
)
session_duration_seconds = _H(
    "session_duration_seconds",
    "Wall-clock duration of an authoring session from create to terminal state.",
    buckets=DEFAULT_DURATION_BUCKETS_S,
)
corrections_per_session = _H(
    "corrections_per_session",
    "Number of correction submissions per accepted session — quality proxy.",
    buckets=DEFAULT_SMALL_INT_BUCKETS,
)
validator_failures_before_success = _H(
    "validator_failures_before_success",
    "Generator validator failures before the first valid HamNoSys output.",
    buckets=DEFAULT_SMALL_INT_BUCKETS,
)

# Summary --------------------------------------------------------------------

daily_cost_usd = _S(
    "daily_cost_usd",
    "Rolling per-call USD cost — backs the today/week cost panels.",
    window=2048,
)


def registry() -> MetricsRegistry:
    """Return the process-wide :class:`MetricsRegistry`."""
    return _REGISTRY


def render_text() -> str:
    """Render the exposition format as a single string."""
    return _REGISTRY.render_text()


def reset_registry() -> None:
    """Zero every counter / gauge / histogram / summary (test helper)."""
    with _registry_lock:
        for metric in _REGISTRY.metrics():
            if isinstance(metric, Counter):
                metric.reset()
            elif isinstance(metric, Gauge):
                for key in list(metric._values):
                    metric._values[key] = 0.0
            elif isinstance(metric, Histogram):
                metric._series.clear()
                if not metric.labels:
                    metric._series[()] = metric._fresh_bucket()
            elif isinstance(metric, Summary):
                metric._series.clear()
                metric._totals.clear()
                if not metric.labels:
                    metric._series[()] = []
                    metric._totals[()] = [0.0, 0]


__all__ = [
    "Counter",
    "DEFAULT_COST_BUCKETS_USD",
    "DEFAULT_DURATION_BUCKETS_S",
    "DEFAULT_LATENCY_BUCKETS_MS",
    "DEFAULT_SMALL_INT_BUCKETS",
    "Gauge",
    "Histogram",
    "MetricsRegistry",
    "Summary",
    "active_sessions",
    "corrections_per_session",
    "corrections_submitted_total",
    "daily_cost_usd",
    "exports_blocked_total",
    "exports_succeeded_total",
    "injection_detections_total",
    "llm_call_cost_usd",
    "llm_call_latency_ms",
    "llm_calls_total",
    "pending_reviews_queue_depth",
    "registry",
    "render_text",
    "reset_registry",
    "reviews_approved_total",
    "reviews_rejected_total",
    "session_duration_seconds",
    "sessions_abandoned_total",
    "sessions_accepted_total",
    "sessions_started_total",
    "validator_failures_before_success",
]

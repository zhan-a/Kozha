"""Simple latency benchmark for ``/api/plan``.

Writes to stdout p50/p95/p99/mean/min/max over ``N`` sequential calls
(default 50) of the same fixed phrase. Used by ``docs/polish/13-perf-budget.md``
to gate observability regressions — the delta between a pre- and post-
instrumentation run must not exceed 20 ms at p50.

Run::

    python3 scripts/bench_translate.py 50
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "server") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "server"))

from fastapi.testclient import TestClient  # noqa: E402

from server import app  # noqa: E402

# Fixed phrase: representative of the common translator visit (a
# scheduled-meeting sentence with a time word, a pronoun, and two
# content nouns). Using the same phrase in both runs means any delta
# comes from instrumentation, not from varying tokenization work.
PAYLOAD = {
    "text": (
        "I will meet my friend at the cafe tomorrow morning "
        "to talk about our holiday plans."
    ),
    "language": "en",
    "sign_language": "bsl",
}


def _percentile(sorted_samples: list[float], p: float) -> float:
    if not sorted_samples:
        return float("nan")
    k = max(0, min(len(sorted_samples) - 1, int(round((p / 100.0) * (len(sorted_samples) - 1)))))
    return sorted_samples[k]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    client = TestClient(app)

    # Warm-up — lazy model loads don't count toward the budget.
    for _ in range(3):
        client.post("/api/plan", json=PAYLOAD)

    samples: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        r = client.post("/api/plan", json=PAYLOAD)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code != 200:
            print(f"unexpected status {r.status_code}: {r.text}", file=sys.stderr)
            return 1
        samples.append(dt_ms)
    samples.sort()

    print(f"N={n}")
    print(f"mean_ms = {statistics.mean(samples):.2f}")
    print(f"p50_ms  = {_percentile(samples, 50):.2f}")
    print(f"p95_ms  = {_percentile(samples, 95):.2f}")
    print(f"p99_ms  = {_percentile(samples, 99):.2f}")
    print(f"min_ms  = {samples[0]:.2f}")
    print(f"max_ms  = {samples[-1]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

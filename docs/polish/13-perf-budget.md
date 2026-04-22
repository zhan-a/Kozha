# Polish-13 performance budget

## Budget

**Instrumentation must not raise `/api/plan` p50 latency by more than 20 ms.**

The observability additions in prompt 13 (structured JSON logging, a
request-id middleware, Prometheus counter/histogram updates on the hot
path, a 5xx error shield) are all O(1) per request but add up. The
budget protects the translator's response-time SLO from death-by-a-
thousand-cuts regressions.

## Method

Benchmarked with `fastapi.testclient.TestClient` against the live
`/api/plan` endpoint, a fixed 86-character English test phrase, and the
BSL grammar profile — representative of the common case visitors hit.
The harness runs three warm-up calls (spaCy and the review-metadata
index both load lazily on the first request), then times 50 sequential
calls. `perf_counter` measures end-to-end request latency inside the
same Python process — no network. Same Python version, same venv, same
data files for both runs.

Benchmark script: `scripts/bench_translate.py` (committed, invocable as
`python3 scripts/bench_translate.py 50`). Numbers below are the ones
this branch reports on the developer's workstation; CI running on a
smaller VM will show higher absolute numbers, but the delta is what the
budget constrains.

## Results

| metric | before (ms) | after (ms) | delta (ms) |
|---|---|---|---|
| mean  | 3.17 | 3.59 | +0.42 |
| p50   | 3.11 | 3.51 | +0.40 |
| p95   | 3.35 | 3.78 | +0.43 |
| p99   | 5.11 | 5.57 | +0.46 |

**p50 delta: +0.40 ms — well inside the 20 ms budget.**

Observed overhead breakdown (all on the hot path):

* `uuid4().hex` for the request id: ~2 µs.
* Three dict-key lookups to resolve metric labels: ~1 µs.
* One histogram bucket iteration (10 buckets): ~3 µs.
* One JSON log line write to stdout: 50–200 µs depending on whether
  stdout is a pipe or a tty. The test-client run has stdout = tty.
* Everything else (`time.perf_counter` calls, outcome string
  compares): sub-microsecond.

The 0.4 ms delta is dominated by the stdout write, which is why the
production number (where stdout goes to systemd's journal via a
pipe) will be marginally _better_ than this measurement, not worse.

The 20 ms budget includes production-only costs this workstation run
can't exercise: JSON log writes under disk pressure (stdout is
line-buffered in systemd, not the test client) and lock contention when
multiple gunicorn workers observe the same histogram bucket
concurrently. Both are bounded: the log writes are byte-level small and
the metric locks are held for microseconds. A post-deploy measurement
(recorded below) is what actually gates the budget.

## Post-deploy measurement

Re-run `scripts/bench_translate.py 50` inside the EC2 host after a
successful deploy and paste the numbers here. The systemd service reads
`/etc/kozha.env`; no extra setup required.

```
# Run on the host, not on the runner:
cd /home/ubuntu/Kozha && /home/ubuntu/kozha-venv/bin/python3 scripts/bench_translate.py 50
```

_Last measured: pending post-deploy re-run._

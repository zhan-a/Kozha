# Observability & Operations Runbook — chat2hamnosys

Companion document for Prompt 18. Four sections:

1. **Signal surface** — what we log, count, and expose, and where it
   lives.
2. **Access & admin** — the endpoints, the auth model, the env knobs.
3. **Runbook playbooks** — "it's slow", "it's wrong", "it's
   expensive", and two more.
4. **Retention, privacy & PII** — what we keep, for how long, and
   what never leaves the process.

Code references point at `backend/chat2hamnosys/obs/` (logger, metrics,
alerts, dashboard, events) and `backend/chat2hamnosys/api/admin.py`.

---

## 1. Signal surface

### 1.1 Events (structured logs)

Every emission goes through a single chokepoint:
`obs.logger.EventLogger.emit(event, **fields)`. The event name must be
in `obs.events.ALL_EVENTS` — unknown names raise
`UnknownEventError`, so typos surface in tests rather than as
silent-new-event drift.

The taxonomy — `<surface>.<noun>.<verb>` — is grouped in
`obs/events.py`:

| Surface      | Events |
|--------------|--------|
| `session.*`  | `created`, `state_changed`, `accepted`, `rejected`, `abandoned` |
| `llm.call.*` | `started`, `succeeded`, `retried`, `failed`, `fallback_used` |
| `parse.*`    | `description.completed`, `description.gaps_found` |
| `clarify.*`  | `question_asked`, `answer_received` |
| `generate.*` | `hamnosys.attempted`, `validated`, `invalid_retry`, `gave_up` |
| `render.*`   | `preview.started`, `preview.succeeded`, `preview.failed`, `cache.hit` |
| `correct.*`  | `submitted`, `applied`, `ambiguous` |
| `review.*`   | `queued`, `approved`, `rejected`, `revision_requested`, `flagged` |
| `export.*`   | `attempted`, `blocked`, `succeeded` |
| `security.*` | `injection_detected`, `budget_exceeded`, `rate_limited` |

Each record is a JSON line containing at minimum `ts`, `level`, and
`event`, plus per-event context (model, latency_ms, cost_usd,
session_id, request_id, user_hash, verdict, reason, etc.).

Severity is picked up by the dashboard from `obs.events.EVENT_LEVEL`;
anything not in the map renders as `info`.

### 1.2 Metrics (Prometheus text exposition)

Handwritten registry under `obs/metrics.py` — no `prometheus_client`
dependency, so the exporter is a string template we can audit line by
line.

Counters:
`sessions_started_total`, `sessions_accepted_total`,
`sessions_abandoned_total`, `llm_calls_total{model,outcome}`,
`corrections_submitted_total`, `reviews_approved_total`,
`reviews_rejected_total`, `exports_succeeded_total`,
`exports_blocked_total{reason}`, `injection_detections_total`.

Gauges: `active_sessions`, `pending_reviews_queue_depth`.

Histograms:
`llm_call_latency_ms{model}`, `llm_call_cost_usd`,
`session_duration_seconds`, `corrections_per_session`,
`validator_failures_before_success`.

Running total: `daily_cost_usd` (summary, bounded window).

### 1.3 Where logs land

`obs.logger.EventLogger` has two sinks, chosen by
`CHAT2HAMNOSYS_LOG_SINK`:

- `file` (default for dev) — one JSONL file per day under
  `CHAT2HAMNOSYS_LOG_DIR` or `<repo>/logs`. Retention trims files
  older than `retention_days` on each emission (default 30).
- `stdout` (prod) — one JSON line per event to stdout, so the host's
  log shipper (systemd-journal → CloudWatch in the current deploy)
  owns storage and retention.

An in-memory ring buffer (`EventLogger.recent()`) keeps the last N
events regardless of sink, powering the dashboard's recent-events
feed without replaying files.

---

## 2. Access & admin

### 2.1 Endpoints (mounted under the API prefix in `api/app.py`)

| Path                              | Auth           | What it's for |
|-----------------------------------|----------------|----------------|
| `GET /health`                     | open           | Liveness. 200 if the process is up. |
| `GET /health/ready`               | open           | Readiness. 200 only if session / sign / token store open and `OPENAI_API_KEY` is set; otherwise 503 + per-check breakdown. |
| `GET /metrics`                    | open           | Prometheus text. |
| `GET /admin/dashboard`            | board reviewer | Funnel, error rates, LLM breakdown, hourly activity, today's numbers. |
| `GET /admin/sessions/{id}`        | board reviewer | Per-session event trace. |
| `GET /admin/cost`                 | board reviewer | 30-day cost breakdown, read from the JSONL archive. |

`board reviewer` auth is handled by `require_board_reviewer()` in
`api/admin.py` — the same role check as the reviewer workflow,
imported lazily to avoid a circular import with `review/dependencies`.
`/metrics` and `/health*` are intentionally open so a load balancer
or Prometheus scraper can reach them without credentials.

### 2.2 Environment knobs

Logger (`obs/logger.py`):

- `CHAT2HAMNOSYS_LOG_SINK` — `file` (default) or `stdout`.
- `CHAT2HAMNOSYS_LOG_DIR` — path for the file sink. Created on first
  emit if absent.

Alerter (`obs/alerts.py`):

- `CHAT2HAMNOSYS_ALERT_WEBHOOK_URL` — if unset, alerts land in a
  `RecordingSink` (tests and dev only).
- `CHAT2HAMNOSYS_ALERT_SILENCE_S` — per-kind suppression window.
  Default 1800s.
- `CHAT2HAMNOSYS_DAILY_BUDGET_USD` — input to the cost-projection
  rule. If unset, the rule is silent.
- `CHAT2HAMNOSYS_PENDING_REVIEW_STALE_H` — staleness threshold for
  pending-review-queue alert. Default 24h.

### 2.3 Alert rules (default set, `obs/alerts.py`)

Each rule is a pure function returning `Alert | None`, evaluated on a
cadence and silenced per-kind.

- `rule_llm_failure_rate` — ≥30% `llm.call.failed` over the last hour
  with ≥5 calls in the window. Severity `error`.
- `rule_session_abandonment` — ≥40% abandon rate over the last hour
  with ≥5 sessions started. Severity `warning`.
- `rule_daily_cost_projection` — linear projection of today's spend
  vs. `CHAT2HAMNOSYS_DAILY_BUDGET_USD`. Severity `error` if
  projected > budget.
- `rule_pending_review_queue_deep` — gauge ≥ 50 **and** the oldest
  queued review is older than `CHAT2HAMNOSYS_PENDING_REVIEW_STALE_H`
  hours. Severity `warning`.
- `rule_injection_detected` — any
  `security.injection_detected` event within the hour.
  Severity `error`.

A broken rule (one that raises) is swallowed by the alerter and does
not block the other rules — see
`test_alerter_swallows_rule_exceptions` in
`tests/test_obs_alerts.py`.

---

## 3. Runbook playbooks

### 3.1 "It's slow" — an author says the UI hangs

1. **Is the process up?** `GET /health`. If 503, it's a process
   crash, not latency — check systemd logs on the host.
2. **Are dependencies healthy?** `GET /health/ready`. A non-`ok`
   `session_store` or `sign_store` means SQLite contention; a failing
   `llm_config` means `OPENAI_API_KEY` isn't in the environment.
3. **Where's the spike?** `GET /admin/dashboard` — the LLM breakdown
   table shows p95 latency per prompt (parse, clarifier, generator).
   If one prompt dominates, that's the culprit.
4. **Is it the LLM provider or us?** Scrape
   `llm_call_latency_ms_bucket` — if the `+Inf` bucket is rising
   without a matching rise in `llm_calls_total{outcome="failure"}`,
   OpenAI is serving slow but not error-ing.
5. **Is the cache warm?** `render.cache.hit` vs.
   `render.preview.started` tells you whether avatar previews are
   hitting the cache. A cold cache + new release is usually the
   explanation.

### 3.2 "It's wrong" — the generated HamNoSys doesn't match the prose

1. **Find the session.** Ask the author for the session ID (shown in
   the UI footer). `GET /admin/sessions/{id}` returns the full event
   trace: prose → parse → questions asked → generator attempts →
   validator verdict → render status.
2. **Read backwards from the validator verdict.** A
   `generate.hamnosys.invalid_retry` followed by
   `generate.hamnosys.gave_up` means the generator ran out of
   budget; the last attempt is the one that shipped.
3. **Is the parser confident?** `parse.description.gaps_found` tells
   you which fields were filled by clarifier answers vs. inferred
   from prose. A gap the author answered with a free-form string is
   a common source of drift.
4. **Cross-check the clarifier.** `clarify.question_asked` rows
   include `rationale`. If the rationale points at a field that
   wasn't actually ambiguous, raise it in the clarifier eval set.

### 3.3 "It's expensive" — someone's asking why the OpenAI bill is up

1. **Today's spend.** `GET /admin/cost` renders the 30-day breakdown
   by day + model. The current day aggregates from the ring buffer;
   prior days are read from the JSONL archive.
2. **Is it volume or model mix?** `llm_calls_total{model,outcome}`
   is labelled per model; a new model ID appearing means someone
   changed configuration without telling ops.
3. **Is a session looping?** A single session emitting many
   `generate.hamnosys.invalid_retry` events is the
   most common abuse pattern — the validator rejects, the generator
   retries, cost compounds. Cap
   `GENERATOR_MAX_ATTEMPTS` in deployment env if the generator eval
   hasn't already pushed it down.
4. **Is budget projection firing?** Check the alerter sink. If
   `daily_cost_projected_over_budget` fires more than once an hour,
   silencing is working but the underlying rate is sustained — cut
   traffic or raise the budget, don't just silence.

### 3.4 "Someone's attacking us" — security tripwires

1. **Injection.** `injection_detections_total` > 0 or a fired
   `injection_detected` alert. Sessions in the trace view show the
   offending field and a redacted snippet (no raw prose ever leaves
   the machine unredacted). Block by IP at the LB if sustained.
2. **Budget exhaustion / rate-limiting.**
   `security.budget_exceeded` carries a `scope` label
   (`session` / `per-ip` / `global_daily`). Per-IP and global_daily
   are the ones that warrant escalation; `session` is self-contained.
3. **Tokens leaked?** `security.rate_limited` includes `client_ip`
   (HMAC-hashed for logs ≥ 1 day old — see section 4). A single IP
   hammering `/sessions/.../answer` outside working hours is a
   credential-theft signal.

### 3.5 "The reviewer queue is growing"

1. **Depth.** `pending_reviews_queue_depth` gauge. The alerter fires
   at depth ≥ 50 **and** oldest queued > staleness threshold, so the
   gauge can climb without paging if reviewers are actively working.
2. **Throughput.**
   `reviews_approved_total + reviews_rejected_total` delta over an
   hour. If depth is climbing but throughput is flat, reviewers are
   offline; if depth is climbing and throughput is high, it's
   inbound volume.
3. **Escalation.** Raise in the Deaf-reviewer channel before paging.
   The two-reviewer rule means throughput can't be brute-forced by
   adding sighted hearing staff.

---

## 4. Retention, privacy & PII

- **JSONL files** rotate daily (`YYYY-MM-DD.jsonl`) and retention
  purges anything older than `retention_days` (default 30) on each
  emit. Storage is bounded by a function of traffic; there is no
  backfill.
- **Stdout logs** in prod are owned by the host's log pipeline. The
  retention story is set at that layer, not in this process.
- **Ring buffer** is volatile and bounded; it survives only until the
  process restarts.
- **User identifiers** never appear in logs raw. `hash_user_id()`
  HMAC-hashes with the `CHAT2HAMNOSYS_SIGNER_ID_SALT` — same salt as
  the session store. Truncated to 16 chars, which is enough to
  correlate within a day and useless across the retention window once
  files are purged.
- **Raw prose** is never logged. Parser events record
  `gaps_found` counts and field names; they do not echo the prose.
- **Raw API keys** are never logged. `llm.call.*` records carry
  `model`, `latency_ms`, `cost_usd`, `input_tokens`, `output_tokens`
  — not the request or response bodies.

If a log ever looks like it might contain PII, treat it as an
incident: purge the offending files, rotate any credentials touched,
and add a redaction at the emission site (not a scrub pass over the
files — the files are append-only).

---

## 5. Local operations cheatsheet

```bash
# Scrape Prometheus locally
curl -s http://localhost:8000/metrics | head -40

# Check liveness + readiness
curl -s http://localhost:8000/health
curl -s http://localhost:8000/health/ready | jq .

# Tail today's JSONL
tail -f backend/chat2hamnosys/logs/$(date -u +%F).jsonl | jq .

# Filter to one session
jq 'select(.session_id=="SID")' backend/chat2hamnosys/logs/*.jsonl

# Enumerate event types actually emitted in the last day
jq -r '.event' backend/chat2hamnosys/logs/$(date -u +%F).jsonl | sort | uniq -c | sort -rn
```

---

## 6. Tests locking in the contract

- `tests/test_obs_logger.py` — taxonomy closed-set, ring buffer,
  daily rotation, retention purge, hash stability.
- `tests/test_obs_metrics.py` — counter/gauge/histogram/summary
  semantics, duplicate registration rejection, exposition format.
- `tests/test_obs_dashboard.py` — funnel / LLM breakdown / error
  rates / hourly activity / today / cost aggregators, plus HTML
  renders.
- `tests/test_obs_alerts.py` — every default rule, silencing window,
  exception-swallowing, webhook URL validation,
  `build_default_alerter` env wiring.
- `tests/test_obs_integration.py` — `/health`, `/health/ready`,
  `/metrics`, full session event sequence, reject path, and a
  20-session light-load check that confirms counters don't drift.

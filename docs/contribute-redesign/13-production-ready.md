# Production-ready deployment gate — prompt 13

Authored: 2026-04-21. Drafted alongside the real-backend wiring of
prompt 13
([prompts/prompt-contrib-13.md](../../prompts/prompt-contrib-13.md)).

This document is the **gate**: the full list of things that must be
true before a deploy of the contribute surface can go out. It is not
a deployment guide. It is the check every operator runs before
promoting a build — if any item here is unmet, do not deploy.

The contribute surface spans the FastAPI sub-app mounted at
`server/server.py:642-647` under `/api/chat2hamnosys/*`, the
static pages in `public/contribute*.html`, and the review console
under `public/chat2hamnosys/review/`. Every endpoint the page calls
is listed in [00-endpoints-map.md](00-endpoints-map.md); every gap
left open after the redesign is listed in
[10-backend-gaps.md](10-backend-gaps.md) and
[11-governance-gaps.md](11-governance-gaps.md). This document
assumes both have been read.

The format of each gate item below is:

- **What must be true** — the condition that has to hold.
- **How to verify** — a concrete command, file check, or API probe.
- **Consequence if unset** — what fails, and how loudly, in
  production. "Silent" consequences are the dangerous ones: call
  them out.

---

## Gate 1 — Database schema present

**What must be true.** Five SQLite databases exist, are writable by
the service account, and have their schema applied. They are created
lazily on first access, so "exist" means *the service can open them
with the configured path*.

| DB | Env var | Default | Holds |
|---|---|---|---|
| Sessions | `CHAT2HAMNOSYS_SESSION_DB` | `./data/sessions.sqlite3` | Authoring sessions + event history |
| Signs | `CHAT2HAMNOSYS_SIGN_DB` | `./data/signs.sqlite3` | `SignEntry` rows after accept |
| Tokens | `CHAT2HAMNOSYS_TOKEN_DB` | `./data/tokens.sqlite3` | Per-session bearer tokens |
| Reviewers | `CHAT2HAMNOSYS_REVIEWER_DB` | `./data/reviewers.sqlite3` | Reviewer bearer tokens (SHA-256 on disk) |
| Contributors | `CHAT2HAMNOSYS_CONTRIBUTOR_DB` | `<DATA_DIR>/contributors.sqlite3` | Registered contributors + captcha history |

The data dir is `CHAT2HAMNOSYS_DATA_DIR` (default `./data`), which
is also where the export path writes `hamnosys_<lang>_authored.sigml`
— the same directory the translator root app serves from under
`/data/*` (see [00-endpoints-map.md § 5](00-endpoints-map.md)).

**How to verify.** Boot the service against the production env and
hit `GET /api/chat2hamnosys/health/ready`. The response shape is
`{status, checks: {session_store, sign_store, llm}}`
(`backend/chat2hamnosys/api/admin.py:186-219`); `session_store` and
`sign_store` must both be `"ok"`. Readiness degrading on `llm` is a
separate concern (gate 2 below) — treat the DB checks as
non-negotiable.

**Consequence if unset.** The service will appear to boot because
the SQLite files are created lazily. The first write will fail only
when an actual request lands, returning 500 with no clear error to
the contributor. This is the worst failure mode in the system:
silent at deploy, loud at first use. Run the readiness probe
explicitly.

---

## Gate 2 — Required secrets provisioned

**What must be true.** Three secrets exist and are non-empty in the
service environment.

| Secret | Required for | Where consumed |
|---|---|---|
| `OPENAI_API_KEY` | Parsing, clarifying questions, correction interpretation, injection classifier | `backend/chat2hamnosys/llm/client.py:285` |
| `CHAT2HAMNOSYS_SIGNER_ID_SALT` | Stable pseudonymous hashing of signer identity; 48 bytes | `backend/chat2hamnosys/README.md:30` |
| `CHAT2HAMNOSYS_CAPTCHA_SECRET` | HMAC of the register-flow captcha challenge | `backend/chat2hamnosys/api/contributors.py:72` |

A fourth secret, `CHAT2HAMNOSYS_CONTRIBUTOR_SECRET`, is optional and
falls back to `CHAT2HAMNOSYS_SIGNER_ID_SALT` when unset
(`contributors.py:87-88`). If operators want the contributor-token
HMAC and the signer-ID hash to be independently rotatable, set both.
Otherwise the single salt is sufficient.

**How to verify.** `GET /api/chat2hamnosys/health/ready` reports
`checks.llm = "ok"` iff `OPENAI_API_KEY` is set
(`admin.py:219`). For the two HMAC secrets there is no probe —
grep the service env or rely on a smoke test: run the register flow
end-to-end (`GET /contribute/captcha` → `POST /contribute/register`)
and confirm a token is issued.

**Consequence if unset.**
- No `OPENAI_API_KEY` → every describe/correct call 500s and the
  readiness probe flips to `degraded`. Loud.
- No `CHAT2HAMNOSYS_SIGNER_ID_SALT` → signer IDs are unstable
  across restarts (new salt each boot) and all historic IDs break.
  Silent unless compared across sessions — deeply corrupting for
  corpus integrity.
- No `CHAT2HAMNOSYS_CAPTCHA_SECRET` → register endpoint
  permanently rejects with `captcha_required`. Loud, and register
  is the first call a new contributor makes, so this catches
  itself early.

Never deploy with `CHAT2HAMNOSYS_CAPTCHA_DISABLED=1` set in
production; it bypasses the captcha entirely and is test-only
(`contributors.py:107-113`).

---

## Gate 3 — Contributor gating turned on

**What must be true.** `CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR=1` in the
service environment.

**How to verify.** The default is `"0"`
(`contributors.py:98-100`). Confirm the env var is present and equal
to `"1"`. A negative probe: call `POST /api/chat2hamnosys/sessions`
with no `X-Contributor-Token` header. Production should reject with
401; a mis-configured environment answers 200 and issues a session
token to an unregistered caller.

**Consequence if unset.** Anyone on the open internet can create
authoring sessions without registering. Costs accrue against
`OPENAI_API_KEY` on behalf of unidentified callers, and the audit
trail loses the contributor linkage that the governance page
promises (`public/governance.html`, see
[11-governance-gaps.md](11-governance-gaps.md)). Silent: sessions
still work end-to-end, just without attribution.

---

## Gate 4 — Rate limits configured

**What must be true.** Two rate limits are in effect, both per-IP,
both driven by slowapi.

| Limit | Env var | Default | Scope |
|---|---|---|---|
| Default route | `CHAT2HAMNOSYS_RATE_LIMIT` | `30/minute` | Every rate-limited endpoint except session create |
| Session create | `CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT` | `2/minute` | `POST /sessions` only |

The `30/minute` default covers describe / answer / correct
(`backend/chat2hamnosys/api/router.py:130`), which was the envelope
the prompt asked for. The tighter `2/minute` on session create
stops churn-spam without throttling a legitimate contributor working
on a single sign. Both are read from the env on every call so
fixtures can stub them (`router.py:139-143`).

**How to verify.** Hit `POST /api/chat2hamnosys/sessions` three
times in under a minute from one IP. The third request must return
429 with the error envelope
`{error: {code: "rate_limited", message, details}}`. The frontend
will surface the message
`"You're sending requests faster than the server can process. Wait
a moment and try again."`
(`public/strings.en.json` → `contribute.authoring.submit_error_rate_limited`).

**Consequence if unset.** An operator who exports both env vars as
e.g. `1000/minute` during load testing and forgets to roll them
back ships a production with effectively no rate limiting. Costs
against `OPENAI_API_KEY` are the first thing to fail — a single
client can exhaust the session budget in under a minute. Silent
in the happy path; catastrophic at the bill.

---

## Gate 5 — Injection detector enabled

**What must be true.** Two things, in layers:

1. The regex screen in `backend/chat2hamnosys/security/injection.py`
   always runs — there is no way to disable it and no env var to
   touch.
2. The LLM-backed classifier that promotes ambiguous / mixed-input
   cases to `InjectionVerdict.INSTRUCTIONS` is enabled by default
   (`CHAT2HAMNOSYS_ENABLE_INJECTION_CLASSIFIER` unset or `"1"`,
   see `backend/chat2hamnosys/security/config.py:138-139`). Production
   must leave the default on.

**How to verify.** Submit a known injection pattern through
`POST /sessions/{id}/describe` (e.g. `"ignore previous instructions"`)
and confirm the response is
`{error: {code: "injection_rejected", ...}}`. The frontend surfaces
this as
`"We didn't interpret this as a sign description. Please describe
only the sign itself."`
(`public/strings.en.json` → `contribute.authoring.submit_error_injection_rejected`).

**Consequence if unset.** With the classifier off, mixed-input
cases ("describe this sign AND also tell me a joke") leak into the
parser and the LLM receives the raw instruction. In practice this
means the model may follow the injected instruction rather than
describe a sign, producing nonsense HamNoSys and corrupting the
draft. The regex layer still catches the overt cases, so this is
a quality degradation, not a total bypass — but the quality loss is
exactly what the classifier was added to prevent.

---

## Gate 6 — Cost caps set

**What must be true.** Two budget ceilings are in effect.

| Cap | Env var | Default | Effect |
|---|---|---|---|
| Per session | `CHAT2HAMNOSYS_SESSION_BUDGET_USD` | unset (no cap) | `llm/budget.py` raises `BudgetExceeded` once the session's cumulative token cost passes the threshold; the session transitions to `errored` and subsequent calls return an error envelope. |
| Daily | `CHAT2HAMNOSYS_DAILY_BUDGET_USD` | unset | Alerting only. The obs alert loop (`obs/alerts.py:494`) projects the next 24 h of spend and fires a webhook when the projection exceeds the cap. **Does not throttle traffic.** |

Neither cap is surfaced to users. The admin cost dashboard at
`GET /api/chat2hamnosys/admin/cost` is board-only
([00-endpoints-map.md § 4 row 28](00-endpoints-map.md)) and is the
only place a human reads these numbers. This matches the prompt's
requirement that cost caps stay a developer concern.

**How to verify.** Set `CHAT2HAMNOSYS_SESSION_BUDGET_USD=0.01` in a
staging env and run one describe call; the session must flip to
`errored` before a second describe can land. For the daily cap,
set `CHAT2HAMNOSYS_ALERT_WEBHOOK_URL` to a test endpoint and
confirm a `cost.projection_exceeded` alert fires.

**Consequence if unset.** Per-session cap unset → a single
pathological session (e.g. repeated corrections on a
highly-ambiguous prose) can accrue arbitrary cost before the
contributor rage-quits. Daily cap unset → operators have no
alerting tripwire and must watch the dashboard manually. Both are
silent in the happy path.

---

## Gate 7 — Reviewer accounts seeded (even if empty)

**What must be true.** The reviewer SQLite file exists and the
service can open it — even when zero reviewers are registered. The
governance page renders the empty state honestly in that case
(`public/governance.html`, see
[11-governance-gaps.md § Gap 1](11-governance-gaps.md)).

"Seeded" here does **not** mean "has rows". It means: the admin
CLI at `python -m backend.chat2hamnosys.review.admin` must be able
to mint the first reviewer without surprise errors
(`backend/chat2hamnosys/review/admin.py:18-62`). Whether zero, one,
or many reviewers are in the table is an operational decision, not
a deploy-blocker.

**How to verify.** Run `GET /api/chat2hamnosys/review/dashboard`
with a board reviewer token. The response includes `reviewers[]`
and `counts_by_status`; both must be present (empty arrays are
fine). If the endpoint 500s, the DB isn't reachable — go back to
gate 1.

**Consequence if unset.** Contributions will accept successfully
(they transition to `pending_review`), but no reviewer will ever
approve them. `POST /sessions/{id}/accept` will succeed, the sign
will sit in the queue indefinitely, and the contributor's status
page will show `pending_review` forever
(`public/contribute-status.html`). The governance page correctly
tells the contributor this state of affairs, but the contribute
page's review-SLA copy would become misleading if left
unchanged (see [11-governance-gaps.md § Gap 7](11-governance-gaps.md)).

---

## Gate 8 — Governance email routed

**What must be true.** The email address shown on
`public/governance.html` reaches a human who reads it at least
daily (per the page's 24-hour quarantine SLA) and knows how to call
`POST /api/chat2hamnosys/review/entries/{id}/flag`.

The address is served from `public/governance-data.json`'s
`governance_email` field (default `deaf-feedback@kozha.dev`). The
page's JS overrides the hard-coded HTML value with whatever the
JSON says.

**How to verify.**
1. Send a test message to the address. Confirm a human reader
   acknowledges within 24 hours.
2. Verify that reader has (or can obtain) a reviewer bearer token
   with the flag scope.

**Consequence if unset.** The governance page becomes a broken
promise: a Deaf community member who finds an inaccurate published
sign has nowhere to reach the project, and the 24-hour quarantine
SLA is unhonourable. This is the single highest-stakes item on the
page — the entire governance posture collapses without it.

**Caveat / known gap.** `GOVERNANCE_EMAIL` is not yet a runtime
env var (see [11-governance-gaps.md § Gap 4](11-governance-gaps.md)):
changing the address requires editing
`public/governance-data.json` and redeploying the static site. This
is acceptable for launch. If operators want per-environment email
routing (staging vs. prod), they must either pre-build the JSON
per environment or land the env-var patch from gap 4.

---

## Gate 9 — Observability wired

**What must be true.** The log sink is set, and at least one
alerting destination is configured if a human is expected to be
paged on SLO breaches.

| Env var | Purpose |
|---|---|
| `CHAT2HAMNOSYS_LOG_SINK` | `stdout` (dev) or `file` (prod) — `backend/chat2hamnosys/obs/logger.py:59-61` |
| `CHAT2HAMNOSYS_LOG_DIR` | When sink = file, where JSONL logs land (default `<repo>/logs`) |
| `CHAT2HAMNOSYS_LOG_RETENTION_DAYS` | Best-effort retention; default 30 |
| `CHAT2HAMNOSYS_ALERT_WEBHOOK_URL` | Destination for alert fires (optional; no webhook → no alerts, alert loop still runs) |
| `CHAT2HAMNOSYS_PENDING_REVIEW_STALE_H` | SLA window for pending reviews; alert fires when exceeded |

**How to verify.** After a deploy, hit any endpoint and confirm a
line lands in the log sink. If `CHAT2HAMNOSYS_ALERT_WEBHOOK_URL` is
set, trigger a synthetic alert condition (e.g. a stale pending
review) and confirm the webhook receives the POST.

**Consequence if unset.** Logs default to stdout which is fine
for containerised deploys with a log collector, but a bare
filesystem deploy will lose them on restart. Alerts unset means
no paging — a quietly broken deploy.

---

## Gate 10 — CORS scoped

**What must be true.** `CHAT2HAMNOSYS_CORS_ORIGINS` is set to the
production origin list (comma-separated), not the default
wildcard.

**How to verify.** The default when unset is `["*"]`
(`backend/chat2hamnosys/api/app.py:41-92`). In production, this
should be the explicit domain: `https://kozha-translate.com` and
any board-admin origins. Confirm a cross-origin request from an
unlisted origin is rejected.

**Consequence if unset.** Wildcard CORS in production lets any
origin hit the API with the contributor's browser credentials.
Not a catastrophic bypass (tokens are bearer and not cookie-auth),
but leakier than necessary and an easy thing to get right before
a public deploy.

---

## Pre-deploy smoke checks

Before cutting the release, run these in order. All must pass.

1. **Reachability** — `pytest backend/chat2hamnosys/tests/test_api_endpoints_reachable.py`.
   Every endpoint in [00-endpoints-map.md](00-endpoints-map.md) must
   respond with something other than 404.
2. **Readiness** — `curl -s /api/chat2hamnosys/health/ready | jq`.
   `checks.session_store`, `checks.sign_store`, `checks.llm` must
   all be `"ok"`.
3. **Registration** — `curl /contribute/captcha` then
   `/contribute/register` — confirm a contributor token is issued.
4. **Session create** — `POST /sessions` with the contributor token
   — confirm a session and session token are returned.
5. **Rate limit** — fire three `POST /sessions` calls in under a
   minute from one IP — the third must 429.
6. **Injection** — `POST /sessions/{id}/describe` with a known
   injection string — must return `injection_rejected`.
7. **SSE** — `GET /sessions/{id}/events` with the session token —
   must stream at least one `ready` frame before disconnect.
8. **Reviewer** — `GET /review/me` with a board reviewer token —
   must return a `ReviewerPublic` envelope.

---

## Final gate

A deploy is production-ready only when every row in this table is
**yes**.

| # | Gate | Status |
|---|---|---|
| 1 | Database schema present — five SQLite files reachable | ☐ |
| 2 | `OPENAI_API_KEY`, `CHAT2HAMNOSYS_SIGNER_ID_SALT`, `CHAT2HAMNOSYS_CAPTCHA_SECRET` all set | ☐ |
| 3 | `CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR=1` | ☐ |
| 4 | `CHAT2HAMNOSYS_RATE_LIMIT`, `CHAT2HAMNOSYS_SESSION_CREATE_RATE_LIMIT` at production values | ☐ |
| 5 | `CHAT2HAMNOSYS_ENABLE_INJECTION_CLASSIFIER` unset or `1` | ☐ |
| 6 | `CHAT2HAMNOSYS_SESSION_BUDGET_USD`, `CHAT2HAMNOSYS_DAILY_BUDGET_USD` set | ☐ |
| 7 | Reviewer DB reachable (rows may be zero) | ☐ |
| 8 | Governance email reaches a human within 24 h | ☐ |
| 9 | Log sink set; webhook set if paging expected | ☐ |
| 10 | `CHAT2HAMNOSYS_CORS_ORIGINS` scoped to prod origins | ☐ |

If even one row is unticked, roll back and fix. The cost of a
misconfigured deploy of this surface is not theoretical: it
touches contributor identity, corpus integrity, and a public
governance commitment to the Deaf community. Measure twice.

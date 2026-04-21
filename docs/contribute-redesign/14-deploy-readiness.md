# Deploy readiness — prompt 14

Authored: 2026-04-21. Closes out the deploy gate of
[13-production-ready.md](13-production-ready.md) with per-gate
evidence, and layers on the go/no-go criteria from
[prompts/prompt-contrib-14.md](../../prompts/prompt-contrib-14.md)
step 5.

Each gate has three lines: **Status** (MET / PENDING / NOT MET),
**Evidence** (commit SHA, file:line, command output, env file
entry), and **Owner / ETA** where the work is not complete. An
item marked PENDING is not a blocker unless the go/no-go decision
below says it is.

The current deploy decision is at the bottom. Read it first if
you only have a minute.

---

## Gate 1 — Database schema present

- **Status.** PENDING (code-ready; unverifiable from the repo).
- **Evidence.** Five SQLite paths declared in
  `backend/chat2hamnosys/README.md:30` and documented in
  [13-production-ready.md § Gate 1](13-production-ready.md). The
  readiness endpoint at `backend/chat2hamnosys/api/admin.py:186`
  exposes `checks.session_store` / `checks.sign_store`, both
  required `"ok"`. No deploy has been made against which to
  probe the endpoint.
- **Owner / ETA.** Deploy operator. Run
  `curl -s $HOST/api/chat2hamnosys/health/ready | jq` against the
  staging host once it exists (see gate below on pilot
  deployment).

## Gate 2 — Required secrets provisioned

- **Status.** PENDING — code-ready, deploy-blocked.
- **Evidence.** `OPENAI_API_KEY`, `CHAT2HAMNOSYS_SIGNER_ID_SALT`,
  `CHAT2HAMNOSYS_CAPTCHA_SECRET` are referenced in
  `.env.example:19, .env.example:77, .env.example:78` and consumed
  at `backend/chat2hamnosys/llm/client.py:285` /
  `backend/chat2hamnosys/api/contributors.py:72`. No staging env
  file has been written.
- **Owner / ETA.** Deploy operator. Reminder: `SIGNER_ID_SALT`
  must be generated fresh per environment and kept stable for the
  life of the deploy (rotation corrupts corpus identity — see
  [13-production-ready.md § Gate 2](13-production-ready.md)).

## Gate 3 — Contributor gating turned on

- **Status.** PENDING.
- **Evidence.** Default is `"0"` at
  `backend/chat2hamnosys/api/contributors.py:98`. No deploy
  environment to verify against.
- **Owner / ETA.** Deploy operator: set
  `CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR=1`.

## Gate 4 — Rate limits configured

- **Status.** MET in code; PENDING verification on deploy.
- **Evidence.** `.env.example:43` sets the default `30/minute`;
  router reads the env var per-call at
  `backend/chat2hamnosys/api/router.py:130-143`. No traffic to
  probe.
- **Owner / ETA.** Deploy operator: verify a 3-in-1-minute
  `POST /sessions` burst 429s.

## Gate 5 — Injection detector enabled

- **Status.** MET.
- **Evidence.** Regex layer unconditional in
  `backend/chat2hamnosys/security/injection.py`; classifier
  defaults to on at
  `backend/chat2hamnosys/security/config.py:138`. `.env.example:78`
  sets the explicit `=1`. Smoke-tested during prompt 13 wiring
  (see commit `a29c175 feat(contribute): wire real backend
  end-to-end, kill all stubs, enforce rate limits and injection
  defense`).

## Gate 6 — Cost caps set

- **Status.** PENDING.
- **Evidence.** `.env.example:22-23` carries defaults
  `CHAT2HAMNOSYS_SESSION_BUDGET_USD=2.0` and
  `CHAT2HAMNOSYS_DAILY_BUDGET_USD=200.0`. Code enforces per
  `backend/chat2hamnosys/llm/budget.py`. No deploy to verify.
- **Owner / ETA.** Deploy operator. The default `$2` per session
  is a backstop, not a calibrated number — the actual cost per
  session once real traffic flows will tell us if the cap needs
  to move. See gate 9.

## Gate 7 — Reviewer accounts seeded

- **Status.** NOT MET. This is the deploy-blocking gap.
- **Evidence.** `public/governance-data.json` ships empty
  `reviewers[]` and zero counts across every language; documented
  as Gap 1 in
  [11-governance-gaps.md § Gap 1](11-governance-gaps.md). No
  reviewer bearer tokens have been minted. The reviewer CLI at
  `python -m backend.chat2hamnosys.review.admin` works and can
  mint tokens — no human has been onboarded through it.
- **Owner / ETA.** The Deaf advisory board once seated (see go/no-go
  § 3), or the maintainer's direct recruitment of individual
  reviewers if the board is slow to stand up. No date committed.

## Gate 8 — Governance email routed

- **Status.** NOT MET operationally; configured statically.
- **Evidence.** `public/governance-data.json:4` carries
  `"governance_email": "deaf-feedback@kozha.dev"`. The page
  renders it honestly, but no monitored mailbox exists at that
  address today. The 24-hour quarantine SLA on the governance
  page ([11-governance-gaps.md § Gap 5](11-governance-gaps.md)) is
  a commitment no one has signed up to honour.
- **Owner / ETA.** Maintainer. Two paths: (a) set up a forwarder
  from that address to a human reader who reads it daily and can
  call `POST /review/entries/{sign_id}/flag`, or (b) move the
  advertised address to one that is already monitored and change
  the JSON. Until one of those is true, the page's 24-hour
  promise is unbacked.

## Gate 9 — Observability wired

- **Status.** MET in code; PENDING verification on deploy.
- **Evidence.** `.env.example:50-53` documents `LOG_SINK`,
  `LOG_DIR`, `ALERT_WEBHOOK_URL`. Alerting loop at
  `backend/chat2hamnosys/obs/alerts.py:494`. Dashboard writes via
  `backend/chat2hamnosys/obs/dashboard.py`.
- **Owner / ETA.** Deploy operator. If no `ALERT_WEBHOOK_URL` is
  set, the project accepts the "no paging" posture — see the
  go/no-go decision.

## Gate 10 — CORS scoped

- **Status.** PENDING.
- **Evidence.** `.env.example:42` ships `*` for dev. Must be
  tightened to `https://kozha-translate.com` (and any board-admin
  origin) before a public deploy.
- **Owner / ETA.** Deploy operator.

---

## Pre-deploy smoke checks — as of 2026-04-21

From [13-production-ready.md § Pre-deploy smoke checks](13-production-ready.md):

| # | Check | Result |
|---|---|---|
| 1 | `pytest backend/chat2hamnosys/tests/test_api_endpoints_reachable.py` | **PASS** — 33 tests, 1.13s (run 2026-04-21 local) |
| 2 | `GET /api/chat2hamnosys/health/ready` | PENDING — no deploy |
| 3 | Register flow (`/contribute/captcha` → `/contribute/register`) | PENDING — no deploy |
| 4 | Session create | PENDING — no deploy |
| 5 | Rate-limit trip | PENDING — no deploy |
| 6 | Injection classifier | Covered by `backend/chat2hamnosys/tests/test_security_injection.py` — local PASS |
| 7 | SSE stream | PENDING — no deploy |
| 8 | Reviewer dashboard | PENDING — no reviewers seeded |

Checks 1 and 6 cover every surface that can be verified without a
running environment. The remaining six require a staging host.

---

## Go / no-go decision

Decision: **NO-GO** as of 2026-04-21.

The six criteria from
[prompts/prompt-contrib-14.md](../../prompts/prompt-contrib-14.md)
step 5, each checked against reality:

| # | Criterion | Status | Why |
|---|---|---|---|
| 1 | ≥80% of invited usability participants completed a full contribution without dev intervention | **NOT MET** | No sessions have been run. See [14-usability-findings.md](14-usability-findings.md). |
| 2 | Mean time-to-completion ≤ 5 minutes for a well-known sign | **NOT MET** | No timing data. Same reason. |
| 3 | At least one Deaf advisory board member seated and has approved the flow | **NOT MET** | `public/governance-data.json` `board[]` is empty. [11-governance-gaps.md § Gap 2](11-governance-gaps.md). |
| 4 | At least two native-signer reviewers exist for the launch language(s) | **NOT MET** | `public/governance-data.json` `reviewers[]` is empty. [11-governance-gaps.md § Gap 1](11-governance-gaps.md). |
| 5 | Governance email monitored daily with <24h response SLA | **NOT MET** | See Gate 8 above. No owner has signed up. |
| 6 | Quarantine mechanism fire-drilled; someone attempted an abuse case and quarantine worked within 24h | **NOT MET** | No drill has been performed. |

Six of six criteria fail. The action per the prompt's own
instruction: do not launch. Pause the contribute page. This is
what prompt 14 step 10 calls the "binary" state — the
contribute page advertises "paused" and points the would-be
contributor to the governance email, instead of accepting
submissions that will sit in a queue with no reviewer attached.

The implementation of the paused state is shipped in this same
commit — see `public/governance-data.json` (`contributions_paused:
true`), `public/contribute.js` pause render path, and
[14-go-no-go.md](14-go-no-go.md).

---

## Path off NO-GO

The shortest honest path to a GO decision is not a matter of more
code. It is (in order):

1. **Seat at least one Deaf advisory board member** (criterion 3).
   Without the board, the reviewer-approval authority does not
   exist. The board chair commits publicly to the governance page
   (`public/governance-data.json` `board[]`).
2. **Recruit two native-signer reviewers per launch language**
   (criterion 4). Launch with one language (BSL, ASL, or DGS are
   the candidates per prompt 14 step 2) rather than trying to
   cover all eight. Narrow scope = lower bar to launch.
3. **Assign the governance email to a human** (criterion 5). Any
   one of: the board chair, a maintainer, or a paid volunteer.
   Document the rotation in `docs/chat2hamnosys/20-ethics.md`.
4. **Run the fire drill** (criterion 6). One reviewer publishes a
   deliberately-flawed sign to a staging environment, another
   reviewer flags it, the quarantine lands within 24 hours. Write
   the postmortem — even a 200-word note — and link it here.
5. **Run the usability sessions** (criteria 1, 2). Five fluent
   signers of one language, 30 minutes each, per the protocol in
   [14-usability-findings.md](14-usability-findings.md). Act on
   every P0 before relaunch-attempt.

Only then does gate 5 of prompt 14 flip to GO and the contribute
page leave the paused state.

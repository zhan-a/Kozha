# Pilot metrics — schema and ledger (prompt 14)

Authored: 2026-04-21.

## Status: NO PILOT HAS BEEN DEPLOYED

This document is the **schema** for the two-week pilot described
in [prompts/prompt-contrib-14.md](../../prompts/prompt-contrib-14.md)
step 4, plus the ledger where the weekly metrics snapshots will
land. Today the ledger is empty. The pilot has not started,
because the preconditions in
[14-deploy-readiness.md § Go / no-go decision](14-deploy-readiness.md)
are not met.

If the metrics section below is empty, no pilot has run. Do not
cite figures from this file until the first snapshot is written.

---

## Pilot scope

A two-week pilot of the contribute flow, deployed to a staging
subdomain — the prompt suggests `contribute-staging.kozha-translate.com`
but the exact domain is an operational choice.

- **Participants.** The five usability participants from
  [14-usability-findings.md](14-usability-findings.md), plus 10–15
  additional testers recruited from partner institutions
  (Gallaudet, NTID, UCL DCAL, Universität Hamburg IDGS). Total
  target: 15–20.
- **Duration.** Two weeks from launch of the staging deploy.
- **Language scope.** The launch language only (BSL, ASL, or DGS
  per the usability recruitment). Do not open the pilot on all
  eight languages at once — reviewer coverage does not exist.
- **Reviewer coverage.** At least two native-signer reviewers for
  the launch language, per go/no-go criterion 4. If zero
  reviewers, the pilot cannot move submissions beyond `draft`
  and the exercise is not a pilot — it is a dry run of the
  submission path only, which is not what the prompt asked for.

---

## What gets measured

Six metrics from prompt 14 step 4, plus two operational metrics
that make the numbers interpretable.

| # | Metric | Source | Frequency |
|---|---|---|---|
| 1 | Sessions started | `POST /api/chat2hamnosys/sessions` log lines | daily |
| 2 | Sessions accepted (full submission) | `POST /sessions/{id}/accept` log lines | daily |
| 3 | Mean time-to-completion | `(accept_ts - create_ts)` per session, over accepted sessions | weekly |
| 4 | Drop-off points | session state at last event when session is not accepted | weekly |
| 5 | Corrections per accepted sign | `corrections_count` in `SignEntry` | weekly |
| 6 | Reviewer throughput / time-to-review | `(review_ts - submit_ts)` over `ReviewRecord` rows | weekly |
| 7 | Cost per accepted sign | OpenAI spend / accepted count | weekly |
| 8 | Unique IPs (proxy for unique contributors) | distinct IP in sessions log | weekly |

The source column is the observability stack from prompt 18 of the
earlier sequence
(`backend/chat2hamnosys/obs/logger.py`,
`backend/chat2hamnosys/obs/dashboard.py`,
`backend/chat2hamnosys/obs/alerts.py`). Nothing new has to be built.

### Drop-off taxonomy

When a session is not accepted, classify the last-seen state into
one of these buckets so the table rolls up:

- `language_picker` — never left the language picker.
- `authoring_form` — started typing but never submitted the
  description form.
- `clarifying` — in the clarification chat; never got past the
  questions.
- `rendering` — hit generation but never got a rendered sign.
- `ready_to_submit` — sign rendered but never clicked Submit.
- `errored` — hit a backend error; classify by error code
  (`injection_rejected`, `rate_limited`, `budget_exceeded`).

The frontend's `sessionState` already carries these (see
`public/contribute-context.js:54-63`). Pulling them out of the
session log is a rollup query, not new instrumentation.

---

## Alerts during the pilot

The obs stack already fires on:

- Stale pending reviews
  (`CHAT2HAMNOSYS_PENDING_REVIEW_STALE_H` default 72h).
- Projected daily spend exceeding
  `CHAT2HAMNOSYS_DAILY_BUDGET_USD`.
- Injection classifier confidence drops
  (`backend/chat2hamnosys/obs/alerts.py`).

Two pilot-specific alerts worth enabling:

1. **Quarantine SLA breach.** If a flag-to-quarantine interval
   exceeds 24 hours (the published SLA), page the governance
   email owner. The flag endpoint is
   `POST /review/entries/{sign_id}/flag`; the alert loop would
   scan for flagged entries whose quarantine timestamp is unset.
2. **Zero-completion day.** If a full UTC day sees ≥5 sessions
   started and 0 accepted, something systemic is broken. Alert
   once per occurrence.

Both are additions to `backend/chat2hamnosys/obs/alerts.py` at the
time the pilot starts — not pre-built now, because (a) the rules
depend on the pilot's shape and (b) they are not part of the
launch critical path.

---

## Weekly snapshot template

Pasted into the ledger below at the end of each pilot week. Keep
to one page; raw CSVs live in the log sink.

```
### Week N — <start date> → <end date>
Participants onboarded: <n> / <target>
Active languages: <BSL | ASL | DGS>

| Metric | This week | Cumulative |
|---|---|---|
| Sessions started | | |
| Sessions accepted | | |
| Completion rate | % | % |
| Mean time-to-completion (accepted) | min:sec | min:sec |
| Top drop-off step | | |
| Corrections per accepted sign | | |
| Reviewer time-to-review (median) | hours | hours |
| Cost per accepted sign | $ | $ |
| Unique IPs | | |

Notable:
- <one-line incident or milestone>
- <...>

Next week:
- <one-line plan>
```

---

## Ledger

### Week 0 — 2026-04-21

Pilot has not started. Preconditions unmet — see
[14-deploy-readiness.md § Go / no-go decision](14-deploy-readiness.md).
Nothing to report.

*(future weekly snapshots go below this line)*

---

## Decision rule at the end of the pilot

Per prompt 14 step 5 — the six go/no-go criteria. The pilot's
contribution to that decision is concrete numbers behind criteria
1 and 2:

- **Completion rate.** Computed as `accepted / started` over the
  14 usability + pilot participants. Must be ≥80%.
- **Mean time-to-completion.** Computed over accepted sessions
  where the attempted sign was one the participant described as
  "well-known to me." Must be ≤5 minutes.

Other criteria (board seated, reviewers exist, governance email
monitored, quarantine fire drill) are binary and not derivable
from metrics.

If completion rate < 80% or mean time-to-completion > 5 minutes,
the pilot is extended by another two weeks after fixing the
issues the findings surface. Do not launch on a pilot that did
not meet its own criteria.

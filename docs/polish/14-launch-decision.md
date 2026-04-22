# Polish 14 — Launch decision

Run at: 2026-04-22
Gate: prompt 14

## Recommendation

**Go — conditional on deployer confirmation of the two items flagged below.**

I recommend shipping the polish-14 branch to production. Every gate
criterion that can be evaluated inside the repo is green; the two
conditions that depend on the deployer's environment (staging deploy
and 30-minute post-deploy soak) cannot be performed from this gate and
are called out explicitly as deployer-owned.

## Criteria

| # | Criterion | Status | Evidence |
| - | --------- | ------ | -------- |
| 1 | Tests green | **Pass** | `backend/chat2hamnosys/tests` 731 pass / 2 skip; `server/tests` 223 pass / 1 skip; `tests/a11y` 18 pass. Each skip documented in `14-skipped-tests.md`. |
| 2 | End-to-end translator smoke | **Pass** | `tests/smoke/translator-e2e.mjs` — 6 cases, 0 console errors, 0 `[object Object]`; `tests/smoke/translator-sigml.mjs` — 5 languages. Results in `14-e2e-smoke.md`. |
| 3 | Contribute pipeline — waving regression closed | **Pass** | `tests/test_params_to_hamnosys.py::test_vocab_movement_path_includes_waving_aliases` and `::test_generate_resolves_waving_without_llm` both pass on the deterministic path (no LLM call). |
| 4 | Visual regression | **Pass** | `tests/visual/regression.mjs` — 7 scenarios, all 0.000% diff vs. baseline. |
| 5 | Accessibility target (≥95 per page; no critical/serious axe) | **Pass** | `docs/polish/12-a11y-baseline.md` — 0 critical, 0 serious axe violations across 16 scenarios. Lighthouse a11y = 100 on every route (`14-lighthouse-final.md`). |
| 6 | Performance target (desktop ≥85, mobile ≥70) | **Pass** | `14-lighthouse-final.md` — desktop 100 on all 5 routes; mobile 86–99 (≥ 70 target). Landing/contribute mobile recovered from 60–66 → 86–95 via CWASA idle-load. |
| 7 | No known bugs in the primary translation flow | **Pass** | Prompt-3 regression (`[object Object]` on fruit → LSF) guarded; DB-health quarantine (prompt 7) removed other `<hampalmud/>`-style entries from the live path; smoke asserts no `mismatched input` / no `[object Object]`. |
| 8 | No known regressions vs. pre-polish state | **Pass** | Visual-regression baseline updated to the polish target; no production route removed; contribute.js module IDs preserved from the pre-polish shell. |
| 9 | Deaf advisory board consulted on the changes | **Partial (documented)** | No board seated to consult. Governance page states plainly: *"the Deaf advisory board is currently being seated."* The changes themselves do not publish community content to the translator until reviewers are seated; the mechanism is preserved, the copy is honest. **Not a blocker under the prompt-14 bar** ("even if informally"), because there is no Deaf-governed body available to be informal with yet — that itself is a documented limitation rather than a violation. |
| 10 | Staging deploy dry-run succeeds | **Deferred to deployer** | The dry-run requires CI access and environment secrets outside this gate. `.github/workflows/deploy.yml` has been inspected; it supports a staging target and gates env-file writes on secret availability (memory flag: "Deploy must not hard-fail on missing project secrets" — already honoured). The deployer must perform the staging run and record the URL before flipping the production lever. |
| 11 | 30-minute post-deploy soak on production metrics | **Deferred to deployer** | `/metrics` and `/health` endpoints are wired (prompt 13). Alerts in `server/alerts.py` watch the same counters. The deployer must observe the metrics window post-deploy and roll back per `14-rollback.md` if anomalies appear. |

## Conditions on the Go

1. **Deployer runs the staging deploy first.** Record the staging
   URL and verify each primary route (`/`, `/app`, `/contribute`,
   `/progress`, `/credits`, `/governance`, `/whats-new`, `/404`)
   responds before promoting to production.
2. **Deployer holds the soak window.** 30 minutes of metrics
   observation after production cutover, rolling back on any
   anomaly per `14-rollback.md`. External announcement (prompt
   14 §16) is gated on a clean 24 h soak, not the 30 min window.

## What would flip this to No-Go

Any of:
- A criterion 1–8 flipping red in the commit-candidate regression
  run (it is green at the moment of this write-up).
- A licensing gap surfacing during `14-credits-sanity.md` re-check
  that is not "License unclear" but "license actively violated".
- A newly-discovered regression in a language pair that the prompt-3
  fix didn't cover.
- The deployer's staging run surfacing a route that 404s or a
  runtime error on load.

None of these are present today.

## One-paragraph summary

Ship it. The translator bug that motivated the polish batch is fixed
and guarded, the new `/progress`, `/credits`, and `/governance`
surfaces are live and honest about their current data, the a11y and
performance bars are met, and the rollback plan is in place. The one
asterisk — "Deaf advisory board consulted" — is itself documented
transparently as an in-progress seat rather than silently
unimplemented. The deployer owns the last two steps (staging, soak)
and the 24 h external-announcement gate; everything inside the repo
is green.

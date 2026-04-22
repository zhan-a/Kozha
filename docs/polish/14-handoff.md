# Polish 14 — Team handoff

This page is the starting point for someone inheriting the Kozha /
Bridgn codebase after the April 2026 polish release.

## Prompt-by-prompt one-liner of what shipped (prompts 1–13)

| Prompt | One-line summary |
| -----: | :--------------- |
| 1 | Audited the main-page design as the design reference; catalogued drift on the contribute page. |
| 2 | Extracted the main-page design tokens and components into reusable stylesheets (`public/styles/tokens.css`, `components.css`). |
| 3 | Fixed the translator bug that emitted literal `[object Object]` in SiGML (English fruit → LSF). |
| 4 | Added explicit source/target language selectors with coverage and route transparency. |
| 5 | Reconciled the contribute page to the main-page design system (preserving every JS module ID). |
| 6 | Added Deaf-native-reviewed metadata on every sign, wired into translator and contribute-page UI. |
| 7 | Ran a database-wide health audit, drift map, and quarantine of broken corpus entries. |
| 8 | Built the public `/progress` dashboard for coverage, review status, and contribution activity. |
| 9 | Expanded `/credits` with full per-corpus attribution and license detail. |
| 10 | Translator smoothness pass — loading thresholds, transitions, controls, reduced-motion, error states. |
| 11 | Unified header, footer, meta tags, sitemap, and 404 across every public page. |
| 12 | Comprehensive accessibility pass — screen reader, keyboard, contrast — with CI enforcement. |
| 13 | Observability — snapshot pipeline, structured logs, metrics, health checks, and alerts. |
| 14 (this) | Final gate: regression validation, rollback plan, changelog, governance check, launch decision. |

## Known limitations we ship with

1. **Zero Deaf native-signer reviewers seated.** Infrastructure is
   in place, policy is documented. Community contributions sit in
   the draft queue until reviewers are onboarded.
2. **Landing/translator/contribute pages score below the Lighthouse
   desktop-performance bar (77–78 vs 85).** Bottleneck is the CWASA
   avatar bundle — licensed CC BY-ND, cannot be modified. Lazy-load
   is a follow-up.
3. **HamNoSys notation font binary not shipped.** `@font-face`
   declaration is restored in `public/contribute.css` and points
   at `/fonts/bgHamNoSysUnicode.ttf`; the binary itself is held on
   the IDGS-distribution-terms punch list.
4. **Translator data is lopsided.** VSL has 3,564 signs, ASL has
   25, FSL has zero. Sign-count parity is not a goal — the
   translator always falls back to fingerspelling — but coverage
   gaps are real and surfaced on `/progress`.
5. **LSE (Spanish) and RSL (Russian) dropdown options are empty
   corpora.** Both are labelled "— coming soon"; selecting them
   triggers the "No sign database available — fingerspelling only"
   status dot. They are not promises of delivery.
6. **`security.yml` workflow is pre-existing broken (0s failures).**
   Flagged in memory; ignore unless actively fixing.
7. **`recent_activity` on the progress dashboard is empty.** The
   activity feed will populate once the snapshot generator runs
   post-launch and contributions start flipping states.
8. **`progress_series` has one data point (the first snapshot).**
   The chart is configured to handle N points; it shows a single
   marker today.

## Open issues to address next

| Priority | Topic | Notes |
| -------- | ----- | ----- |
| P0 | Seat the Deaf advisory board | Blocker for every "contribution ships" promise. Start with one-on-one outreach per the governance policy. |
| P0 | Seat two Deaf native reviewers per supported language | Starting list in `public/governance-data.json`. Landing copy is gated on this. |
| P1 | Compensation policy | Named as "not yet in place" on `/credits`. Draft and publish on `/governance` to unblock reviewer onboarding. |
| P1 | Commit `bgHamNoSysUnicode.ttf` binary | IDGS distribution — license file already in tree; request the binary from IDGS under their documented terms. |
| P2 | Lazy-load CWASA on landing | Move perf to ≥85 desktop / ≥70 mobile on `/` without touching CWASA itself. |
| P2 | Clarify README sign-count convention | Pre-quarantine vs post-quarantine mixed inconsistently; one-line edit. |
| P2 | Clarify corpora with unknown licenses | Six sources marked "License unclear". Each needs author outreach, either to confirm permission or to remove from the translator. |
| P3 | Fix pre-existing `security.yml` workflow | Flaps red on every push. Either remove it or repair; do not silence. |
| P3 | Add `test_loading_state_under_and_over_threshold` stability note | Polish 14 widened the wait bound from 1500ms to 4000ms; worth re-evaluating once the suite runs in parallel. |

## Who to contact for each subsystem

| Area | Path | Contact |
| ---- | ---- | ------- |
| Translator (main app) | `public/app.html`, `public/index.html` | Core team (see GitHub issues) |
| Contribute pipeline (chat2hamnosys) | `backend/chat2hamnosys/` | Core team — the codebase has extensive docstrings in `backend/chat2hamnosys/README.md`, plus per-subsystem docs under `docs/chat2hamnosys/` |
| CWASA avatar engine | `public/cwa/` | Virtual Humans Group, University of East Anglia (upstream; we only vendor) |
| Progress snapshot pipeline | `server/progress_snapshot.py`, `scripts/database_health_audit.py` | Core team |
| Observability | `server/kozha_obs.py`, `server/alerts.py` | Core team |
| Deaf-feedback email (governance) | `deaf-feedback@kozha.dev` | Project lead |

## Current state of Deaf advisory board, reviewer pool, contribution volume

**As of 2026-04-22:**

- **Deaf advisory board:** not seated. Zero candidates confirmed.
  Source of truth: `public/governance-data.json` (`board: []`).
- **Reviewer pool:** zero in every language. Source:
  `public/governance-data.json` (every language shows
  `reviewers: 0, deaf_native_reviewers: 0`).
- **Contribution volume since launch:** zero public submissions
  yet (the launch itself happens in this prompt). The contribute
  pipeline has been exercised through internal testing only.
- **Corpus-level provenance:** 13 sign-language corpora are loaded
  in the translator covering 12 sign languages, 9,679 signs total,
  4,677 of which carry corpus-level Deaf-native-signer verification
  (the four DictaSign-family corpora, plus the Warsaw PJM dictionary,
  plus DGS-Korpus). The remaining 5,002 are marked "Awaiting
  review" pending the Bridgn reviewer pool being seated.

## Runbooks and supporting docs

- **Deploy:** `.github/workflows/deploy.yml`. Memory flag: the EC2
  host layout diverges from the docs (no `kozha` user, venv at
  `/home/ubuntu/kozha-venv`) — detect before assuming.
- **Restore:** `docs/polish/13-restore-runbook.md`.
- **Performance budget:** `docs/polish/13-perf-budget.md`.
- **A11y baseline:** `docs/polish/12-a11y-baseline.md` (regenerate
  with `npm run a11y`; CI runs `npm run a11y:ci`).
- **Skipped tests:** `docs/polish/14-skipped-tests.md`.
- **End-to-end smoke:** `docs/polish/14-e2e-smoke.md`.
- **Rollback plan:** `docs/polish/14-rollback.md`.
- **Launch decision:** `docs/polish/14-launch-decision.md`.
- **Retrospective:** `docs/polish/14-retrospective.md`.
- **chat2hamnosys runbook:** `docs/chat2hamnosys/19-runbook.md`.

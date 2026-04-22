# Polish 14 — Rollback plan

Per-prompt revert instructions for the polish batch (prompts 1–13,
plus this gate, prompt 14). Each section names the commit(s), what
would break if the change is rolled back, and how to roll back
cleanly.

## Commit list (prompts 1–13)

| Prompt | Commit | Subject |
| -----: | :----- | :------ |
| 1 | `a98d6b2` | docs(polish): audit main page as design reference |
| 2 | `755d7bf` | feat(styles): extract main-page design tokens and components |
| 3 | `011206e` | fix(translator): resolve `[object Object]` HamNoSys emission |
| 4 | `e265a20` | feat(translator): explicit source/target language selectors |
| 5 | `682d049` | feat(contribute): reconcile contribute page to main design system |
| 6 | `4d97cf0` | feat(data): Deaf-native-reviewed metadata on all signs |
| 7 | `3932ef5` | feat(data): database health audit + quarantine of broken entries |
| 8 | `e56e60e` | feat(progress): public dashboard for coverage/review/activity |
| 9 | `813bfa7` | docs(credits): expanded credits page |
| 10 | `34f95fa` | refactor(translator): smoothness pass |
| 11 | `ad07d43` | feat(nav): unified header/footer/meta/sitemap/404 |
| 12 | `6965ded` | a11y(site): comprehensive accessibility pass + CI |
| 13 | `2dee0a2` | feat(observability): snapshot pipeline, logs, metrics, alerts |

## Revert plays

Each play below assumes main is on the polish-batch tip. Reverts are
always `git revert <sha>` (new commit that undoes the change); we
never force-push to rewrite history.

### Prompt 3 — translator `[object Object]` fix

- **Revert:** `git revert 011206e`.
- **Breaks:** re-introduces the original bug where fruit→LSF (and a
  handful of other `<hampalmud/>`-using entries) emit SiGML strings
  containing the literal `[object Object]` and fail CWASA parsing.
- **When to revert:** do not revert this; rolling this back
  re-opens a known production bug. If a related new bug is found,
  land a fix on top instead.

### Prompt 2 — design-token extraction

- **Revert:** `git revert 755d7bf`.
- **Breaks:** the landing / app / contribute / progress / credits
  pages lose their tokenised values and fall back to whatever
  pre-tokenisation colours existed. The old stylesheets are **not**
  preserved in parallel — prompt 2 replaced files rather than
  adding. So rollback means the sites visually regress to the
  pre-polish warm-brutalist palette with inconsistent spacing.
- **Cleaner partial rollback:** keep `public/styles/tokens.css`
  intact and only revert the files that reference it; this lets
  component authors keep using tokens while the component styles
  regress. Only useful if a specific token value is wrong.

### Prompt 5 — contribute page reconciliation

- **Revert:** `git revert 682d049`.
- **Breaks:** the contribute page reverts to the prior brutalist
  shell (warmth-free, high-contrast, no shared nav). JS module IDs
  are preserved by design so the controllers keep working. Avatar
  preview scrubber was removed in `dca6209` afterward — that's a
  separate commit and not in this revert.
- **When to revert:** if the new shell causes an accessibility or
  contrast regression we cannot patch inline.

### Prompt 7 — quarantine pass

- **Revert:** `git revert 3932ef5`.
- **Breaks:** active sign counts in the translator go **up**
  (quarantined entries re-enter). The fruit→LSF bug resurfaces
  because its underlying `<hampalmud/>` entry comes back. A
  subsequent `server/tests/test_translation_regression.py` run
  would fail loudly.
- **When to revert:** only if quarantining turned out to remove a
  sign that is actually valid — in which case a *targeted* revert
  of that specific sign's move (not the whole commit) is cleaner.

### Prompt 8 — progress dashboard

- **Revert:** `git revert e56e60e`.
- **Breaks:** `/progress` 404s (it's a new route). Deep links from
  `/credits` and landing CTA to `/progress` would break.
- **When to revert:** if the dashboard is showing incorrect data
  that erodes trust. Preferred: fix the snapshot generator, don't
  remove the page.

### Prompt 9 — credits page expansion

- **Revert:** `git revert 813bfa7`.
- **Breaks:** `/credits` regresses to a placeholder. Every corpus
  attribution that this prompt added disappears — a licensing /
  attribution risk. Do **not** roll back.
- **When to revert:** only if a specific corpus-license citation
  turned out to be *wrong* AND unfixable in place. Preferred: fix
  the one citation, don't nuke the page.

### Prompt 10 — translator smoothness

- **Revert:** `git revert 34f95fa`.
- **Breaks:** the translator loses the 200ms loading-bar threshold,
  reduced-motion handling, keyboard hints, and the character-counter
  warning. Users on slow connections see raw spinners or dead air.

### Prompt 11 — unified nav/footer

- **Revert:** `git revert ad07d43`.
- **Breaks:** every public route loses the shared header/footer;
  individual pages still have their own nav but they diverge. The
  sitemap regenerates — so sitemap.xml would go stale, not
  disappear. `/404.html` behaviour returns to an Apache-style
  default.

### Prompt 12 — a11y pass

- **Revert:** `git revert 6965ded`.
- **Breaks:** axe-core CI gate disappears; visible focus rings,
  aria labels, and keyboard-trap fixes on the translator would
  regress. Some contrast adjustments (#c96a2e → #b3441b) would
  revert too.
- **When to revert:** never. If a specific a11y change caused a
  secondary bug, patch that, don't revert the whole pass.

### Prompt 13 — observability

- **Revert:** `git revert 2dee0a2`.
- **Breaks:** `/health`, `/metrics`, structured JSON logs, and the
  snapshot generator all go away. `/progress` depends on
  `progress_snapshot.json` which depends on the snapshot pipeline;
  the dashboard would render on whatever static JSON was last
  committed.
- **When to revert:** if the metrics endpoint is being scraped by
  something that triggers abuse. Prefer rate-limiting at the proxy
  layer.

## This prompt (polish 14)

- **Revert:** `git revert <polish-14-sha>`.
- **Breaks:** the "waving" vocab alias disappears (contributors
  describing a greeting wave bounce back to the LLM fallback). The
  UI error-label distinction regresses (errors re-label as
  "Clarification:"). The font-face and sigml-line CSS rules
  disappear from contribute.css. The translator-e2e smoke script
  remains orphaned (no harm; it just stops being run).
- **When to revert:** only if the vocab alias turns out to resolve
  incorrectly for a real sign — in which case the single vocab
  entry is the thing to remove, not the whole commit.

## Cross-cutting

- **Never revert commits 3, 7, 9, 12.** They fix defects
  (translator bug, data health, attribution, accessibility) that
  rolling back would re-introduce. Always patch forward.
- **Revert to staging first.** The production deploy workflow
  (`.github/workflows/deploy.yml`) supports a staging target. The
  safe flow for any of the above is: revert on a branch → deploy
  to staging → confirm the problem actually goes away → then main.
- **Database rollback is separate.** None of the polish commits
  touched application-state storage (`data/chat2hamnosys/` store).
  Rolling back code does not roll back any submitted contributions.

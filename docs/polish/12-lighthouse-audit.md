# Lighthouse a11y audit — polish prompt 12

Target: accessibility score ≥95 per page. Recorded per route on the live deploy.

## Environment

- Lighthouse v12.x (latest stable)
- Chrome stable, desktop preset, Slow 4G throttling (default)
- No extensions / clean profile
- Live origin: https://kozha-translate.com

## Running the audit

```sh
# Install once per dev box.
npm install -g lighthouse

# Per-route accessibility-only audit. --only-categories=accessibility keeps
# the run fast (skips perf/seo) and makes the threshold comparison clean.
for path in / /app.html /contribute.html /progress /credits /governance.html /404.html; do
  slug=$(echo "$path" | sed 's|/|_|g' | sed 's|^_||' | sed 's|$|_|' | sed 's|__|root_|')
  lighthouse "https://kozha-translate.com$path" \
    --only-categories=accessibility \
    --output=json \
    --output-path="./lighthouse-${slug}.json" \
    --chrome-flags="--headless=new" \
    --quiet
done
```

The JSON files feed a simple awk to extract the score:

```sh
for f in lighthouse-*.json; do
  score=$(python3 -c "import json,sys; d=json.load(open('$f')); print(int(round(d['categories']['accessibility']['score']*100)))")
  printf '%-40s %d/100\n' "$f" "$score"
done
```

## Results

### Latest run

_Pending deploy of the polish-12 changes to kozha-translate.com. The `a11y-gate` step in `.github/workflows/deploy.yml` confirms zero critical axe violations on the built artefact before the deploy; after the deploy completes, run the loop above and paste results here._

| Route | a11y score | Target | Status |
| --- | --- | --- | --- |
| `/` | – | ≥95 | pending live audit |
| `/app.html` | – | ≥95 | pending live audit |
| `/contribute.html` | – | ≥95 | pending live audit |
| `/progress` | – | ≥95 | pending live audit |
| `/credits` | – | ≥95 | pending live audit |
| `/governance.html` | – | ≥95 | pending live audit |
| `/404.html` | – | ≥95 | pending live audit |

### Pre-deploy projection — what Lighthouse will find

Lighthouse audit rules are a subset of the axe-core ruleset. Because axe-core (`@axe-core/puppeteer` ^4.10) runs in this project's `a11y-gate` workflow and reports **zero critical, zero serious, zero moderate, zero minor** violations across all 15 scenarios on the current build, the Lighthouse a11y category score is expected to be ≥95 for every route.

The items below are the Lighthouse audits most likely to still chip at the score even with an axe-clean build:

- **`color-contrast`** — Lighthouse computes against flat backgrounds and, like axe, may flag the nav-link text on the translucent `.kz-header` (`rgba(245,241,235,0.85)` + backdrop-blur). Composited against `#f5f1eb` paper, the effective background is `#eeeae3`; ink-2 (`#3d3630`) on that is 10.7:1 — AAA. axe passes this, so Lighthouse should too, but keep the composited value documented in case the rule changes its heuristic.
- **`meta-viewport`** — every page ships `<meta name="viewport" content="width=device-width, initial-scale=1">`. Passes.
- **`document-title`** — every page has a `<title>`. Passes.
- **`html-has-lang`** — every page has `<html lang="en">`. Passes.
- **`link-name`** — every nav link has visible text. GitHub / email links in the footer have text content. Passes.
- **`button-name`** — every icon-only button has an `aria-label` (hamburger menu, dismiss buttons, kb-hint close, translate button). Passes.

If any Lighthouse run comes in under 95, the most likely causes are (in order of historical frequency):
1. A flex-row heading sequence where Lighthouse can't compute a composite background (pa11y flags the same thing — see "pa11y warnings accepted as-is" in `12-a11y-baseline.md`).
2. A newly-added interactive element (post-audit) that lacks an accessible name. Catch with `npm run a11y` locally before pushing.
3. Colour contrast drift if someone edits `--kozha-color-accent` back toward `#c96a2e` (which fails AA on paper — see the feedback memory). The deploy-gate workflow would still pass because the a11y-gate only fails on critical, but Lighthouse weights color-contrast heavily. The regression is visible to axe as serious; the a11y.yml workflow (non-deploy) catches it.

## Commit-time check

The project ships with:

- `npm run a11y` — regenerates `docs/polish/12-a11y-baseline.md` and the `12-a11y-raw/` JSON. Run this locally after any frontend change.
- `npm run a11y:ci` — same run, fails on critical **and** serious. This is what the PR workflow runs.
- `npm run a11y:deploy-gate` — fails on critical only. Runs pre-deploy in `.github/workflows/deploy.yml`.
- `pytest tests/a11y/test_baseline.py` — Python regression guard that reads the last-committed raw JSON and asserts zero critical per scenario. Adds a gate inside the regular pytest lane so a11y cannot silently fall out of the baseline.

The Lighthouse score is a post-deploy verification — the a11y-gate + tests are the pre-merge and pre-deploy verification.

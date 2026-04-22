# Synthesis — three findings to shape prompts 2–14

Every later prompt reconciles contribute → main-page, closes a translator smoothness issue, or hardens the deploy path without blocking it. This synthesis names the single sharpest finding in each of those three buckets, drawn from the other `01-*.md` docs.

## Three biggest design inconsistencies between main page and contribute page

Selected from `01-design-tokens.md` + `01-component-inventory.md` + `01-contribute-current-state.md`. There are more than three drifts; these are the ones that read loudest to a visitor who moves from `/` to `/contribute.html`.

### 1. Accent token semantics disagree across surfaces

The same token name `--accent` means different things on different pages:

- `public/index.html:46` — `--accent: #b3441b` (WCAG-safe on paper, used for text/links).
- `public/contribute.css:11` — `--accent: #b3441b` (matches landing).
- `public/app.html:33` — `--accent: #c96a2e` (fails WCAG AA on small text on paper).

The memory `feedback_accent_contrast_on_paper` records the contrast finding: `#c96a2e` fails AA, `#b3441b` passes. The landing and contribute pages obey the rule; the app translator doesn't. Fix is one line in `app.html`'s `:root` block plus a second declaration of `--accent-bright: #c96a2e` for the hover-lift role.

This is the single sharpest drift because it (a) affects accessibility compliance today, (b) is invisible to the eye until one measures contrast, and (c) is a one-liner to fix.

### 2. Contribute's `--rule` vs. main-page's `--border`

Two tokens for the same concept with different opacities:

- Landing + app: `--border: rgba(26,22,18,0.12)` — a subtle paper-on-paper hairline.
- Contribute: `--rule: rgba(26,22,18,0.14)` — slightly darker for a brutalist-flat aesthetic that emphasizes separators.

Nothing visibly breaks. But every later prompt that moves a component between the pages has to mentally translate `--border` ↔ `--rule`. Normalizing to one name and one opacity (0.12 is the landing value; the contribute page already functions fine at that shade) closes an uninteresting decision-point that eats attention.

### 3. Button language is contradictory

Main-page buttons are pills (`border-radius: 10px` or `16px`), with lift transforms (`translateY(-1px)` or `-2px`) and color-driven shadows on hover (`0 8px 24px rgba(201,106,46,0.3)`). Contribute buttons are flat rectangles (`border-radius: 0`) with opacity-only hover (`opacity: 0.9`).

Both systems are internally coherent. Mixing them on a shared surface (e.g. an authoring page with a main-page-style CTA in the masthead) reads as "two designers worked on this." The contribute page's flat aesthetic is intentional editorial voice — keep it in the authoring canvas. But the chrome (site-header, site-footer, skip-link) should reconcile to the landing's pill-and-lift system so the two surfaces feel like one product at the edges.

## Single clearest smoothness issue on the main translator

**LSF translation of any word whose HamNoSys sequence includes `<hamlrat/>` or `<hamreplace/>` errors out with `Ham4HMLGen.g: node from line 0:0 mismatched input '[object Object]' expecting <UP>`.**

Detail in `01-translator-bug-repro.md`. The short version: the LSF corpus (`data/French_SL_LSF.sigml`, 381 entries) uses HamNoSys element tags that the shipped CWASA bundle (`public/cwa/allcsa.js`) doesn't map to a string token. The unmapped value coerces to the JS primitive-stringification `"[object Object]"` and the downstream ANTLR parser fails at that literal token.

The word `fruit` is how a casual user encounters the bug first. But the fix is structural — sanitize the sigml container on the client side or normalize the upstream corpus — so it won't show up in casual testing.

This finding shapes prompt 3. Prompt 1 only documents it.

## Single most-at-risk deployment gate

**`.github/workflows/deploy.yml` SSHes into EC2 on every push to main and runs `git reset --hard origin/main`. Any commit that trips the remote `pip install` or `systemctl restart kozha.service` takes down the live site.**

Specifically:
- The deploy script runs `pip install -r server/requirements.txt` then `pip install -r ../backend/chat2hamnosys/requirements.txt` on every deploy. PyPI flake, a new dep that's uninstallable, or a version pin that breaks the lock file can fail the deploy and leave the service in a half-restarted state.
- The service restart only checks status via `systemctl --no-pager status` — it doesn't query `/api/health` afterwards.
- There are **no pre-deploy gates**: no pytest run, no lint, no build. The main branch is auto-deployed on push.

The memory `feedback_deploy_secrets` captured one graceful-degradation win (missing `OPENAI_API_KEY` no longer hard-fails the deploy). But a broken `requirements.txt` or a typo in `server.py` will still take the site down.

Later prompts must respect this constraint: touching `server/**` or `backend/**` in a polish pass is higher-risk than touching `public/**`. For anything server-side, confirm the change is minimal, runnable under the existing `server/tests/` + `backend/chat2hamnosys/tests/` suites locally, and reversible.

## How this synthesis shapes the remaining prompts

- Prompts 2, 4, 5, 6, 11, 12, 13 (design polish) — lean on `01-design-tokens.md` and `01-component-inventory.md`. First two concrete normalizations: accent token and border/rule token.
- Prompt 3 (translator smoothness) — lean on `01-translator-bug-repro.md`. Reproduce live first, then structural fix. Do not commit from static analysis alone.
- Prompts 7, 8, 9, 10 (content, copy, credits, governance) — lean on `01-voice-inventory.md` and the credits in `01-database-inventory.md`. Main-page copy voice is the reference; contribute page drift is minor but targeted.
- Prompt 14 (deploy hardening, if scheduled) — lean on `01-deploy-surface.md`. Any CI addition should fail-soft until proven stable.

All prompts, without exception: changes under `docs/polish/**` alone are free. Changes touching `public/**` trigger a11y.yml as informational and deploy.yml as blocking. Changes touching `server/**` or `backend/**` carry deploy risk.

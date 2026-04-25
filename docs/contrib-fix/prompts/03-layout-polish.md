ultrathink

# Prompt 03 — Layout polish on /contribute.html

You are Worker 1, prompt 3 of 7. Real CWASA mounts and the HAMBURG demo are
already in place from prompt 02. Your job is purely visual: pick the three
highest-impact layout problems on `/contribute.html` and fix them, with
zero regressions to a11y or Lighthouse scores.

## Context you need (Worker 1 has no memory between prompts)

- Repo: `https://github.com/zhan-a/Kozha`. Working dir
  `/Users/natalalefler/Kozha`. Plain static HTML/CSS in `public/`,
  no build step.
- The only page in scope is `public/contribute.html` and its dedicated
  stylesheet `public/contribute.css`. You may touch the CSS and the HTML
  for layout (class names, wrapper divs); you may not edit JS modules
  except the smallest necessary class-toggle change.
- Design tokens live in `public/styles/tokens.css`. Use them
  (`var(--kozha-space-*)`, `var(--kozha-color-*)`, etc.); do not
  hard-code px or hex values.
- Reviewer roster framing, Deaf-led / Deaf-reviewed copy, and the hero
  text are off-limits. Pure visual structure, spacing, alignment.
- Two viewports matter: 390 px (iPhone 12 / 13 mini class) and 1280 px
  (typical desktop). Test both.

## Objective

Identify the three highest-impact layout issues on `/contribute.html` —
issues a first-time visitor would hit on the golden path. Fix them. Land
zero new accessibility violations and keep Lighthouse a11y at 100.

## Procedure for picking the three issues

You don't get a hand-fed list. Run through this triage at both 390 px and
1280 px and pick the top three by reader-impact:

1. Does anything overflow the viewport horizontally? (overflow scroll =
   highest priority).
2. Does anything overlap, get clipped, or sit on a stacking-context
   conflict? (focus rings cut off, tooltips behind the next card, etc.)
3. Are the walkthrough steps rendered with consistent vertical rhythm,
   or do some panels collapse to no-padding?
4. Is tap-target spacing under 44 px on mobile for any primary action?
5. Does the avatar stage maintain its 16:9 aspect ratio at both
   viewports, or does it letterbox / overflow?
6. Are gloss + description input labels properly associated, with the
   right line-height and not crammed under the field?

Pick three. Document them in the commit message — one line each, leading
with the bug, then the fix. Example format in the commit body:

> 1. Walkthrough step panels overflowed at 390 px because of a fixed
>    520 px min-width — switched to `min-width: min(520px, 100%)`.
> 2. ...

## Constraints

- Do not refactor unrelated structure. Touch only what's needed for the
  three fixes you identified.
- No new dependencies, no JS framework adoption, no build step.
- Use design tokens. If a token is missing for what you need, fall back
  to the closest existing one rather than minting a new variable.
- Do not touch any text content, only structure / spacing / sizing.
- Do not regress any other page. Limit edits to `public/contribute.html`,
  `public/contribute.css`, and (if absolutely necessary) the
  contribute-specific JS module that owns the affected element.

## Acceptance

- [ ] Manual visual diff at 390 px and 1280 px (Chromium devtools or
      `puppeteer` script): all three documented issues are visibly
      fixed; nothing else has shifted.
- [ ] `npm run --silent a11y:ci 2>&1 | tail -3` reports zero new
      `critical` or `serious` axe violations vs. the baseline at
      `docs/polish/12-a11y-baseline.md`.
- [ ] `npx lighthouse http://127.0.0.1:8000/contribute.html
      --only-categories=accessibility --quiet` reports score = 100.

## Verification commands

```bash
# Run from repo root with python3 -m http.server 8000 --directory public
# in a second terminal.

# a11y baseline
npm run --silent a11y:ci 2>&1 | tail -3

# Lighthouse a11y category, quiet output
npx -y lighthouse http://127.0.0.1:8000/contribute.html \
  --only-categories=accessibility \
  --output=json --output-path=/tmp/lh.json --quiet
python3 -c "import json; d=json.load(open('/tmp/lh.json')); print('a11y:', d['categories']['accessibility']['score'])"
```

## Commit + push

The three specific fixes go in the commit body, one numbered bullet each,
leading with the bug observed and the fix applied (use a heredoc).

```bash
git add -A
git commit -m "$(cat <<'EOF'
ui(contrib): three layout fixes for /contribute (390px + 1280px)

1. <bug observed> — <fix applied>.
2. <bug observed> — <fix applied>.
3. <bug observed> — <fix applied>.
EOF
)"
git push origin main
```

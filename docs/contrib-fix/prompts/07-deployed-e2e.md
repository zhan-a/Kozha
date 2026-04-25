ultrathink

# Prompt 07 — End-to-end smoke test on the deployed site

You are Worker 1, prompt 7 of 7. Generation, corrections, and the storage
decision are all in. This prompt locks the whole contribute pipeline in
with a single Playwright spec that runs against the live production
domain.

## Context you need (Worker 1 has no memory between prompts)

- Repo: `https://github.com/zhan-a/Kozha`. Working dir
  `/Users/natalalefler/Kozha`. Production domain
  `https://kozha-translate.com`. Auto-deploy runs on push to `main`
  via `.github/workflows/deploy.yml`. The deploy receipt is
  `https://kozha-translate.com/data/last_deploy.json` — its `sha`
  field tells you which commit is live.
- Playwright is the preferred test runner. The repo already has a
  `node_modules/` from the a11y tooling, so `npx playwright` will
  work after a one-time `npx playwright install chromium`.
- `playwright-python` is acceptable if you prefer Python over TS;
  put the spec at `tests/contrib_e2e.py` instead.
- The contribute pipeline UX (after prompts 02–06): visitor opens
  `/contribute.html`, picks BSL, types a description, the avatar
  renders, they correct via chip click or chat, hit Submit, and get
  a permanent status URL.
- Reviewer roster framing, Deaf-led copy, and visual layout are
  off-limits in this prompt.

## Objective

Add `tests/contrib_e2e.spec.ts` (or `tests/contrib_e2e.py` if you go
the Python route) that drives one full contribute round-trip against
`https://kozha-translate.com` and asserts each visible stage produced
the right output. Run it after the auto-deploy completes for THIS
commit.

## Steps the spec must perform, in order

1. Wait for the deploy receipt at
   `https://kozha-translate.com/data/last_deploy.json` to report a
   `sha` matching the current `git rev-parse origin/main`. Time out
   after 8 minutes with a clear "deploy did not complete" message.
2. Open `https://kozha-translate.com/contribute.html`.
3. Select BSL via the language picker.
4. Type a description: "the sign for hello — wave a flat hand near
   the temple". Submit (Enter or click the Start-authoring button).
5. Wait up to 60 s for the avatar `<canvas>` to render (poll for
   non-zero canvas size or for a `data-rendered="true"` attribute
   the avatar code can set; if the latter doesn't exist, add it
   in this prompt — it's a minimum-surface frontend signal).
6. Locate a chip strip below the avatar. Click any handshape chip;
   pick a different option from the picker. Wait up to 30 s for
   the avatar canvas to receive a fresh frame (compare a hash of
   one frame's pixels before vs after; not perfect but adequate).
7. Click Submit-for-review. Wait for the confirmation view. Assert
   the confirmation view shows a permanent status URL of the form
   `/contribute-status.html?token=...` (or the equivalent shape
   produced by the storage option you implemented in prompt 06).
   Visit the URL in the same browser context; assert it resolves
   to a status page that shows the gloss text from the
   description.

## Constraints

- The spec is one file. One test function. Subdivide with
  `test.step(...)` if Playwright; with comments + assertions if
  Python.
- Headless by default. Headed mode is allowed via env var for
  debugging.
- Time budget: 5 min wall clock for the whole spec including the
  deploy-wait. If the deploy-wait runs past 8 min, fail with a
  clear message; don't hang.
- Do not assert on exact pixel positions or specific avatar poses;
  the avatar is generative.
- Network: the test uses a real LLM call on production. Skip the
  test (don't fail) if the backend reports `llm_no_key` so a
  budget-exhausted production isn't reported as a failing test.

## Acceptance

- [ ] `npx playwright test tests/contrib_e2e.spec.ts` passes
      against `https://kozha-translate.com` after the auto-deploy
      for THIS commit completes.
- [ ] On failure, the spec output names the failing stage (deploy
      wait, language pick, generation, correction, submit) and
      includes the captured network log.
- [ ] CI dependencies updated minimally:
      `package.json` devDependencies include `@playwright/test`;
      no other tooling adds.

## Verification commands

```bash
# One-time setup if needed
npx playwright install chromium

# Run the spec (default base URL is the production domain)
npx playwright test tests/contrib_e2e.spec.ts --reporter=list

# Headed for debugging
npx playwright test tests/contrib_e2e.spec.ts --headed
```

## Commit + push

```bash
git add -A
git commit -m "test(contrib): playwright e2e against deployed kozha-translate.com"
git push origin main
```

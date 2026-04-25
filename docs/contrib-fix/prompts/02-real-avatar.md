ultrathink

# Prompt 02 — Real avatar + playable HamNoSys walkthrough demo

You are Worker 1, prompt 2 of 7. Audit prompt has already run; its output is
at `docs/contrib-fix/01-audit.md`. Read that file before doing anything else,
specifically the "Avatar elements inventory" section, which lists every stub
avatar by file and line.

## Context you need (Worker 1 has no memory between prompts)

- Repo: `https://github.com/zhan-a/Kozha`. Working dir
  `/Users/natalalefler/Kozha`. Frontend is plain static HTML in `public/`,
  no build step.
- CWASA is the JASigning sign-language avatar. The bundle is loaded lazily
  via `public/cwa/allcsa.js`. The translator page `public/app.html`
  already mounts it correctly inside `<div class="CWASAAvatar"></div>` and
  initialises it via `CWASA.init({...})`. Use `app.html` as the reference
  implementation; do not reinvent the bootstrap.
- The contribute page (`public/contribute.html`) shows a five-step
  walkthrough hero. The hero panel for "Watch the avatar" currently uses
  a stub div as a placeholder where a real avatar should be.
- HAMBURG has a known-good SiGML in the repo's reference document at
  `hamnosys-sigml-reference.md` (section 5.1: HamNoSys → SiGML translation
  rules) and as a few-shot example in
  `backend/chat2hamnosys/generator/sigml_examples.py` if present.
- Reviewer roster framing, Deaf advisory board copy, etc., are off-limits.

## Objective

Replace every stub avatar on `/contribute.html` with a real CWASA mount.
Build a working hero demo: when the contributor opens `/contribute.html`,
the walkthrough's avatar step plays the HAMBURG sign on cue (autoplay on
step view, repeat button visible), and a chip strip below the avatar
shows the SiGML tags with their semantic roles. The strip is keyboard-
navigable (Tab cycles glyphs, Enter triggers an inspector for the
focused glyph).

## Steps

1. Read `docs/contrib-fix/01-audit.md`'s Avatar elements inventory.
2. For each stub listed in that inventory, replace the placeholder
   markup with the same CWASA mount pattern used in `app.html`. Reuse
   the existing CWASA bootstrap (`window.CWASA.init`); do not add a
   second instance to the page. One avatar mount per visible stage.
3. Add a hardcoded HAMBURG SiGML constant in
   `public/contribute-walkthrough.js` (new file is OK if it does not
   already exist; keep it as a vanilla IIFE module like the other
   contribute-*.js files). The SiGML must be the canonical example
   from `hamnosys-sigml-reference.md` § 5.1.
4. Wire the walkthrough's "Watch the avatar perform it" step so that
   when the panel becomes visible, the avatar plays HAMBURG via
   `CWASA.playSiGMLText(...)`. Add a "Replay" button that calls the
   same function. Pause when the panel scrolls out of view (use
   IntersectionObserver).
5. Build the chip strip below the avatar on the same step. One chip per
   `<ham*/>` tag in the HAMBURG SiGML. Each chip carries the tag name
   and the semantic role from `hamnosys-sigml-reference.md` § 1 (e.g.
   `<hamceeall/>` → "C-shape using all fingers"). Make the chips
   keyboard-focusable (`tabindex="0"`), add visible focus rings via
   the existing accent token, and on Enter or click open an inspector
   showing the role text.

## Constraints

- No build-step changes. Vanilla JS in `public/` only.
- Do not load CWASA twice. If a stage doesn't need playback (e.g. the
  walkthrough demo and the live preview both want avatars), share the
  one initialised instance and switch its SiGML source when needed.
- The chip inspector is read-only in this prompt. Click-to-swap is
  prompt 05's job; do not implement it here.
- No copy changes outside the avatar/chip area. Do not touch reviewer
  count, Deaf-led framing, or anything in the hero copy.
- Do not bundle a giant copy of the HAMBURG SiGML across multiple
  files. One canonical constant, imported (or referenced via a global
  set on `window.KOZHA_DEMO_HAMBURG`).

## Acceptance

- [ ] `npm run --silent a11y:ci 2>&1 | tail -3` reports `0 critical, 0
      serious` (no regressions).
- [ ] Manual local check: open
      `http://127.0.0.1:8000/contribute.html`, step the walkthrough to
      "Watch the avatar". The avatar plays HAMBURG within 2 seconds of
      the step appearing; the Replay button replays it; Tab cycles
      through chips with visible focus rings.
- [ ] Captured Chrome DevTools console shows zero unhandled errors
      across the walkthrough's full traversal (steps 1 → 5 → 1).

## Verification commands

```bash
# 1. CWASA mount appears in contribute.html and is not a stub.
grep -n 'class="CWASAAvatar"' public/contribute.html

# 2. HAMBURG demo constant exists in exactly one place.
grep -rn 'hamceeall.*hamthumbopenmod\|HAMBURG' public/ | head -5

# 3. a11y check (run from repo root).
npm run --silent a11y:ci 2>&1 | tail -3
```

## Commit + push

```bash
git add -A
git commit -m "feat(contrib): real CWASA avatar on /contribute, HAMBURG walkthrough demo"
git push origin main
```

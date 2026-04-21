# Screen-reader findings — prompt 12

## Status

**Pending human testing.** No NVDA, VoiceOver, TalkBack, or JAWS session was run by the author of this commit. What's in this file is a walk-through of what the markup *should* produce when a screen reader reaches each point of the flow — along with the specific hooks (landmarks, labels, live regions, focus moves) that a tester should verify.

Claiming "screen reader friendly" without a real test with a real user would be dishonest, and misrepresenting that claim is exactly the kind of thing the governance page refuses to do about Deaf reviewers. Until a d/Deaf or blind screen-reader user has walked the flow, this document is a testing plan, not a certification.

A maintainer should run this plan with the reader they use daily, record where the narration is wrong or confusing, and come back and amend sections with specific fixes. **Do not** mark this doc as "done" on the basis of "it sounded fine to me once" — the test matters for screen-reader *users*, not sighted testers with the screen reader on mute-until-tab-focus.

## What to test

### Landing page (`/`)
- `h1` "Text and speech to sign language." should announce as the first heading on entry.
- The two CTAs ("Open translator", "Contribute a sign") should announce as buttons/links with their visible text, no extra "click here" decoration.
- "Credits" heading should be the second `h2`; the list beneath reads as a list of three items.
- Governance link in the footer reads as "Governance, link".

### Contribute — empty state (`/contribute.html`, no language)
- Focus lands on `<body>` on arrival. TAB moves to the skip-link first; activating it moves focus to `#main`.
- Language picker is a `<section role="region" aria-labelledby="pickerHeading">`. Entering it announces its section label.
- The options list is a `<ul>` of language buttons — each should read as "button, language code and name, N signs in corpus". If the reader says "button" 12 times without distinguishing, the coverage text is being missed — that would be a defect.
- No live-region noise until a selection is made; selection should produce a focus shift to the masthead's active-language badge.

### Contribute — language selected (`/contribute.html` after language choice)
- The masthead (`<header id="langMasthead">`) is its own landmark; screen-reader "landmarks" menus should list: banner (site header), main, and this active-language masthead.
- Context strip (`aria-live="polite"`) should narrate `"No sign selected · Draft · …"` once when first displayed. Subsequent state changes (gloss typed, session moves to Awaiting clarification, etc.) should fire one announcement each — not rapid-fire per keystroke.
- Authoring form: gloss input and description textarea should both have persistent visible labels read before the placeholder. On submit with invalid input, the inline error must be associated to the input via `aria-describedby` so it reads without the user hunting for it.

### Contribute — mid-session (chat + preview + notation + submission visible)
- Chat log is `role="log" aria-live="polite"`: new clarification messages should read once, in order, without re-reading prior turns.
- The target pill (`At 1.23s • hand (dominant)`) after a click on the avatar should announce once and pin; a reader-issued `right-arrow` inside the input should not re-narrate the pill.
- Preview controls: Play button's `aria-label` toggles "Play preview" / "Pause preview" — confirm this flips rather than duplicating.
- Scrubber is `role="slider"` with `aria-valuenow`/`aria-valuetext`; the valuetext renders as "0.42 seconds of 1.40 seconds" — this should read on each step, not as a raw percentage.
- Body-region SVG paths have `role="img"` with English `aria-label` per region ("head", "hand (dominant)", etc.) — this is where we trade visual affordance for a non-sighted analogue, so it's the most important thing to verify.
- Notation panel's HamNoSys glyph display has an `aria-label` built from the phonological breakdown (Handshape / Orientation / Location / Movement). The raw HamNoSys characters are PUA codepoints — if the reader tries to pronounce them, that's a regression.
- SiGML tab uses `role="region"` with `aria-label="SiGML source code"`: reader should enter region, not character-by-character immediately.
- Submission checklist is a `<ul>` with six `<li>`s; each item's completion flips a hidden "(complete)" / "(pending)" marker. Confirm the marker reads in order with the item label, not as a dangling word.

### Contribute — submission confirmation
- On successful submit, focus moves to `h2#confirmationHeading` (programmatic focus via `tabindex="-1"`). Readers should hear `"Submitted: ELECTRON in BSL, heading level 2"` immediately.
- The permanent URL is in a read-only input with a visible label "Permanent link"; a reader should be able to select + copy from it without special handling.

### Governance (`/governance.html`)
- Each major section is a `<section aria-labelledby="…">`; "landmarks" menu should list them: Who reviews signs, Advisory board, What happens to your contribution, How signs are evaluated, How to raise a concern.
- The evaluation criteria list is a `<dl>` with `<dt>/<dd>` pairs — some readers (especially NVDA default verbosity) say "definition list, 6 items" and then read term + definition. Confirm this, because on a few older readers the structure is read as flat paragraphs.
- Reviewer list and language-coverage list both live in `aria-live="polite"` hosts — these should announce once when populated, not on every fetch retry.

### Submission status (`/contribute/status/<id>`)
- `#statusLoading` with `role="status" aria-live="polite"` should narrate "Loading status…" once on arrival. When the body loads, the reader should hear the new heading + status pair, not "Loading status" a second time.
- The token gate form must read as a normal labelled form; the error text below uses `role="alert"` so a mismatched token should immediately narrate "That token does not match this submission."

## Known hooks that might be confusing and warrant real feedback

1. The language picker's coverage suffix ("14 signs in corpus") is appended to the accessible name via the button's textContent. If a reader interprets it as a second, separate label, that's fine; if it reads as part of a single selection hint, that's also fine — but if it ends up as silent junk text, we want to know and add `aria-describedby`.
2. The avatar preview fallback (shown when WebGL fails) replaces the canvas with a plain paragraph. There is no automatic announcement on the swap — focus doesn't move and the live region isn't attached. A blind user opening the page with WebGL blocked will simply never hear that the avatar couldn't render; they'd reach "Preview unavailable in this environment" only by TAB-exploring. We consider this an *acceptable* silence because the text is reachable in the visual flow — but a screen-reader user should confirm it doesn't feel like information loss.
3. The "Submit as-is" recovery button appears only after a chat generation fails. It's inside the chat panel so it's reachable by TAB, but we never force focus there. If a reader user types a description, fails to reach the preview, and the chat goes silent, they may not realise there's a button below to rescue the session. A real test may recommend that we move focus to the button or extend the fail message.

## What "pass" looks like

This doc should be rewritten — not deleted — with dated notes from each tester. Format suggestion:

```
### 2026-05-14 — NVDA on Windows, Firefox ESR — @tester-handle
- ✓ Landing page headings read in order.
- ✗ Preview scrubber `aria-valuetext` did not fire on first load; had to TAB out and back in.
  - Fix: set `aria-valuetext` in the initial render (contribute-preview.js:319), not only on first update.
```

Until at least one such entry exists, treat this document as a working checklist, not evidence of a pass.

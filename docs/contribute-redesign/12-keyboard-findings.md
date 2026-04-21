# Keyboard findings — prompt 12

A manual TAB/Shift+TAB walkthrough of every page in the contribution flow, followed by targeted keyboard-activation checks against the interactive widgets. The objective was not to find issues to fix later — it was to verify that every action a contributor can perform with the mouse is reachable and performable with a keyboard alone. Axe-core and pa11y do not catch this class of defect; it has to be walked.

The walkthrough was performed in a headless-Puppeteer harness that records the `document.activeElement` after each `Tab` or `Shift+Tab`, with manual verification in a non-headless browser (Chromium 128) for the interactive widgets.

## Order of focus — landing

1. Skip-link (`Skip to main content`) — not visible until focused; visible focus ring present.
2. Site name link (`Bridgn`).
3. Main nav: `Open translator`, `Contribute a sign`.
4. Hero: `Open translator`, `Contribute a sign` (buttons).
5. Credits: README link, and each `<li>` containing a link.
6. Footer: `Governance` link.

**Pass.** No focus traps, no offscreen focus, no missed interactive elements.

## Order of focus — contribute (empty state)

1. Skip-link.
2. Site header: site name, back-to-site link.
3. Language picker options (in DOM order, which matches visual order).

Enter / Space on a language option selects it and sets `focus` onto the active-language change button inside the masthead. There is no visible focus loss on this transition.

**Pass.** The picker acts as a radio-like group but is built as plain buttons; roving focus is not implemented because one tab-stop per language is acceptable for a 10-item list. Revisit if the list ever grows beyond ~20.

## Order of focus — contribute (language selected, pre-session)

1. Skip-link.
2. Site header links.
3. Masthead: `change` button, `copy` button (hidden until a session exists, so skipped here).
4. Context strip (not focusable — it's status, not interactive).
5. Authoring form: gloss input, description textarea, Deaf-native checkbox, `Start authoring` submit.
6. Footer links.

**Pass.** The inline errors, when they appear, are `aria-describedby` targets — they do not take focus on their own; focus stays on the invalid input so the user can immediately correct it.

## Order of focus — contribute (mid-session)

Focus order follows DOM order, which we verified matches visual reading order top-to-bottom:

1. Masthead buttons (change, copy).
2. Chat panel: option buttons (if present), target-pill clear (if present), input, Send, Submit as-is (if error shown).
3. Avatar preview: stage (focusable via `tabindex="0"` on `#avatarCanvas`), body-region SVG paths (role="button"), Play, Loop, scrubber, speed buttons.
4. Notation panel: HamNoSys tab, SiGML tab, Copy HamNoSys / Copy SiGML / Download .sigml (context-dependent).
5. Submission panel: `Discard`, `Save draft and leave`, `Submit for review`.
6. Footer.

### Keyboard-activation spot-checks

- **Skip-link**: Enter moves focus into `#main`. ✓
- **Language picker option**: Enter + Space both select. ✓
- **Change-language button**: Enter opens the discard modal; focus moves to the modal's Cancel button. Modal is focus-trapped (TAB cycles between Cancel and Discard). Escape dismisses. ✓
- **Copy session URL** (`contextCopyBtn`): Enter copies, toast reads "Copied session URL" on screen and via aria-live. ✓
- **Chat option button**: Enter posts the option via `/answer`. ✓
- **Chat target-pill clear (×)**: Enter clears the pending correction target. ✓
- **Avatar region SVG paths**: Enter + Space both set the correction target. Focus remains on the SVG node; the pill inside the chat panel updates without stealing focus. ✓
- **Play button**: Space plays/pauses (matches the conventional media button). ✓
- **Scrubber**:
  - ArrowLeft / ArrowRight step by 100 ms (SCRUB_STEP_MS).
  - ArrowDown / ArrowUp: also step by 100 ms (same as Left/Right by spec).
  - PageUp / PageDown step by 1 s (SCRUB_LARGE_STEP_MS).
  - Home / End jump to 0 / duration. ✓
- **Speed radio group**: Arrow keys move within the group; TAB exits. ✓
- **Notation tabs**: Enter + Space activate; ArrowLeft/ArrowRight move within the tablist (`role="tablist"` wiring). Focus follows activation. ✓
- **Copy HamNoSys / Copy SiGML**: Enter copies; the inline "(copied)" span reveals for 1.6 s. ✓
- **Submission buttons**: Enter on `Submit for review` fires `/accept`. On success, focus moves to `#confirmationHeading` (verified via `document.activeElement`). ✓
- **Discard modal**: TAB, Shift+TAB stay inside the modal; Escape cancels. ✓
- **Token-prompt form** (on deep-link resume without a stored token): TAB reaches input → submit; Enter submits. ✓
- **Status-page token form**: same as above. ✓

## Escape-key behaviour

Escape is wired at three specific scopes:

1. Discard modal: closes and restores focus to the trigger button.
2. Chat correction-target pill: clears the pending target when the input has focus.
3. Notation tab focus: not wired (Escape on a tab is a no-op, matching WAI-ARIA authoring practice).

Escape does **not** close the hint strip — that's a deliberate choice. The hint is meant to be dismissed only by the close button, so it doesn't silently vanish if a user is already hitting Escape for the modal above it. Confirmed with keyboard walk.

## Focus visibility

Global `:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }` rule (contribute.css) applies to every interactive element. Verified that no focus ring is suppressed via `outline: none` on buttons, inputs, or custom widgets. The SVG body-region paths explicitly re-apply the visible outline on `:focus-visible` so keyboard users can see which region is selected before pressing Enter.

## Tab-order surprises (resolved before this walkthrough)

- *Prior state*: The confirmation view's `Another sign` / `Change language` / `Back to Bridgn` buttons used to appear before the copy-link button in DOM order, which felt wrong visually. Reordered during prompt 10 to match visual order. Current order: copy button, then nav buttons.
- *Prior state*: The target-pill clear (×) was not reachable by keyboard because the clear-button was rendered only via a CSS `::after`. Changed to a real `<button>` with `aria-label="Clear correction target"` during this pass. Verified it now gets a stop.
- *Prior state*: Speed buttons used `<div role="button">` without tabindex. Fixed to real `<button>` elements in a `<fieldset>` / `role="radiogroup"` container.

## Items that cannot be tested headlessly

- **OS-level keyboard shortcuts** (Cmd+Tab, Alt+Tab) — out of scope.
- **Browser accessibility tree overlays** (devtools "Accessibility" panel) — manually inspected in Chromium; every interactive element has a non-empty accessible name.
- **Physical keyboard behaviour of the CWASA 3D canvas** — CWASA is third-party and does not forward arrow keys. Our scrubber is the authoritative seek input; the canvas is marked `role="img"` so keyboard users do not land on it as if it were interactive.

## Result

No blocking keyboard-accessibility defects observed. Every interactive control is reachable by TAB in a sensible order, activatable with Enter (and Space where the widget type implies it), and returns focus to a reasonable location after modal / confirmation transitions.

If a later change reintroduces a non-focusable interactive widget, the axe-core CI (prompt 12 step 9) will flag it via `button-name`, `interactive-supports-focus`, or `link-name` — this file documents the baseline so deviations are easy to spot.

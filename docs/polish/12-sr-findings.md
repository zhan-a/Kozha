# Screen-reader findings — polish prompt 12

Captured against the polish-12 build on 2026-04-22. Updated as screen-reader sessions complete. Automated axe/pa11y coverage is in `12-a11y-baseline.md`; this file is the human-testing record.

Bridgn's primary audience is Deaf users. For users who read the sign output directly the avatar is the content and captions are secondary. For Hard-of-Hearing (HoH) users, users switching between languages, and sighted users relying on assistive tech for vocabulary work, captions are the content. The test matrix is designed around the second group.

---

## What's been automated away

The following concerns have machine coverage; they are not re-tested here.

- **ARIA attribute validity and role-allowed-attr** — axe-core across 15 scenarios, 0 violations.
- **Colour contrast on all text/bg pairs axe can compute** — axe-core, 0 violations. The `--accent` downshift from `#c96a2e` → `#b3441b` on `app.html` closed the primary-button + active-sidebar contrast failures. Caption text promoted to `--kozha-color-ink-2` (AAA on white).
- **Missing form labels / orphaned inputs** — pa11y on 11 scenarios, 0 errors.
- **Heading order** — axe-core, 0 violations. Footer column titles promoted from `h3` → `h2` site-wide so `h1 → h2` works on pages with no main-content `h2` (404).
- **Landmarks / regions** — axe-core, 0 region violations. The translator render panel is now a `<section aria-label="Sign preview and playback">`.

See `12-a11y-baseline.md` for the per-scenario numbers. **Every axe impact level is at zero across every scenario.** The only items below are things text-based automation cannot see.

---

## Environments

Target matrix, in the order we plan to cover it:

| SR | OS | Browser | Status |
| --- | --- | --- | --- |
| VoiceOver | macOS 14 | Safari 17 | pending |
| VoiceOver | iOS 17 | Safari | pending |
| NVDA | Windows 11 | Firefox ESR | pending |
| TalkBack | Android 14 | Chrome | pending |
| Narrator | Windows 11 | Edge | pending (fallback if NVDA unavailable) |

Rationale for this ordering:

1. **VoiceOver / Safari first** — the most widely used SR/browser pair among Deaf/HoH users in the UK, and what the core team develops on. Every regression will be caught here first.
2. **iOS VoiceOver** — mobile BSL usage is high. iOS gestures differ enough from macOS VO that we test both.
3. **NVDA / Firefox** — the reference open-source pair. NVDA is closest to what accessibility consultants exercise.
4. **TalkBack** — Android coverage for parity. The `CWASAAvatar` WebGL canvas behaves differently on TalkBack than on VO; worth confirming.
5. **Narrator** — only if NVDA is unavailable on a given test machine. Narrator is less representative of how blind users interact with the web in 2026.

---

## Test scenarios per screen reader

Each session walks the five anchor flows. Results go in the "Findings" subsection below once the session runs.

### 1. Landing → translator

Navigate from `/` to `/app.html` via the hero CTA. Expect SR to announce, in order:

- "Skip to main content, link" when focus lands on the skip link (Tab on page load).
- "Bridgn, link" logo.
- "Main navigation, navigation" landmark entering.
- "Translate, link; Contribute, link; Progress, link; About, link" when arrowing through the nav list (NVDA/JAWS mode) or tabbing (VO).
- "Open translator, link" — the primary CTA. Activating moves to `/app.html`.
- On landing in `/app.html`: the visually-hidden `<h1>Translator</h1>` announces as "Translator, heading level 1". This is the SPA-hygiene anchor for SR users even though the page is not an SPA.

### 2. Translator — text → sign

On `/app.html`:

- Focus text area via Tab. SR announces "Enter your phrase, edit text, character count below".
- Type "Good morning, everyone." SR echoes characters (per default SR setting).
- Press Enter or Tab to "Translate to sign language, button". Activate.
- **Critical announcement**: `#translatorAnnouncer` polite live region fires "Translating." — this is the new prompt-12 announcement.
- When CWASA plays: `renderStatus` announces "Playing…". `captionGloss` + `captionSource` update inside `role="status" aria-live="polite"` — SR hears "Current sign and source word: ELECTRON. electron." for each sign in sequence.
- When playback ends: `#translatorAnnouncer` fires "Translation complete. N signs ready." and `renderStatus` flips to "Ready".
- Failure path: if translation returns original text, announcer fires "Translation failed. Try again or switch language pair." and the `#translatorError` panel (`role="alert"`) reads its title + body.

### 3. Contribute — language → gloss → description → submit

On `/contribute.html`:

- Language picker: focus lands on `#pickerSelect` → SR announces "Which sign language are you contributing to?, combo box, 1 of 12".
- After selecting: `#langMasthead` `aria-live="polite" aria-label="Active language and session"` announces the current language + session id.
- Gloss input: SR reads "Gloss — the English word or short phrase for this sign., edit text, required" (aria-describedby pointing at any inline error).
- Description textarea: aria-describedby points at hint + error + char count.
- Chat panel (`role="log"` with `aria-live="polite"` and `aria-relevant="additions"`): each assistant message appends and is announced.
- HamNoSys display updates with `aria-label` describing the symbol count.
- Submission checklist items announce their state via accessible names.

### 4. Progress — per-language table

On `/progress.html`:

- Landing announces snapshot timestamp via `#progressGenerated` `aria-live="polite"`.
- Table headers render as `<button>` inside `<th>`. The aria-sort is now on the `<th>` (fixed this pass) — so arrowing over headers SR announces "Language, column header, sorted ascending, button".
- Activating a sort header toggles the direction and re-renders. `aria-sort` updates on the `<th>`; the live region does NOT re-announce (the sort change is user-triggered via Enter, so SR will speak the new focus context after the re-render).
- Chart: `<svg role="img" aria-labelledby="progressChartTitle progressChartDescription">` — SR reads the title + the plain-language trend description (e.g. "Grew from 5,400 on 2026-01-01 to 7,103 on 2026-04-01"). No animation; `prefers-reduced-motion` is irrelevant here.
- Help-wanted lists announce as lists of links with the target word as the accessible name.

### 5. 404 recovery

On `/404.html`: SR reads "404, Page not found, heading level 1, That page doesn't exist, Return to Bridgn, link". The footer column h2s follow the h1 properly (fixed this pass).

---

## Deaf-user perspective

Automated tools cannot assess two concerns that matter specifically for Deaf users:

- **Signing clarity vs screen-reader narration.** A Deaf user watching the avatar will rely on the captions' *text* content, not on aria-live announcements. The caption strip uses `--kozha-text-body` (13px) at AAA contrast on the avatar backdrop — verified this pass. Gloss uses the serif (ELECTRON); source uses sans (electron). For BSL signers who know the sign, the gloss is confirmation; for learners, the source word anchors the semantic meaning. Both lines are always visible.
- **Sign video alternatives for heavy instructions.** On the contribute/governance pages, long instructional text blocks (governance policy, review criteria) are candidates for a signed video alternative — not automated-testable, not yet authored. Tracked as a follow-up for the Deaf advisory board once seated.

---

## Findings

### VoiceOver on macOS 14 / Safari 17 — pending

Session notes to be filled in. Template:

- **Nav**: … (expected: logo → skip link → nav links → hamburger (collapsed on ≥900px so skipped) → page CTA; actual: …)
- **Translator**: … (expected: announcer fires exactly twice per translate — "Translating" then "Translation complete"; actual: …)
- **Contribute**: … (expected: chat log announces every system message immediately; actual: …)
- **Progress**: … (expected: column header sort state audible on focus; actual: …)

### VoiceOver on iOS 17 / Safari — pending

Session notes to be filled in. iOS gesture-driven SR has different affordances (rotor for headings/landmarks/forms). Expected: rotor-by-headings on `/app.html` reads "Translator" as the only h1, then card titles as h2s.

### NVDA on Windows 11 / Firefox ESR — pending

Session notes to be filled in. Browse mode vs Focus mode interaction:

- In browse mode, arrow keys read content linearly; the token-chip `role="group"` announces "Signed tokens, group" when entered, then button-by-button.
- In focus mode, Tab should take focus through the translator controls in visual order.

### TalkBack on Android 14 / Chrome — pending

Session notes to be filled in. Specific concerns:

- The WebGL avatar canvas: TalkBack is expected to announce the canvas as an image with the accessible label from the outer `<div role="img" aria-label="Sign language avatar animation…">`. Confirm the label updates as signs progress (currently static; a follow-up is to push live updates into the label as each sign starts).

---

## Queued follow-ups from SR sessions

When a human-testing session finds an issue, it lands here with a severity and an owner. "High" ships the fix this cycle; "moderate" goes to the next polish pass; "low" stays on the list.

_(No entries yet — human-testing sessions pending.)_

---

## Out-of-scope honesty

- **Lighthouse a11y score** — recorded in `12-lighthouse-audit.md`, not here.
- **Keyboard-only walkthrough** — axe's keyboard-related rules pass, but live interaction (Escape on modal, Enter on sort header, Space on play/pause) needs a human. Current status: the translator modal is defined with `aria-modal="true" role="dialog"`, the sort headers are real `<button>`s (Enter/Space works by default), Escape on modal is wired in `contribute.js`. Spot-checking in the dev server showed all three work; formal session notes go under the SR sessions above once they happen.
- **Cognitive-load audit for Deaf users new to the contribute flow** — out of scope for this pass; belongs to a discovery round with the Deaf advisory board.

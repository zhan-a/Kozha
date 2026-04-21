# A11y baseline — prompt 12

Captured by `scripts/a11y/run.mjs` on 2026-04-21 15:38:11 UTC.

Runs axe-core (via `@axe-core/puppeteer`, tags `wcag2a/aa wcag21aa wcag22aa best-practice`) and pa11y (HTML_CodeSniffer, `WCAG2AA`) against each scenario below. Raw JSON per scenario is in `12-a11y-raw/`.

## Summary
| id | scenario | axe crit | axe serious | axe mod | axe minor | pa11y error | pa11y warn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| landing | Landing page / (index.html) | 0 | 0 | 0 | 0 | 0 | 22 |
| contribute-empty | Contribute — empty state, language picker | 0 | 0 | 0 | 0 | 1 | 5 |
| contribute-after-language | Contribute — language selected, empty authoring area | 0 | 0 | 0 | 0 | 0 | 0 |
| contribute-mid-session | Contribute — mid-session (chat + preview + notation + submit visible) | 0 | 0 | 0 | 0 | 0 | 0 |
| governance | Governance page | 0 | 0 | 0 | 0 | 0 | 3 |
| status-draft | Submission status — draft | 0 | 0 | 0 | 0 | 0 | 1 |
| status-pending_review | Submission status — pending_review | 0 | 0 | 0 | 0 | 0 | 1 |
| status-under_review | Submission status — under_review | 0 | 0 | 0 | 0 | 0 | 1 |
| status-validated | Submission status — validated | 0 | 0 | 0 | 0 | 0 | 1 |
| status-rejected | Submission status — rejected | 0 | 0 | 0 | 0 | 0 | 1 |
| status-quarantined | Submission status — quarantined | 0 | 0 | 0 | 0 | 0 | 1 |

## Landing page / (index.html)

- URL: `http://127.0.0.1:50569/`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 22 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > nav`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > nav > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > nav > a > span`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > nav > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > nav > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > nav > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > nav > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_3.1_3_1.H48` — If this element contains a navigation section, it is recommended that it be marked up as a list.
  - `html > body > nav > div`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > nav > div > a:nth-child(1)`
- **warning** `WCAG2AA.Principle1.Guideline1_3.1_3_1.H48` — If this element contains a navigation section, it is recommended that it be marked up as a list.
  - `#main-content > section:nth-child(1) > div:nth-child(1) > div:nth-child(4)`
- …and 12 more (see raw)

## Contribute — empty state, language picker

- URL: `http://127.0.0.1:50569/contribute.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 1 error, 5 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_3.1_3_1.H48` — If this element contains a navigation section, it is recommended that it be marked up as a list.
  - `#contributePaused > p:nth-child(4)`
- **error** `WCAG2AA.Principle3.Guideline3_2.3_2_2.H32.2` — This form does not contain a submit button, which creates issues for those who cannot submit the form using the keyboard. Submit buttons are INPUT elements with type attribute "submit" or "image", or BUTTON elements with type "submit" or omitted/invalid.
  - `#byoKeyForm`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `#toast`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `#modalBackdrop`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `#hintStrip`

## Contribute — language selected, empty authoring area

- URL: `http://127.0.0.1:50569/contribute.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 0 warning

## Contribute — mid-session (chat + preview + notation + submit visible)

- URL: `http://127.0.0.1:50569/contribute.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 0 warning

## Governance page

- URL: `http://127.0.0.1:50569/governance.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 3 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `#governanceReviewers > p`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `#governanceBoard > p`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `#governanceEmailPlain`

## Submission status — draft

- URL: `http://127.0.0.1:50569/contribute/status/sim-draft`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 1 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Submission status — pending_review

- URL: `http://127.0.0.1:50569/contribute/status/sim-pending_review`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 1 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Submission status — under_review

- URL: `http://127.0.0.1:50569/contribute/status/sim-under_review`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 1 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Submission status — validated

- URL: `http://127.0.0.1:50569/contribute/status/sim-validated`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 1 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Submission status — rejected

- URL: `http://127.0.0.1:50569/contribute/status/sim-rejected`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 1 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Submission status — quarantined

- URL: `http://127.0.0.1:50569/contribute/status/sim-quarantined`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 1 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Moderate violations accepted as-is

None. All axe violations at every impact level are fixed in code.

## pa11y warnings accepted as-is

- **`WCAG2AA.Principle1.Guideline1_3.1_3_1.H48`** on the landing hero-actions div — pa11y heuristically suggests any link cluster inside a section be a list. The hero actions are two primary CTA buttons, not navigation. Semantically they belong in a flex row; a `<ul>` would be artificial. No user impact — both buttons are reached by Tab.
- **`WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206`** on `#toast`, `#modalBackdrop`, `#hintStrip` — these use `position: fixed` by function (overlay toast, confirm modal, dismissable keyboard hint). `position: fixed` is inherent to the pattern; the content inside wraps on narrow viewports (see `@media (max-width: 640px)` override). No two-dimensional scrolling in practice.
- **`WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,...`** on the SiGML `<pre>` element — preformatted XML for a reviewer is legitimately not reflowable. `overflow-x: auto` and `max-height: 360px` bound the element; `tabindex="0"` lets keyboard users scroll within it. A hands-on reviewer at 400% zoom on a phone still sees a usable scroll container.
- **`WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha`** on `#governanceReviewers > p`, `#governanceBoard > p`, `#governanceEmailPlain` — these elements have a ~3–4% ink-tint background (`rgba(21,19,15,0.03–0.04)`) for visual grouping. Foreground is solid `--ink` (#15130f); composite background on paper is #eeeae3 / #efebe4, giving >15:1 contrast. pa11y cannot compute composites and warns on any transparency; axe correctly passes these.

## Out of scope for this tool — requires human review

- Screen-reader behaviour under NVDA, VoiceOver, and TalkBack. Tracked in `12-sr-findings.md` with honest "pending human testing" entries.
- Keyboard navigation under live interaction (not simulated by axe). Tracked in `12-keyboard-findings.md`.
- Cognitive load audit (affordance count per screen, error-remedy pairing, concurrent-action load). Tracked in `12-cognitive-load.md`.
- Real-device testing across iPhone SE, iPad portrait, desktop 1440px, 4K. Tracked in `12-responsive-findings.md`.
- CWASA 3D avatar semantics — the canvas renders a signing body that cannot be inspected by text-based automation. axe treats it as an opaque block; captions and play/pause are the accessible surface. Deaf users cross-check the gloss + description captions against what the avatar signs.
- HamNoSys font rendering across Firefox, Safari, Chrome, and mobile Safari — the `@font-face` declaration points at `/fonts/bgHamNoSysUnicode.ttf`. If the binary is missing, the font stack falls through to system fonts. Browser-matrix verification is hands-on.

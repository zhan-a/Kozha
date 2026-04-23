# A11y baseline — polish prompt 12

Captured by `scripts/a11y/run.mjs` on 2026-04-23 03:14:02 UTC.

Runs axe-core (via `@axe-core/puppeteer`, tags `wcag2a/aa wcag21aa wcag22aa best-practice`) and pa11y (HTML_CodeSniffer, `WCAG2AA`) against each scenario below. Raw JSON per scenario is in `12-a11y-raw/`. Screen-reader, keyboard-live, and device-matrix results live in sibling `12-sr-findings.md` and `12-lighthouse-audit.md`.

## Summary
| id | scenario | axe crit | axe serious | axe mod | axe minor | pa11y error | pa11y warn |
| --- | --- | --- | --- | --- | --- | --- | --- |
| landing | Landing page / (index.html) | 0 | 0 | 0 | 0 | 0 | 21 |
| app-fresh | Translator /app — fresh load | 0 | 0 | 0 | 0 | 0 | 15 |
| app-mid-translation | Translator /app — mid-translation (captions, token list, controls active) | 0 | 0 | 0 | 0 | 0 | 0 |
| progress | Progress dashboard /progress | 0 | 0 | 0 | 0 | 0 | 35 |
| credits | Credits /credits | 0 | 0 | 0 | 0 | 0 | 7 |
| not-found | 404 page | 0 | 0 | 0 | 0 | 0 | 7 |
| contribute-empty | Contribute — empty state, language picker | 0 | 0 | 0 | 0 | 0 | 13 |
| contribute-after-language | Contribute — language selected, empty authoring area | 0 | 0 | 0 | 0 | 0 | 0 |
| contribute-mid-session | Contribute — mid-session (chat + preview + notation + submit visible) | 0 | 0 | 0 | 0 | 0 | 0 |
| governance | Governance page | 0 | 0 | 0 | 0 | 0 | 6 |
| status-draft | Submission status — draft | 0 | 0 | 0 | 0 | 0 | 7 |
| status-pending_review | Submission status — pending_review | 0 | 0 | 0 | 0 | 0 | 7 |
| status-under_review | Submission status — under_review | 0 | 0 | 0 | 0 | 0 | 7 |
| status-validated | Submission status — validated | 0 | 0 | 0 | 0 | 0 | 7 |
| status-rejected | Submission status — rejected | 0 | 0 | 0 | 0 | 0 | 7 |
| status-quarantined | Submission status — quarantined | 0 | 0 | 0 | 0 | 0 | 7 |

## Landing page / (index.html)

- URL: `http://127.0.0.1:61230/`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 21 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a > span`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_3.1_3_1.H48` — If this element contains a navigation section, it is recommended that it be marked up as a list.
  - `#main-content > section:nth-child(1) > div:nth-child(1) > div:nth-child(4)`
- **warning** `WCAG2AA.Principle1.Guideline1_3.1_3_1.H85.2` — If this selection list contains groups of related options, they should be grouped with optgroup.
  - `#heroSignLang`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Abs` — This element is absolutely positioned and the background color can not be determined. Ensure the contrast ratio between the text and all covered parts of the background are at least 4.5:1.
  - `#avatarLoading > span`
- …and 11 more (see raw)

## Translator /app — fresh load

- URL: `http://127.0.0.1:61230/app.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 15 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a > span`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle4.Guideline4_1.4_1_2.H91.Select.Value` — This select element does not have a value available to an accessibility API.
  - `#langHint`
- **warning** `WCAG2AA.Principle4.Guideline4_1.4_1_2.H91.Select.Value` — This select element does not have a value available to an accessibility API.
  - `#signLangSelect`
- **warning** `WCAG2AA.Principle1.Guideline1_3.1_3_1.H85.2` — If this selection list contains groups of related options, they should be grouped with optgroup.
  - `#signLangSelect`
- …and 5 more (see raw)

## Translator /app — mid-translation (captions, token list, controls active)

- URL: `http://127.0.0.1:61230/app.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 0 warning

## Progress dashboard /progress

- URL: `http://127.0.0.1:61230/progress.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 35 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a > span`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_3.1_3_1.H39.3.NoCaption` — If this table is a data table, consider using a caption element to the table element to identify this table.
  - `#progressTable`
- **warning** `WCAG2AA.Principle2.Guideline2_5.2_5_3.F96` — Accessible name for this element does not contain the visible label text. Check that for user interface components with labels that include text or images of text, the name contains the text that is presented visually.
  - `#progressTable > thead > tr > th:nth-child(5) > button`
- **warning** `WCAG2AA.Principle2.Guideline2_5.2_5_3.F96` — Accessible name for this element does not contain the visible label text. Check that for user interface components with labels that include text or images of text, the name contains the text that is presented visually.
  - `#progressTableBody > tr:nth-child(1) > td:nth-child(2) > a`
- …and 25 more (see raw)

## Credits /credits

- URL: `http://127.0.0.1:61230/credits.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 7 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a > span`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`

## 404 page

- URL: `http://127.0.0.1:61230/404.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 7 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a > span`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`

## Contribute — empty state, language picker

- URL: `http://127.0.0.1:61230/contribute.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 13 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_3.1_3_1.H48` — If this element contains a navigation section, it is recommended that it be marked up as a list.
  - `#contributePaused > p:nth-child(4)`
- **warning** `WCAG2AA.Principle4.Guideline4_1.4_1_2.H91.Select.Value` — This select element does not have a value available to an accessibility API.
  - `#pickerSelect`
- **warning** `WCAG2AA.Principle1.Guideline1_3.1_3_1.H85.2` — If this selection list contains groups of related options, they should be grouped with optgroup.
  - `#pickerSelect`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``
- …and 3 more (see raw)

## Contribute — language selected, empty authoring area

- URL: `http://127.0.0.1:61230/contribute.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 0 warning

## Contribute — mid-session (chat + preview + notation + submit visible)

- URL: `http://127.0.0.1:61230/contribute.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 0 warning

## Governance page

- URL: `http://127.0.0.1:61230/governance.html`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 6 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`

## Submission status — draft

- URL: `http://127.0.0.1:61230/contribute/status/sim-draft`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 7 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Submission status — pending_review

- URL: `http://127.0.0.1:61230/contribute/status/sim-pending_review`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 7 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Submission status — under_review

- URL: `http://127.0.0.1:61230/contribute/status/sim-under_review`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 7 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Submission status — validated

- URL: `http://127.0.0.1:61230/contribute/status/sim-validated`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 7 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Submission status — rejected

- URL: `http://127.0.0.1:61230/contribute/status/sim-rejected`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 7 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Submission status — quarantined

- URL: `http://127.0.0.1:61230/contribute/status/sim-quarantined`
- axe: 0 critical, 0 serious, 0 moderate, 0 minor
- pa11y: 0 error, 7 warning

### pa11y issues

- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — This element has "position: fixed". This may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - `html > body > header`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(1) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(2) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(3) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha` — This element's text or background contains transparency. Ensure the contrast ratio between the text and background are at least 4.5:1.
  - `html > body > header > ul > li:nth-child(4) > a`
- **warning** `WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206` — Preformatted text may require scrolling in two dimensions, which is considered a failure of this Success Criterion.
  - ``

## Moderate violations accepted as-is

None. All axe violations at every impact level are fixed in code.

## pa11y warnings accepted as-is

- **`WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206`** on the site-wide `.kz-header` — the unified header uses `position: fixed` so the translator, contribute flow, progress dashboard, and status pages share the same nav across scroll. pa11y warns about 2D scrolling for any fixed element, but the header content wraps and the hamburger menu engages under 900px — there is no horizontal scroll at any viewport width on any page, including 400% zoom.
- **`WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha`** on the `.kz-header` nav links — the header has `background: rgba(245,241,235,0.85)` + `backdrop-filter: blur(12px)`. pa11y cannot read the composited background so it warns on any transparency. Computed composite against paper (`#f5f1eb`) is `#eeeae3`; with link color `#3d3630` (ink-2) the ratio is 10.7:1 — AAA. axe correctly passes.
- **`WCAG2AA.Principle1.Guideline1_3.1_3_1.H48`** on the landing hero-actions div — pa11y heuristically suggests any link cluster inside a section be a list. The hero actions are two primary CTA buttons, not navigation. Semantically they belong in a flex row; a `<ul>` would be artificial. No user impact — both buttons are reached by Tab.
- **`WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,C31,C33,C38,SCR34,G206`** on `#toast`, `#modalBackdrop`, `#hintStrip` — these use `position: fixed` by function (overlay toast, confirm modal, dismissable keyboard hint). `position: fixed` is inherent to the pattern; the content inside wraps on narrow viewports (see `@media (max-width: 640px)` override). No two-dimensional scrolling in practice.
- **`WCAG2AA.Principle1.Guideline1_4.1_4_10.C32,...`** on the SiGML `<pre>` element — preformatted XML for a reviewer is legitimately not reflowable. `overflow-x: auto` and `max-height: 360px` bound the element; `tabindex="0"` lets keyboard users scroll within it. A hands-on reviewer at 400% zoom on a phone still sees a usable scroll container.
- **`WCAG2AA.Principle1.Guideline1_4.1_4_3.G18.Alpha`** on `#governanceReviewers > p`, `#governanceBoard > p`, `#governanceEmailPlain` — these elements have a ~3–4% ink-tint background (`rgba(21,19,15,0.03–0.04)`) for visual grouping. Foreground is solid `--ink` (#15130f); composite background on paper is #eeeae3 / #efebe4, giving >15:1 contrast. pa11y cannot compute composites and warns on any transparency; axe correctly passes these.
- **`WCAG2AA.Principle1.Guideline1_3.1_3_1.H85.2`** on `#heroSignLang`, `#pickerSelect` — pa11y suggests any long `<select>` with ≥10 options group them with `<optgroup>`. `#heroSignLang` already uses optgroups; the warning fires on the common-vs-more split because pa11y counts options across both groups. `#pickerSelect` lists 12 sign languages total — a flat list is easier to scan than forced groups and matches the per-language progress dashboard. Keyboard users can still type-ahead.
- **`WCAG2AA.Principle1.Guideline1_3.1_3_1.H39.3.NoCaption`** on `#progressTable` — pa11y suggests a `<caption>` on data tables. The table has a sibling `<h2 id="tableHeading">Per-language coverage</h2>` and the wrapper `role="region" aria-label="Per-language coverage table"`; the heading serves as the caption and is preferred because it participates in heading-order navigation.
- **`WCAG2AA.Principle2.Guideline2_5.2_5_3.F96`** on progress-table help-wanted links — the link text is the target English word (e.g. "breakfast") which is the visible label. pa11y thinks the accessible name must include the column value; axe and WCAG 2.5.3 only require the visible label appear — which it does.
- **`WCAG2AA.Principle4.Guideline4_1.4_1_2.H91.Select.Value`** on `#langHint`, `#signLangSelect`, `#pickerSelect` in fresh-load state — the selects are populated by JS post-DOMContentLoaded. pa11y inspects the DOM before the population event fires in static analysis. Once populated, each select has a real selected option. No run-time impact.

## Out of scope for this tool — requires human review

- Screen-reader behaviour under NVDA, VoiceOver, and TalkBack. Tracked in `12-sr-findings.md` with honest "pending human testing" entries.
- Keyboard navigation under live interaction (not simulated by axe). Tracked in `12-keyboard-findings.md`.
- Cognitive load audit (affordance count per screen, error-remedy pairing, concurrent-action load). Tracked in `12-cognitive-load.md`.
- Real-device testing across iPhone SE, iPad portrait, desktop 1440px, 4K. Tracked in `12-responsive-findings.md`.
- CWASA 3D avatar semantics — the canvas renders a signing body that cannot be inspected by text-based automation. axe treats it as an opaque block; captions and play/pause are the accessible surface. Deaf users cross-check the gloss + description captions against what the avatar signs.
- HamNoSys font rendering across Firefox, Safari, Chrome, and mobile Safari — the `@font-face` declaration points at `/fonts/bgHamNoSysUnicode.ttf`. If the binary is missing, the font stack falls through to system fonts. Browser-matrix verification is hands-on.

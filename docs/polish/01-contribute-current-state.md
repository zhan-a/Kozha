# Contribute page — current state and drift from main pages

The contribute page (`public/contribute.html` + `public/contribute.css` + seven `contribute-*.js` modules) already exists as a through-designed flow from the 14-prompt `docs/contribute-redesign/` series. It reads like a coherent editorial artifact — a near-opposite aesthetic from the landing's "warm product card" vibe. Where it succeeds and where it drifts from the main-page reference is documented below.

## Token drift

See `01-design-tokens.md` for the full table. The contribute page's `:root` (contribute.css:6-17) declares:

```css
--ink: #1a1612;
--paper: #f5f1eb;
--accent: #b3441b;          /* matches landing */
--accent-bright: #c96a2e;   /* matches landing */
--rule: rgba(26, 22, 18, 0.14);
--muted: #5e5c57;
--max-width: 880px;
```

| drift | main page | contribute |
|---|---|---|
| `--accent-light` | `#faeee0` | not declared — every accent-light surface is drawn with `rgba(179, 68, 27, 0.xx)` opacity instead |
| `--border` opacity | `0.12` | `--rule` at `0.14` |
| body `--ink-3` / muted | `#5e5c57` | `--muted: #5e5c57` (same value, different token name) |
| page max-width | `1100px` landing, `1280px` app | `880px` |
| border-radius | `10px` / `16px` / `999px` / `24px` | `0` everywhere (explicit `border-radius: 0`) |
| shadow | cards lift with warm shadows | zero shadows on the page (brutalist flat) |
| card background | white | transparent on paper, or `rgba(21,19,15,0.02)` tint |
| link treatment | nav: no underline; body: underline on `.footer-logo span` only | every `<a>` underlined with `text-underline-offset: 3px` |
| button radius | `10px` / `16px` pills | `0` (flat rectangles) |
| focus ring | border-color shift (landing) / 3px box-shadow (app) | `outline: 2px solid var(--accent); outline-offset: 3px` |
| skip-link bg | `var(--accent)` | `var(--ink)` |
| serif fallback stack | `'Instrument Serif', serif` | `'Instrument Serif', Georgia, serif` — extra `Georgia` fallback |

**Why the brutalist divergence exists.** `docs/contribute-redesign/` picked a distinctive authorial voice for the flow to signal "this is a workspace, not a marketing page." The radius-zero, underlined-everything, no-shadow aesthetic is intentional. The question for later prompts is whether to keep the aesthetic but pull in the main-page tokens for the tokens that *aren't* aesthetic choices — specifically the accent palette and border opacity. The zero-radius and flat-link treatment are defensible; the token name drift and max-width difference are not.

## Component drift

### Site header vs. main-page nav

`public/contribute.html:26-29`:

```html
<header class="site-header">
  <a href="/" class="site-name">Bridgn</a>
  <a href="/app.html" class="back-link">Open translator</a>
</header>
```

No hamburger. No skip-link with accent. No `.nav-cta` group with ghost + primary CTAs. The landing uses a `<nav>` landmark with `aria-label="Main navigation"`; contribute uses `<header>` semantically (correct — it's a banner) but that landmark is duplicated by the page's `<body>` level banner role, which axe may flag.

The site-name logo's accent span is missing: landing renders `Bridg<span>n</span>` coloring the `n` in accent; contribute renders flat `Bridgn` — no accent. Copy drift only; visual drift is meaningful.

NORMALIZE: accept the minimalist header (no CTAs), but add the accent `<span>n</span>` treatment so the wordmark is consistent with landing and app footer.

### HTML structure bug

`public/contribute.html:99` opens `<section class="lang-masthead">`. `public/contribute.html:116` closes it with `</header>`. This is an unbalanced element: a `<section>` should not close with `</header>`. Browsers paper over it with implicit closing, but an HTML validator will flag the mismatch, and a linter (`htmlparser2`, `htmlhint`) will reject it. Risk: none today (the page renders); long-term: any linter we add will start failing. Low-effort fix in a later prompt — swap `</header>` for `</section>`.

### Buttons

Contribute has four button patterns, all flat (zero radius):

1. **Primary** — solid accent, paper text: `.authoring-submit`, `.submission-submit`, `.chat-send`, `.token-prompt button`. Hover: `opacity: 0.9`. Disabled: `opacity: 0.4; cursor: not-allowed`.
2. **Outline** — transparent, `1px solid var(--ink)`: `.chat-submit-as-is`. Hover: `opacity: 0.75`.
3. **Ghost link** — no border, underlined text, muted color: `.change-btn`, `.chat-discard-btn`, `.authoring-summary-edit`, `.context-copy-btn`, `.token-prompt-discard`. Hover: color shifts to ink.
4. **Pill** — rounded `.chat-option-btn` with 1px rule border, no fill. Hover: border shifts to ink, slight opacity reduce.

Main pages have **two** button patterns (primary solid / ghost bordered) plus the danger variant. There is no 4-pattern button system on the main pages. NORMALIZE: the contribute's pattern #3 ("ghost link") is unique to the contribute flow and can stay; patterns #1, #2, #4 should reconcile to landing's `.btn-primary` / `.btn-ghost` + a small chip variant.

### Focus state

Contribute has the most complete focus system of the three surfaces:

```css
:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
.field-input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-color: var(--ink); }
```

Landing has *no* `:focus-visible` rules. App has some but only on inputs. NORMALIZE: promote the contribute focus treatment to a shared default.

### Avatar stage

Contribute's stage is 16:9, backdrop-layered, with body-region overlay for click-to-correct. Main-page avatar (landing `.demo-avatar-area` / app `.avatar-stage`) is fixed-aspect 364×240 / 384×320 with a simple spinner fallback. The main-page version is "marketing-appropriate preview"; the contribute version is "review workspace." Different purposes, both correct — no drift to normalize.

## Copy drift

The contribute page's copy (see `public/strings.en.json:contribute.*`) follows the landing voice. Specific matches:

- Short declarative sentences with em-dashes for appositional clauses.
- Imperative second-person for labels (`Start authoring`, `Submit for review`, `Discard this draft`).
- No exclamation points. No persona voice. No welcome copy.

Points of drift to verify in prompt 4:

| landing copy | contribute analogue | status |
|---|---|---|
| `Bridgn translates text and speech into sign language animations…` | *(no such lede on contribute)* | missing — contribute dives straight into the picker |
| `Deaf-governed Contributions` / `reviewed by two Deaf native signers before it ships` | covered implicitly in `Governance` nav link | maybe worth a one-liner above the picker |
| `Open-source research project` eyebrow | *(no such eyebrow on contribute)* | missing — if added, matches site voice |
| `© 2025–2026 Bridgn. Open-source research project.` | contribute footer has only three links + no copyright | drift |
| `Volunteer` / `Get involved` CTAs | contribute page itself is the CTA target | not drift |

### Stale copy

`public/strings.en.json:22` still references databases by their SignAvatars/algerianSignLanguage-avatar / bdsl-3d-animation / text_to_isl / KurdishSignLanguage / VSL / syntheticfsl / signtyper authors. This is correct attribution but not displayed anywhere that I can find on the contribute flow. It's wired into `landing.*` and doesn't surface in the `contribute.*` keys. This is surfaced as its own prompt-level concern; noted here for tracing.

## Scripts loaded on contribute.html

`public/contribute.html:582-588`:

```
/contribute-byokey.js
/contribute-context.js
/contribute-chat.js
/contribute-notation.js
/contribute-preview.js
/contribute-submit.js
/contribute.js
```

Total: 8 files (including `/i18n.js` from the head). Total minified size unknown but not trivial. A future prompt might consider bundling; out of scope here.

Main-page `index.html` is a single file with inlined script, ~1580 lines. App.html is a single file, ~1760 lines. The two architectures — bundle-vs-split — are different answers to the same problem, and neither needs to migrate to the other to polish the page.

## Summary of drift to act on (non-aesthetic)

1. **HTML validity** — fix the `</header>` closing a `<section>` in contribute.html:116.
2. **Token naming** — rename `--rule` to `--border` and set opacity to `0.12` to match the main-page convention.
3. **Accent-light** — add `--accent-light: #faeee0` to contribute.css and swap the ad-hoc `rgba(179, 68, 27, 0.xx)` fills where appropriate.
4. **Logo wordmark** — add accent `<span>n</span>` treatment to the contribute site-name.
5. **Stale description on app.html** — update `<meta name="description">` to reflect the full 15-language catalog; not contribute-specific but main-page.
6. **Unify skip-link** — accent background everywhere, not ink on contribute.

Aesthetic differences (zero radius, underlined links, no shadows, flat buttons) are intentional and stay.

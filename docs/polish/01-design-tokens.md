# Design tokens — main page (index.html + app.html)

Scope. This doc is the factual extraction of tokens *as declared and used* on the two main-page surfaces. It is the design reference every later prompt reconciles to. Values were read directly from `public/index.html` (inline `<style>`) and `public/app.html` (inline `<style>`). Both pages ship CSS inline; there is no shared stylesheet on the main-page surface.

Where the same concept has multiple values, the row is flagged NORMALIZE.

## Color

### Ink (body text + near-black)

| token | index.html | app.html | where used |
|---|---|---|---|
| `--ink` | `#1a1612` | `#1a1612` | body text, headings, avatar fallback text |
| `--ink-2` | `#3d3630` | `#3d3630` | nav links, secondary headings, btn-ghost text |
| `--ink-3` | `#5e5c57` | **`#685d54`** — NORMALIZE | hero-sub copy, muted captions, footer copy, field placeholders |

`--ink-3` is the one ink drift between the two main pages. `#5e5c57` is the palette the contribute page also uses (as `--muted`), so the canonical value is `#5e5c57`. `app.html`'s `#685d54` should move to `#5e5c57`.

### Paper (backgrounds)

| token | index.html | app.html | where used |
|---|---|---|---|
| `--paper` | `#f5f1eb` | `#f5f1eb` | body, nav backdrop, demo-input bg |
| `--paper-2` | `#ede8e0` | `#ede8e0` | btn-ghost:hover, input-tabs bg, sidebar-icon bg, token-chip bg |
| `--paper-3` | `#e4ddd3` | `#e4ddd3` | declared but unused on both pages |

Also appears as literal in CSS, not via token: `#e9e6df` — avatar-backdrop & avatar-fallback bg on contribute.css. Not used on main pages.

### Accent (orange)

| token | index.html | app.html | where used |
|---|---|---|---|
| `--accent` | `#b3441b` *(text-safe)* | **`#c96a2e`** — NORMALIZE | primary button bg, links on paper, eyebrow text, focus ring |
| `--accent-2` | `#c96a2e` *(bright)* | `#e8843e` — NORMALIZE | button hover bg, translate-btn hover |
| `--accent-bright` | `#c96a2e` | — (not declared) | step-num glyph on "How it works" dark section |
| `--accent-light` | `#faeee0` | **`#f5e4d6`** — NORMALIZE | hero-eyebrow bg, contribute-banner bg, chip:hover bg |

This is the most important normalization: `--accent` must be `#b3441b` on surfaces that render small text on paper (WCAG AA 4.5:1). `app.html` declares `--accent: #c96a2e` and applies it to 12px+ text and focus rings — which fails AA at small body sizes. The memory `feedback_accent_contrast_on_paper` records this trade-off; the resolution is to use `#b3441b` for text/links and reserve `#c96a2e` as a hover-lift shade.

`--accent-light` has two different values. `#faeee0` (index) is a whiter, paler wash; `#f5e4d6` (app) is a warmer cream. Unify on `#faeee0` — it has higher contrast with the accent text for the eyebrow pill.

### Semantic

| token | value | where |
|---|---|---|
| `--green` | `#2d6a4f` | status-dot.ready on index; status-badge.ready on app |
| `--green-light` | `#d4edda` | app.html only; status-badge.ready bg |
| error red | `#c0392b` | app.html `.btn-danger`, `.char-count.over`, `.demo-record-btn.recording` (literal, no token) |
| error red (hover) | `#e74c3c` | app.html `.demo-record-btn.recording:hover` (literal) |
| error red (bg tint) | `#fdf0ee` | app.html `.btn-danger:hover` bg (literal) |

NORMALIZE: promote error red to a `--danger` token (`#c0392b`) with `--danger-hover: #e74c3c` and `--danger-light: #fdf0ee`. No token exists on either main page today.

### Border / hairline

| token | value | where |
|---|---|---|
| `--border` | `rgba(26,22,18,0.12)` | both main pages, everywhere hairlines exist |
| focus glow | `rgba(201,106,46,0.1)` | app.html input `:focus` box-shadow ring |
| hover accent stroke | `rgba(201,106,46,0.2)` | feature-card border on hover, contribute-banner border |
| accent bright alpha | `rgba(201,106,46,0.3)` | btn-primary hover shadow |

The contribute page uses `--rule: rgba(26,22,18,0.14)` instead. NORMALIZE: one token, one opacity — `0.12` wins because both main pages use it.

## Typography

### Font stacks

| role | index.html | app.html |
|---|---|---|
| body | `'DM Sans', system-ui, -apple-system, sans-serif` | `'DM Sans', sans-serif` — NORMALIZE |
| display | `'Instrument Serif', serif` | `'Instrument Serif', serif` |
| mono | `ui-monospace, 'Menlo', monospace` (log-area on app.html) | same |

Both pages declare Google Fonts preconnect + the same family request:
`Instrument+Serif:ital@0;1 & DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300`.

NORMALIZE: extend app.html's body stack to include the system fallbacks — `'DM Sans', system-ui, -apple-system, sans-serif`.

### Type scale (sizes actually applied)

Sorted from small → large; all values are from concrete rules (not just declared in a var).

| px | index.html use | app.html use |
|---|---|---|
| 10 | `.demo-lang-label` letter-small caps | — |
| 11 | — | `.sidebar-section-label`, `.status-badge::before` sizing, `.field-label`, `.nav-badge` |
| 12 | `.hero-eyebrow`, `.footer-copy`, `.section-label`, `.stat-label`, `.demo-chip`, `.demo-translate-btn`, `.demo-status`, `.demo-lang-select`, `.demo-status-bar`, status-dot text, `.footer-copy` | `.status-badge`, `.chip`, `.token-chip`, `.log-area`, `.char-count`, `.hint`, `.sidebar-section-label` |
| 13 | `.footer-links a`, `.avatar-loading span`, `.demo-file-input` | `.nav-link-sm`, `.input-tab`, `.check-label`, `.avatar-switcher label` + `select`, `.app-footer` |
| 14 | `.nav-links a`, `.btn-ghost`, `.btn-primary`, `.feature-body`, `.step-body`, `.skip-link` | `textarea`, `input[type=text]`, `select`, `.btn`, `.btn-primary`, `.btn-secondary`, `.btn-danger`, `.skip-link` |
| 15 | `.demo-input` | `.btn-translate` |
| 16 | `.btn-large`, `.btn-large-ghost`, `.contribute-text p`, mobile hero-sub, demo-record/process btn mobile | — |
| 17 | — | `.card-title` |
| 18 | `.hero-sub` | — |
| 20 | — | `.feature-title` (on landing only), `.picker-prompt` (contribute) |
| 22 | `.nav-logo` (landing), `.step-title` | — |
| 32 | `.stat-num` | — |
| 36 | `.contribute-text h2` | — |
| 56 | `.step-num` | — |
| clamp(32,4vw,48) | `.section-title` | — |
| clamp(44,6vw,72) | `.hero h1` | — |
| clamp(36,8vw,48) | mobile `.hero h1` | — |

NORMALIZE: the type scale has 13 distinct sizes between 10px and 18px. A refined scale (11, 12, 13, 14, 15, 16, 18) with purposeful use would let every later prompt reach for a nameable step.

### Weight

Declared: 300, 400, 500, 600. No `700` in use anywhere on the main pages. Common pattern: 500 for nav links and fine labels, 600 for buttons/CTAs and uppercase eyebrows, 300 for hero-sub, default 400 for body. Instrument Serif is always at its default weight.

### Letter-spacing and line-height

| value | where |
|---|---|
| `letter-spacing: -0.02em` | `.hero h1`, `.section-title`, `.contribute-text h2` |
| `letter-spacing: -0.01em` | `.nav-logo` |
| `letter-spacing: 0.04em` | `.stat-label` |
| `letter-spacing: 0.06em` | `.nav-badge` (app) |
| `letter-spacing: 0.08em` | `.demo-lang-label`, `.field-label`, `.hero-eyebrow` (partial) |
| `letter-spacing: 0.1em` | `.section-label`, `.sidebar-section-label` |
| `line-height: 1` | `.stat-num`, `.step-num`, `.hamburger` |
| `line-height: 1.05` | `.hero h1` |
| `line-height: 1.15` | `.section-title` |
| `line-height: 1.5` | `.hint`, default input text |
| `line-height: 1.6` | `.contribute-text p` |
| `line-height: 1.7` | `.hero-sub`, `.feature-body`, `.step-body` |

## Spacing scale (unique non-zero values actually applied)

Every distinct margin/padding value that appears at least once, regardless of axis, in px unless noted. `rem` not used on main pages.

`2, 3, 4, 6, 7, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 48, 56, 64, 72, 96, 100, 140`

NORMALIZE: this is 25 distinct values — far too granular. A canonical scale like `4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96` would cover all meaningful steps. Values 2/3/6/7/10/14/18/22 should be eliminated in the reconciliation pass.

## Border radius

| token/value | where |
|---|---|
| `--radius` = `16px` | `.btn-large*`, `.feature-card`, `.avatar-wrapper`, `.card`, `.avatar-stage`, `.btn-translate` |
| `--radius-sm` = `10px` | `.btn-ghost`, `.btn-primary`, `.demo-input`, `.demo-lang-select` (8px on app), `.demo-record-btn`, `.demo-process-btn`, `.skip-link`, `.btn`, `.input-tabs`, `.hamburger`, `.log-area`, `.avatar-switcher select` |
| `8px` | `.demo-lang-select` (index), `.sidebar-icon`, `.input-tab`, `.row` inputs (app) |
| `24px` | `.hero-demo-card`, `.contribute-banner` |
| `999px` | pills: `.hero-eyebrow`, `.demo-chip`, `.demo-translate-btn`, `.status-badge`, `.chip`, `.token-chip`, `.nav-badge` |
| `2px` | `.playback-progress`, `.playback-progress-bar` |
| `50%` | pulse dots, spinner |

NORMALIZE: `24px` radius appears twice and is inconsistent with `--radius: 16px`. Either drop 24px or add a `--radius-lg: 24px` token. `8px` drift vs `--radius-sm: 10px` should collapse to one value.

## Shadow

| value | where |
|---|---|
| `0 24px 64px rgba(26,22,18,0.08), 0 4px 12px rgba(26,22,18,0.04)` | `.hero-demo-card` resting |
| `0 12px 32px rgba(26,22,18,0.08)` | `.feature-card:hover` |
| `0 8px 24px rgba(201,106,46,0.3)` | `.btn-large:hover` |
| `0 6px 20px rgba(201,106,46,0.3)` | `.btn-translate:hover` (app) |
| `0 4px 12px rgba(201,106,46,0.3)` | `.btn-primary:hover` (app) |
| `0 1px 4px rgba(26,22,18,0.08)` | `.input-tab.active` |

NORMALIZE: create `--shadow-card-rest`, `--shadow-card-hover`, and `--shadow-btn-hover`. Six ad-hoc values become three semantic tokens.

## Transitions and easings

| value | where |
|---|---|
| `all 0.15s` | `.sidebar-item`, `.demo-chip`, `.chip`, `.input-tab` |
| `all 0.2s` | nearly every button and control hover |
| `all 0.25s` | `.feature-card`, `.btn-large`, `.btn-large-ghost` |
| `color 0.2s` | text-only transitions on nav links, footer links |
| `border-color 0.2s`, `box-shadow 0.2s` | `.demo-input:focus`, input:focus |
| `width 0.3s` | `.playback-progress-bar` |
| `max-height 0.3s ease` | mobile sidebar collapse |

Keyframes on the main pages: `pulse` (2s ease-in-out infinite), `spin` (0.8s linear infinite), `fadeUp` (0.6s ease forwards, staggered `.delay-1…5` at 0.1s through 0.5s).

NORMALIZE: two timings survive — `120ms` / `200ms` / `250ms`. Collapse `0.15` and `0.2` to `150ms` (or pick one). Keep `0.25s` for big-move transitions (card hover, button large).

## Breakpoints

| breakpoint | index.html | app.html |
|---|---|---|
| mobile | `@media (max-width: 400px)` | `@media (max-width: 600px)` |
| tablet | `@media (max-width: 900px)` | `@media (max-width: 1099px)` |
| desktop | — (default) | `@media (min-width: 1100px)` |
| reduced motion | `@media (prefers-reduced-motion: reduce)` | — (not declared on app.html) |

NORMALIZE: two different tablet breakpoints (900 vs 1099). Pick one — 1024 or 1100. app.html's lack of `prefers-reduced-motion` is a bug; the spinner and status-dot `loading` animation should stop under reduced motion.

## Layout constants

| var | value | notes |
|---|---|---|
| page max-width (landing) | `1100px` | `.hero`, `.section`, `.how-inner`, `.footer-inner` |
| page max-width (app) | `1280px` | `.app-shell`, `.app-footer` |
| sidebar width (app) | `260px` | `--sidebar` |
| avatar hero size | `364×240` | `#heroAvatar` |
| avatar app size | `384×320` | `.CWASAAvatar.av0` on app |
| nav height (landing) | `64px` | |
| nav height (app) | `56px` | |

NORMALIZE: nav heights differ. Either unify on `64px` or explicitly document why the app chrome is shorter (to give the avatar more space).

## Z-index scale

| z | where |
|---|---|
| `100` | nav (landing), app-nav, .modal-backdrop (contribute) |
| `200` | .skip-link both pages |
| `50` | hint-strip (contribute) |
| `60` | toast (contribute) |
| `10` | lang-masthead (contribute) |
| `3` | avatar-regions (contribute) |
| `2` | avatar-loading (landing) |
| `1` | avatar-pulse (contribute) |

NORMALIZE: make a five-step scale (`1, 10, 50, 100, 200`) with semantic names (`--z-backdrop`, `--z-nav`, `--z-modal`, `--z-toast`, `--z-skip`).

## Summary: the seven things to normalize

1. `--accent` must be `#b3441b` everywhere it appears in text/links on paper. app.html currently uses `#c96a2e` (fails AA).
2. `--ink-3` must be `#5e5c57` on app.html (currently `#685d54`).
3. `--accent-light` must be `#faeee0` (contribute.css and app.html currently drift to `#f5e4d6`).
4. `--border` opacity is `0.12`; the contribute page's `--rule` at `0.14` should normalize.
5. Body font stack must be `'DM Sans', system-ui, -apple-system, sans-serif` on every surface (app.html drops the fallbacks).
6. Type scale should collapse from 13 distinct sizes between 10–18px to a seven-step scale.
7. Radius `24px` and `8px` drifts should collapse to `--radius: 16px` and `--radius-sm: 10px`.

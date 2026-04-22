# Component inventory — main pages

Every distinct UI component pattern on `public/index.html` and `public/app.html`, with its markup skeleton, applied styles, states, and ARIA role. This is the set of patterns contribute-page components reconcile to.

## Nav bar (landing)

Markup skeleton (`public/index.html:760-775`):

```html
<nav aria-label="Main navigation">
  <a href="/" class="nav-logo">Bridg<span>n</span></a>
  <ul class="nav-links">
    <li><a href="#features">Features</a></li>
    …
  </ul>
  <div class="nav-cta">
    <a href="/contribute.html" class="btn-ghost">Volunteer</a>
    <a href="/app.html" class="btn-primary">Open translator →</a>
    <button class="nav-hamburger" …></button>
  </div>
</nav>
```

Styles: fixed-top, 64px tall, translucent paper backdrop `rgba(245,241,235,0.85)` + `backdrop-filter: blur(12px)`, bottom hairline, 48px horizontal padding.

States:
- `.nav-links a:hover` → color `var(--accent)`.
- `.nav-hamburger` hidden ≥ 901px; shown below with `nav-hamburger svg { display:block }`.
- `.nav-links.open` on mobile becomes a vertical stack dropped below the nav, full paper-opacity, bordered bottom.

ARIA: `aria-label="Main navigation"` on nav. Hamburger has `aria-label="Menu"` and `aria-expanded` toggled by JS. Logo's nested `<span>` colors the `n` in accent — purely decorative, no SR impact.

## Nav bar (app translator)

Markup (`public/app.html:624-634`):

```html
<nav class="app-nav" aria-label="Main navigation">
  <div>… logo + <span class="nav-badge">Translator</span></div>
  <div class="nav-right">
    <a class="nav-link-sm">Contribute a sign</a>
    <a class="nav-link-sm">← Home</a>
    <button class="hamburger">☰</button>
  </div>
</nav>
```

Differences from landing:
- 56px tall (landing is 64px).
- `sticky` rather than `fixed` — it scrolls off on shorter screens.
- `.nav-badge` chip in accent-light shows page context.
- No CTAs — just quieter text links.
- Hamburger uses literal unicode char `☰` instead of SVG; landing uses inline SVG three-bar icon.

## Primary CTA (large button)

Canonical definition (landing, `.btn-large`):

```css
font-family: 'DM Sans', sans-serif;
font-size: 16px;
font-weight: 600;
color: white;
background: var(--accent);
border-radius: var(--radius);   /* 16px */
padding: 14px 28px;
transition: all 0.25s;
```

Hover: `background: var(--accent-2); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(201,106,46,0.3);`.

Variants:
- `.btn-large-ghost` — 1px border, white bg, translateY(-1px) on hover.
- `.btn-primary` (nav) — smaller, 8px 20px padding, 14px text.
- `.btn-translate` (app) — full-width, 13px padding, 15px text, 6px/20px shadow.

ARIA: these are plain `<a>` or `<button>` tags. `.demo-translate-btn` has explicit `aria-label="Translate to sign language"` on index.html:810.

## Secondary / ghost button

Landing `.btn-ghost`:

```css
font-size: 14px; font-weight: 500;
color: var(--ink-2);
background: none;
border: 1px solid var(--border);
border-radius: var(--radius-sm);  /* 10px */
padding: 8px 18px;
```

Hover: `background: var(--paper-2); color: var(--ink);`.

App `.btn-secondary`: same palette, `border: 1px solid var(--border)`, no radius var referenced (inherits 10px).

App `.btn-danger`: `color: #c0392b; border: 1px solid rgba(192,57,43,0.2);` hover `background: #fdf0ee;`. Danger buttons are a visual pattern only on app.html.

## Text input

Landing `.demo-input` (textarea):

```css
width: 100%;
background: var(--paper);
border: 1px solid var(--border);
border-radius: var(--radius-sm);
padding: 12px 16px;
font-size: 15px;
min-height: 56px;
transition: border-color 0.2s;
```

Focus: `border-color: var(--accent);` — no box-shadow ring.

App `textarea, input, select`:

```css
border-radius: var(--radius-sm);
padding: 10px 14px;
font-size: 14px;
transition: border-color 0.2s, box-shadow 0.2s;
```

Focus: `border-color: var(--accent); box-shadow: 0 0 0 3px rgba(201,106,46,0.1);` — has a 3px accent-tinted focus ring. Landing doesn't.

NORMALIZE: focus treatment diverges. The app's 3-ring focus is the accessible pattern; adopt it on the landing `.demo-input` too.

## Select dropdown

Landing `.demo-lang-select` (12px text, 7px/10px padding, 8px radius) — smaller than app's `select` (14px text, 10px/14px padding, 10px radius). Focus: landing uses border-color only; app adds box-shadow.

## Tab chips (mode switcher)

Landing `.demo-chip` in `.demo-controls`:

```css
background: var(--paper);
border: 1px solid var(--border);
border-radius: 999px;
padding: 6px 14px;
font-size: 12px;
```

Active/hover: `background: var(--accent-light); border-color: rgba(201,106,46,0.3); color: var(--accent);`.

App equivalent is `.input-tab` inside `.input-tabs` — a different pattern: `background: var(--paper-2)` 4px-padded container with children that have `background: white` and `color: var(--accent)` when active, with a 1px shadow. Visually a segmented control, not a pill chip.

NORMALIZE: pick one. Landing's pill is the pattern the contribute page follows.

## Hero eyebrow (status pill)

Landing `.hero-eyebrow`:

```html
<div class="hero-eyebrow">Open-source research project</div>
```

```css
background: var(--accent-light);
border: 1px solid rgba(201,106,46,0.2);
color: var(--accent);
font-size: 12px;
font-weight: 600;
letter-spacing: 0.08em;
text-transform: uppercase;
padding: 6px 14px;
border-radius: 999px;
```

With an `::before` pulsing dot (`animation: pulse 2s ease-in-out infinite`) that `prefers-reduced-motion: reduce` disables.

## Feature card

Landing `.feature-card`:

```css
background: white;
border: 1px solid var(--border);
border-radius: var(--radius);
padding: 32px 28px;
transition: all 0.25s;
```

Hover: `transform: translateY(-3px); box-shadow: 0 12px 32px rgba(26,22,18,0.08); border-color: rgba(201,106,46,0.2);`.

Contains `.feature-title` (Instrument Serif, 20px) and `.feature-body` (14px, line-height 1.7).

## Step card (dark section)

Landing `.step-item`:

```css
background: rgba(26,22,18,0.95);   /* on a 1x1 grid with 2px gap, reads as near-black */
padding: 40px 32px;
```

Three-card grid sits on the `.how-section` ink background. Step number is Instrument Serif 56px in `--accent-bright`. Step title is Instrument Serif 22px in `--paper`. Body is 14px in `rgba(245,241,235,0.75)`.

## Hero demo card

`.hero-demo-card` — white, 24px radius, `padding: 28px`, double-shadow stack `0 24px 64px rgba(26,22,18,0.08), 0 4px 12px rgba(26,22,18,0.04)`. Contains the full translator demo (tabs, lang selects, input, avatar area). This is the "look at me I do a thing" marketing surface.

## Contribute banner

Landing `.contribute-banner`:

```css
background: var(--accent-light);
border: 1px solid rgba(201,106,46,0.2);
border-radius: 24px;
padding: 56px 64px;
display: flex; justify-content: space-between; gap: 40px;
```

Contains an `<h2>` Instrument Serif 36px + a `<p>` 16px + a `.btn-large` CTA. On mobile collapses to a single column.

## Footer

Landing `footer`:

```css
border-top: 1px solid var(--border);
padding: 40px 48px;
```

`.footer-inner` is a flex-wrap row of logo, `ul.footer-links`, copyright copy.

App `.app-footer` is minimal: 16/32px padding, single `/governance.html` link, no logo.

## Sidebar (app only)

`.sidebar` — 260px column of button-styled items. `.sidebar-item` is a full-width button with a 28×28 accent-tintable icon square + label. Active state is white bg, accent border, accent text. No ARIA role on the container; the parent `<aside aria-label="Sidebar navigation">` provides the landmark. On tablet (<1099px) it collapses to a horizontal row that can be opened via hamburger.

Known issue: it is an `<aside>` inside a container that already has no `<main>` landmark (`.content-area` has `role="main"` but also `id="main-content"`), which duplicates the main role and creates the a11y issue in [`docs/contribute-redesign/12-a11y-baseline.md`](../contribute-redesign/12-a11y-baseline.md).

## Avatar stage

Both pages use CWASA's mount points:

```html
<div class="CWASAAvatar av0" id="…"></div>
<div class="CWASAGUI av0" style="display:none"></div>
```

Landing's `.demo-avatar-area` is 260px tall with a pre-mount `.avatar-loading` overlay (spinner + "Loading avatar…"). App's `.avatar-wrapper` is a 384×320 stage with a separate token-chip list under it showing played/unplayed glosses.

Contribute has a richer stage: `aspect-ratio: 16/9`, backdrop layer, a subtle pulse during generation, and an invisible SVG body-region overlay for click-to-correct targeting. None of that exists on the main pages.

## Status dot / badge

Two patterns:

Landing `.status-dot`:

```css
display: inline-flex; align-items: center; gap: 6px;
::before  {  width: 7px; height: 7px; border-radius: 50%;  }
.status-dot.ready::before { background: var(--green); }
.status-dot.loading::before { background: var(--ink-3); animation: pulse …; }
```

App `.status-badge` — a full pill with bg + text:

```css
background: var(--paper-2); color: var(--ink-3);
padding: 4px 10px; border-radius: 999px;
::before { width: 6px; height: 6px; background: currentColor; }
.status-badge.ready { background: var(--green-light); color: var(--green); }
.status-badge.loading { background: var(--accent-light); color: var(--accent); }
```

NORMALIZE: pick one and use it everywhere. The app's filled pill reads louder and signals state better to sighted users; the landing's dot is subtler. For the contribute page's frequent state changes, the filled pill is probably correct.

## Skip link

Both pages declare:

```css
.skip-link {
  position: absolute; top: -100%; left: 16px;
  background: var(--accent); color: white;
  padding: 8px 16px; border-radius: var(--radius-sm);
  font-size: 14px; font-weight: 600; z-index: 200;
}
.skip-link:focus { top: 8px; }
```

Contribute.css uses `background: var(--ink); color: var(--paper);` instead — a different visual. NORMALIZE to accent/white.

## Animations catalog

| name | keyframes | duration | used by |
|---|---|---|---|
| `pulse` | opacity 1↔0.5, scale 1↔0.8 | 2s ease-in-out infinite | `.hero-eyebrow::before`, `.status-dot.loading::before` (1.5s) |
| `spin` | rotate 0→360deg | 0.8s linear infinite | `.spinner` (avatar loading) |
| `fadeUp` | opacity 0→1, translateY 24px→0 | 0.6s ease forwards | `.fade-up` with `.delay-1` (0.1s)…`.delay-5` (0.5s) |

All are gated behind `@media (prefers-reduced-motion: reduce)` on landing. App.html does not honor reduced-motion — the spinner and status-dot pulse run regardless.

## Summary: components the contribute page must reconcile

1. Skip link → accent, not ink.
2. Focus ring → accept the app's 3px accent-tinted box-shadow on all inputs.
3. Button system → three sizes (large, default, small); two styles (primary, ghost); one accent danger variant.
4. Status indicator → filled-pill badge, not bare dot.
5. Radius scale → 10px (controls), 16px (cards), 999px (pills). Drop 24px.
6. Hairlines → `var(--border)` at `rgba(26,22,18,0.12)`, not `--rule` at 0.14.

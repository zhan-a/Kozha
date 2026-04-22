# Design system — shared tokens and components

This is the token and component layer extracted from the canonical main-
page surfaces (`public/index.html` + `public/app.html`), as inventoried
in [`01-design-tokens.md`](./01-design-tokens.md) and
[`01-component-inventory.md`](./01-component-inventory.md). Everything
here is dev-only build convention — nothing ships an external CSS
processor; the rules are plain CSS with custom properties.

## Files

| file | what it is |
|---|---|
| `public/styles/tokens.css` | Every design token as a `--kozha-*` custom property on `:root`. |
| `public/styles/components.css` | Reusable `.kz-*` component classes, each wrapped in `:where()` so they contribute zero specificity. |
| `docs/polish/01-design-tokens.md` | Source-of-truth inventory every token in `tokens.css` was drawn from. |
| `docs/polish/01-component-inventory.md` | Source-of-truth inventory for the component rules in `components.css`. |

Both stylesheets are linked from `public/index.html` and
`public/app.html` before any other stylesheet (ahead of the Google Fonts
CSS and `cwa/cwasa.css`). They are defined but currently *parallel* to
the legacy inline `<style>` blocks — legacy still owns the rendered
pixels, and the pixel-diff CI (`npm run visual`) enforces that.

## Tokens (`--kozha-*`)

Naming convention: `--kozha-<group>-<role>[-<variant>]`. Groups in use:

| group | examples | purpose |
|---|---|---|
| `color` | `--kozha-color-ink`, `--kozha-color-accent`, `--kozha-color-paper-2` | All palette values. |
| `font` | `--kozha-font-sans`, `--kozha-font-serif`, `--kozha-font-mono` | Family stacks. |
| `text` | `--kozha-text-body`, `--kozha-text-hero-h1`, `--kozha-text-card-title` | Type sizes (role-based, not px-named). |
| `weight` | `--kozha-weight-medium`, `--kozha-weight-semibold` | 300 / 400 / 500 / 600. |
| `tracking` | `--kozha-tracking-tight`, `--kozha-tracking-widest` | Letter-spacing steps. |
| `leading` | `--kozha-leading-snug`, `--kozha-leading-loose` | Line-height steps. |
| `space` | `--kozha-space-1` … `--kozha-space-24` | 12-step scale: 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96 px. |
| `radius` | `--kozha-radius-sm`, `--kozha-radius`, `--kozha-radius-lg`, `--kozha-radius-pill` | 10 / 16 / 24 / 999. |
| `shadow` | `--kozha-shadow-card-rest`, `--kozha-shadow-btn-hover-lg` | Semantic shadow presets. |
| `duration` / `ease` | `--kozha-duration-base`, `--kozha-ease-standard` | Motion timing. |
| `layout` | `--kozha-layout-page-max`, `--kozha-layout-sidebar` | Page and sidebar widths, nav heights. |
| `z` | `--kozha-z-nav`, `--kozha-z-skip` | Stack order. |

### Key normalizations

These values came from `01-design-tokens.md`'s "Summary: the seven
things to normalize" list, which picked winners between the drifts on
`index.html` and `app.html`:

| token | winning value | loser |
|---|---|---|
| `--kozha-color-accent` | `#b3441b` (WCAG AA on paper) | `#c96a2e` (fails AA at body sizes on `app.html`) |
| `--kozha-color-ink-3` | `#5e5c57` | `#685d54` (`app.html`) |
| `--kozha-color-accent-light` | `#faeee0` | `#f5e4d6` (`app.html`) |
| `--kozha-color-border` | `rgba(26,22,18,0.12)` | `rgba(26,22,18,0.14)` (contribute) |
| `--kozha-text-body` | `14px` | `13px` (less common) |
| `--kozha-radius` / `--kozha-radius-sm` | `16px` / `10px` | drifts to `24px` / `8px` |

Both `#c96a2e` and `#faeee0` are still present in the palette as
`--kozha-color-accent-hover` / `--kozha-color-accent-light` —
`--accent-hover` is the brighter shade for button hover lift, not for
small text.

## Components (`.kz-*`)

Reusable classes in `components.css`:

| class | matches | when to use |
|---|---|---|
| `.kz-btn-primary` | landing `.btn-large` (16/600/accent/white) | Hero or banner CTA. |
| `.kz-btn-secondary` | landing `.btn-large-ghost` (16/500/ink-2/white + hairline) | Ghost equivalent. |
| `.kz-link` | `.nav-links a` (14/500/ink-2 → accent on hover) | Quiet text link on paper. |
| `.kz-input` | landing `.demo-input` (paper-filled textarea/input) | Text input surfaces. |
| `.kz-card` | landing `.feature-card` (white/hairline/16px radius, hover lift) | Content card. |
| `.kz-header` | landing `nav` (fixed, 64px, translucent paper, blur hairline) | Top nav region. |
| `.kz-footer` | landing `footer` (hairline top, 40/48px padding) | Bottom footer. |

### Zero-specificity wrapping

Every rule is `:where(.kz-*) { … }` rather than `.kz-* { … }`. This
gives the class rule a specificity of `0,0,0,0` so it cannot win over
legacy type selectors (`nav`, `footer`) or compound class selectors
(`.nav-links a`). That's the contract for the parallel-mode rollout —
the kz-\* class is *strictly weaker* than whatever legacy rule the
element already has, so visual is unchanged when both are applied.

When legacy inline rules are retired in later prompts, the `:where()`
wrapping stays. Zero-specificity design-system rules are the pattern we
want long-term: they never collide with local overrides, and pages keep
the power to override tokens or rules without having to match
specificity ladders.

## How to use

### Build a new UI surface

1. Reach for tokens, not literals. A new color, size, or spacing value
   should come from the `--kozha-*` set.
2. Reach for components. If the surface has a primary CTA, it should
   carry `.kz-btn-primary`. If the surface has a card, it should carry
   `.kz-card`. Don't re-implement buttons or cards.
3. If a token you need isn't in the set, **open an issue or propose a
   design review** — then add it to `tokens.css`. The whole point of
   this layer is convergence toward a small named set. The inventory
   documents had to flag 13 distinct body-text sizes and 25 distinct
   spacing values; we are actively narrowing that.

### Extend an existing component

Edit `components.css`. Keep the `:where()` wrapping. Reference tokens,
not literals. If the extension is a genuine variant (e.g. a small size
or danger color), add a new class like `.kz-btn-primary--sm` /
`.kz-btn-danger`, not an option prop via attribute selectors.

### Migrate a legacy surface

1. Verify the canonical appearance of the legacy surface matches the
   `.kz-*` class (comparison is straightforward — legacy sets the same
   property set as the kz rule).
2. Remove the legacy declaration from the inline `<style>`; keep the
   element's `.kz-*` class.
3. Run `npm run visual` and ensure diff stays under 0.5%. If it
   doesn't, the legacy class had a property the component class
   doesn't cover — add it to the component, or add a modifier class.

### Visual regression workflow

`tests/visual/regression.mjs` renders `/` and `/app.html` at
1440×900 and 390×844 through headless Chrome, screenshots each, and
pixel-diffs against `tests/visual/baseline/*.png`. A diff image for
each scenario goes to `tests/visual/diff/`.

- `npm run visual` — compare current state against baseline. Fails
  when any scenario drifts above **0.5%** of pixels.
- `npm run visual:baseline` — overwrite baselines. Do this only after
  a deliberate design change that has been reviewed.

CWASA avatar surfaces and all animations are hidden / frozen in the
test environment so pixel output stays deterministic across runs.

## Do not invent new tokens without a design review

Every token in `tokens.css` is sourced from concrete values observed
on the main-page surfaces. If you need a new color, size, or spacing
value — the correct move is to:

1. Check whether an existing token covers the intent.
2. If not, explain the need in a design review (even a brief async
   note), so the new token is chosen with the same rigor as the
   existing set.
3. Only then add it to `tokens.css`, with a comment on where it's used.

The system is small on purpose. Keeping it small is how we prevent
the pre-refactor drift (accent colors failing AA, 25 spacing values,
3 different focus-ring treatments) from coming back.

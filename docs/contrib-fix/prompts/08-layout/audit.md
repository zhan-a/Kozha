# Prompt 08 — Authoring-page layout: collapse the empty void below the chat

Companion to `docs/contrib-fix/prompts/08-layout` (this directory). Prompt 07
fixed the avatar mount; this prompt rebuilt the surrounding page so the
chat, avatar, and parameters columns share the viewport rhythmically
instead of stacking into a hollow tower with empty space below.

## Symptom

Loaded `/chat2hamnosys/index.html` at 1440×900, dropped a single chat
turn that returned `preview.sigml`, and observed:

- the chat panel ended ~480 px from the bottom of the viewport;
- the right column (avatar + inspector) was even shorter — the
  inspector body collapsed to ~120 px;
- the page outer-scrolled by ~200 px even though every panel was
  visibly hollow;
- nothing useful sat in the void.

The page read as broken: a vertical gulf opened, then the footer
element (or just empty paper) sat below the fold.

## Root cause

Three rules compounded.

### 1. The outer container forced 100vh of height

`public/chat2hamnosys/styles.css` (pre-fix):

```css
body { min-height: 100vh; display: flex; flex-direction: column; }
.layout {
  flex: 1;
  display: grid;
  grid-template-columns: minmax(320px, 1fr) minmax(360px, 1.2fr);
  grid-template-rows: minmax(0, 1fr) minmax(0, auto);
  grid-template-areas: "chat preview" "chat inspect";
  ...
  min-height: calc(100vh - var(--topbar-h));
}
```

`.layout` had `min-height: calc(100vh - var(--topbar-h))`, so it always
filled to viewport bottom. The grid then had two rows: a 1fr row
(preview) and an auto row (inspector). On a 900 px tall viewport with
~120 px of inspector content and ~520 px of preview content, the right
column totalled ~640 px — short of the 844 px the layout demanded. The
1fr row absorbed the slack as a 200+ px hollow band beneath the avatar;
the chat column (spanning both rows) absorbed it on the left as the
chat-log's empty scroll region.

### 2. The chat panel had a hard `min-height: 60vh`

`.panel-chat { min-height: 60vh; }` made the chat column at least 540
px tall on a 900 px viewport regardless of message count. Combined
with the layout's 100vh minimum, the column locked into a tall hollow
shape that never shrank.

### 3. The page itself outer-scrolled

Body had `min-height: 100vh` (not `height: 100vh`). When the right
column genuinely grew past 100vh on a short viewport, the page
silently outer-scrolled instead of letting the inner columns own the
overflow. There was no internal-scroll discipline anywhere.

### 4. The mobile-tabs panel switcher was redundant

`<nav class="mobile-tabs">` showed below 880 px and switched the visible
panel via `.is-active`. With three panels but only one visible at a
time, mobile users had to play tab golf to inspect the avatar while
typing. The new responsive grid replaces this with always-visible
stacked sections.

## Fix

**Pinned page height + internal scroll discipline.** The body is now
`height: 100dvh; overflow: hidden`, so the page is exactly one screen.
Every panel uses `display: flex; flex-direction: column; min-height: 0;
overflow: hidden`, which delegates overflow to its scroll-region child
(`.chat-log`, `.preview-stage`, `.inspector-body`). No void can open
because the rows are sized by definite tracks and any extra content
turns into internal scroll, not outer page scroll.

**Three-column desktop grid.** `chat | preview | inspect` with a
1.0fr / 1.2fr / 0.7fr weighting (max-width 1500 px). The widths are
product weight, not equal thirds: chat is the input target, preview
is the centerpiece, status/inspector is supporting metadata.

**Two-column tablet (768–1023 px).** Inspector folds into a one-row
strip across the top — collapsed by default so it shows just its
header (Collapse/Expand button still works). Chat and preview share the
remaining row.

**One-column mobile (< 768 px).** Stack top-to-bottom: inspector
(collapsed), preview, chat. Chat at the bottom is the iOS / messaging
convention — it gives the on-screen keyboard the room it expects. The
preview row uses `minmax(180px, 32dvh)` so the avatar stays visible
but doesn't push the composer off-screen.

**Sticky composer.** `.chat-form { flex-shrink: 0 }` keeps the input
pinned to the bottom of the chat column at every viewport. The user
never has to scroll the column to find Send.

**Avatar mount survives the responsive override.** Prompt 07 made the
mount `aspect-ratio: 1/1; min-height: 320px` so its CWASA percentage
chain resolves. On mobile we override to `height: 100%; min-height: 0;
aspect-ratio: auto` because the parent grid row is already a definite
track. The percentage chain still resolves against the row, so the
prompt-07 fix is preserved, just with a smaller floor on small screens.

## Files

- `public/chat2hamnosys/styles.css` — body height, layout grid, panel
  overflow discipline, sticky composer, three responsive blocks.
- `public/chat2hamnosys/index.html` — dropped `<nav class="mobile-tabs">`,
  added `aria-label="Authoring workspace"` to `<main>`, updated panel
  comments.
- `public/chat2hamnosys/app.js` — removed `setupTabs()` /
  `switchMobileTab()` / `state.activeMobileTab` and the three legacy
  call sites; added `setInspectorExpanded()` plus a `matchMedia`
  listener so the inspector body is collapsed by default below 1024 px
  but stays out of the way once the user has clicked the toggle.
- `tests/c2h_avatar_visible.spec.ts` — removed the mobile-tabs click;
  loosened the avatar's min-height floor on mobile (180 px instead of
  320 px) to reflect the new viewport-aware sizing.
- `tests/c2h_layout_no_outer_scroll.spec.ts` (new) — five viewports
  (1440, 1280, 1024, 768, 414), asserts outer scrollHeight ≤ 1.05×
  viewport, asserts the resolved column count, asserts the chat
  composer pinned to panel bottom, asserts mobile composer reachable
  without outer scroll, asserts the legacy `.mobile-tabs` element is
  gone, asserts desktop tab order is chat → preview → inspect.

## Acceptance check

- No empty void below the chat at any viewport ≥ 320 px — body is
  pinned to 100dvh and the layout fills it via grid.
- Three / two / one-column responsive break at 1024 / 768 — verified
  by `c2h_layout_no_outer_scroll.spec.ts` column-count assertions.
- Composer is sticky — `flex-shrink: 0` on `.chat-form` plus the
  panel's flex column.
- Outer page does not scroll — `body { overflow: hidden }`.
- Accent-contrast rule from `feedback_accent_contrast_on_paper.md`
  preserved (no `--accent` overrides in this change set).
- Avatar fix from prompt 07 preserved (mount `aspect-ratio: 1/1;
  min-height: 320px` on desktop / tablet; relaxed only on mobile where
  the grid row already provides a definite parent).

## Out of scope

Re-skinning chat bubbles, the avatar shell, or the authoring API.

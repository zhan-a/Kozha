# Fake-avatar audit — `/contribute`, landing, translator, extension

Tracking deliverable for the prompt asking us to eliminate every avatar
that is not the real CWASA engine driven by real SiGML pulled from
`data/`. Where CWASA legitimately cannot run we fall back to a
**SiGML + HamNoSys snapshot card** (Pattern B) — never to an illustrated
silhouette or static avatar render.

## Scan scope

- All HTML under `public/` (incl. `public/_deprecated/` ignored).
- All HTML under `extension/`.
- All CSS-only "avatar" mock illustrations (silhouette + caption shapes).
- All `<img>`, `<canvas>`, `<video>` placeholders in those files.
- All `<div class="avatar*">` whose JS handler logs a sign name without
  actually calling into CWASA.

## Findings

| # | File | Lines | What it was | Replacement | Pattern |
|---|------|------:|-------------|-------------|---------|
| 1 | `public/contribute.html` | 161–168 | `c2-hero__cwasa-poster` SVG silhouette inside the hero card. The card said `BSL · ELECTRON` even though `hamnosys_bsl_version1.sigml` has no `electron` entry — doubly fake (silhouette + invented gloss). | Replaced the SVG silhouette with a **SiGML + HamNoSys snapshot card** rendering the BSL `name(v)#1` gloss (real entry, line 2303 of `hamnosys_bsl_version1.sigml`). The card now shows the gloss, the HamNoSys glyphs in `HamNoSysUnicode`, and a `<pre>` with the SiGML excerpt. A real `<button>` ("▶ Play with avatar") swaps the snapshot for a live CWASA mount on click and toggles `aria-pressed`. | **B** (default) → **A** on click |
| 2 | `public/contribute.html` | 344–356 | `c2-viz-4__avatar-poster` SVG silhouette inside the walkthrough Step 4 mount, with the caption "avatar loads on first interaction". | Replaced with a HamNoSys snapshot block that renders the real DGS `HAMBURG2^` HamNoSys glyphs (matches the SiGML the walkthrough already plays). The Replay button takes ownership of the canvas the moment CWASA is ready. The snapshot is hidden by `:has(canvas)` once CWASA injects its WebGL canvas (existing pattern, kept). | **A** (silhouette poster removed) |
| 3 | `public/contribute.css` | 2977–3010, 3556–3579 | Silhouette-fill rules (`.c2-cwasa-poster-head/torso/arm-a/arm-b`) + `.c2-viz-4__avatar-poster` SVG layout + `:has(canvas)` hide rules. | Silhouette CSS removed. New `.c2-snapshot-card` styles render the HamNoSys + SiGML preview using design tokens. The `:has(canvas)` hide rule is preserved against the snapshot card class so live CWASA still covers the snapshot. | — |

### Confirmed real CWASA mounts (no action needed)

| File | Element | Status |
|------|---------|--------|
| `public/index.html:1153` | `<div class="CWASAAvatar av0" id="heroAvatar">` | Real, lazy-loaded, plays an actual SiGML excerpt. Gold-standard. |
| `public/app.html:1890` | `<div class="CWASAAvatar av0">` (in `#avatarStage`) | Real translator engine. |
| `public/contribute.html:850` | `<div class="CWASAAvatar av0" id="avatarCanvas">` | Real preview after submit. |
| `public/chat2hamnosys/index.html:130` | `<div class="CWASAAvatar av0">` (in `#cwasaMount`) | Real fallback inside the authoring tool. |
| `extension/panel.html:15` | `<div class="CWASAAvatar av0">` | Real engine inside the Chrome extension panel iframe. |

### Honest empty states (kept; not fakes)

| File | Element | Why kept |
|------|---------|----------|
| `public/index.html:1149–1152` | `.avatar-loading` spinner + "Loading avatar…" | Loading indicator, not a fake avatar render. |
| `public/app.html:1892–1895` | `.avatar-placeholder` "Ready when you are." 🤟 | Empty state before a translation; an icon, not an avatar imitation. |
| `public/chat2hamnosys/index.html:134–136` | `#previewPlaceholder` "Send a description to generate a preview." | Empty state. |

## Gloss-vs-payload alignment

The prompt called out that signs named in demo captions must be the
**actual** corpus entries:

- Hero card: gloss `name(v)#1`, source `data/hamnosys_bsl_version1.sigml`,
  attribution "BSL · DictaSign Corpus, CC BY-NC-SA". HamNoSys + SiGML
  payload mirrors the entry verbatim.
- Walkthrough Step 4: gloss `HAMBURG2^`, source `data/German_SL_DGS.sigml`,
  attribution "DGS · DGS-Korpus / SignAvatars". The HamNoSys snapshot
  matches the SiGML the existing JS plays.

A new spec at `tests/contrib_demo_signs.spec.ts` parses both demo cards,
reads the gloss caption + the SiGML payload data attribute, and asserts
that the payload contains the gloss tag and that all `<ham*-/>` tags in
the payload appear (in order) in the named corpus file's entry for that
gloss.

## Mobile + reduced-motion behaviour

- Both demo cards default to **Pattern B** (snapshot card) on
  `prefers-reduced-motion: reduce`. The play button still works, but
  no auto-play and no canvas hand-off in the IntersectionObserver.
- On viewports < 600 px wide the hero card stays Pattern B (no auto
  swap). The walkthrough step keeps its existing scroll-into-view
  behaviour but skips auto-play under reduced motion.

## Accessibility

- Each snapshot card has a plain-language `aria-label` describing the
  sign (e.g. "BSL sign for 'name': flat hand near the forehead, two
  fingers extended, brushing outward").
- The play button is a real `<button type="button">` with `aria-pressed`
  toggling between `false` (snapshot showing) and `true` (CWASA playing
  in the same slot).
- Text inside the cards uses `var(--accent)` (`#b3441b`, WCAG AA on
  paper) — not `--accent-bright` (`#c96a2e`, fails AA at small text).

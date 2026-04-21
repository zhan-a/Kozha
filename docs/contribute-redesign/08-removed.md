# Removed — the prompt-8 audit trail

Everything taken out (or explicitly left untouched) during the prompt-8
"strip clutter" pass, with a one-line reason for each. This doc is the
audit trail for the non-goals list in
[01-design-principles.md](01-design-principles.md) and for the copy
decisions in [08-copy-changes.md](08-copy-changes.md).

---

## `public/index.html` — rewritten end-to-end

The whole file was rewritten. The rewrite keeps: one sentence about
Bridgn, one primary CTA (Open translator), one secondary CTA
(Contribute a sign), a minimal Credits section linking to the README,
and a one-line footer. Everything below was in the pre-rewrite file
and is no longer served.

### Hero region

- `.hero-eyebrow` badge "Open-source research project" — marketing
  badge, no function. Ran a 2s infinite `pulse` animation on page load.
- `@keyframes pulse` and its animated dot on the eyebrow — decorative,
  page-load animation.
- `.hero-sub` two-sentence copy "Bridgn converts typed text, recorded
  speech, or uploaded video into sign language animations. It runs
  entirely in your browser — no accounts, no servers, no cost." —
  collapsed to one sentence.
- `.hero-stat` block (three stat items: "BSL / British Sign Language",
  "On-device / Speech recognition", "Free / Always for educators") —
  promotional statistics, forbidden by
  [01-design-principles.md §Non-goals #14](01-design-principles.md).
- `.hero-demo-card` — the embedded live demo card with input modes,
  language selectors, CWASA avatar, `MediaRecorder` ASR, and ffmpeg
  video pipeline. Reason: fails the "Deaf user on a 3G connection in
  a rural area" test — the card loaded CWASA (`/cwa/cwasa.css`,
  `/cwa/allcsa.js`), a preloaded SiGML database, Google Fonts, and
  pulled `transformers.js`, `@ffmpeg/ffmpeg` on demand. The live demo
  is now reached via the "Open translator" link.
- `fade-up` / `.delay-1`-`.delay-5` page-load animations and
  `@keyframes fadeUp` — marketing motion, forbidden by
  [§Non-goals #12](01-design-principles.md).
- Avatar loading spinner and `@keyframes spin` — only ever rendered
  because the demo card embedded CWASA; removed with the card.

### Preloads and external assets

- `<link rel="preload" href="/data/hamnosys_bsl_version1.sigml">` —
  only needed for the embedded demo.
- `<link rel="preload" href="/cwa/cwasa.css">` and
  `<script src="/cwa/allcsa.js">` — avatar engine; only the
  `/app.html` translator needs these.
- `<link href="https://fonts.googleapis.com/…">` (Google Fonts:
  Instrument Serif + DM Sans) — replaced with the `system-ui` stack
  already used on `contribute.html`. No external font network
  request on first paint.

### Sections removed wholesale

- `<section class="section" id="features">` — six feature cards
  ("Voice-to-Sign", "Text-to-Sign", "Video Input", "STEM Vocabulary",
  "Automatic Planning", "Open Source"). Reason: promotional
  scaffolding that a 3G user has no reason to wait for.
- `<div class="how-section" id="how-it-works">` — three-step
  "How it works" panel. Reason: the translator itself explains
  itself to a first-time user in a handful of clicks; an infographic
  duplicates that.
- `<section class="section"><div class="contribute-banner">` —
  "Contribute to the project. / Bridgn's sign database has limited
  vocabulary and needs help. / Get involved". Reason: "needs help"
  is the inspirational framing prompt 8 §5 specifically rules out,
  and the nav already carries a Contribute link.

### Scripts removed

Every `<script>` block in the old `index.html` existed to power the
hero demo card or its language switcher. All removed:

- `navHamburger` toggle — no hamburger on the new nav.
- `demo-chip` tab switcher — no demo tabs.
- CWASA init + `showAvatarError` — no embedded avatar.
- Sign-DB loaders (`loadSigmlUrl`, `loadAlphabetSigmlUrl`,
  `loadConceptCsvUrl`, `SIGN_LANG_DB`, `switchSignLanguage`) — no
  embedded translator.
- `levenshtein` / `similarity` / `mapToAvailable` /
  `fingerspellWord` / `buildSigml` / `heroTranslateText` /
  `heroTranslateIfNeeded` / `heroTranslateBtn` click handler — no
  translate control.
- `pickAsrModel` / `ensureASR` / `getAsrOptions` and the IIFE that
  wired the mic button to transformers.js — no on-device ASR.
- The `ensureFFmpeg` IIFE and its video-file handler — no ffmpeg
  video pipeline.

### Meta + structured data

- `twitter:card`, `og:*` strings trimmed to plain descriptive copy
  ("Translate typed text, recorded speech, and uploaded video into
  sign-language animations."). No marketing verbs.
- JSON-LD `WebApplication` entry kept, description updated to match
  the new one-sentence pitch.

---

## `public/app.html` — minimal touch

Explicitly out of scope for redesign. One copy change:

- Top-nav link text "Contribute" → "Contribute a sign" (line 604).
  Reason: align the label with the landing page and the new
  contribute page's voice.

### Not changed (deferred)

- Sidebar `.sidebar-icon` emoji (🤟 / 🎙️ / 🎬 / 🗄️ / ⚙️ / 📋) —
  the emoji-audit rule in prompt 8 §6 would remove these, but
  prompt 8 §2 says "do NOT redesign" app.html, and the icons sit
  inside styled 28×28 chrome containers whose removal would shift
  layout. Deferred to a later prompt that opts app.html into scope.
- `--accent: #c96a2e` palette token — fails AA on `#f5f1eb` for
  normal text (3.41:1). Used in `.nav-link-sm:hover`,
  `.nav-badge`, `.nav-logo span`, and other interactive chrome.
  Not fixed here for the same reason: a palette shift is a visual
  redesign, which prompt 8 §2 rules out.
- No "Help us grow the dictionary" banner exists — the prompt's
  instruction is vacuous for the current app; the removal rule is
  recorded here for the audit trail.

---

## `extension/` — only the one link added

- `extension/popup.html`: added a single `<footer id="popup-footer">`
  element containing a link "Contribute a sign on
  kozha-translate.com" pointing at
  `https://kozha-translate.com/contribute.html` with
  `target="_blank" rel="noopener"`.
- `extension/popup.css`: added scoped rules for `#popup-footer` and
  `#popup-footer a` only. Existing button, status, and layout
  styling is untouched.

### Not changed (deferred)

- `#popup-sign-btn` background `#c96a2e` with `#fff` text — 3.76:1,
  fails AA for normal text at 13px. Per prompt 8 §3 "Do not
  redesign the extension," left as-is. Recorded here as a known
  AA gap for a later extension-focused pass.
- Extension popup header branding "Kozha" — the brand inconsistency
  with the website's "Bridgn" is noted in
  [08-copy-changes.md §6](08-copy-changes.md#6-observations-on-copy--applied-decisions)
  and deferred.

---

## `public/contribute.css` — one AA fix

- `--muted: rgba(21, 19, 15, 0.55)` → `rgba(21, 19, 15, 0.62)`.
  Reason: composited over `--paper`, the pre-existing value
  resolved to `#7b7872` at 3.98:1 on paper — fails AA for normal
  text (the token is used at 13-14px in several places including
  `.field-meta`, `.notation-legend-title`, and `.notation-sigml-summary`).
  The new alpha resolves to `#6b6863` at 5.02:1, comfortably
  passing AA. No visual regression at larger sizes; muted text is
  now a hair darker everywhere.

---

## Site-wide CSS audit — animations, stock images

### `@keyframes` that ran on page load

All of the following were removed via the index.html rewrite:

- `@keyframes pulse` — hero-eyebrow dot, ran forever.
- `@keyframes fadeUp` + `.fade-up` / `.delay-*` — page-load fade-in
  ladder.
- `@keyframes spin` — avatar-loading spinner (only fired during the
  demo card's CWASA boot, which is gone).

### Kept (interaction-triggered only)

- `@keyframes chat-progress-slide` in `contribute.css` — activates
  only while the chat panel is waiting for a response; also
  respects `prefers-reduced-motion` (falls back to a static bar at
  60% opacity).
- `@keyframes spin` in `chat2hamnosys/styles.css` — spinner rendered
  only when an active operation sets the `.is-loading` state.
- `@keyframes regionPulse` in `chat2hamnosys/styles.css` — 0.2s
  pulse on click/tap inside the avatar region picker. Interaction
  feedback.
- `@keyframes` inside `public/_deprecated/contribute-pre-redesign.html` —
  deprecated file, never served. Deliberately not touched so the
  before/after diff remains legible to a later maintainer.

### Stock images

None to remove. `public/` contains no raster or vector files beyond
`favicon.ico`, the HamNoSys glyph font under `public/fonts/`, and
CWASA engine assets under `public/cwa/` — all semantic/functional,
none decorative. No hero photography, no illustrated mascots.

---

## Emoji audit

### Removed

- Hero demo-card tab chips in `index.html` ("✏️ Text", "🎙 Voice",
  "🎬 Video") — removed with the whole demo card.

### Kept, flagged as functional

- None elsewhere in authored surfaces after this pass.

### Deferred (out of scope)

- `app.html` sidebar emoji — see the app.html section above.
- `README.md` `🌐 **Live Demo:**` emoji (line 5) — the README is a
  repo-facing artifact, not a web-facing surface under this prompt.

---

## WCAG AA contrast — recheck

`axe-core` is not available in this environment; contrast was
recomputed by hand with the W3C luminance formula. Results on the
surfaces prompt 8 §7 names:

- `/` (`public/index.html`, new) — all authored pairs ≥ 5.0:1. The
  strictest pair is `--accent: #b3441b` on `--paper: #f7f3ec` at
  5.05:1 (interactive focus ring and em accent), both of which
  comfortably meet AA for normal text. Body text (`--ink` on
  `--paper`) is 16.8:1.
- `/app` (`public/app.html`) — body text passes; hover accent
  token fails AA as above. Not in scope.
- `/contribute` (`public/contribute.html`) — after the `--muted`
  fix, the weakest authored pair is muted labels at 5.0:1 on paper
  (from `#6b6863` over `#f7f3ec`). All active-state colors use
  `--ink` or `--accent` which pass.
- `/contribute` mid-session (same file, after the authoring form
  mounts) — uses the same tokens, same ratios. No session-specific
  styling drops below AA.

No axe-reported violation simulated for the authored surfaces
after the `--muted` fix. See
[08-copy-changes.md](08-copy-changes.md) for the copy side.

---

## Landing-page credits section

The prompt asked to keep "the existing credits section" intact on
the landing page. No such section existed before this pass — the
credits were (and still are) kept in full detail in `README.md §
Credits`. To honor the intent of the instruction (the licences
matter; attribution must be reachable from the site), a minimal
credits block was added to the new landing page that names CWASA,
HamNoSys, and the sign-language databases by project name and
links to the README for full licence terms. This is an addition,
not a removal, and is recorded here so the audit trail does not
mis-describe it as "kept intact."

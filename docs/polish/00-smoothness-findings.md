# Smoothness walkthrough — current-state findings

Walkthrough of `public/app.html` captured on 2026-04-22 against the prompt-10
acceptance list. Each finding is tagged S1 (broken / blocks a normal flow),
S2 (jarring, reads as unpolished), or S3 (mildly annoying).

## Loading states

- **S1 — no 200ms threshold.** The status badge flips to `Loading…` the
  instant `translateBtn` is clicked, even for fast same-language lookups
  that complete in <100ms. A flash of "Translating… / Ready" back-to-back
  reads as a flicker.
- **S1 — loading is a badge, not a progress bar.** `renderStatus` is a
  pill at the top-right of the render panel. The playback progress bar
  (`#playbackProgress`) already exists for post-translate playback but
  isn't used during the translate step itself. No visual tie between
  "we're working" and the avatar area.
- **S2 — no horizontal progress surface during translate.** The chat
  generation indicator convention (thin bar, indeterminate motion) is
  not mirrored.

## Layout stability

- **S1 — avatar area resizes on mobile.** `.CWASAAvatar.av0` is fixed
  `384×320` on desktop but `width:100%; height:auto` at ≤1099px. Until
  CWASA mounts, the wrapper collapses to 0px, then pops to its target
  height when the WebGL canvas appears.
- **S2 — no neutral placeholder pose before first play.** The avatar
  stage shows a blank white box until CWASA is ready and the first
  SiGML plays. There is no 16:9 aspect ratio reservation.
- **S2 — coverage counter can push layout.** `.coverage-counter` has a
  `min-height: 18px` but its `innerHTML` change between empty and a
  multi-segment string (signed · fingerspelled · omitted) can still
  force a paint reflow if the string wraps.
- **S3 — token list doesn't reserve max lines.** `min-height: 28px`
  is enough for one row; two-row phrases still shift the controls
  below.

## Avatar pose transitions

- **S2 — no inter-sign neutral pose pause.** `simulatePlayback` ticks
  through chips at 1200ms/token but CWASA plays one large SiGML
  concatenation, so the avatar chains poses without a rest beat. At
  ×2 speed the transitions pop visibly.
- **S3 — "Stop" is abrupt.** `CWASA.stop(0)` snaps to neutral with no
  ease-out.

## Avatar backdrop

- **S1 — stark white, not the contribute-page light gray.** Contribute
  uses `#e9e6df` on the avatar backdrop; app.html uses `background:
  white` with `border: 1px solid var(--border)`. The literature the
  contribute page references recommends a non-white neutral for
  prolonged Deaf viewing. App translator diverges.
- **No `--kozha-color-surface-avatar` token yet.** Prompt 2 extracted
  tokens but didn't add this one. Adding it here so every surface
  shares a single definition.

## Playback controls

- **S1 — only a Stop button.** No play/pause toggle. No loop. No speed
  control (extension exposes speed; main site does not). No timeline
  scrubber.
- **S1 — no keyboard shortcuts.** Space does nothing on the avatar
  stage. Arrow keys do nothing.
- **S2 — time / duration not displayed.** Users cannot tell how much
  of a long translation is left to play.

## Captions

- **S1 — no caption strip.** The token chips under the avatar serve a
  similar purpose but are oriented as a progress tracker, not as a
  readable current/next caption. The Deaf community standard is a
  prominent gloss + a smaller English source context.

## Error recovery

- **S2 — errors render in the status badge, not inline.** A failed
  translation shows "Translation unavailable — using original text" in
  the top-right pill. On mobile the pill is below the fold from the
  input, so the user sees nothing happen.
- **S2 — no retry affordance.** Users must re-click Translate. No
  explicit "Try again, or switch to a supported language pair."
- **Good:** the input textarea is already preserved on failure —
  nothing clears it today.

## Input textarea

- **S1 — Enter submits a newline instead of translating.** The
  textarea has no Enter handler; native behavior is to insert `\n`.
  Shift+Enter has no special meaning.
- **S2 — no auto-grow.** The textarea ships at 3 rows; long inputs
  trigger an inner scrollbar at ~4 lines.
- **S2 — no auto-translate toggle.** The page has no concept of
  automatic translation on typing; the Translate button is the only
  trigger. That's actually the safer default per this prompt — we'll
  keep it that way and only add opt-in auto-translate.

## Character count

- **S2 — counter always visible.** `0 / 10000` shows from first
  paint, even on an empty textarea. Adds visual noise. The counter
  only flips to red when length *exceeds* 10000 — the hard-stop is
  silent (maxlength attribute), no warning approaching the limit.

## Keyboard shortcuts surfaced once

- **S2 — no discovery surface at all.** Nothing tells a first-time
  user that Enter submits or that playback has any keys. Even once
  we add the keys, a user won't know.

## Mobile layout

- **S2 — avatar width:100%; height:auto collapses.** Already noted
  above. In addition, the render panel sits below the input on mobile,
  which is correct, but the caption/chips/controls area is
  inconsistent in spacing.
- **S3 — not all tap targets reach 44px.** The avatar Switcher and
  status badge are below the 44px threshold. Per the existing
  `@media (max-width: 1099px)` block, `.btn` hits 44px min-height but
  `.avatar-switcher select` does not.

## Extension

- **S2 — popup has no speed control on the popup surface** (speed is
  passed via `set_speed` but the popup never surfaces it). No
  play/pause. No loop. Smoothness parity with the main site is a
  follow-on — minimum-change here: keep the popup working, add a
  speed select and a hairline pause/play.

## Prefers-reduced-motion

- **S1 — not honored in app.html.** The landing honors it; the app
  does not. Every transition we add must honor it, and the existing
  `transition: all 0.2s` on buttons should collapse to no transition
  under `reduce`.

## Summary — what prompt 10 must close

| Tag | Item                                               |
|-----|----------------------------------------------------|
| S1  | 200ms loading threshold + progress bar surface     |
| S1  | Avatar 16:9 reservation + neutral placeholder pose |
| S1  | Avatar backdrop token + apply on main translator   |
| S1  | Play/pause, loop, speed, scrubber — all visible    |
| S1  | Keyboard shortcuts (space, arrows, Enter)          |
| S1  | Caption strip (gloss large, source small)          |
| S1  | Enter submits, Shift+Enter newlines                |
| S1  | Prefers-reduced-motion honored                     |
| S2  | Layout-stable counters and lists                   |
| S2  | Inter-sign neutral pose pause (300ms)              |
| S2  | Inline error recovery with retry                   |
| S2  | Auto-grow textarea                                 |
| S2  | 80%-threshold character counter, warn at 8000      |
| S2  | First-visit keyboard hint strip                    |
| S2  | Mobile tap targets ≥ 44×44                         |
| S2  | Extension parity (minimum-change)                  |
| S3  | Inter-sign fade on Stop (optional stretch)         |

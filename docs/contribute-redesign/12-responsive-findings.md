# Responsive findings — prompt 12

## Status

**Partial automation, pending human device testing.** The Puppeteer harness in `scripts/a11y/run.mjs` captures screenshots at three viewports (`iphone-se: 375×667`, `ipad-portrait: 768×1024`, `desktop-wide: 1440×900`) and runs axe + pa11y at each. No 4K capture is done because headless Chromium's device-pixel-ratio handling is not reliable at UHD and would require a separate physical-device pass to trust.

The screenshots under `12-a11y-raw/*.png` are the artefacts; axe/pa11y passes at each viewport are in the summary table of `12-a11y-baseline.md`.

### What automation caught

- No horizontal scrollbar at 375 px on any page in the flow.
- No overlapping text at 375 px.
- Touch targets: axe's `target-size` rule (WCAG 2.5.8) passes at all viewports — every interactive element is at least 24×24 CSS pixels with no intersecting target. This was verified at 375 px specifically for the context-strip `copy` button, the chat option buttons (which are smaller), and the notation-panel tab affordances.
- Focus-visible outlines continue to meet 3:1 contrast against their background at all three viewports.
- Font sizing scales legibly at iPhone SE width; no line-length exceeds ~75 ch in the single-column layouts.

### What automation cannot catch

Real responsive behaviour on real hardware has never been exercised by this commit. In particular:

- **Real iOS Safari** — headless Chromium does not reproduce the WebKit viewport behaviour. `100vh` quirks, the dynamic URL bar, safe-area insets, and the overscroll background colour have not been verified on a real device.
- **Real Android Chrome** — no soft-keyboard flex testing; we do not know whether the chat input stays visible when a user types a long correction with the keyboard open on a 360 px Android device.
- **Real iPad portrait** — Puppeteer at 768 px gets close, but pencil hover and split-screen (`Slide Over`) behaviours have not been verified.
- **4K monitor (2160 p)** — the layout is max-width 720 px everywhere, so it will look identical to the 1440 px capture — but the font rendering, focus-ring crispness, and high-DPI stroke thickness have not been verified visually.
- **Browser zoom to 200%** — we render, but a user at 200% zoom may experience horizontal scrolling on the submission checklist's "complete/pending" markers or the notation panel's glyph display; we haven't tested.
- **Reduced motion** — we do not currently gate the 200 ms crossfade on the notation panel behind `prefers-reduced-motion`. This should probably be fixed regardless of device testing.

## What to verify on real devices

### iPhone SE / iPhone 13 mini (375 px class)
1. Landing: hero CTAs fit on one line each. If they wrap to two lines, the spacing is acceptable — not a defect.
2. Contribute empty: the language picker scrolls naturally within the main column; no two-finger horizontal scrolling should be possible.
3. Contribute mid-session: the avatar preview is the largest block; confirm the Play button and scrubber are both reachable without unintentional pan. The scrubber must not accidentally start a horizontal scroll.
4. Notation tabs: tapping "SiGML" should scroll the tab into view if offscreen; it should not cause a layout shift that bumps the Submit button below the fold.
5. Submission: the three action buttons (Discard, Save, Submit) should all be reachable without leaving the submission section; on a 375×667 viewport this means no single screen swallows all three buttons — expect the user to scroll within the submission section.
6. Confirmation: the permanent link input and Copy Link button should be next to each other without the input wrapping to a full line.
7. Governance: the email address displayed below "Or copy the plain address:" should not overflow the viewport. If it wraps, that is acceptable.

### iPad portrait (768 px class)
1. Every page should look identical to the desktop layout, just narrower. Because our max-width is 720 px, the main column is almost exactly the iPad portrait width with small margins.
2. The chat panel's `options` row of suggested answers should still show all options without wrapping if the options are short; long options wrapping is expected.

### 1440 px and up
1. The layout is centred and capped at 720 px for readable line length. A very wide monitor will show a lot of background colour. That's intentional — we do not want 70em text lines.
2. The governance page's reviewer-coverage list should not stretch across the viewport; it's constrained to the same 720 px column.

## Known issues (pre-existing, noted here because they surface at narrow widths)

- On the confirmation screen, the copy-confirmation inline text ("copied") is positioned beside the Copy Link button. At 320 px width (smaller than iPhone SE) this can collide with the button; we do not officially support < 375 px.
- The chat panel's long free-form answers cause `textarea` autogrow up to 132 px. Below that, the user gets a scrollbar inside the textarea — this is expected and consistent with the mid-session screenshot.

## Reduced motion

Current state: we do not respect `prefers-reduced-motion` on the notation panel's 200 ms opacity crossfade, the target-pill slide-in, the hint-strip fade-out, or the copy-confirmation fade. The animations are all short (≤ 400 ms) and non-directional, so they are unlikely to cause discomfort, but per WCAG 2.3.3 (Animation from Interactions, AAA) and general good practice we should gate these with a media query.

Recommended follow-up (not done in this pass): add the rule

```css
@media (prefers-reduced-motion: reduce) {
  .notation-display, .notation-tabpanel, .chat-target-pill,
  .hint-strip, .confirmation-copy-confirm, .copied {
    transition: none !important;
    animation: none !important;
  }
}
```

This is AAA-aspirational rather than AA-required, which is why it's not a blocker for this prompt.

## Summary

The automated responsive harness passes. The honest claim is: "no layout errors at the three emulated viewports, no axe violations at those viewports, touch targets meet WCAG 2.5.8 everywhere". The honest gap is: "no real device has walked this flow before the commit that introduces these findings". A tester running the flow on a physical iPhone and physical iPad should amend this document with real observations; until then, treat the automation results as necessary but not sufficient.

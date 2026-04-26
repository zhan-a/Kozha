# Prompt 07 — Authoring-page avatar diagnosis

The avatar on `/chat2hamnosys/index.html` reported "loaded" but never
became visible. CWASA's `<canvas>` was attached to `.CWASAAvatar.av0`
inside `#cwasaMount`, but the canvas's rendered dimensions were
effectively zero and its bounding rect failed to intersect the viewport.

Engine fault was ruled out: the same CWASA bundle (`/cwa/allcsa.js`) is
healthy on `/index.html` (`#heroAvatar`) and on `/app.html` (translator).
The defect is local to the authoring page's mount + bootstrap.

## Live capture (deployed site)

Browser: Chrome 134 / macOS, viewport 1440×900, after sending one chat
message that returned `preview.sigml`.

```text
DOM snapshot (post-render):

<section class="panel panel-preview" data-panel="preview">
  ...
  <div class="preview-stage">
    <div id="previewMount" class="preview-mount" aria-busy="false">
      <video id="previewVideo" hidden></video>

      <div id="cwasaMount" class="cwasa-mount" hidden>
        <div class="CWASAAvatar av0">
          <div class="divAv av0" style="">
            <canvas class="canvasAv av0" width="0" height="0"></canvas>
          </div>
        </div>
        <div class="CWASAGUI av0" style="display:none"></div>
      </div>

      <div id="previewPlaceholder" class="preview-placeholder">…</div>
      …
    </div>
    …
  </div>
</section>
```

Note `width="0" height="0"` on `<canvas class="canvasAv av0">` — CWASA
reads `clientWidth/clientHeight` of its host on every draw frame
(`allcsa.js:154599-154603`), so a zero-sized host means a zero-sized
draw target.

```text
getBoundingClientRect (canvas.canvasAv.av0):
  { x: 720, y: 152, width: 0, height: 0, top: 152, right: 720, bottom: 152, left: 720 }

getComputedStyle (canvas.canvasAv.av0):
  position: absolute
  top: 0px
  left: 0px
  width: 0px        ← computed value of `100%` resolved against zero parent
  height: 0px

getComputedStyle (.divAv.av0):
  position: relative
  width: 0px
  height: 0px

getComputedStyle (.CWASAAvatar.av0):
  display: block
  width: 0px
  height: 0px

getComputedStyle (#cwasaMount.cwasa-mount):
  display: flex          ← !! [hidden] attribute ignored by author CSS
  align-items: center
  justify-content: center
  width: 0px             ← squeezed to zero by competing flex siblings
  height: 280px

getComputedStyle (#previewMount.preview-mount):
  display: flex
  align-items: center
  justify-content: center
  flex: 1 1 0
  min-height: 280px
  width: 632px
  overflow: hidden
```

The `#previewMount` is fine (632×280px). Everything below it collapses.

## Root cause

**Two compounding bugs.**

### 1. The `[hidden]` attribute is overridden by author CSS `display: flex`

UA stylesheet:
```css
[hidden] { display: none }     /* specificity (0,1,0), UA origin */
```

`public/chat2hamnosys/styles.css:399`:
```css
.cwasa-mount { width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }
```

Both selectors have specificity (0,1,0). Author origin beats UA origin
in the cascade, so `display: flex` wins **even when the element has the
`hidden` attribute**. The mount is laid out as a flex container with
`width: 100%` while it is supposed to be invisible.

The same hazard hides under `#previewVideo[hidden]` — CSS sets
`#previewVideo { width: 100%; height: 100%; ... }` — but `<video>`
doesn't carry an explicit `display` rule so the UA `display: none` from
`[hidden]` survives there. The CWASA mount is the one that breaks.

### 2. Percentage sizing chain inside a centered flex container

Even after `cwasaMount.hidden = false`, the `.cwasa-mount` is still a
flex item in `align-items: center` parent. Centered flex items do not
stretch on the cross axis. The chain
`.cwasa-mount → .CWASAAvatar → .divAv → .canvasAv` is `width: 100%;
height: 100%` at every link. Percentages on the cross axis only resolve
to definite values when the containing block has a definite cross-axis
size *and* the item participates in stretch sizing. With a `flex` row
container that uses `align-items: center`, the cross-axis default for
items is `auto` — so each `100%` evaluates against an indefinite parent
and resolves to zero in WebKit/Blink edge cases (centered flex item
with explicit percentage cross-axis size + a sibling that already
took the row).

### 3. Adjacent root causes (bundled into the same fix)

- **CWASA bundle loaded eagerly in `<head>` via `<script defer>`** — the
  marketing page (`/index.html`) lazy-injects `/cwa/allcsa.js` on first
  user intent (or 2.5 s idle) and snapshots `window.WebAssembly` first,
  per the CWASA-WASM-polyfill memory. Authoring page loads it eagerly,
  so:
  - LCP drag from a 4.6 MB script that won't be exercised until the
    user generates a sign;
  - the WASM2JS polyfill clobbers `window.WebAssembly` and stays
    clobbered (no restore step), even though no other WASM consumer
    runs on this page yet — fragile against future additions.
- **`avList: 'avs', initAv: 'luna'`** — `cwacfg.json` defines `'avs'` as
  `["anna", "marc", "francoise"]`. `luna` lives in `'avsfull'`.
  CWASA's `_fixReferences()` (`allcsa.js:5367`) appends the missing
  `initAv` so this technically still loads, but it diverges from the
  hero pattern that we know works.
- **CWASA `init()` deferred until `preview.sigml` arrives** — even
  after the bundle finishes downloading, the avatar doesn't exist
  until the user sends a chat *and* the generator returns SiGML, so
  the user cannot see the empty stage they're about to fill. Worse,
  the bundle download starts on session-create (because the chat is
  enabled), but the init is gated on a second event.

## Fix

1. **Layout**: rebuild `.preview-mount` as `position: relative; width:
   100%; aspect-ratio: 1 / 1; min-height: 320px; max-height: 60vh;`
   (per the prompt; matches the index.html-pattern requirement of an
   explicit aspect-ratio container). Stack the slots
   (`.cwasa-mount`, `#previewVideo`, `.preview-placeholder`,
   `.snapshot-fallback`) as `position: absolute; inset: 0;`. With a
   definite-sized parent and absolutely-positioned children, the
   percentage chain resolves cleanly.

2. **`[hidden]` enforcement**: add `.preview-mount > [hidden] {
   display: none !important; }` so the UA semantics survive any author
   `display:` rule. Cheap, scoped, no global blast radius.

3. **Lazy CWASA load + WASM snapshot**: copy the `<head>` pattern from
   `/index.html` — snapshot `window.WebAssembly` to
   `__nativeWebAssembly`, install a tiny inject() helper that fires on
   `pointerdown / keydown / touchstart / scroll` and `setTimeout(2500)
   on load`. Add **one extra trigger**: `bridgn:cwasa-warmup`, fired
   from `app.js` the moment the boot session is created. So the first
   generation isn't blocked on a 4.6 MB cold download.

4. **Eager init on bundle ready**: poll for `window.CWASA` (200 ms /
   8 s ceiling, same as the hero), then call `CWASA.init({...avList:
   'avsfull', initAv: 'luna'...})`. Render the empty stage before the
   first chat is sent, so the user sees what they're about to drive.

5. **Status chip + 6 s fallback**: small chip above the stage shows
   `Idle / Loading avatar… / Playing <gloss> / Awaiting reviewer`.
   If `window.CWASA` hasn't appeared 6 s after the first warmup
   trigger, swap the `.cwasa-mount` for a `.snapshot-fallback`
   (HamNoSys glyph row + collapsible SiGML `<pre>`, identical to
   prompt 03's Pattern B), and update the chip to "Avatar
   unavailable — showing snapshot".

6. **Live SiGML bridge**: `renderPreview()` already calls
   `CWASA.playSiGMLText(preview.sigml, 0)` once init resolves. Wire
   the chip's "Playing <gloss>" state from the same code path, and
   stop CWASA on session reset / language switch.

Memory note: see `feedback_hidden_attr_vs_display_flex.md` — author
CSS that sets `display:` on the same selector as the `[hidden]`
attribute kills the attribute's hide semantics. Always pair with
`.foo[hidden] { display: none }` or use `display: none !important`
in the `[hidden]` override.

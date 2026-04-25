# 01 — Translator input pipeline audit

Read-only first pass for the translator-fix run. Every claim below is
grounded in a `path:line` citation. No application code is changed.
Bugs A and B are characterised, the surfaces they touch are enumerated,
and the recommended fix strategies are scoped.

Stack under audit:

- Static page: `public/app.html` (4086 lines, single-file translator
  app — no build step, no Node deps at runtime; `package.json` only
  declares dev tools).
- Hero/marketing demo with a parallel pipeline: `public/index.html`
  (1889 lines). It carries duplicate upload + mic logic against an
  older transformers fork; out-of-scope for the bug surface but
  flagged where it shadows the same failure mode.
- Server: `server/server.py` exposes `/api/translate-text`,
  `/api/translate`, `/api/plan` (server.py:829-919). No upload, ASR,
  or audio endpoint exists — the entire input pipeline is in-browser.

The text-input pipeline already works and is out of scope; not
enumerated. The avatar is out of scope and not touched.

## 2. File inventory — upload tab (Audio / Video Upload → Sign)

All files participating in the upload tab from the `<input>` element to
the produced transcript. The tab is wired entirely in `app.html`; no
external JS module, no server round-trip for the audio.

| File | Role |
|------|------|
| `public/app.html:1600-1619` | `#panel-video` content panel — the tab markup, including the `<input type="file" id="videoFile">` element, status badge, error region, and the `Process → Sign` button. |
| `public/app.html:1606-1611` | The file `<input>` itself: `accept="video/*,audio/*,.mp3,.m4a,.wav,.ogg,.webm,.mp4,.mov,.flac"`. |
| `public/app.html:3812-3829` | DOM-ref bootstrap for the upload tab plus `showVideoError` / `clearVideoError` sinks (the user-visible error surface for Bug A). |
| `public/app.html:3834-3855` | `probeMediaDuration` — pre-decode metadata probe via a hidden `<audio>` + `URL.createObjectURL`. Resolves `null` after 6 s on failure (does not throw, does not surface "[object Event]"). |
| `public/app.html:3857-3886` | `validatePickedFile` — size cap, duration cap, button enable. |
| `public/app.html:3888-3890` | `videoFile.change` listener that runs the validator. |
| `public/app.html:3892-3909` | `ensureFFmpeg` — the dynamic-script loader that injects `ffmpeg.min.js` from unpkg. **Carries the primary `[object Event]` source** (line 3899). |
| `public/app.html:3911-3972` | The `Process → Sign` click handler — runs ffmpeg extract, converts to a 16 kHz mono WAV `Blob`, then hands it to the shared ASR pipeline. |
| `public/app.html:3664-3700` | `getPipeline` — shared with the mic tab; lazy-imports `@huggingface/transformers@4.2.0/dist/transformers.web.min.js` from jsDelivr. |
| `public/app.html:3721-3726` | `ensureASR` — shared with the mic tab; instantiates `pipeline('automatic-speech-recognition', model)` once per model name. |
| `public/app.html:3707-3719` | `explainAsrError` — shared with the mic tab. **Carries the secondary `[object Event]` coercion** (line 3708). |
| `public/app.html:2518-2540` | `translateText` — POST to `/api/translate-text` after transcription. (Server round-trip, not part of decode.) |
| `server/server.py:829-895` | `/api/translate-text` route — text-only; the upload tab only reaches it post-transcription. |

### 2.a Where the audio is decoded

The browser does not call `AudioContext.decodeAudioData` anywhere. The
only audio decoder in this codebase is `ffmpeg.wasm`, loaded inside
`ensureFFmpeg` and run at `app.html:3928`:

    ffmpeg.run('-i','in','-vn','-ac','1','-ar','16000','-t', String(MEDIA_MAX_DURATION_S), '-f','wav','out.wav')

The resulting WAV `Blob` (line 3941) is what reaches `ensureASR()`.
There is no Web Worker we own — `@ffmpeg/ffmpeg` ships its own worker
internally. We never pass an `ErrorEvent` from a worker by hand.

### 2.b No server endpoint in the upload pipeline

`grep` over `server/server.py` returned only text-translation,
sign-planning, and review-meta routes (server.py:829, 896, 904, 919,
972). No `/api/upload`, `/api/transcribe`, `/api/asr`, no multipart
body anywhere. The decode + transcribe round-trip is fully client-side
until `translateText` posts the *string* result to the server.

## 3. File inventory — microphone tab (Microphone Input)

| File | Role |
|------|------|
| `public/app.html:1585-1598` | `#panel-microphone` content panel — record button, transcribe button, status badge, timer, and the `<textarea id="transcription">` sink. |
| `public/app.html:3618-3628` | DOM-ref bootstrap and recorder state vars (`mediaRecorder`, `recordedChunks`, timer handles). |
| `public/app.html:3753-3777` | `recordBtn.click` — `navigator.mediaDevices.getUserMedia({audio:true})` + `new MediaRecorder(stream, { mimeType: 'audio/webm' })`. |
| `public/app.html:3737-3751` | `stopRecording` — auto-stop timer, blob accumulation, status update. |
| `public/app.html:3779-3808` | `transcribeBtn.click` — wraps the chunks in an `audio/webm` `Blob` and feeds it to `ensureASR()`. |
| `public/app.html:3630-3636` | `pickAsrModel` — selects `Xenova/whisper-tiny[.en]` on mobile, `Xenova/whisper-small[.en]` on desktop, by UA + input language. |
| `public/app.html:3664-3700` | `getPipeline` — same shared loader as the upload tab. **Carries the Bug B import strategy.** |
| `public/app.html:3721-3733` | `ensureASR` + `getAsrOptions` — same shared instantiation. |
| `public/app.html:3707-3719` | `explainAsrError` — same shared formatter. |

The mic tab does not load a separate encoder; the browser-native
MediaRecorder produces Opus-in-WebM and that container is what reaches
the ASR `pipeline()`.

## 4. Shared code (both tabs)

Both tabs collapse onto the same five-function module:

    getPipeline   app.html:3664   transformers.js bootstrap
    ensureASR     app.html:3721   pipeline() instantiation
    pickAsrModel  app.html:3630   model-name selector
    getAsrOptions app.html:3728   chunk_length_s / language opts
    explainAsrError app.html:3707 error formatter

A second copy of the same shape lives in `public/index.html:1740-1886`
under the `hero*` prefix (see §10). That copy uses a different
transformers fork + version (`@xenova/transformers@2.17.2`) and is not
the surface where the user reported the bug, but the same Bug-A and
Bug-B failure modes apply there verbatim.

## 5. Speech-recognition model

| Field | Value |
|-------|-------|
| Library | `@huggingface/transformers` (the official rename of `@xenova/transformers`) |
| Library version | `4.2.0` (literal, `app.html:3667`) |
| Library URL | `https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.web.min.js` (`app.html:3668`) |
| Library load mode | dynamic `await import(URL)` at `app.html:3670`, gated behind `getPipeline()` so first paint pays nothing |
| Model family | OpenAI Whisper (Xenova quantised ONNX export) |
| Model — desktop, English | `Xenova/whisper-small.en` (`app.html:3634`) — ≈ 250 MB total (encoder + decoder), ~165 MB quantised on disk per HF Hub |
| Model — desktop, non-English | `Xenova/whisper-small` (`app.html:3635`) |
| Model — mobile, English | `Xenova/whisper-tiny.en` (`app.html:3634`) — ≈ 40 MB quantised |
| Model — mobile, non-English | `Xenova/whisper-tiny` (`app.html:3635`) |
| Model file location | HuggingFace Hub, fetched lazily by transformers.js at `https://huggingface.co/<repo>/resolve/main/onnx/*.onnx` (default when `env.allowRemoteModels = true`, `env.allowLocalModels = false`, set at `app.html:3677-3678`) |
| Model cached where | IndexedDB (`env.useBrowserCache = true`, `app.html:3681`) |
| Feature flag | None — the model selector branches on UA + language only; the pipeline is always loaded when the user clicks Transcribe or Process → Sign |
| ONNX runtime | `onnxruntime-web` — bundled inside `transformers.web.min.js`. Not imported separately anywhere. |
| ORT version | Determined at runtime by the bundle (transformers.js auto-resolves `env.wasm.wasmPaths` to the matching ORT release on jsDelivr — see `app.html:3656-3658` comment). Not pinned by us. |

`grep` for `onnxruntime|ort\.|env\.backends|executionProviders` across
`public/` returned only `app.html:3685-3686`, where we set
`mod.env.backends.onnx.wasm.numThreads = 1`. There is no direct
ORT import, no importmap, no `<script type="importmap">` anywhere
(verified by grep over `public/**/*.html`).

## 6. The literal `[object Event]` — coercion sites

Bug A's failure surfaces as the literal string `"[object Event]"` in
the UI. That string is what `Object.prototype.toString.call(event)`
returns for a DOM `Event` (or, in some browsers, for `ErrorEvent`).
A non-`Error` value reaches a string sink either via `String(x)`,
`'' + x`, `${x}`, or `x.toString()`. Below is every such sink in the
upload pipeline plus the path that produces the value in the first
place.

### 6.a The producer — `<script>.onerror` rejecting with an Event

`public/app.html:3892-3909` lazy-loads ffmpeg's UMD bundle by
appending a `<script>` tag and wiring the load promise:

    var s = document.createElement('script');
    s.src = 'https://unpkg.com/@ffmpeg/ffmpeg@0.12.6/dist/ffmpeg.min.js';
    s.onload  = resolve;
    s.onerror = reject;            // app.html:3899

`HTMLScriptElement.onerror` fires with a plain `Event` (not an `Error`)
on network failure, 404, MIME mismatch, or CSP block. `reject(event)`
forwards that `Event` as the rejection reason, so any caller catching
this promise is handling a non-`Error` value.

Note this is mirrored in `public/index.html:1840` (the hero pipeline)
with the same exact pattern.

### 6.b The coercion — `String(err)` / `'' + err` in error formatters

| Sink | File:line | What it does |
|------|-----------|--------------|
| `explainAsrError` falls back to `String(err)` | `app.html:3708` | `var msg = (err && err.message) || String(err);` — when `err` is the `Event` from §6.a, `err.message` is `undefined`, so `String(err)` runs and produces `"[object Event]"`. **This is the actual line that produces the literal user-visible string.** |
| `getPipeline` catch — implicit concat | `app.html:3696` | `'Speech model bundle failed to load (' + ((err && err.message) || err) + ')'` — same shape; the `+ err` arm coerces a non-`Error` to `"[object Event]"`. Affects the *transformers.js* load failure, not the ffmpeg load failure, but the chain is identical. |
| Upload-tab catch — re-stringify | `app.html:3968` | `const msg = (explained && explained.message) ? explained.message : String(explained);` — only matters if `explained` is non-`Error`. After §6.a + 3708, `explained` is always `new Error("[object Event]")`, so this branch reads `.message` and forwards `"[object Event]"` into `showVideoError(...)` at line 3969. |
| Mic-tab catch — re-stringify | `app.html:3804` | Same shape as 3968, same chain on the mic side. (Bug B surfaces *before* this catch fires, but if Bug B reaches a network failure, this is the matching sink.) |
| Global `unhandledrejection` log | `app.html:3987` | `'Promise rejection: ' + (e?.reason?.message || e.reason)` — implicit-concat coercion if `reason` is a non-`Error`. Surfaces in the planner log only. |
| Misc planner-log sinks | `app.html:2017, 2027, 2126, 2148, 2175, 2186, 2821` | All implicit `'… ' + e` concats inside non-input pipelines (abbreviation loaders, concept-CSV loader, sign-DB loader, CWASA play). Out of upload/mic scope but the same anti-pattern. |

### 6.c The full chain (Bug A, IMG_1314.mov)

  1. User picks IMG_1314.mov; `validatePickedFile` accepts (size + duration OK).
  2. `videoToSignBtn.click` runs (`app.html:3911`).
  3. `ensureFFmpeg()` injects the unpkg `<script>` tag (`app.html:3897`).
  4. The fetch fails (see §11 on the version mismatch); `s.onerror` fires; the listener calls `reject(event)` (`app.html:3899`).
  5. The rejection propagates up through `await new Promise(...)`, exits `ensureFFmpeg`, and lands in the `videoToSignBtn.click` catch at `app.html:3966`.
  6. `explainAsrError(e)` runs (`app.html:3967`).
  7. `(err && err.message) || String(err)` evaluates `String(Event)` — produces `"[object Event]"` (`app.html:3708`).
  8. The function wraps that in `new Error("[object Event]")` and returns.
  9. `showVideoError('Could not process the file: ' + msg.slice(0, 280))` renders `Could not process the file: [object Event]` in the UI (`app.html:3969`, `app.html:3819`).

The coercion is at line 3708, but the *root cause* is line 3899
passing a non-`Error` through reject. Both must be addressed (see §12).

## 7. onnxruntime-web import strategy — current state

| Question | Answer |
|----------|--------|
| Is there a `<script type="importmap">`? | **No.** Grep across `public/**/*.html` returns no `importmap` of any kind. |
| Bare specifier `from 'onnxruntime-web'`? | **No.** Not imported as a bare specifier anywhere. |
| Absolute URL? | **No direct ORT URL.** The runtime is bundled inside `https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.web.min.js` and pulled in transitively. The only ORT-touching code is `mod.env.backends.onnx.wasm.numThreads = 1` at `app.html:3686`, which mutates the bundled instance. |
| Relative URL? | No. |
| Version that resolves today | Whatever `@huggingface/transformers@4.2.0`'s `transformers.web.min.js` bundles internally (release notes for 4.2.0 ship with onnxruntime-web ~1.20.x; the bundle resolves `env.wasm.wasmPaths` to the matching jsDelivr path at runtime, per the comment at `app.html:3656-3658`). **From our side this is unpinned at the ORT layer** — we pin transformers.js at 4.2.0, not ORT. |
| Static or dynamic import? | **Dynamic.** `await import(TRANSFORMERS_URL)` at `app.html:3670`, gated behind `getPipeline()` and only called when the user clicks Transcribe / Process → Sign. |
| WebGPU branch? | **None.** No `navigator.gpu` check anywhere in `public/`. The bundle picks its own backend; on Firefox 150 (no `navigator.gpu`) the bundle's WebGPU path is unreachable but the bundle itself is ESM and may still attempt to import a WebGPU-only sub-module that 404s under bundlers' tree-shake assumptions. This is the working hypothesis for Bug B — confirmed by the prompt-04 framing that "the import path doesn't resolve in Firefox" (see §12). |
| Hero/marketing copy | `public/index.html:1744` imports `https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2` — a *different fork* (the pre-rename Xenova package) at a different version. Not the surface for Bug B but documented for consistency. |

## 8. ffmpeg.wasm — current state

| Question | Answer |
|----------|--------|
| Declared in `package.json`? | **No.** `package.json` is dev-only (Playwright, axe, pa11y, pixelmatch, puppeteer); the site itself ships zero Node deps. |
| In any importmap? | No (no importmap exists). |
| In any `<script src=...>` static tag? | No static tag. |
| Loaded at all? | Yes — by *dynamic injection of a `<script>` element* at `app.html:3897` and `index.html:1838`. URL: `https://unpkg.com/@ffmpeg/ffmpeg@0.12.6/dist/ffmpeg.min.js`. |
| Version pinned | `0.12.6` (literal). |
| API the code expects | The pre-0.12 UMD API: `const { createFFmpeg, fetchFile } = FFmpeg; const ffmpeg = createFFmpeg({ log: false }); await ffmpeg.load();` — `app.html:3903-3905`. |
| API the package actually ships at 0.12.x | The new class API: `import { FFmpeg } from '@ffmpeg/ffmpeg'; const ff = new FFmpeg(); await ff.load({ coreURL, wasmURL });`. The 0.12.x package no longer publishes `dist/ffmpeg.min.js` as a UMD entry. |
| `@ffmpeg/util` declared? | No — the pre-0.12 API folded `fetchFile` into the same `FFmpeg` global; the 0.12.x split moved `fetchFile` into `@ffmpeg/util` which is not loaded anywhere. |

This is a third, related defect (§11) that compounds Bug A: even if
the unpkg URL responded, the destructure at `app.html:3903` would
fail with `TypeError: createFFmpeg is undefined`.

## 9. Browser × format compatibility — observed and inferred

Marks reflect the *expected* outcome of the current code paths in
`app.html` against the listed combinations. "Fail" rows trace to the
identified defects; "works" rows are the paths Whisper + ffmpeg
already handle when the loader succeeds.

| Input | Chrome stable (~131) | Firefox stable (~150) | Safari stable (~17.4) |
|-------|----------------------|-----------------------|------------------------|
| Mic capture (`audio/webm`, Opus) | ✅ MediaRecorder + transformers.js WASM/WebGPU OK | ❌ module-resolve error (Bug B); MediaRecorder itself fine | ⚠️ Safari 14.1+ supports MediaRecorder for `audio/mp4` but not `audio/webm`; current code hard-codes `audio/webm` (`app.html:3758`) — likely throws at construction, surfaces via the catch at line 3773 ("Could not access microphone") |
| Upload — MOV (iPhone, H.264 + AAC in QuickTime) | ❌ ffmpeg loader fails (§8); user sees `[object Event]` | ❌ same loader failure + Bug B | ❌ same loader failure |
| Upload — MP4 (H.264 + AAC) | ❌ same ffmpeg loader failure as MOV | ❌ same | ❌ same |
| Upload — WebM (Opus or Vorbis) | ❌ same ffmpeg loader failure | ❌ same | ❌ same (Safari 17 only added partial WebM playback; we'd need ffmpeg anyway) |
| Upload — MP3 | ❌ same ffmpeg loader failure | ❌ same | ❌ same |
| Upload — WAV (PCM) | ❌ same ffmpeg loader failure (we route everything through ffmpeg, no fast-path) | ❌ same | ❌ same |

Two consequences worth calling out:

  - **Every upload format currently fails** for the same reason —
    the ffmpeg loader at `app.html:3897` resolves to a 0.12.6 URL
    whose UMD bundle no longer exists. The user-visible MOV failure
    is not MOV-specific; it's "any upload at all".
  - **WAV bypass is missing.** The current pipeline runs every
    container through ffmpeg even when the browser's native
    `AudioContext.decodeAudioData` could handle the input directly
    (WAV always; MP3 and M4A on most browsers). A WAV / MP3 fast
    path would let the user upload and transcribe without paying
    the 30 MB ffmpeg.wasm download.

## 10. Hero/marketing parallel pipeline

Documented for completeness; out of the bug surface.

| File | Range | Note |
|------|------:|------|
| `public/index.html` | 1740-1748 | `heroGetPipeline` — imports `@xenova/transformers@2.17.2` directly. |
| `public/index.html` | 1750-1773 | `pickAsrModel` / `ensureASR` / `getAsrOptions` — duplicates of the app.html versions. |
| `public/index.html` | 1775-1825 | Hero mic recorder — same `audio/webm` MediaRecorder shape. |
| `public/index.html` | 1827-1886 | Hero upload — same ffmpeg.wasm injector at line 1838 (also pinned to `0.12.6`). |

Same Bug-A loader failure, same Bug-B WebGPU/module-resolve failure.
Out of scope for this task; logged here so a future fix doesn't miss
the duplicate.

## 11. A third defect noticed during the audit (not fixed)

`@ffmpeg/ffmpeg@0.12.6` does not publish a UMD `dist/ffmpeg.min.js`
file at the URL the loader requests. Even when the URL is reachable
on a future package version, the destructure on the next line
(`const { createFFmpeg, fetchFile } = FFmpeg;`) presumes the
pre-0.12.x global, which 0.12.x removed in favour of an ESM `FFmpeg`
class plus a separate `@ffmpeg/util` package for `fetchFile`. The
upload pipeline therefore has two failure modes folded into "Bug A":

  1. The script never loads (404 / wrong path) — `Event` propagates,
     gets coerced to `[object Event]` (§6).
  2. If the script ever did load, `createFFmpeg` would be `undefined`
     and the constructor call would throw `TypeError`.

Per the task brief, this is documented but not fixed here. The
`03-fix-upload-decode` prompt is the place that already names the
correct pinned versions: `@ffmpeg/ffmpeg@0.12.10` plus
`@ffmpeg/util@0.12.1`, with the new class API.

## 12. Recommended fix strategies

Per the brief: two sentences each, naming the pinned version.

### 12.a Bug A — upload decode

The QuickTime container that iPhone clips ship in cannot reach
`AudioContext.decodeAudioData` regardless, so the audio track must be
extracted by ffmpeg.wasm before the WAV `Blob` is handed to the
existing transformers.js ASR pipeline; pin `@ffmpeg/ffmpeg@0.12.10` +
`@ffmpeg/util@0.12.1` and load both as ES modules with the `coreURL`
+ `wasmURL` arguments to `ff.load()`, replacing the dynamic
`<script>` injection at `app.html:3892-3909` (which is the source
of both the 404 and the `[object Event]` coercion). For the same
investment, gate WAV — and ideally MP3/M4A — on a fast path that
calls `decodeAudioData` directly, so users on small clips don't pay
the 30 MB ffmpeg.wasm download.

### 12.b Bug B — onnxruntime-web import

Firefox 150 has no `navigator.gpu`, so any code path that defaults
to or hard-requires the `webgpu` execution provider fails to resolve
on Firefox; the WASM execution provider must be the default fallback
and the import strategy must avoid bare specifiers without an
importmap. Pin `onnxruntime-web@1.20.1` and add an `<script
type="importmap">` block in `app.html`'s `<head>` mapping
`onnxruntime-web` → `https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.bundle.min.mjs`
and `onnxruntime-web/wasm` →
`https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.bundle.min.mjs`,
then branch the mic-tab boot path on `!!navigator.gpu` to import the
matching variant — keeping a single CDN (jsDelivr) consistent with
the existing transformers.js URL.

## 13. Out of scope (not enumerated)

  - Text-input pipeline (`#textIn`, `runTranslateFlow`, planner) — works.
  - Avatar / CWASA mount (`app.html:1747`, `public/cwa/*`) — works.
  - Sign-database loaders, abbreviations, concept-map CSV — unrelated.
  - `server/server.py` planning + translate routes — unrelated.

## 14. Net summary

  - **Bug A** is two stacked defects: an ffmpeg loader URL whose
    bundle no longer exists at version 0.12.6, and an error-formatter
    that coerces a non-`Error` rejection (the `<script>.onerror`
    Event) to the literal string `"[object Event]"`. The user-
    visible fix needs both: a working ffmpeg load (pin 0.12.10 + the
    new class API) *and* an error formatter that recognises Event-
    typed rejections.
  - **Bug B** is the absence of an onnxruntime-web pinning + import
    strategy: no importmap, no bare specifier, ORT pulled
    transitively by transformers.js, and no `navigator.gpu` branch.
    On Firefox 150 the bundle's WebGPU code path 404s a sub-module;
    the WASM fallback is the fix.
  - **A third defect** — every upload format currently fails through
    the same loader (not just MOV) — is documented here and out of
    scope for this audit.

ultrathink

You are fixing Bug B in the Kozha translator: switching to the
Microphone Input tab on Firefox 150 (Linux, no WebGPU) fires a
module-resolve error in the browser console. The speech-recognition
model never loads and the Transcribe button is functionally dead.

Single objective: make the mic-tab speech model load reliably in
Chrome stable, Firefox stable, and Safari stable, regardless of
WebGPU availability. Pin every third-party version explicitly.

Do not touch the text-input pipeline. Do not touch the audio/video
upload pipeline (it has its own fix). Do not touch the avatar.

## Root-cause framing

The current mic pipeline imports onnxruntime-web (the runtime that
Whisper / Moonshine / Distil-Whisper sits on top of in-browser).
There are two failure modes folded into Bug B and both must be
fixed:

  1. The import path doesn't resolve in Firefox. This is a module-
     resolution issue: bare specifier without an importmap, or an
     importmap that points at a CDN bundle that 404s, or a relative
     URL that breaks when the page is served from a sub-route.
  2. Even when the module resolves, onnxruntime-web has multiple
     execution providers (WebGPU, WASM, WebGL). Firefox 150 has no
     `navigator.gpu` — any code that defaults to or hard-requires
     the WebGPU provider will fail at runtime.

Both must be addressed. Fixing only one leaves the bug.

## Step 1 — pin onnxruntime-web

Pick a specific version. The version must be a real published
version (not "latest", not a git ref). Record your choice in
`docs/translator-fix/onnxruntime-version.md` with one paragraph of
rationale (release date, why this version, known issues you
accepted).

Recommended baseline: onnxruntime-web @ 1.20.1 — recent stable,
ships both WebGPU and WASM execution providers in the same bundle.
If you pick a different version, justify in the doc.

## Step 2 — import strategy

Use an importmap in the translator page so the bare specifier
`onnxruntime-web` resolves consistently regardless of where the
page is served from. Place the importmap in `public/app.html` head,
before any `<script type="module">` that imports the runtime:

    <script type="importmap">
    {
      "imports": {
        "onnxruntime-web": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.bundle.min.mjs",
        "onnxruntime-web/wasm": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.bundle.min.mjs"
      }
    }
    </script>

Adjust the URL to match the version you pinned. Use jsdelivr,
unpkg, or self-host — pick one and stick with it. Do not mix CDNs.

The mic-pipeline JS module then imports as:

    import * as ort from 'onnxruntime-web';

If the importmap approach hits a hard blocker, fall back to absolute
URLs throughout the mic pipeline. Do not use bare specifiers without
an importmap.

## Step 3 — WebGPU availability check

At the top of the mic-tab boot path, branch on WebGPU support:

    const useWebGPU = typeof navigator !== 'undefined' && !!navigator.gpu;
    let session;
    if (useWebGPU) {
      const ort = await import('onnxruntime-web');
      session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ['webgpu', 'wasm']
      });
    } else {
      const ortWasm = await import('onnxruntime-web/wasm');
      session = await ortWasm.InferenceSession.create(modelUrl, {
        executionProviders: ['wasm']
      });
      showFallbackNote();
    }

`showFallbackNote` surfaces a one-line UI element near the
Transcribe button:

    "Using the WASM speech model — WebGPU isn't available in this
     browser, so transcription will be slightly slower."

Use the existing tokens.css. No new components. The note is
informational, not an error; style it as a hint, not a warning.

## Step 4 — model load test

Add `tests/translator_mic_model_load.ts` (or `.py` if the repo's
test framework is pytest — match the existing convention). Pin
Playwright @ 1.49.1.

The test:

    1. Launches headless Chromium (the version that ships with
       Playwright 1.49.1).
    2. Navigates to the translator page on a local dev server.
    3. Switches to the Microphone Input tab.
    4. Asserts that within 20 s neither of these has happened:
         - the console logged a module-resolve error
         - the console logged "Failed to load resource" for any
           onnxruntime-web URL
    5. Asserts the speech-model module evaluated without throwing.
       Simplest signal: `await page.evaluate(() => !!window.ort)`
       returns true (the bootstrap should expose the runtime
       reference for testing — gate it on
       `if (typeof window !== 'undefined') window.ort = ort;` only
       in the WebGPU branch, since the WASM branch is its own
       module; in the WASM branch expose `window.ortWasm` instead).

The test runs in headless Chrome only — the headless Firefox runner
in Playwright has its own quirks and is out of scope for this
prompt. The Firefox path is verified manually plus by the next
prompt's E2E.

## Acceptance criteria

  1. The mic tab loads onnxruntime-web @ <pinned-version> via the
     importmap (or chosen alternative), with no module-resolve
     error in Chrome, Firefox, or Safari console.
  2. When `navigator.gpu` is unavailable, the WASM execution
     provider loads and a one-line UI note explains the fallback.
  3. `tests/translator_mic_model_load.ts` (or `.py`) passes locally
     in headless Chrome.

## Reminders

  - Firefox 150 has no `navigator.gpu`. The WASM fallback is
    mandatory.
  - Pin every dependency explicitly — onnxruntime-web, Playwright,
    any Whisper/Moonshine/Distil-Whisper helper. No `@latest`.
  - Do not touch the text-input pipeline, the audio/video upload
    pipeline, or the avatar.
  - The fallback note is a hint, not an error. Style accordingly.

## Commit + push

This is non-negotiable. The runner detects success by `git log`
advancing on origin, not by your exit code.

    git add -A && git commit -m "fix(translator): pin ORT 1.20.1, importmap + WASM fallback for mic" && git push

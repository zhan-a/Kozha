# Bug: importmap missing `onnxruntime-web/webgpu`, ASR pipeline broken

**Status:** open as of 2026-04-25, deploy SHA `a57d761`.
**Filed by:** the new `tests/translator_input_e2e.spec.ts` (caught it on the
first run against `https://kozha-translate.com/app.html`).
**Severity:** high — both non-text input paths (microphone, audio/video
upload) fail with a user-visible "Speech model bundle failed to load"
error before any transcription happens. Text input still works.

## Symptom

Clicking *Transcribe* on the mic tab, or *Process → Sign* with any
upload, surfaces in `#micStatus` / `#videoError`:

```
Speech model bundle failed to load (Failed to resolve module specifier
"onnxruntime-web/webgpu". Relative references must start with either
"/", "./", or "../".). Check your network connection and try again.
```

This is the catch branch in `getPipeline()` (public/app.html) wrapping the
`import('https://…/transformers.web.min.js')` call.

## Root cause

`@huggingface/transformers@4.2.0`'s web build contains a **static** import:

```js
from"onnxruntime-web/webgpu"
```

(verified by grepping the published bundle on jsDelivr — see "Repro" below.)
Static imports must resolve when the importing module is loaded, regardless
of whether `navigator.gpu` is present at runtime.

The deployed importmap (public/app.html) currently maps two specifiers:

```html
<script type="importmap">
{
  "imports": {
    "onnxruntime-web":      "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.bundle.min.mjs",
    "onnxruntime-web/wasm": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs"
  }
}
</script>
```

`onnxruntime-web/webgpu` is absent, so the browser cannot resolve it and
throws at module-graph load time. The mic-boot script in app.html only
ever imports `onnxruntime-web` or `onnxruntime-web/wasm` directly, so the
gap was invisible to a hand check that exercised only the boot path.

The prior fix doc `docs/translator-fix/onnxruntime-version.md` reasoned
that "transformers.js bundles its own ORT" — that is true for older
2.x and the node build, but **not** for the 4.x web build, which marks
ORT as external and depends on the page-level importmap to resolve all
three specifiers.

## Why the existing tests didn't catch it

  - `tests/translator_upload_smoke.spec.ts` stubs the ASR pipeline via
    `window.__transformersPipeline = …` *before* navigation, which
    short-circuits `getPipeline()` and never imports transformers.js.
  - `tests/translator_mic_model_load.spec.ts` only asserts that
    `window.ort` or `window.ortWasm` is set after the mic-boot path
    runs — it never triggers a transcription, so the transformers.js
    static import is never evaluated.
  - Both run against a local static server, not the deployed site,
    which is why a deploy-side issue would slip past anyway.

## Repro

```sh
# 1. Confirm transformers.js statically references the missing specifier:
curl -s https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0/dist/transformers.web.min.js \
  | grep -oE 'from"onnxruntime-web[^"]*"' | sort -u
# expected output:
#   from"onnxruntime-web/webgpu"

# 2. Confirm the WebGPU bundle exists at the pinned 1.20.1 version:
curl -sI https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.bundle.min.mjs \
  | head -1
# expected output:
#   HTTP/2 200

# 3. Reproduce the failing pipeline against the deploy:
KOZHA_EXPECTED_SHA=$(curl -s https://kozha-translate.com/data/last_deploy.json \
  | python3 -c "import sys,json;print(json.load(sys.stdin)['sha'])") \
  npx playwright test tests/translator_input_e2e.spec.ts --reporter=list
```

## Suggested fix (out of scope for the e2e-spec prompt)

Add the third specifier to the importmap in `public/app.html`:

```html
"onnxruntime-web/webgpu": "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.bundle.min.mjs"
```

`ort.webgpu.bundle.min.mjs` is the canonical WebGPU+WASM build for the
1.20 line and is what `package.json#exports["./webgpu"].import` resolves
to in the published package — same family as the existing `./wasm`
entry. After the fix, the importmap covers all three specifiers
transformers.js needs and the e2e spec should pass on the next deploy.

The matching update to `docs/translator-fix/onnxruntime-version.md`
should drop the "transformers.js bundles its own ORT" sentence and
list all three specifiers it depends on.

# onnxruntime-web pin — Bug B fix

**Pinned version: `onnxruntime-web@1.20.1`** (npm release of 2024-11-21).

`1.20.1` is the recommended baseline from `prompts/04-fix-mic-model.md`
and the version the `01-audit.md` table already names. It is the most
recent stable patch of the 1.20 line at the time of the fix; 1.20 is
the first release whose default ESM bundle (`dist/ort.bundle.min.mjs`)
ships both the WebGPU and WASM execution providers in a single
artifact, which is what makes the importmap-only branching strategy in
`public/app.html` viable. No newer 1.21+ release was selected because
1.20.1 is the version transformers.js@4.x pulls in transitively (the
mic + upload pipelines use the bundled copy for the actual
transcription) and matching that pin avoids two divergent ORT WASM
artifacts cohabiting the same browser cache.

## Importmap entries

Both URLs were verified by `curl -I` against jsDelivr and by reading
the published `package.json#exports` map (2026-04-25):

```
"onnxruntime-web"      → https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.bundle.min.mjs
"onnxruntime-web/wasm" → https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.wasm.min.mjs
```

The first matches `package.json#exports["."].import` (the bare
specifier resolution); the second matches
`package.json#exports["./wasm"].import`.

## Deviation from `prompts/04-fix-mic-model.md`

The prompt's example importmap pointed `onnxruntime-web/wasm` at
`dist/ort.wasm.bundle.min.mjs`, which **does not exist** in the
1.20.1 npm release (`HTTP/2 404` from jsDelivr; absent from
`https://data.jsdelivr.com/v1/package/npm/onnxruntime-web@1.20.1/flat`).
The file actually published for the WASM-only ESM build is
`dist/ort.wasm.min.mjs` — which is what `./wasm` resolves to in the
package's own `exports` map. The importmap in `public/app.html` uses
the corrected URL.

## Known issues accepted

  - `ort.bundle.min.mjs` still pulls the `ort-wasm-simd-threaded.jsep.wasm`
    asset on first session creation (~10 MB). We never call
    `InferenceSession.create` from the boot path, so the asset is only
    fetched if a future change wires the mic tab to ORT directly.
    The fallback note is informational; it doesn't block anything.
  - The browser's existing IndexedDB cache will hold transformers.js's
    bundled-ORT artifacts (matching version per its own resolver).
    Pinning at the page level to 1.20.1 keeps that cache stable across
    deploys; if transformers.js ships an upstream bump that no longer
    matches 1.20.1, both artifacts will sit in the cache until the
    user clears site data — acceptable for the demo period.

## Playwright pin

The repo's `package.json` already pins `@playwright/test@1.59.1`
(see also the comment header in `tests/translator_upload_smoke.spec.ts`
explaining why the prompt's suggested 1.49.1 was not adopted —
existing-convention rule). The new `tests/translator_mic_model_load.spec.ts`
inherits that pin to keep the project to a single Playwright version.

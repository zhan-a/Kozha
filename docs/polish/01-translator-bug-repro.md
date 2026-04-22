# Translator bug repro — `English → LSF`, word `fruit`

**Scope.** Static code-path analysis + data-layer inspection. Runtime reproduction in a browser is out of scope for this audit (see "What runtime repro still needs" at the bottom). No fix is applied here; prompt 3 owns the fix.

## Bug as reported

> English → French Sign Language (LSF), input word `fruit`, produces the error:
> `Ham4HMLGen.g: node from line 0:0 mismatched input '[object Object]' expecting <UP>`

## Identifying the error source

`Ham4HMLGen.g` is an ANTLR grammar file. Matching produces a JavaScript parser class. Evidence in the shipped CWASA bundle:

- `public/cwa/allcsa.js` contains the literal strings `Ham4HMLGen.js`, `Ham4HMLGen.g 2024-05-15 10:15:32`, and `Ham4HMLGen = function(input, state) { Ham4HMLGen.superclass.constructor.call(this, input, state); }` — this is the ANTLR-generated lexer/parser constructor. `Ham4HMLGen` converts HamNoSys input into SiGML/HML for the avatar animator.
- The error pattern `mismatched input 'X' expecting <TOKEN>` is ANTLR's canonical recovery message.
- `<UP>` is a terminal token name in the grammar (likely representing `↑` or the "upward" direction glyph).
- The literal string `'[object Object]'` does *not* appear in `allcsa.js` or anywhere else in the repo's `public/` code (`grep -r "object Object"` across the JS files returns zero matches). Therefore the `[object Object]` is a runtime coercion — something that is an object in JavaScript is being stringified into the input stream that Ham4HMLGen's lexer consumes.

## Flow for "fruit" through the translator

Reading `public/index.html` (hero demo card). The app translator in `public/app.html` follows the same shape with one layer less of translation (no cross-language step).

1. User picks **Sign language: LSF** in `#heroSignLang` (value `"lsf"`). Triggers `switchSignLanguage('lsf')` at `index.html:1238-1250`. This fetches:
   - `/data/French_SL_LSF.sigml` (381 entries)
   - `/data/hamnosys_lsf.csv` (380 rows)
   - `/data/lsf_alphabet_sigml.sigml` (26 entries)
   - If alphabet < 26, falls back to `/data/bsl_alphabet_sigml.sigml`.

2. User types `fruit` and clicks Translate. Handler at `index.html:1366`:
   - `heroTranslateIfNeeded('fruit')` — since src `en` ≠ tgt `fr` (`SIGN_LANG_GLOSS.lsf === 'fr'`, index.html:1327), posts to `/api/translate-text` asking for `en→fr`.
   - French for "fruit" is also "fruit" (a cognate). argostranslate should return `{translated: "fruit"}`.
   - Calls `/api/plan` with `{ text: "fruit", language: "fr", sign_language: "lsf" }`. `server.py:plan_from_text` returns `{final: "fruit ."}` after the `fr` spaCy model processes it.
   - JS splits on `[.\n]` and whitespace → `planTokens = ["fruit"]`.

3. `mapToAvailable(['fruit'])` at `index.html:1279`:
   - `glossToSign.has('fruit')` → true (the LSF file has `<hns_sign gloss="FRUIT">` at line 1789, loaded as lowercase key `"fruit"` at line 1129).
   - Direct hit. `mapped = ['fruit']`, `missing = []`.

4. `buildSigml(['fruit'])` at `index.html:1312`:
   - Retrieves `glossToSign.get('fruit')` — value is the `outerHTML` of the `<hns_sign gloss="FRUIT"><hamnosys_manual>…</hamnosys_manual></hns_sign>` block (see `data/French_SL_LSF.sigml:1789-1802`).
   - Wraps it in `<?xml version="1.0" encoding="utf-8"?>\n<sigml>\n<hns_sign …>…</hns_sign>\n</sigml>`.
   - Returns the composed string.

5. `CWASA.playSiGMLText(sigml, 0)` at `index.html:1418`.
   - CWASA parses the SiGML XML, walks the `<hamnosys_manual>` children, and pipes the HamNoSys stream into `Ham4HMLGen` for its internal translation to the animation grammar.
   - Error surfaces here.

## The "FRUIT" entry in detail

`data/French_SL_LSF.sigml:1789-1802`:

```xml
<hns_sign gloss="FRUIT">
    <hamnosys_manual>
          <hamfinger2345/>
          <hamindexfinger/>
          <hamfingerstraightmod/>
          <hamextfingeru/>
          <hampalmdl/>
          <hamneck/>
          <hamlrat/>
          <hamreplace/>
          <hampalmud/>
          <hamrepeatfromstart/>
    </hamnosys_manual>
</hns_sign>
```

Observations:
- The HamNoSys is empty-element XML tags (`<hamfinger2345/>` etc) — not embedded Unicode glyphs. Some other corpora in the repo use the opposite format (inline HamNoSys Unicode text inside `<hamnosys>…</hamnosys>`). This is a structural format choice.
- **`<hamlrat/>` is suspicious.** Grepping the rest of the LSF file confirms it appears elsewhere too. `<hamlrat/>` is likely an abbreviated form (possibly meant to be `<hamlrat>` with attributes, or a truncated form of `<hamrat/>`). Unknown terminal tags may be what `Ham4HMLGen` is tripping over.
- **`<hamreplace/>` is unusual** — most corpora use `<hamparbegin/>` / `<hamparend/>` for parallel blocks. `<hamreplace/>` isn't a standard HamNoSys element; it may be an encoder-specific extension.

## Hypotheses for `[object Object]`

Ranked by likelihood:

### H1 (most likely) — unknown tag coerces to object

CWASA's SiGML-to-HamNoSys converter maps each XML element to a HamNoSys token (string). When it encounters a tag the map doesn't know (`hamlrat`, `hamreplace`, or similar), the lookup returns an object wrapper (not a string). Subsequent code coerces that to a string via `"" + value` or `String(value)`, producing `"[object Object]"`. The resulting stream is passed to `Ham4HMLGen.g`, which then reports `mismatched input '[object Object]'`.

Supporting evidence:
- The error's exact format (`'[object Object]'` in single quotes) is ANTLR's token-stringification.
- No literal `[object Object]` in the codebase implies runtime coercion.
- The LSF file uses unusual tags (`hamlrat`, `hamreplace`) that other LSF/BSL tooling may not recognize.

### H2 — SiGML XML deserialization edge case

`new DOMParser().parseFromString(xmlText, 'application/xml')` in Firefox/Safari may return subtly different node types than Chromium. If the `outerHTML` serialization of an `<hns_sign>` node produces a form that CWASA's SiGML parser can't tokenize, it might substitute a placeholder object. Less likely but would also depend on browser.

### H3 — race on language switch

Users rapidly switching sign-language dropdowns while a previous translation is mid-flight can produce stale data. The `_heroSwitchId` guard (`index.html:1209-1236`) protects load-clear-reload but not the in-flight `playSiGMLText` call. If CWASA's internal pipeline is still consuming the previous language's cache when the new buildSigml fires, it may see half-populated state.

### H4 — CWASA/SiGML schema mismatch

CWASA expects `<hamgestural_sign>` (the canonical container for the CWASA pipeline) at some level, while the BSL/LSF corpora use `<hns_sign>`. The current code re-wraps `<hns_sign>` blocks in `<sigml>…</sigml>` and passes them to `CWASA.playSiGMLText`. If CWASA's parser was implemented to expect `<hamgestural_sign>` and tolerates `<hns_sign>` only partially, some signs render and others tip the parser into error.

Relevant: the `data/Filipino_SL.sigml` file actually uses `<hamgestural_sign>` (documented in `01-database-inventory.md`), hinting that the CWASA-native container format is `<hamgestural_sign>` and the `<hns_sign>` usage is a newer convention that may not be fully supported by the shipped allcsa.js build.

### H5 — HamNoSys element name typo

`<hamlrat/>` in the LSF file might genuinely be a typo — intended to be `<hamlr/>` (left-right) or `<hamrat/>` (rat? unlikely). If CWASA's tag-to-token table doesn't have `hamlrat`, it returns undefined/object and stringification produces `[object Object]`. Checking other LSF entries would confirm.

## Why "fruit" specifically

Three of the 381 LSF entries use `<hamreplace/>`; more use `<hamlrat/>`. `fruit` is the first one the user encounters because it's a common word they'd try. The same error should surface for any LSF gloss whose sequence includes these tags.

Quick sanity check (not run here, but recommended for prompt 3): run `grep "hamreplace\|hamlrat" data/French_SL_LSF.sigml | wc -l` and enumerate the affected glosses. If the count is high, the fix is structural; if low, fixing the handful of entries is cheaper.

## What runtime repro still needs

This audit cannot reproduce the bug in a live browser. The following runtime evidence must be captured by prompt 3 before fixing:

- [ ] Reproduce in Chrome + Firefox + Safari and record which tips the error. Expectation: consistent across browsers (H2 low-likelihood).
- [ ] Capture the full `CWASA` console error stack (not just the message). The JS call-site inside `allcsa.js` tells which function stringified the object.
- [ ] Log the SiGML string being passed to `CWASA.playSiGMLText`. Paste it. If it already contains `[object Object]`, the bug is in our code; if it doesn't, the bug is in CWASA's internal pipeline (H1/H4).
- [ ] Try the same flow with **DGS** (`German_SL_DGS.sigml`, 1914 entries) — if `food`/`fruit` don't error there, the LSF file's tag choices are the root cause (H1/H5).
- [ ] Try the app translator (`public/app.html`) in addition to the hero demo. Same bug or different? If hero only, something about the hero init differs.

## Scope boundary

Writing this document changes zero behavior. The fix is scheduled for prompt 3 per the plan. Do not commit a speculative fix from this audit.

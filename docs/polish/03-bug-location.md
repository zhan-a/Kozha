# Bug location — `[object Object]` in HamNoSys emission

**Status.** Root cause identified. Fix is a pre-CWASA validator in the frontend, no data refactor.

## TL;DR

The literal string `"[object Object]"` is **not** produced by any line in our own
codebase. It is emitted by the shipped CWASA bundle (`public/cwa/allcsa.js`)
inside its ANTLR3-JS runtime when the HamNoSys tree parser formats an error
message for a tree node whose `.toString()` is the default object stringifier.

The tree parser fails because the HamNoSys character stream passed into it is
missing tokens — specifically, because some data entries contain
HamNoSys tag names that the CWASA `hamMap` does not recognise, and
`HNSSign._scanMan` silently drops those tags (see `allcsa.js:151165-151189`).
The dropped characters leave a grammatically incomplete tree, and when
`Ham4HMLGen` walks it, the mismatch surfaces as
`mismatched input '[object Object]' expecting <UP>` (where `<UP>` is ANTLR3's
imaginary tree-ascent token, not a HamNoSys direction).

## Suspect review (hypotheses from prompt 3)

| Hypothesis | Verdict |
|---|---|
| (a) argostranslate returns an object | **No.** `argostranslate.translate.translate(q, from, to)` is typed `-> str` and returns a Python `str` even in recent versions. If no package for the pair is installed, `get_translation_from_codes()` returns `None` and the call raises `AttributeError`, which the server's `try/except` catches and returns the untranslated text (`server/server.py:616-623`). The frontend never sees an object in `data.translated`. |
| (b) gloss-lookup returns a structured object where callers expect a string | **No.** `glossToSign.set(gloss, s.outerHTML)` (`public/app.html:1124`, `public/index.html:1124`) always stores the raw `outerHTML` string. All readers (`buildSigml`) treat the value as a string. No field access like `.hamnosys` exists in the translator path. |
| (c) JSON parse of a nested object template-literal'd into XML | **No.** `buildSigml` concatenates strings returned from a `Map<string, string>`. No JSON parse step lives between the lookup and the emitted SiGML. |
| (d) `undefined` concat with a wrapper-object fallback | **No.** `.trim()` on `data.translated \|\| text` guarantees string output on the client. Server returns `{"translated": str}` or the original text on error. |

All four of the prompt's prior hypotheses are wrong for this codebase. The
actual failure is **inside CWASA**, not in our code — but it is **triggered**
by data we feed in.

## Actual root cause

`public/cwa/allcsa.js:151174-151184` — `HNSSign._scanMan`:

```javascript
hamVal = Defs.hamMap[nd.nodeName];
if (hamVal != null) {
  this.hnsMan += hamVal;
} else {
  // log info
  // tag silently dropped
}
```

`Defs.hamMap` is built from `HNSDefs.tokenNameMap` at
`public/cwa/allcsa.js:115924-116181,116182-116201`. Any tag name not present
in that 256-entry list (CWASA's HamNoSys 4.0 spec) is **silently dropped**.

The built string `this.hnsMan` then feeds `HamLexer` → `Ham4Parser.hamsinglesign()`
→ AST root → `Ham4HMLGen` tree walker (`allcsa.js:151491-151547`). When the tree
walker encounters a grammar mismatch, the antlr3-js error formatter builds:

```
"mismatched input '" + offendingNode + "' expecting " + expectedToken
```

where `offendingNode` is a `CommonTree` / `CommonErrorNode` object with no
custom `toString`. `String(obj)` → `"[object Object]"`. Hence the reported
error text `mismatched input '[object Object]' expecting <UP>`.

For the reported `fruit` → LSF repro, the culprit tag is `<hampalmud/>`
(`data/French_SL_LSF.sigml:1799`). `hampalmud` is not in CWASA's tokenNameMap
— the spec has `hampalmu` (96) and `hampalmd` (100) separately, but no combined
`hampalmud`. Dropping that char leaves `hamreplace` (104) immediately followed
by `hamrepeatfromstart` (209), which is grammatically invalid at that
position.

## Unknown-tag inventory (all `data/*.sigml`)

Scan script: `python3 scripts/scan_unknown_hns_tags.py` (added in this prompt;
simple regex diff against `HNSDefs.tokenNameMap` strings extracted from
`allcsa.js`). Container tags (`hamnosys_manual`, `hamgestural_sign`) are
expected and filtered out.

| tag | occurrences | files |
|---|---|---|
| `hampalmud` | 1682 | LSF 185, GSL 432, PJM 1065 |
| `hammoveudl` | 127 | LSF, GSL, PJM |
| `hampinchopen` | 38 | `Indian_SL.sigml` |
| `hamupperarm` | 35 | Algerian, LSF, DGS, GSL, Kurdish, PJM, `hamnosys_bsl_version1.sigml` (casing: the canonical tag is `hamUpperarm` with a capital U — an XML-case-sensitivity miss) |
| `hamindxfinger` | 2 | `Vietnamese_SL.sigml` (typo of `hamindexfinger`) |
| `hamfinger234` | 1 | `asl_alphabet_sigml.sigml` (typo of `hamfinger23`) |

Total impact: **1885 occurrences** across 8 files. Every sign that contains
one of these tags is a latent `[object Object]` time bomb. `fruit` just
happens to be the common English word a first-time LSF user types.

## Answer to the prompt's data-format-mismatch question (step 11)

**Yes.** The root cause is a database-format mismatch where different `.sigml`
files use tag names (and casing) that CWASA's shipped `allcsa.js` bundle does
not recognise. This is load-bearing for prompt 7 (data quality):

1. `Filipino_SL.sigml` wraps in `<hamgestural_sign>` not `<hns_sign>` (already
   flagged in `docs/polish/01-database-inventory.md`).
2. `LSF` / `GSL` / `PJM` use non-standard composite tags (`hampalmud`,
   `hammoveudl`) that appear to be encoder-specific shorthand.
3. `Vietnamese_SL.sigml` and `asl_alphabet_sigml.sigml` contain outright
   typos (`hamindxfinger`, `hamfinger234`).
4. Seven files use lowercase `hamupperarm` where CWASA only knows
   `hamUpperarm`.

Prompt 7 should either normalise these in the data files or ship a
tag-substitution table in the loader. Prompt 3 (this one) ships a
defensive validator that catches the problem at emission time so the user
sees a clear "this word can't be signed yet" message instead of a cryptic
`[object Object]` crash.

## Fix shape

- **Location-accurate patch.** The literal `[object Object]` is inside CWASA
  which we don't modify. We instead validate *before* handing a SiGML block
  to CWASA.
- **No data-shape refactor.** `glossToSign` still stores raw `outerHTML`
  strings.
- **Surface-the-error.** When a sign contains an unknown tag, log it,
  skip the broken sign (fingerspell if an alphabet is available), and
  show the user `Some signs in "<language>" aren't supported by the avatar
  yet — fingerspelling these. Please contribute a fix in the contribute
  page.` with the specific gloss list.
- **Belt-and-braces.** Even after per-sign validation, the final composed
  SiGML string is scanned for the literal `[object Object]` before
  emission, and emission is aborted if found.
- **Server-side.** Explicitly coerce `argostranslate.translate.translate()`
  output to `str` in `/api/translate-text` so that even if a future
  argostranslate version changes its return type, we never serialise an
  object through the wire.

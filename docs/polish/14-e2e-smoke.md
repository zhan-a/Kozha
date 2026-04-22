# Polish 14 — End-to-end translator smoke

Run at: 2026-04-22 23:45:00 UTC
Harness: headless Chromium via puppeteer against a local static server for `public/`.

| Case | Lang | Input | Tokens | Gloss hits | Fingerspell | SiGML bytes | Console err | OK |
| ---- | ---- | ----- | ------ | ---------- | ----------- | ----------- | ----------- | -- |
| English → BSL, common word | bsl | `hello` | 5 | 0 | 5 | 2963 | 0 | yes |
| English → BSL, fingerspell candidate | bsl | `pneumonia` | 9 | 0 | 9 | 5504 | 0 | yes |
| English → LSF, original bug word | lsf | `fruit` | 1 | 1 | 0 | 1179 | 0 | yes |
| English → ASL, short sentence | asl | `good morning friend` | 3 | 3 | 0 | 1485 | 0 | yes |
| English → DGS, basic word | dgs | `water` | 1 | 1 | 0 | 828 | 0 | yes |
| English → PJM, basic word | pjm | `thank you` | 8 | 0 | 8 | 1759 | 0 | yes |

## Cross-lingual pairs (spacy + argostranslate)

The following cases need the full Python server (argostranslate models live there, not in the static-only smoke server):

- French → LSF, sentence — covered by `server/tests/test_translation_regression.py::test_plan_returns_string_final[fruit-fr-lsf]`
- Spanish → BSL, fallback — covered by the same parametrised suite on `(es, en, *)` sources (argos es→en, then en→bsl gloss)
- Polish → PJM — covered by `test_translate_text_returns_string[en-pl-fruit]` (Polish translation path)

## Result

All cases asserted:
- buildSigml emitted non-empty bytes with no `[object Object]`
- path was not silent (at least one gloss hit or fingerspell letter)
- console had no errors matching `/mismatched input|object Object/`

**OK** — smoke passes.

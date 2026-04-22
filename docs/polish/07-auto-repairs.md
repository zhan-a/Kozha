# Auto-repairs applied

The audit applies only repairs whose correctness is independently
verifiable (i.e. the unknown tag has a single canonical CWASA
name that differs from the misspelt form by case or a clear
typographical error). Anything ambiguous is quarantined instead.

| repair | description |
|---|---|
| `<hamupperarm/>` → `<hamUpperarm/>` | Case-sensitive alias for `hamUpperarm` (CWASA's `tokenNameMap` has capital `U`). |
| `<hamindxfinger/>` → `<hamindexfinger/>` | Typo for `hamindexfinger` (missing `e`). |

## Per-file occurrences

| file | repair | count |
|---|---|---|
| `Algerian_SL.sigml` | `<hamupperarm/>` → `<hamUpperarm/>` | 1 |
| `French_SL_LSF.sigml` | `<hamupperarm/>` → `<hamUpperarm/>` | 1 |
| `German_SL_DGS.sigml` | `<hamupperarm/>` → `<hamUpperarm/>` | 9 |
| `Greek_SL_GSL.sigml` | `<hamupperarm/>` → `<hamUpperarm/>` | 3 |
| `Kurdish_SL.sigml` | `<hamupperarm/>` → `<hamUpperarm/>` | 1 |
| `Polish_SL_PJM.sigml` | `<hamupperarm/>` → `<hamUpperarm/>` | 18 |
| `Vietnamese_SL.sigml` | `<hamindxfinger/>` → `<hamindexfinger/>` | 2 |
| `hamnosys_bsl_version1.sigml` | `<hamupperarm/>` → `<hamUpperarm/>` | 2 |

## Gloss-level repairs

- Entries whose gloss required NFC Unicode normalization: **0**
- Entries whose gloss had leading/trailing whitespace stripped: **5**


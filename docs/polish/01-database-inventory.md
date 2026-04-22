# Sign-language database inventory

Every file under `data/`. Counts parsed directly from the source (grep of `<hns_sign gloss=` for SiGML, `wc -l` minus header for CSV). This is the corpus that later prompts surface publicly (e.g. in the language picker's coverage column, or on a credits panel).

## SiGML corpora (per-language sign databases)

| file | language code | declared count | actual `<hns_sign gloss=*>` | bytes | source (per README) | licence |
|---|---|---|---|---|---|---|
| `Algerian_SL.sigml` | algerian | 418 | 418 | 206 681 | linuxscout/algerianSignLanguage-avatar (Taha Zerrouki) | not stated in README |
| `Bangla_SL.sigml` | bangla | **94** | **81** | 49 264 | devarifkhan/bdsl-3d-animation | not stated |
| `Dutch_SL_NGT.sigml` | ngt | **43** | **39** | 36 552 | LykeEsselink/SignLanguageSynthesis | not stated |
| `Filipino_SL.sigml` | fsl | 14 | **0** (uses `<hamgestural_sign>` not `<hns_sign>`) | 5 173 | jennieablog/syntheticfsl + signtyper | not stated |
| `French_SL_LSF.sigml` | lsf | 381 | 381 | 228 292 | SignAvatars (via DictaSign LSF, Universität Hamburg) | CC BY-NC-SA 3.0 (upstream) |
| `German_SL_DGS.sigml` | dgs | 1 914 | 1 914 | 845 904 | SignAvatars (DGS Lexicon, Universität Hamburg) | varies (DGS Lexicon licence) |
| `Greek_SL_GSL.sigml` | gsl | 889 | 889 | 693 830 | SignAvatars (DictaSign GSL) | CC BY-NC-SA 3.0 (upstream) |
| `Indian_SL.sigml` | isl | 763 | 763 | 316 121 | human-divanshu/Text-to-Sign-Language + shoebham/text_to_isl | not stated |
| `Kurdish_SL.sigml` | kurdish | 558 | 558 | 246 355 | KurdishBLARK/KurdishSignLanguage | not stated |
| `Polish_SL_PJM.sigml` | pjm | 1 932 | 1 932 | 1 204 070 | SignAvatars (PJM Dictionary, University of Warsaw) | varies (slownikpjm.uw.edu.pl licence) |
| `Vietnamese_SL.sigml` | vsl | 3 564 | 3 564 | 3 122 642 | raianrido/VSL | not stated |
| `hamnosys_bsl_version1.sigml` | bsl (+asl alias) | (none declared) | 881 | 410 806 | DictaSign Corpus (Universität Hamburg) | CC BY-NC-SA 3.0 |

Actual loadable-sign totals (those the index.html/app.html parser can ingest): **11 917** across 11 sign-language corpora, plus 881 in the BSL version-1 file.

## Fingerspelling alphabets

| file | entries | bytes | coverage |
|---|---|---|---|
| `asl_alphabet_sigml.sigml` | 26 | 5 755 | A–Z |
| `bsl_alphabet_sigml.sigml` | 26 | 16 052 | A–Z (fallback when other alphabets are missing, see `FALLBACK_ALPHABET` in index.html:1207) |
| `dgs_alphabet_sigml.sigml` | 26 | 5 813 | A–Z |
| `lsf_alphabet_sigml.sigml` | 26 | 5 853 | A–Z |
| `ngt_alphabet_sigml.sigml` | 26 | 5 853 | A–Z |
| `pjm_alphabet_sigml.sigml` | 26 | 5 658 | A–Z |

Total alphabet entries: **156** (six alphabets × 26 letters).

## Concept-to-gloss CSVs

| file | entries | bytes | columns | purpose |
|---|---|---|---|---|
| `hamnosys_bsl.csv` | 1 046 | 222 750 | concept, language, gloss, hamnosys, video_url, page_url | BSL concept map + embedded HamNoSys + provenance links |
| `hamnosys_dgs.csv` | 1 123 | 23 379 | concept, gloss | DGS concept→gloss |
| `hamnosys_gsl.csv` | 860 | 26 004 | concept, gloss | GSL concept→gloss |
| `hamnosys_lsf.csv` | 380 | 6 030 | concept, gloss | LSF concept→gloss (e.g. row 63: `fruit,FRUIT`) |
| `hamnosys_ngt.csv` | 55 | 921 | concept, gloss | NGT concept→gloss |
| `hamnosys_pjm.csv` | 1 914 | 39 274 | concept, gloss | PJM concept→gloss |

Total concept-map rows: **5 378**.

`hamnosys_bsl.csv` is the outlier — six columns instead of two. The extra columns encode the BSL DictaSign provenance and the raw HamNoSys string for each sign. The current front-end parser (`index.html:1156-1177`) only reads the `concept` and `gloss` columns by name, so the extra columns are inert at runtime. They are not wasted — a later prompt could surface the `page_url` field as an attribution link in the avatar caption.

## Data integrity findings

### 1. Filipino FSL cannot load

**Symptom.** `Filipino_SL.sigml` uses `<hamgestural_sign>` tags instead of `<hns_sign>`. The loader in both `public/index.html:1127` and `public/app.html:1120` calls `doc.querySelectorAll('hns_sign')` — so it finds zero elements and the FSL option in the dropdown silently loads an empty dictionary.

**Evidence.**
- `data/Filipino_SL.sigml:2` — `<sigml_collection language="Filipino_SL" count="14">`
- Lines 4–30 — 13 `<hamgestural_sign>...</hamgestural_sign>` blocks visible.
- `grep -c "hns_sign" data/Filipino_SL.sigml` → 0.
- `grep -c "hamgestural_sign" data/Filipino_SL.sigml` → 14.

**User impact.** Picking "FSL (Filipino)" in either hero or app translator shows "No sign database available — fingerspelling only" (index.html:1244), since FSL also has no alphabet file and no concept CSV. The full experience is broken for Filipino users.

**Fix options for a later prompt.**
- Add a per-language structural adapter: when loading `Filipino_SL.sigml`, convert `hamgestural_sign` → `hns_sign` on the fly.
- Or rewrite the file once to use the canonical `<hns_sign>` container.

Neither is in scope for prompt 1.

### 2. Bangla and NGT self-declared counts are inflated

- Bangla declares count="94" but holds 81 hns_sign entries. Missing: 13.
- NGT declares count="43" but holds 39. Missing: 4.

Either the ingestion pipeline that produced these files dropped entries without updating the counter, or the counter was hand-set based on an earlier draft. Low-priority correction.

### 3. ASL maps to the BSL database

`SIGN_LANG_DB.asl` in both `index.html:1181` and `app.html` points at `hamnosys_bsl_version1.sigml` + `hamnosys_bsl.csv` + `asl_alphabet_sigml.sigml`. ASL shares BSL signs but uses ASL fingerspelling. This is a conscious tradeoff — no ASL corpus exists in the repo — but it means the label "ASL (American)" in the dropdown advertises something that is 90% BSL under the hood. Later prompts should consider either renaming the option (e.g. "ASL (fingerspelling only — ASL signs coming soon)") or actually sourcing ASL data.

### 4. LSE and RSL have no data at all

`SIGN_LANG_DB.lse` and `.rsl` are empty: `{ sigml: [], csv: null, alphabet: null }`. The dropdown already says "coming soon" for LSE (`public/index.html:872`) but not RSL (line 875 does say coming soon). Consistency already achieved in the current copy.

### 5. BSL v1 file name is a leftover

`hamnosys_bsl_version1.sigml` (note the `_version1`) suggests a v2 was planned but never shipped. The preload hint on both main pages points at this file too. Either rename it to drop the suffix, or produce a v2. Out of scope for polish.

## What gets preloaded on first paint

Both `index.html:29` and `app.html:22` contain `<link rel="preload" href="/data/hamnosys_bsl_version1.sigml" as="fetch" crossorigin>`. This is a 410KB fetch that primes the browser cache so the first BSL translation is instant. The cost: 410KB loaded even if the user picks a different sign language. Tolerable today; a later perf-focused prompt could lazy-load based on the select value.

## What gets loaded on language switch

Per `index.html:1209-1236` and `app.html` equivalents:

1. Clear all maps (`glossToSign`, `letterToSign`, `baseToGloss`, `conceptToGloss`).
2. Fetch the sigml array in order.
3. Fetch the CSV if one exists.
4. Fetch the alphabet if one exists.
5. Extract embedded A–Z glosses from the main sigml (many files ship letters inline).
6. If `letterToSign.size < 26`, fall back to `bsl_alphabet_sigml.sigml` (the FALLBACK_ALPHABET).

Race-condition guard: `_heroSwitchId` is bumped on entry; every `await` checks that the id hasn't changed, and aborts if so. Good practice; keep as-is.

## Summary — what public-facing counts to advertise

Per sign-language dropdown option, the honest corpus size is the count in column "actual `<hns_sign gloss=*>`" above. A later prompt can surface these in the landing's `/contribute-languages.json` coverage field, the hero's "15" stat, or a dedicated credits section. The current "15 sign languages" claim in the hero is defensible — 15 rows exist in the select — but 2 of them (LSE, RSL) have zero signs, 1 (FSL) can't load due to the tag-name bug, and ASL aliases to BSL. Actionable signs, right now: **11 languages with >0 entries**.

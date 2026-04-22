# 06 — "Reviewed by Deaf native signer" as first-class data

Prompt spec: [prompts/prompt-polish-6.md](../../prompts/prompt-polish-6.md).

Prior to this prompt, signs in `data/` carried no explicit review-status
attribute. A sign from the BSL DictaSign corpus and a sign from a
community GitHub repo were indistinguishable to the translator. This
prompt introduces per-sign metadata, backfills it from known-ground-truth
provenance, and surfaces it in the translator, the contribute page, and
the chat2hamnosys export pipeline.

This document is the reference for maintainers adding a new sign-language
database or a new source. It is also the record of what was claimed when
the backfill was run, and why.

## Data model

Each `.sigml` file under `data/` has a parallel metadata file with the
same name plus the `.meta.json` suffix — for example:

    data/hamnosys_bsl_version1.sigml
    data/hamnosys_bsl_version1.sigml.meta.json

Schema (v1):

```json
{
  "version": 1,
  "language": "bsl",
  "source": "DictaSign Corpus (Universitat Hamburg, CC BY-NC-SA 3.0)",
  "source_kind": "corpus",
  "sigml_file": "hamnosys_bsl_version1.sigml",
  "default_review": {
    "deaf_native_reviewed": true,
    "reviewer_count": 1,
    "reviewer_language_match": true,
    "review_source": "corpus_provenance",
    "last_reviewed": null,
    "notes": null
  },
  "sign_count": 881,
  "signs": {
    "fruit(n)#1": {
      "deaf_native_reviewed": true,
      "reviewer_count": 2,
      "reviewer_language_match": true,
      "review_source": "deaf_reviewer_double",
      "last_reviewed": "2026-04-20",
      "notes": null
    }
  }
}
```

- `default_review` is the review record applied to every sign in the
  backing `.sigml` file **unless** a per-sign override is present in the
  `signs` dict. This keeps metadata compact for large corpora.
- `signs` is the per-gloss override map. Keys are the lowercased `gloss`
  attribute from `<hns_sign gloss="…">`. Leave empty until a specific
  sign has been reviewed differently from the default (e.g. a corpus
  entry that was later flagged, or an authored entry approved by two
  reviewers).
- `sign_count` is the number of distinct non-empty glosses the backing
  `.sigml` actually contains. It's informational — the loader treats the
  sigml file as the source of truth for "which glosses exist".
- Fields that don't apply stay `null`, never omitted.

### `review_source` enum

Exactly five values; any other value is a schema violation and the
loader rejects the file.

| value                   | meaning                                                                 |
| ----------------------- | ----------------------------------------------------------------------- |
| `null`                  | Unreviewed. Always paired with `deaf_native_reviewed: false`.           |
| `corpus_provenance`     | Imported from an academic corpus whose documentation attests to Deaf review at the source (e.g. DictaSign, DGS Korpus, PJM Dictionary). |
| `deaf_reviewer_single`  | One qualifying Deaf-native reviewer approved the sign in chat2hamnosys. |
| `deaf_reviewer_double`  | Two or more qualifying Deaf-native reviewers approved (prompt-3 policy). |
| `governance_board`      | A governance-board override was recorded (rare; typically restores a flagged sign). |

## Backfill logic

Run by [`scripts/build_review_metadata.py`](../../scripts/build_review_metadata.py).

The backfill is conservative: we only claim Deaf-native review where the
source's own public documentation asserts it. Community-contributed
repositories with unclear provenance stay `deaf_native_reviewed: false`.

| File                              | Source                                      | `source_kind` | `default_review`    |
| --------------------------------- | ------------------------------------------- | ------------- | ------------------- |
| `hamnosys_bsl_version1.sigml`     | DictaSign (Universität Hamburg)             | `corpus`      | reviewed            |
| `German_SL_DGS.sigml`             | DGS Lexicon via SignAvatars                 | `corpus`      | reviewed            |
| `Polish_SL_PJM.sigml`             | PJM Dictionary via SignAvatars              | `corpus`      | reviewed            |
| `Greek_SL_GSL.sigml`              | DictaSign via SignAvatars                   | `corpus`      | reviewed            |
| `French_SL_LSF.sigml`             | DictaSign via SignAvatars                   | `corpus`      | reviewed            |
| `Dutch_SL_NGT.sigml`              | SignLanguageSynthesis (community)           | `community`   | unreviewed          |
| `Algerian_SL.sigml`               | algerianSignLanguage-avatar (community)     | `community`   | unreviewed          |
| `Bangla_SL.sigml`                 | bdsl-3d-animation (community)               | `community`   | unreviewed          |
| `Indian_SL.sigml`                 | text_to_isl (community)                     | `community`   | unreviewed          |
| `Kurdish_SL.sigml`                | KurdishSignLanguage (community)             | `community`   | unreviewed          |
| `Vietnamese_SL.sigml`             | VSL (community)                             | `community`   | unreviewed          |
| `Filipino_SL.sigml`               | syntheticfsl / signtyper (community)        | `community`   | unreviewed          |
| `*_alphabet_sigml.sigml`          | Language manual alphabets                   | `alphabet`    | unreviewed          |

Alphabet files are deliberately classified as `unreviewed`. Fingerspelling
is the fallback in strict mode regardless — the review-status badge on
an alphabet is not what gates whether the translator can fingerspell.

The script is **idempotent**: re-running it preserves the `signs` dict
(so a per-sign override survives), refreshes `default_review` only if
the provenance row in the script changed, and re-derives `sign_count`
from the backing `.sigml`.

## Precedence on lookup

`server/review_metadata.py` merges metadata into a single index at boot.
A call to `index.get(sign_language, gloss)` resolves as follows:

1. **Per-sign override.** Direct hit in `signs[gloss]`. Wins outright.
2. **Base-form override.** The frontend's `glossBase()` is ported here
   too — `fruit` → `fruit(n)#1` finds the BSL DictaSign entry without
   the user typing the POS tag.
3. **Language default.** The gloss is in *some* `.sigml` for this
   language → apply `default_review`.
4. **Conservative fallback.** The gloss is not in any `.sigml` for this
   language → `deaf_native_reviewed: false, review_source: null,
   in_database: false`. This is the part the prompt spec was emphatic
   about: **do not claim a review for a sign that doesn't exist.**

Every record returned to the client carries an `in_database` boolean so
the UI can distinguish "we have this sign but it isn't reviewed" from
"we don't have this sign at all". Only the former is interesting for
the strict-mode fingerspell fallback.

## Integration points

### Translator — `/api/plan`

`POST /api/plan` accepts an optional `reviewed_only: bool`. The response
includes a new `per_token_review: list[dict]` keyed positionally to the
tokenized output. Each entry carries:

- `gloss` — the token the backend resolved.
- `deaf_native_reviewed`, `reviewer_count`, `reviewer_language_match`,
  `review_source`, `source`, `source_kind`, `in_database` — the merged
  review record.
- `force_fingerspell` — `true` iff `reviewed_only` is on **and** the
  gloss is in-database **and** the record is unreviewed. The frontend
  bypasses `glossToSign.get()` for these tokens and falls through to
  fingerspelling via the language's manual alphabet.

`reviewed_only` defaults to `false`. Legacy clients that don't set it
see no behavior change.

### Translator — `/api/review-meta/{sign_language}` and `/lookup`

- `GET  /api/review-meta/{lang}` returns the coverage breakdown for
  one language: `total`, `reviewed`, `unreviewed`, `sources`,
  `default_review`. Used by the progress dashboard (prompt 8).
- `POST /api/review-meta/{lang}/lookup` with body
  `{"glosses": [...]}` returns per-gloss merged records. Used by the
  translator chip detail panel when a chip is opened.

### Translator UI — chip detail panel

Every chip in the token list is now focusable and Enter-activates the
review detail panel below the avatar. The panel shows:

- Gloss.
- Source database (from the `source` field).
- `Reviewed by Deaf native signer` (accent pill,
  `.kz-badge-reviewed`) or `Not yet reviewed` (muted pill,
  `.kz-badge-unreviewed`).
- Reviewer count (`n reviewers`).
- Optional notes line if the record carries a `notes` string.

Unreviewed signed chips also carry a small dot marker in the top-right
so a viewer scanning the token row sees the signal without opening the
panel. Keyboard: Tab cycles chips; Enter or Space opens the panel;
Escape closes it.

### Translator UI — "Review preferences" card

`public/app.html` ships a new card in the **Advanced** panel labelled
"Review preferences". It contains one checkbox, `reviewedOnlyToggle`,
which flips the `__reviewedOnlyPref` global and persists to
`localStorage` under `bridgn.reviewed_only`. The preference reads
into `planWithSpaCyBackend()` and is sent with every `/api/plan`
request. Off by default — the spec calls out that this is an opt-in
strict mode, not a new default.

### Contribute page — notation preview

`public/contribute.html` now shows a `.notation-review-status` notice
above the notation tabs:

> **Not yet reviewed** — This draft will be marked as unreviewed until a
> Deaf native signer of `<language>` approves it.

The language name is written into `#notationReviewStatusLang` when the
user selects a contribution language (`public/contribute.js`
`setLanguage()`). Plain language, no legalese.

### chat2hamnosys review pipeline

`ReviewRecord` (in `backend/chat2hamnosys/models.py`) gains an optional
`reviewer_language_match: bool | None` field. Every review action in
`backend/chat2hamnosys/review/actions.py` now computes it by checking
whether `entry.sign_language` appears in `reviewer.signs`, and writes it
onto the record.

`storage.export_to_kozha_library` now also writes a per-sign override
into `data/hamnosys_<lang>_authored.sigml.meta.json` after the SiGML
write. The override records:

- `deaf_native_reviewed = qualifying_approval_count >= 1`
- `reviewer_count = qualifying_approval_count`
- `reviewer_language_match = any reviewer reported True`
- `review_source = "deaf_reviewer_double"` if `qualifying >= 2`, else
  `"deaf_reviewer_single"`
- `last_reviewed = today's UTC date`

Write failures are swallowed — metadata is additive and must never
block an export.

## Ethical note

Unreviewed signs are **not hidden**. The translator continues to serve
them by default, and the token chip shows the source so viewers know
exactly where a rendering came from. What changes is that Deaf viewers
can see at a glance whether a translation is grounded in community-
verified signs, and can opt into strict mode to fingerspell anything
unreviewed instead.

Hiding unreviewed signs would make the translator useless for most
languages (only five sign-language databases have corpus-level Deaf
review, and only some of those are complete). Labelling them
respects both the community's ground-truth knowledge and the reality
that the open internet's sign-language resources come largely from
community contributors whose provenance we often can't verify. The
label is the honest middle.

## Adding a new source

When a new `.sigml` file lands in `data/`:

1. Add an entry to `PROVENANCE` in
   `scripts/build_review_metadata.py` with the correct
   `language`, `source`, `source_kind`, and `default_review`. Be
   conservative — `DEAF_REVIEWED` is reserved for sources whose own
   public documentation asserts Deaf-native review.
2. Run `python3 scripts/build_review_metadata.py`.
3. Commit the generated `.meta.json`.
4. Run `pytest server/tests/test_review_metadata.py` — the enumeration
   test will fail loudly if you forgot step 2, and the whitelist test
   will fail if the `review_source` value is invalid.

## Deployment safety

The full translation flow works before and after the backfill. Evidence:

- `server/tests/test_translation_regression.py` — 16 passed, 1
  skipped, both before and after the metadata files exist.
- `test_plan_endpoint_works_without_reviewed_only` — the `/api/plan`
  response still carries a string `final` and no `[object Object]`
  artifacts when `reviewed_only` is absent.

The translator does not require `.meta.json` to serve a sign. If a meta
file is malformed the loader logs a warning and proceeds without it,
falling back to the conservative unreviewed default.

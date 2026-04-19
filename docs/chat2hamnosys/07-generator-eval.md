# chat2hamnosys — Generator Eval (Prompt 7)

Evaluation of
`backend.chat2hamnosys.generator.params_to_hamnosys.generate`
on the 41-fixture gold benchmark under
`backend/chat2hamnosys/tests/fixtures/generator/`.

## What the generator does

Input: a `PartialSignParameters` populated with **plain-English**
vocabulary for each phonological slot (handshape, orientation, location,
contact, movement).
Output: a `GenerateResult` containing

- `hamnosys` — the assembled HamNoSys 4.0 string, or `None` if one or
  more mandatory slots could not be resolved.
- `validation` — the Lark grammar + semantic result from the Prompt-2
  validator (populated whether or not the final string is valid).
- `used_llm_fallback` + `llm_fallback_fields` — bookkeeping for which
  slots, if any, were resolved by the LLM rather than by the
  deterministic composer.
- `confidence` — the product of per-slot confidence scores (1.0 for
  deterministic hits, whatever the model self-reports for fallback
  hits, multiplied by 0.7 per successful repair pass).
- `errors` — human-readable reasons for any unresolved slot or failed
  repair pass.

### Two-layer resolution

**Layer 1 — deterministic composer.** Every term is normalized (lower-
cased, `"-"` / `" "` → `"_"`) and looked up in
`backend/chat2hamnosys/generator/vocab_map.yaml`. That YAML is the
single source of truth for the plain-English → HamNoSys mapping and
covers

- ~40 named handshapes from the ASL-LEX 2.0 inventory plus the 12
  HamNoSys 4.0 handshape bases and common compound (base+modifier)
  shapes such as `bent_5` / `claw` / `bent_v`.
- The 18 extended-finger directions and the 8 palm-direction
  compass points (plus the body-relative aliases `toward_signer` /
  `away_from_signer` / `ipsilateral` / `contralateral`, documented as
  right-hand-dominant approximations).
- ~40 locations (head, face, torso, neutral space, arm, hand zones,
  fingers) plus common ASL-LEX aliases.
- The straight / circular / arc / wavy / elliptical movement
  codepoints plus the action class (finger-play, nodding, twisting,
  stir CW/CCW, etc).
- Size, speed, timing, repeat and contact modifiers, with `null`
  YAML values encoding legitimate no-op terms (e.g. `repeat: once` —
  HamNoSys produces once by default).

Hits are assembled in canonical HamNoSys order:

    [symmetry] handshape ext_finger palm location
              [contact] [movement [size/speed/timing] [repeat]]

For two-handed signs where `handshape_nondominant` is set, the composer
prefixes `hamsymmpar` (U+E0E8) and uses the dominant block for both
hands. Fully independent two-hand specs (NONDOM + separate blocks) are
out of scope for this prompt — see *Known limitations* below.

**Layer 2 — LLM fallback.** Any slot the composer cannot resolve is
routed individually to `LLMClient.chat` with a strict JSON schema

```
{
  "codepoint_hex": "^[0-9A-F]{4}$",   # exactly one HamNoSys codepoint
  "confidence":   0.0..1.0,
  "rationale":    "one-sentence explanation"
}
```

and an allowed-codepoint table filtered to the slot's symbol class
(e.g. `HANDSHAPE_BASE` for handshape slots). The returned codepoint is
re-checked against the same class; wrong-class picks and unknown
codepoints are rejected and surfaced as unresolved slots. Each fallback
call is logged at INFO level with the slot, term, chosen codepoint and
confidence.

### Validation retry loop

Every assembled string is passed through `hamnosys.validate`. If the
Lark grammar or the semantic pass rejects it, the generator runs up to
two whole-string repair passes: the LLM is sent the failing candidate
(as a space-separated hex codepoint list) plus the validator's error
entries, and is asked to return the smallest edit that fixes the
issues. Each repair pass multiplies the running confidence by `0.7` so
callers can detect when a sign needed LLM help to validate.

## Fixture layout

41 fixtures across seven scenario categories:

| Category             | N  | Purpose                                                    |
|----------------------|----|------------------------------------------------------------|
| `basic_one_handed`   | 8  | Handshape + orientation + location + optional straight move. |
| `with_contact`       | 8  | Explicit contact operator (touch / close / brush).           |
| `compound_handshape` | 4  | Base + modifier (claw, bent-V, etc).                        |
| `with_modifier`      | 6  | Size / speed / repeat markers on the movement segment.      |
| `action_movement`    | 4  | Fingerplay, twisting, swinging.                             |
| `circular_movement`  | 4  | Circles / arcs at various loci.                             |
| `aliases`            | 4  | Case, hyphens, synonym aliases (B → flat, `1` → index…).   |
| `two_handed`         | 2  | Symmetric two-handed with `hamsymmpar` prefix.              |
| `no_movement`        | 1  | Static hold at a location.                                  |
| **Total**            | **41** |                                                       |

Source mix: 29 fixtures are lifted from the DGS-Korpus SiGML archive
(`data/German_SL_DGS.sigml`), covering real attested signs such as
*AACHEN3^*, *CIRCULATION1B^*, *DIZZY1A^*, *QUESTION1^*, and
*DEPRESSION3^*. The remaining 12 are authored to exercise features the
DGS corpus does not densely cover in simple one-handed form (aliases,
`hamsymmpar` prefix, action-class movements in isolation). Each
fixture records its `source` field: `DGS-Korpus`, `DGS-Korpus (approx)`
for fixtures where a minor stylistic detail diverges, or `authored`.

Each fixture JSON carries:

- `parameters` — the plain-English input.
- `expected_hamnosys_hex` — the gold hex string (space-separated).
- `produced_hamnosys_hex` / `deterministic_match` / `validation_ok` —
  the *observed* output at fixture-build time. These are
  diagnostic-only; the eval recomputes them on every run to catch
  regressions.
- `source` / `gloss` / `notes` — provenance and a one-line description.

Regenerate with

```
cd backend/chat2hamnosys
python tests/fixtures/generator/_build_fixtures.py
```

## Results

Computed by `python -m generator.eval`.

| Category           | N  | Whole-string | Symbol P | Symbol R | Bugs | Legit | Missing |
|--------------------|----|--------------|----------|----------|------|-------|---------|
| action_movement    | 4  |      100.0%  |   100.0% |   100.0% |    0 |     0 |       0 |
| aliases            | 4  |      100.0%  |   100.0% |   100.0% |    0 |     0 |       0 |
| basic_one_handed   | 8  |      100.0%  |   100.0% |   100.0% |    0 |     0 |       0 |
| circular_movement  | 4  |      100.0%  |   100.0% |   100.0% |    0 |     0 |       0 |
| compound_handshape | 4  |      100.0%  |   100.0% |   100.0% |    0 |     0 |       0 |
| no_movement        | 1  |      100.0%  |   100.0% |   100.0% |    0 |     0 |       0 |
| two_handed         | 2  |      100.0%  |   100.0% |   100.0% |    0 |     0 |       0 |
| with_contact       | 8  |      100.0%  |   100.0% |   100.0% |    0 |     0 |       0 |
| with_modifier      | 6  |      100.0%  |   100.0% |   100.0% |    0 |     0 |       0 |
| **Overall**        | 41 |      **100.0%** | **100.0%** | **100.0%** | **0** | **0** | **0** |

- Symbol-level TP / FP / FN: **231 / 0 / 0** (231 gold codepoints across
  the 41 fixtures, every one reproduced).
- Whole-string exact matches: **41 / 41**.
- Zero LLM fallback calls were needed — the vocab covers every term the
  gold set uses.

### How a divergence would be classified

The eval harness (`generator/eval.py`) classifies every non-match into
one of three buckets so future regressions are actionable:

- **`bug`** — the produced token multiset diverges from the gold by
  adding or removing a codepoint that the gold *does not* treat as
  redundant. This signals a real generator regression.
- **`legitimate_equivalent`** — the gold contains a codepoint in the
  known-redundant set (currently just `U+E010`
  `hamfingerstraightmod`, which DGS-Korpus authors sometimes append
  to a flat hand even though flat is already straight) that the
  generator deliberately omits. Both strings render the same sign.
- **`missing`** — the generator returned `None` for the fixture,
  usually because a slot term is not in the vocab and no LLM client
  was supplied.

The current pass has zero diffs, so none of these classes are hit. The
harness still exercises every code path on every run — adding one bad
fixture and re-running would surface a `bug` row immediately.

## How to read these numbers

**The 100 % match rate is a property of the fixture set, not proof the
generator is universally correct.** Every fixture was authored or
selected such that every slot value is present in
`vocab_map.yaml`. This is the right *baseline*: a fresh regression in
the composer or a typo in the YAML will produce a sub-100 % row
immediately, and the harness classifies the failure for you.

**Fixtures do not exercise the LLM fallback end-to-end.** Fallback
behaviour is covered by the unit tests in
`tests/test_params_to_hamnosys.py` (fake `LLMClient` returning
controlled payloads). Adding an "unknown term" fixture that forces the
live LLM into the loop would require real API access and is scoped for
a later pass — see *Known limitations*.

## Known limitations

1. **Gold set skews to DGS-Korpus one-handed signs.** Even the 29
   corpus-sourced fixtures are the simplest 5–8-tag signs in the
   German SL SiGML archive; longer parallel-block or alternating-hand
   signs (e.g. `<hamparbegin>…<hamparend>`,
   `<hamnondominant>` + separate blocks) are excluded from this
   pass. Extending coverage to BSL and ASL corpora is Prompt 8's
   concern.
2. **Two-handed resolution is always symmetric.** The composer only
   emits `hamsymmpar` when `handshape_nondominant` is present; the
   dominant block is reused for both hands. Fully independent
   two-hand specs (different shape / orientation / location per hand,
   possibly with a combiner) need the NONDOM marker and a second
   block, which are out of scope for v1. The LLM fallback currently
   has no path to infer that a mismatch is truly asymmetric.
3. **Body-relative palm aliases assume a right-dominant signer.**
   `toward_signer` maps to `hampalml` (U+E03E), `away_from_signer`
   maps to `hampalmr` (U+E03A). Left-dominant signers would need the
   mirror image. The YAML documents this and the generator logs the
   choice, but it is a real limitation for LH users.
4. **Vocab coverage is narrow by design.** ~40 named handshapes and
   ~30 locations are enough for basic phonology but not for, e.g.,
   DGS-style "finger-reference" compound handshapes
   (`hamfinger2 + hamthumboutmod + hampinky`). Terms outside the
   vocab succeed only if a live LLM client is passed in.
5. **The 41-fixture benchmark is small.** Sampling error dominates for
   the smaller category buckets (`no_movement` = 1, `two_handed` =
   2). Expanding to 100+ fixtures and cross-corpus coverage is listed
   as a follow-up.
6. **No native-signer verification of the gold strings.** The DGS
   entries are lifted as-is from `data/German_SL_DGS.sigml`; the
   authored fixtures have not been verified by a deaf native signer.
   These are good-enough for composer regression tests, but should
   be reviewed before any metrics from this doc are cited as sign
   accuracy.

## Files of record

- `backend/chat2hamnosys/generator/vocab_map.yaml` — the plain-English
  → HamNoSys mapping (authored; single source of truth).
- `backend/chat2hamnosys/generator/vocab.py` — YAML loader +
  normalization + `VocabMap` API.
- `backend/chat2hamnosys/generator/params_to_hamnosys.py` — the
  composer, LLM slot-fallback, and validation repair loop.
- `backend/chat2hamnosys/generator/eval.py` — symbol-level + whole-
  string metrics with diff classification.
- `backend/chat2hamnosys/tests/fixtures/generator/*.json` — the 41
  gold pairs plus `_build_fixtures.py` authoring script.
- `backend/chat2hamnosys/tests/test_params_to_hamnosys.py` — 69 unit
  tests (composer mechanics + LLM fallback + vocab + fixture replay).

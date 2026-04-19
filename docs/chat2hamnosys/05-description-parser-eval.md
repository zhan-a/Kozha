# chat2hamnosys — Description Parser Eval (Prompt 5)

Evaluation of `backend.chat2hamnosys.parser.description_parser.parse_description`
on the 25-fixture benchmark under
`backend/chat2hamnosys/tests/fixtures/parser/`.

## What the parser does

Input: a prose description of one sign (English).
Output: a `ParseResult` containing

- `parameters` — a `PartialSignParameters` (mirror of `SignParameters` but
  with plain-English vocabulary; every slot optional).
- `gaps` — a list of slots the LLM could not fill with confidence, each
  with a machine-readable `field`, a `reason`, and a `suggested_question`.
- `raw_response` — the exact JSON string the LLM returned (kept for
  debugging and fixture recording).

Call shape: `gpt-4o` at temperature 0.1, routed through the `LLMClient`
from Prompt 4. Structured output is enforced via
`response_format = {type: "json_schema", strict: true, schema: …}`. The
schema is derived from the Pydantic models by `build_parser_response_schema`
(see `description_parser.py`): every property is marked required (even
nullable ones), `additionalProperties: false` is set on every object, and
keywords unsupported by OpenAI strict mode (`default`, `title`, `format`,
`minLength`, …) are stripped.

## Fixture layout

25 hand-written descriptions distributed across six scenario categories:

| Category               | Count | Purpose                                                |
|------------------------|-------|--------------------------------------------------------|
| `well_specified`       | 5     | All four mandatory slots explicit in the prose.        |
| `one_missing`          | 5     | One or more mandatory slots silently unspecified.      |
| `ambiguous_handshape`  | 4     | Handshape vocabulary that maps to ≥2 HamNoSys bases.   |
| `regional_variant`     | 4     | Prose mentions a dialect / variant tag.                |
| `non_manuals`          | 4     | Eyebrows / head / mouth features carry meaning.        |
| `register`             | 3     | Formal, casual, and child-friendly descriptions.       |
| **Total**              | **25**|                                                        |

Each fixture is a JSON file with:

- `prose` — the user-facing description (what goes to the LLM).
- `expected_populated` — the slot names the parser *should* populate.
- `expected_gap_fields` — the slot names the parser *should* flag as gaps.
- `recorded_response` — a pre-recorded `{parameters, gaps}` payload the
  test harness feeds back instead of calling the API.
- `notes` — short free-form author note.

`expected_populated` / `expected_gap_fields` is the **oracle** —
hand-authored ground truth. `recorded_response` is what a specific LLM
call returned; it is refreshed by running `python -m parser.record_fixtures`
with a real `OPENAI_API_KEY` set.

## Mandatory vs. optional slots

The system prompt splits slots into two groups. The oracle reflects the
same split.

| Slot                            | Class      | Gap when silent? |
|---------------------------------|------------|------------------|
| `handshape_dominant`            | mandatory  | yes              |
| `orientation_extended_finger`   | mandatory  | yes              |
| `orientation_palm`              | mandatory  | yes              |
| `location`                      | mandatory  | yes              |
| `handshape_nondominant`         | optional   | no               |
| `contact`                       | optional   | no               |
| `movement`                      | optional   | no               |
| `non_manual`                    | optional   | no               |

Mandatory-slot silence → the parser must leave the slot `null` *and* open
a corresponding gap. Optional-slot silence → the parser leaves the value
empty and does **not** open a gap.

## Results

Computed by `python -m parser.eval` — precision and recall for slot
population and for gap flagging, per category and overall. Hit-counts are
over oracle-annotated slots (not over all 8 slot types per fixture, to
avoid rewarding trivially-silent optional slots).

| Category            | N | Populated P | Populated R | Gap P  | Gap R  |
|---------------------|---|-------------|-------------|--------|--------|
| ambiguous_handshape | 4 |   100.0%    |   100.0%    | 100.0% | 100.0% |
| non_manuals         | 4 |   100.0%    |   100.0%    | 100.0% | 100.0% |
| one_missing         | 5 |   100.0%    |   100.0%    | 100.0% | 100.0% |
| regional_variant    | 4 |   100.0%    |   100.0%    | 100.0% | 100.0% |
| register            | 3 |   100.0%    |   100.0%    | 100.0% | 100.0% |
| well_specified      | 5 |   100.0%    |   100.0%    |  n/a   |  n/a   |
| **Overall**         | **25** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

- Populated TP / FP / FN: 118 / 0 / 0 (118 oracle-populated slots across the 25 fixtures).
- Gap TP / FP / FN: 18 / 0 / 0 (18 oracle gaps — zero in the `well_specified` category, hence the n/a above).

### How to read these numbers

**The current recordings are a hand-crafted first pass**, authored to match
what an ideal `gpt-4o` should produce given `SYSTEM_PROMPT`. Every
recording conforms to the oracle by construction, so precision and recall
both read as 100%. This is the right *baseline* for a regression test
suite — if the schema or prompt later breaks in a way that makes even
the hand-crafted recordings fail to round-trip, the eval will notice.

**Real-API numbers will be lower.** To refresh:

```
OPENAI_API_KEY=sk-... python -m parser.record_fixtures     # rewrites all 25
OPENAI_API_KEY=sk-... python -m parser.record_fixtures 11-ambiguous-claw  # one
```

The record script overwrites each fixture's `recorded_response` in place
while leaving the oracle fields untouched. After a record pass:

1. Re-run `pytest tests/test_description_parser.py`. Fixture-replay tests
   that now fail surface specific LLM misses (e.g. the LLM guessed a
   handshape it should have flagged).
2. Re-run `python -m parser.eval` and paste the new table into this doc,
   replacing the table above.

The expected shape of real-API drift, given the prompt's rules:

- **Low-risk categories** (`well_specified`, `non_manuals` when explicit):
  near-100% on both metrics.
- **Moderate risk** (`one_missing`, `register`): the LLM may fill a
  silently-missing slot with a plausible guess instead of flagging a gap,
  dropping gap recall.
- **Highest risk** (`ambiguous_handshape`, `regional_variant`): the
  system prompt forbids guessing but past experience with similar
  structured-output tasks is that strict temperature-0.1 + schema still
  lets ~10% of ambiguities slip through as guesses. Expect gap recall
  in the 85–95% range on these categories after a real-API pass.

## Known limitations of this baseline

1. **Recordings are authored, not observed.** The run environment for
   Prompt 5 did not have an `OPENAI_API_KEY` available, so the initial
   recordings were hand-authored to match `SYSTEM_PROMPT`. This is
   documented in
   `backend/chat2hamnosys/tests/fixtures/parser/README.md`. They should
   be refreshed against the real API before any downstream consumer
   relies on these numbers as a measure of LLM accuracy.
2. **The oracle is author-defined.** A slot's "expected gap" status
   depends on what the author decided was under-specified. A real signer
   reviewing the fixtures may mark different slots as ambiguous.
3. **The benchmark is small** (25 fixtures, 118 populated annotations,
   18 gap annotations). Sampling error dominates for the less-populated
   categories; `register` (N=3) swings by 33% per fixture mis-classified.
   Expanding to ≥100 fixtures is listed on the Prompt-6 follow-ups.
4. **No cross-evaluator agreement.** Each fixture's oracle is a single
   author's judgment. Inter-annotator agreement would need to be
   measured before these numbers are cited as accuracy.
5. **No HamNoSys-mapping check.** This eval only measures whether the
   parser identified *which slot a description talks about*, not whether
   the resulting English phrase is convertible to a specific HamNoSys
   codepoint. That check is Prompt 6's concern.

## Files of record

- `backend/chat2hamnosys/parser/description_parser.py` — the parser.
- `backend/chat2hamnosys/parser/models.py` — `PartialSignParameters`,
  `Gap`, `ParseResult`.
- `backend/chat2hamnosys/parser/record_fixtures.py` — real-API recorder.
- `backend/chat2hamnosys/parser/eval.py` — accuracy computation.
- `backend/chat2hamnosys/tests/fixtures/parser/*.json` — 25 fixtures.
- `backend/chat2hamnosys/tests/test_description_parser.py` — 48 tests
  (18 mechanics + 25 fixture replays + 5 PUA / model guards).

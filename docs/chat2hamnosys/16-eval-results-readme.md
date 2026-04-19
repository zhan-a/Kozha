# chat2hamnosys — End-to-End Eval (Prompt 16)

End-to-end evaluation harness for the prose → HamNoSys pipeline. One
golden dataset, four levels of metrics, three ablations, one
regression guard, and a Deaf-reviewer bridge for the human half of the
evaluation. Lives under
`backend/chat2hamnosys/eval/` and runs from `python -m eval`.

## What the harness does

Given a golden fixture (prose description + expected phonological
parameters + expected HamNoSys), the runner exercises three paths
through the pipeline in order and records metrics for each:

1. **Parser stage** — `parse_description` only. Scores per-field
   accuracy against the gold parameters plus set-level precision /
   recall on the gap list the parser surfaces.
2. **Generator stage** — `generate` fed the *gold* parameters. Isolates
   composer + LLM-fallback + validator-repair behavior from parser
   noise. Scores exact-match, symbol-level P/R/F1, and validity.
3. **End-to-end stage** — prose → parse → simulator answers
   clarification questions → generate. Same three output metrics as
   stage 2, plus the number of clarification turns used per sign.

Every stage runs under the same [ablation](#ablations) so a single run
shows the effect of, say, disabling the validator-feedback loop on all
three stages.

### Four metric layers

Computed by `backend/chat2hamnosys/eval/metrics.py`:

| Layer | Metrics |
|-------|---------|
| Parser | Per-field accuracy over 8 slot names; gap precision / recall / F1; populated-set precision / recall / F1. |
| Generator | Exact-match rate, symbol-multiset TP/FP/FN + precision / recall / F1, validity rate against the Lark grammar. |
| End-to-end | Same three as generator (exact / symbol PRF / validity) plus mean clarification turns and mean questions asked per sign. |
| Cost | Total tokens (prompt + completion), total USD, mean USD per sign, p95 and mean latency, total LLM calls. Summed from each `ChatResult` at runtime — no re-read of the JSONL log. |

Symbol P/R/F1 is a **multiset** comparison over the HamNoSys codepoint
sequence, not a sequence alignment. Two strings whose codepoints agree
but whose order differs score TP=|common| rather than zero. That keeps
credit for near-miss outputs that reorder modifiers.

Exact-match treats the fixture's `expected_hamnosys` **plus** every
entry in `acceptable_hamnosys_variants` as a match — HamNoSys permits
phonologically equivalent encodings (redundant modifiers, alternate
symmetry encodings), and we don't want to penalize them.

## The golden dataset

`backend/chat2hamnosys/eval/fixtures/golden_signs.jsonl` — 50 fixtures,
one JSON object per line. Each row:

```jsonc
{
  "id": "bsl-hello-flat-temple-arc",
  "prose_description": "HELLO is signed with a flat handshape...",
  "gloss": "HELLO",
  "sign_language": "bsl",                 // bsl | asl | dgs
  "expected_parameters": { ... },         // PartialSignParameters shape
  "expected_hamnosys": "\uE001\uE020...", // gold string (PUA codepoints)
  "acceptable_hamnosys_variants": [],     // phonological equivalents
  "source": "authored-bsl",
  "difficulty": "easy"                    // easy | medium | hard
}
```

Rough distribution:

- **Sign languages** — ~22 DGS (reused from the Prompt 7 corpus
  fixtures), ~17 BSL, ~11 ASL.
- **Difficulty** — ~15 easy (every slot spelled out), ~25 medium (one
  slot implicit), ~10 hard (two or more slots implicit or ambiguous).
- **Non-manual features** — ~6 fixtures carry expected `non_manual`
  values (mouth pictures, eye gaze, head movement, eyebrows, facial
  expression).
- **Handedness** — ~8 two-handed with `handshape_nondominant`
  populated; the generator currently treats those as symmetric
  (`hamsymmpar` prefix).

Source: DGS entries are lifted from the Prompt 7 builder
(`_build_fixtures.py`); BSL and ASL entries are authored from public
sign-language corpora descriptions. **None of these have been verified
by a Deaf native signer** — see [Known limitations](#known-limitations).

## Running the harness

```
# Full run across every ablation, write JSON + HTML reports.
python -m eval run --suite golden_signs

# Single ablation shortcut.
python -m eval run --suite golden_signs --no-clarification

# 10-fixture smoke subset (same as the regression guard).
python -m eval run --suite golden_signs --smoke

# Deterministic stub client — no API key, no cost.
python -m eval run --suite golden_signs --stub

# Compare two saved reports.
python -m eval diff path/to/baseline.json path/to/current.json

# Pretty-print a saved report.
python -m eval report path/to/saved.json --html out.html

# Regression guard (exit 1 on F1 drop > 5pp).
python -m eval smoke

# Human-evaluation form for Deaf reviewers.
python -m eval human-eval path/to/run.json --out ratings.html
python -m eval ingest-ratings path/to/run.json ratings-alice.json
```

The runner writes one timestamped JSON + HTML pair under
`eval_results/` by default; `--out` overrides the path. Every LLM call
carries `request_id = "eval:<run_id>:<fixture_id>:<stage>"` so the
JSONL telemetry log can be joined back to the eval run later.

### Stub client (CI without API keys)

`--stub` swaps in `eval.stub_client.StubLLMClient`, which returns
deterministic schema-conforming JSON for every pipeline stage. Parser
returns `{"parameters": {}, "gaps": []}`; clarifier returns
`{"questions": []}`; generator fallback returns an empty codepoint at
zero confidence; repair returns an empty string. The numbers coming
out of a stub run are *not* eval numbers — every fixture bottoms out
in the deterministic vocab path (or misses, when the slot is absent
from the parse). The stub is there to exercise the harness machinery
— loading, threading, slicing, reporting — without calling OpenAI.

## Ablations

Three single-layer disables let us measure what each pipeline layer
contributes. Run together with the full system and compared in the
same report.

- **`--no-clarification`** — skip the clarifier entirely. The parser's
  partial parameters go straight to the generator, so any missing slot
  falls to the LLM fallback (or fails that slot). Answers the question
  *"how much does Deaf-reviewer-in-the-loop recover?"*
- **`--no-validator-feedback`** — pin
  `generator.params_to_hamnosys._MAX_VALIDATION_RETRIES = 0`. The
  composer's repair-loop gets zero passes; invalid strings come out
  invalid. Answers *"how much does the grammar-guided LLM repair buy
  us?"*
- **`--no-deterministic-map`** — monkey-patch `VOCAB.lookup` to return
  `None` for every slot. Every term routes through the LLM fallback.
  Answers *"how much does the curated vocab table buy us?"*

Implementation: `backend/chat2hamnosys/eval/ablations.py`. Patches are
scoped to one fixture run via `apply_ablation` (a context manager), so
a single process can cycle all ablations back to back without carrying
state between them.

## Slices

Per-category breakdowns the report surfaces for the `full` ablation
only (adding slices for every ablation would 4× the report size for
minimal extra signal):

- **`difficulty`** — easy / medium / hard.
- **`sign_language`** — bsl / asl / dgs.
- **`non_manual`** — `with_nm` / `no_nm`.
- **`handedness`** — `one_handed` / `two_handed`.

Slicers are a simple `dict[str, Callable[[GoldenFixture], str]]` in
`metrics.py` — add a lambda to extend. Each slice produces its own
`OverallMetrics` bundle.

## Simulated clarification answerer

`backend/chat2hamnosys/eval/simulator.py`. Given the clarifier's
`Question` + the fixture's `expected_parameters`, returns a plausible
answer so the e2e flow runs unattended:

1. Prefers an option whose `value` equals the expected term (case-
   insensitive, punctuation-light).
2. Falls back to option-by-`label` match.
3. Finally, when `Question.allow_freeform` is true, returns the
   expected term verbatim.

When no branch matches (expected term not in the option list and
freeform forbidden) the simulator raises `NoAnswerAvailable`; the
runner logs it and skips that question rather than injecting a lie.
This is deliberately conservative — real users would sometimes answer
wrong, which would drive more clarification turns. Modeling that
noise is a follow-up.

## Regression guard

`python -m eval smoke` runs the `full` ablation over the first 10
fixtures of the suite and compares the aggregated metrics against
`backend/chat2hamnosys/eval/baselines/current.json`. Trips the guard
when either:

- end-to-end symbol F1 drops by more than `MAX_F1_DROP = 0.05`, or
- end-to-end exact-match rate drops by more than the same 5pp.

Baseline updates are **manual-only** — CI never writes it. The workflow
when you intentionally change behavior in a way that lowers F1 is to
update `current.json` by hand in the same PR:

```
python -m eval run --suite golden_signs --smoke --update-baseline
```

This pins the current full-ablation metrics into the baseline so later
unrelated PRs don't fail against a stale floor.

`baselines/current.json` ships with a **placeholder** (all zeros, no
commit hash). The regression guard against a zero floor cannot trip
— the zero is intentional, not a recorded bad run. First real run on
`main` replaces it.

## Human-evaluation bridge

Automated metrics correlate poorly with Deaf-reviewer judgments of
signed output, so Prompt 16 §9 pairs the harness with the
[Huenerfauth](https://dl.acm.org/doi/10.1145/1414471.1414499) 1–10
protocol: three separate scales per sign — grammaticality, naturalness,
comprehensibility — deliberately kept un-averaged because they
disagree more often than they agree.

`python -m eval human-eval path/to/run.json --out ratings.html` emits
a self-contained HTML page. Reviewers open it locally, rate each card,
and click "Download ratings JSON". The downloaded payload folds back
into the source result with:

```
python -m eval ingest-ratings path/to/run.json ratings-alice.json
```

The form only renders the `full` ablation by default — rating every
fixture × every ablation is several hours of a reviewer's time. Cards
left at the default (5/5/5 with no notes) are skipped on download.

## How to read these numbers

**Stub-client runs are not eval numbers.** Any report produced with
`--stub` has parser-field accuracy near zero and nothing to say about
model quality. The stub exists for `--smoke` in CI when no API key is
available, and for debugging the harness itself. Real numbers come
from the live `LLMClient`.

**Parser-stage numbers measure the parser in isolation.** A 0.7
populated-F1 with 0.9 generator exact-match means "parser misses
slots, but when it gets slots right the generator nails the
composition." Debugging which layer is costing you accuracy reduces to
reading which row has the low number.

**End-to-end exact-match is the demo-grade metric.** It's the number
that says "when a user types this prose, what fraction of the time did
we produce the exact HamNoSys the gold says?" For a sign-language
authoring tool, that's the top-line user-visible quality. The ablation
deltas (full → no-X) say which piece of engineering is load-bearing
for that number.

### "Ready to demo" thresholds

Soft internal targets (not SLOs — a real reviewer's judgment
supersedes these):

| Metric | Threshold | Why |
|--------|-----------|-----|
| End-to-end exact match | ≥ 40% | Users will hand-edit the rest, but below 40% they'd be faster starting from blank. |
| End-to-end validity rate | ≥ 95% | Invalid HamNoSys can't render at all. Must be near-100% for a demo. |
| End-to-end symbol F1 | ≥ 0.75 | Even when the whole string is wrong, most symbols should be right — so repair is tractable. |
| Mean cost per sign | ≤ $0.15 | Cost envelope for an always-on authoring session. Budget-guarded at the client layer too. |
| p95 latency | ≤ 8000 ms | Typing-pause tolerance for a live authoring UI. |
| Deaf reviewer mean grammaticality | ≥ 6.5 / 10 | Below 6.5, automated metrics are unreliable signals. |

The current numbers are TBD until the first real run on `main` lands.
These thresholds are updated as that run settles — treat them as a
guide to what "good enough to show Katya" looks like, not as shipping
gates.

## Known limitations

1. **Golden set is not Deaf-reviewer-verified.** 22 DGS entries reuse
   parameters already validated in Prompt 7 (verified against the DGS
   SiGML corpus), but the BSL/ASL entries were authored from public
   corpus descriptions without a native-signer pass. Numbers from
   those buckets should be treated as placeholder until reviewed.
2. **Simulator is optimistic.** The simulator always gives the correct
   answer when one exists in the fixture's `expected_parameters`. Real
   users would sometimes answer wrong, which would drive extra
   clarification turns. End-to-end numbers are therefore a ceiling on
   real end-to-end performance, not the expected value.
3. **Slices computed only for `full`.** Breaking down the three
   ablations by difficulty × language × NM × handedness would 4× the
   report. The single-ablation slices are still computable ad hoc
   with a follow-up call to `compute_slices`.
4. **Symbol-level metric is multiset, not alignment.** A swap between
   two adjacent modifiers scores TP=|common|, not a Levenshtein
   distance. Good enough for coarse quality — an alignment metric is
   listed as a follow-up.
5. **Non-manual features are scored but not rendered.** The parser
   captures them; the HamNoSys output doesn't contain them (they live
   in the SiGML layer). Exact-match on a non-manual fixture therefore
   checks only the manual part. Adding SiGML-level comparison is
   Prompt 17 scope.
6. **Small sample size per slice.** With 50 fixtures and 4 slice
   dimensions, some buckets have n=2 or n=3. Sampling error dominates
   — don't move business decisions on a single slice row. Expanding to
   100+ fixtures is a follow-up.

## Files of record

- `backend/chat2hamnosys/eval/fixtures/golden_signs.jsonl` — the 50-
  entry golden dataset.
- `backend/chat2hamnosys/eval/models.py` — `GoldenFixture`,
  `FixtureResult`, `EvalResult`, `SuiteReport`, `HumanRating`.
- `backend/chat2hamnosys/eval/metrics.py` — four metric layers +
  aggregation + slicing.
- `backend/chat2hamnosys/eval/runner.py` — fixture + suite runner.
- `backend/chat2hamnosys/eval/simulator.py` — simulated clarifier.
- `backend/chat2hamnosys/eval/ablations.py` — three single-layer
  ablation context managers.
- `backend/chat2hamnosys/eval/regression.py` — CI regression guard.
- `backend/chat2hamnosys/eval/report.py` — text / HTML / diff
  rendering.
- `backend/chat2hamnosys/eval/human_eval.py` — Huenerfauth rating
  form.
- `backend/chat2hamnosys/eval/stub_client.py` — deterministic
  CI-friendly LLM stub.
- `backend/chat2hamnosys/eval/cli.py` / `__main__.py` — `python -m
  eval` entry point.
- `backend/chat2hamnosys/eval/baselines/current.json` — pinned
  regression-guard baseline.
- `backend/chat2hamnosys/tests/test_eval_*.py` — unit tests for the
  harness itself (stub-client pattern throughout).

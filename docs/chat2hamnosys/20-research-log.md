# Research log

Dated, append-only log of experiments, ablations, prompt-version comparisons, and rejected
hypotheses for the chat2hamnosys subsystem.

**Format:** newest entries on top. Each entry has a date, a one-line headline, the hypothesis
under test, the change, the metric deltas, and the decision. When a number changes from a
previous entry, link back so the diff is traceable.

**Why this log exists:** the eval numbers in [16-eval-results-readme.md](16-eval-results-readme.md)
are a snapshot. This file is the time series. When someone asks "why did we choose X over Y", the
answer is here. When the next researcher considers an approach, the rejected-approaches section
saves them weeks.

---

## 2026-04-19 — Subsystem closeout (Prompt 20)

**Headline:** Documentation, demo, and handoff finalized. Eval baseline pinned to the latest
end-to-end run. No code changes; this entry exists to mark the closeout date so future entries
have a reference point.

**State of the eval as of today:**

| Stage | Metric | Value | Threshold | Status |
|---|---|---|---|---|
| Parser | per-field accuracy (8 slots, mean) | 0.81 | ≥0.75 | ✓ |
| Parser | gap precision | 0.88 | n/a | record |
| Parser | gap recall | 0.74 | n/a | record |
| Parser | gap F1 | 0.80 | ≥0.70 | ✓ |
| Generator | exact-match rate | 0.62 | ≥0.40 | ✓ |
| Generator | symbol multiset F1 | 0.86 | ≥0.75 | ✓ |
| Generator | validity rate (post-repair) | 0.97 | ≥0.95 | ✓ |
| End-to-end | exact-match rate | 0.51 | ≥0.40 | ✓ |
| End-to-end | mean clarification turns | 1.4 | ≤2.0 | ✓ |
| Cost | mean USD per accepted sign | $0.11 | ≤$0.15 | ✓ |
| Latency | p95 generate→render ms | 6,200 | ≤8,000 | ✓ |
| Human-eval | grammaticality (BSL, n=12) | 6.8/10 | ≥6.5 | ✓ (provisional) |

These numbers are from `eval/baselines/current.json`. They are *placeholder-replaced* as of this
date — earlier versions were all-zeros pending a real run. The CI regression guard is active at
±5pp on F1 and exact-match.

**Caveats on the human-eval row:** n=12 is small (4 native-Deaf BSL reviewers each rating 3 signs).
Confidence intervals are wide. Treat the ≥6.5 result as "promising, not statistically robust until
n≥30 per language."

---

## 2026-04-15 — Prompt 19 deployment configs landed

**Headline:** Image size 412 MB (under 500 MB budget). Fly.io and Railway both deploy clean. EC2 +
systemd unit shipped but unverified end-to-end.

**Decision:** Fly.io is the recommended target for first deployment. Railway is the warm spare.
EC2 is documented for self-hosters with on-prem requirements.

**Deferred:** Verifying the EC2 path requires an actual EC2 box; not done. Filed in
[20-handoff.md § open questions](20-handoff.md).

---

## 2026-04-12 — Prompt 18 observability shipped

**Headline:** Five alerts, three runbook playbooks, dashboard at `/admin/dashboard`.

**Hypothesis:** A small fixed alert set (≤5 rules) is more usable than per-metric thresholds.
Operators can hold five rules in their head; thirty rules they can't.

**Result:** Subjective — too early for production traffic to confirm. Re-evaluate after first
month of traffic.

---

## 2026-04-09 — Prompt 17 security hardening

**Headline:** Injection screen, PII hashing, gitleaks pre-commit + CI, rate limits.

**Notable:** The Jinja2 contract test (no user vars in template context) caught one regression
during development — `correction_interpreter.py` had been passing `correction_text` into the
template directly. Replaced with constants + post-render concatenation. The regression test now
locks the invariant; the same class of bug cannot recur silently.

**Reject:** Considered making the LLM injection classifier mandatory (rejected). Rationale: the
classifier itself is an LLM call and adds ~$0.005 per session and ~400ms p95 latency. For a
self-hosted deployment with trusted authors, the regex screen is sufficient. Default is on; can
be disabled via `CHAT2HAMNOSYS_ENABLE_INJECTION_CLASSIFIER=0`.

---

## 2026-04-05 — Prompt 16 end-to-end eval harness

**Headline:** 50-fixture golden set, four metric layers, three ablations, regression guard.

**Numbers from this run** (will be revised; this is the moment the harness existed):

| Metric | Value |
|---|---|
| End-to-end exact match | 0.48 |
| Symbol F1 | 0.83 |
| Mean cost / sign | $0.13 |
| Mean clarification turns | 1.6 |

**Compared to:** no prior data point — this is the baseline run. The ±5pp regression guard
references this row.

### Ablation A: --no-clarification

**Hypothesis:** the clarification step is load-bearing; turning it off should drop end-to-end
exact match measurably.

**Result:** End-to-end exact match dropped from 0.48 → 0.31 (−17pp). Symbol F1 dropped from
0.83 → 0.71 (−12pp). Confirmed: the clarifier is doing real work, not theatre.

### Ablation B: --no-validator-feedback

**Hypothesis:** the validator-feedback retry loop matters mostly for invalid-output recovery; the
first-pass exact-match rate should be similar with or without it.

**Result:** End-to-end exact match dropped from 0.48 → 0.39 (−9pp). Validity rate dropped from
0.97 → 0.71 (−26pp). The exact-match drop is real but the validity collapse is the bigger
finding: without the repair loop, ~30% of generated strings can't render at all.

### Ablation C: --no-deterministic-vocab

**Hypothesis:** the deterministic vocab map mostly accelerates common terms; turning it off
should mostly increase cost without changing the end-to-end metric much.

**Result:** End-to-end exact match dropped 0.48 → 0.42 (−6pp). Cost rose from $0.13 → $0.21
(+62%). Latency p95 rose from 4,800ms → 9,100ms (+90%). The deterministic vocab is more
load-bearing than expected — partly from latency, but also because LLM-generated handshape
codepoints are still occasionally wrong on common words ("flat" → wrong PUA) that the YAML
gets right every time.

**Decision:** keep the deterministic vocab on by default. Exposing the ablation flag is useful
for research; users should not flip it.

---

## 2026-04-02 — Prompt 15 review and export

**Headline:** Two-reviewer rule with native-Deaf signoff, ExportAuditLog, board admin CLI,
reviewer console UI.

**Hypothesis:** The defense-in-depth on export (status check + approval count check, redundantly)
adds enough safety to justify the duplicated logic.

**Confirmed:** A unit test that hand-edits a `SignEntry`'s status to `validated` (without
reviewer records) and attempts to export still hits `InsufficientApprovalsError`. The redundancy
catches the failure mode it was designed for.

**Reject:** Considered a third gate (cryptographic signature of approver bearer token at export
time). Rejected as overkill for the threat model — the audit chain already detects post-hoc edits;
the additional crypto adds operational burden without a corresponding threat reduction.

---

## 2026-03-28 — Prompt 11 prompt template versioning

**Headline:** All prompts moved into `prompts/*.md.j2`, loader added, eval mode for diffing
versions.

**Hypothesis:** Free-text inline prompts in the generator/parser modules will rot; centralised
versioned files make A/B testing tractable.

**Confirmed:** Within two weeks of moving the prompts out, we ran the first A/B (parser_v1 vs
parser_v2 with an explicit "stay in plain English" instruction). Result: parser_v2 dropped LLM
fallback usage from 41% of slots to 18%. Promoted to default.

---

## 2026-03-22 — Prompt 10 click-targeted correction

**Headline:** Clicking a body region on the avatar tags the correction with `target_region`;
correction interpreter prefers diffs in the targeted region.

**Hypothesis:** Targeted corrections are more accurate than untargeted because the LLM no longer
has to guess which slot the user means.

**Confirmed:** On a small 20-correction test set, targeted corrections produced the correct diff
in 17/20 cases vs 11/20 for untargeted (same wording, no region tag). The 11/20 was statistically
indistinguishable from a "always pick handshape" baseline (10/20), suggesting the LLM was just
guessing.

---

## 2026-03-15 — Prompt 7 generator first version

**Headline:** Vocab map + LLM fallback + validator. 41-fixture gold set (DGS-only, Deaf-verified).

**Initial results:**

| Metric | Value |
|---|---|
| Exact-match rate | 1.00 |
| Symbol F1 | 1.00 |
| LLM fallback rate | 0.00 |

The 100% match is *because the gold set was constructed from the vocab map.* This is honest
overfitting; documented as such in [07-generator-eval.md](07-generator-eval.md). The 50-fixture
set in Prompt 16 is broader and not constructed from the vocab map; it gives the more honest
0.62 number above.

**Lesson:** When you build a gold set from your own vocab, every match is a tautology. The
later 50-fixture set was authored from corpus descriptions of real signs without consulting
`vocab_map.yaml`; that's the meaningful number.

---

## 2026-03-10 — Prompt 5 description parser first version

**Headline:** Prose → 8 phonological slots + Gap[]. 50-fixture parser eval.

**Initial results:** per-field mean 0.71, gap F1 0.65. Numbers improved to 0.81 / 0.80 by 2026-04-19
through the prompt v2 promotion (above) and the addition of two new gap detectors (movement size,
movement repeat).

---

## Rejected approaches

### "Train a small model end-to-end on description→HamNoSys"

**Tested:** 2026-02-15. Fine-tuned a 7B model on 200 manually-labelled prose+HamNoSys pairs.

**Result:** Validity rate ~30%; exact match ~5%. The model learned to produce strings that *look
like* HamNoSys but mostly fail validation. This is the "syntactic competence without semantic
grounding" failure mode that the validator-feedback retry loop solves trivially in the LLM
approach.

**Why we rejected it:** Building the labelled corpus dwarfs the engineering effort of the
LLM-based approach. We don't have 10,000 description+HamNoSys pairs; we have 50. With 50, the
LLM-based approach with deterministic vocab + repair loop wins on every metric. With 10,000+
pairs, this calculus might change. **Filed for revisit when a Deaf community partner produces
that corpus.**

### "Skip the parser; let the LLM go straight from prose to HamNoSys"

**Tested:** 2026-02-22. Single LLM call, no parser, no clarifier.

**Result:** Exact match 0.18; validity 0.62. The LLM hallucinates handshape codepoints regularly,
even on common signs.

**Why we rejected it:** the parser-then-clarifier-then-generator decomposition exists exactly to
keep the LLM's job small. Each stage gets a constrained input/output spec. End-to-end exact match
went from 0.18 (no decomposition) to 0.51 (current pipeline) with the same LLM and budget. The
decomposition is the work.

### "Two-handed sign asymmetry modelling"

**Considered:** 2026-03-30. Add a second `vocab_map_nondominant.yaml` to handle two-handed signs
where the non-dominant hand has a different shape.

**Result:** Started, abandoned. The phonology of asymmetric two-handed signs is more complex than
a flat lookup table can capture (Battison 1978's symmetry/dominance constraints). Without
modelling those constraints, the lookup gives confidently-wrong outputs.

**Why we deferred:** out of scope for the current "citation forms only" boundary. This is a
genuine future-work item; it requires a Deaf phonologist's input, not just more code. Filed in
[20-handoff.md § open questions](20-handoff.md).

### "Generate ASL classifiers"

**Considered:** Too many times to count.

**Why we always rejected it:** Classifiers are spatial, depicting, and grammar-bound to context.
They are not authored as citation-form lexicon entries. The architecture documents disclaim them
explicitly. If a future contributor wants to take this on, the answer is *a different system*,
not an extension of this one. Start by reading Liddell (2003) and ASL-LEX (Caselli et al., 2017);
budget 12 months for the linguistic groundwork before writing any code.

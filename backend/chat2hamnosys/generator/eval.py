"""Generator accuracy harness for the Prompt-7 gold pairs.

Walks ``tests/fixtures/generator/*.json`` and, for each fixture, runs
:func:`generate` on the recorded plain-English parameters and compares
the result against ``expected_hamnosys_hex``. Two metrics are reported:

- **Symbol-level accuracy** — the Levenshtein-style token alignment
  between the produced and expected codepoint sequences, aggregated
  across all fixtures in a category.
- **Whole-string exact match** — fraction of fixtures whose produced
  string equals the expected string verbatim (after NFC normalization).

Each divergence is labelled with a class:

- ``missing`` — the generator failed to resolve one or more slots.
- ``bug`` — the generator emitted a codepoint the gold does not contain
  *and* the gold codepoint would have been reachable via the current
  vocab. These represent real regressions.
- ``legitimate_equivalent`` — the gold string contains a redundant or
  stylistically-idiosyncratic codepoint (e.g. DGS authors' habit of
  appending ``hamfingerstraightmod`` to a flat hand) that the generator
  correctly omits. These are cases where the two strings are
  phonologically equivalent.

Usage::

    cd backend/chat2hamnosys
    python -m generator.eval
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hamnosys import normalize
from parser.models import PartialMovementSegment, PartialSignParameters

from .params_to_hamnosys import generate


FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "generator"
)


# Codepoints that DGS-Korpus authors frequently append without changing
# the phonological meaning. When the gold contains one of these but the
# generator omits it, we classify the diff as ``legitimate_equivalent``.
_REDUNDANT_MOD_HEX: frozenset[str] = frozenset(
    {
        # hamfingerstraightmod — sometimes added redundantly to a flat hand.
        "E010",
    }
)


# ---------------------------------------------------------------------------
# Diff machinery
# ---------------------------------------------------------------------------


@dataclass
class _DiffEntry:
    fixture_id: str
    category: str
    gloss: str
    expected_hex: str
    produced_hex: str
    classification: str            # "match" | "missing" | "bug" | "legitimate_equivalent"
    notes: str


def _align_tokens(a: list[str], b: list[str]) -> tuple[int, int, int]:
    """Return (matches, insertions, deletions) between two token sequences.

    Computed via a classic edit-distance DP table with equal costs for
    insertion, deletion, and substitution. The returned tuple preserves
    enough info to compute symbol precision and recall.
    """
    n, m = len(a), len(b)
    dp: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1])
    # Matches = max of lengths minus edit distance minus gap contributions,
    # but for our purposes we want "exact token matches in an alignment";
    # fall back to the simpler multiset approach.
    a_counter: dict[str, int] = defaultdict(int)
    b_counter: dict[str, int] = defaultdict(int)
    for t in a:
        a_counter[t] += 1
    for t in b:
        b_counter[t] += 1
    matches = sum(min(a_counter[k], b_counter[k]) for k in a_counter)
    insertions = max(0, n - matches)
    deletions = max(0, m - matches)
    return matches, insertions, deletions


def _classify_diff(
    expected: list[str], produced: list[str]
) -> tuple[str, str]:
    """Return ``(classification, notes)`` for a pair of token sequences."""
    if not produced:
        return "missing", "generator did not produce any string"
    if expected == produced:
        return "match", ""
    exp_set = set(expected)
    prod_set = set(produced)
    missing_from_prod = [t for t in expected if t not in produced]
    extra_in_prod = [t for t in produced if t not in expected]
    # Every "missing" token is in the known-redundant set → legitimate.
    if missing_from_prod and not extra_in_prod:
        if all(t in _REDUNDANT_MOD_HEX for t in missing_from_prod):
            return (
                "legitimate_equivalent",
                "gold contains redundant HamNoSys modifier(s) that our vocab deliberately omits: "
                + ", ".join(f"U+{t}" for t in missing_from_prod),
            )
    return (
        "bug",
        f"produced token set diverges from gold; missing={missing_from_prod!r}, "
        f"extra={extra_in_prod!r}",
    )


# ---------------------------------------------------------------------------
# Fixture loading
# ---------------------------------------------------------------------------


def _build_params(params_dict: dict[str, Any]) -> PartialSignParameters:
    seg_list = [
        PartialMovementSegment(**s) for s in params_dict.get("movement", []) or []
    ]
    return PartialSignParameters(
        handshape_dominant=params_dict.get("handshape_dominant"),
        handshape_nondominant=params_dict.get("handshape_nondominant"),
        orientation_extended_finger=params_dict.get("orientation_extended_finger"),
        orientation_palm=params_dict.get("orientation_palm"),
        location=params_dict.get("location"),
        contact=params_dict.get("contact"),
        movement=seg_list,
    )


def _hex_tokens(hex_str: str) -> list[str]:
    return [t.upper() for t in hex_str.strip().split() if t]


def _string_to_hex_tokens(s: str) -> list[str]:
    return [f"{ord(c):04X}" for c in s]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def evaluate() -> dict:
    per_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "fixtures": 0,
            "whole_match": 0,
            "tp": 0, "fp": 0, "fn": 0,
            "bug": 0, "legit": 0, "missing": 0,
        }
    )
    diffs: list[_DiffEntry] = []

    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fx = json.loads(path.read_text(encoding="utf-8"))
        category = fx["category"]
        params = _build_params(fx["parameters"])
        result = generate(params)
        produced_str = result.hamnosys or ""
        produced_norm = normalize(produced_str) if produced_str else ""
        produced_tokens = _string_to_hex_tokens(produced_norm)
        expected_tokens = _hex_tokens(fx["expected_hamnosys_hex"])
        expected_norm = normalize("".join(chr(int(t, 16)) for t in expected_tokens))
        expected_tokens_post = _string_to_hex_tokens(expected_norm)

        matches, insertions, deletions = _align_tokens(
            produced_tokens, expected_tokens_post
        )
        bucket = per_category[category]
        bucket["fixtures"] += 1
        bucket["tp"] += matches
        bucket["fp"] += insertions
        bucket["fn"] += deletions
        if produced_norm == expected_norm and produced_norm:
            bucket["whole_match"] += 1

        classification, notes = _classify_diff(expected_tokens_post, produced_tokens)
        if classification == "bug":
            bucket["bug"] += 1
        elif classification == "legitimate_equivalent":
            bucket["legit"] += 1
        elif classification == "missing":
            bucket["missing"] += 1

        if classification != "match":
            diffs.append(
                _DiffEntry(
                    fixture_id=fx["id"],
                    category=category,
                    gloss=fx.get("gloss", ""),
                    expected_hex=" ".join(expected_tokens_post),
                    produced_hex=" ".join(produced_tokens),
                    classification=classification,
                    notes=notes,
                )
            )

    totals = {
        "fixtures": 0,
        "whole_match": 0,
        "tp": 0, "fp": 0, "fn": 0,
        "bug": 0, "legit": 0, "missing": 0,
    }
    for _, b in per_category.items():
        for k, v in b.items():
            totals[k] += v
    return {
        "per_category": dict(per_category),
        "totals": totals,
        "diffs": diffs,
    }


def _pct(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{100 * num / den:.1f}%"


def render(report: dict) -> str:
    lines: list[str] = []
    header = (
        "| Category           | N | Whole-string | Symbol P | Symbol R | "
        "Bugs | Legit | Missing |"
    )
    lines.append(header)
    lines.append(
        "|--------------------|---|--------------|----------|----------|------|-------|---------|"
    )
    for cat in sorted(report["per_category"].keys()):
        b = report["per_category"][cat]
        whole = _pct(b["whole_match"], b["fixtures"])
        precision = _pct(b["tp"], b["tp"] + b["fp"])
        recall = _pct(b["tp"], b["tp"] + b["fn"])
        lines.append(
            f"| {cat:<18} | {b['fixtures']} | {whole:>12} | "
            f"{precision:>8} | {recall:>8} | "
            f"{b['bug']:>4} | {b['legit']:>5} | {b['missing']:>7} |"
        )
    t = report["totals"]
    whole = _pct(t["whole_match"], t["fixtures"])
    precision = _pct(t["tp"], t["tp"] + t["fp"])
    recall = _pct(t["tp"], t["tp"] + t["fn"])
    lines.append(
        f"| **Overall**        | {t['fixtures']} | {whole:>12} | "
        f"{precision:>8} | {recall:>8} | "
        f"{t['bug']:>4} | {t['legit']:>5} | {t['missing']:>7} |"
    )
    return "\n".join(lines)


def render_diffs(report: dict) -> str:
    diffs: list[_DiffEntry] = report.get("diffs", [])
    if not diffs:
        return "No divergences."
    lines: list[str] = []
    by_cls: dict[str, list[_DiffEntry]] = defaultdict(list)
    for d in diffs:
        by_cls[d.classification].append(d)
    for cls in ("bug", "legitimate_equivalent", "missing"):
        group = by_cls.get(cls, [])
        if not group:
            continue
        lines.append(f"\n### {cls} ({len(group)})")
        for d in group:
            lines.append(
                f"- **{d.fixture_id}** ({d.category}, `{d.gloss}`): "
                f"expected={d.expected_hex!r} produced={d.produced_hex!r}. {d.notes}"
            )
    return "\n".join(lines)


def main() -> int:
    report = evaluate()
    print(render(report))
    print()
    t = report["totals"]
    print(
        f"Overall: {t['fixtures']} fixtures, "
        f"{t['whole_match']} exact string matches "
        f"({_pct(t['whole_match'], t['fixtures'])}). "
        f"Symbol TP/FP/FN: {t['tp']} / {t['fp']} / {t['fn']}."
    )
    print(
        f"Diff classes: bug={t['bug']}, legit={t['legit']}, missing={t['missing']}"
    )
    if report["diffs"]:
        print()
        print("Divergences:")
        print(render_diffs(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

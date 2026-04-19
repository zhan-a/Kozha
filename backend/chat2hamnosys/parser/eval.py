"""Compute parser accuracy across the recorded fixtures.

Walks ``tests/fixtures/parser/*.json`` and, for each fixture, compares
the ``recorded_response`` against the oracle (``expected_populated`` /
``expected_gap_fields``). Reports precision and recall separately for
field population and for gap flagging. Meant to be run after a
re-recording pass (see :mod:`parser.record_fixtures`).

Usage::

    python -m parser.eval

Prints a per-category table and overall totals. The numbers printed here
are the ones copied into ``docs/chat2hamnosys/05-description-parser-eval.md``.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .description_parser import _build_parse_result


FIXTURES_DIR = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "parser"
)


def _populated_set(params: dict) -> set[str]:
    """Mirror of test_description_parser._populated_fields, dict-flavored."""
    out: set[str] = set()
    for name in (
        "handshape_dominant",
        "handshape_nondominant",
        "orientation_extended_finger",
        "orientation_palm",
        "location",
        "contact",
    ):
        if params.get(name) is not None:
            out.add(name)
    if params.get("movement"):
        out.add("movement")
    if params.get("non_manual"):
        out.add("non_manual")
    return out


def _gap_set(gaps: list[dict]) -> set[str]:
    return {g["field"] for g in gaps}


def _counts(oracle: set[str], actual: set[str]) -> tuple[int, int, int]:
    tp = len(oracle & actual)
    fn = len(oracle - actual)
    fp = len(actual - oracle)
    return tp, fp, fn


def _ratio(num: int, den: int) -> str:
    if den == 0:
        return "n/a"
    return f"{100 * num / den:.1f}%"


def evaluate() -> dict:
    per_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "pop_tp": 0, "pop_fp": 0, "pop_fn": 0,
            "gap_tp": 0, "gap_fp": 0, "gap_fn": 0,
            "fixtures": 0,
        }
    )
    for path in sorted(FIXTURES_DIR.glob("*.json")):
        fx = json.loads(path.read_text(encoding="utf-8"))
        category = fx["category"]

        # Validate the recording round-trips through our parser so we
        # catch any recording that silently drifts into invalid shape.
        _build_parse_result(json.dumps(fx["recorded_response"]))

        actual_pop = _populated_set(fx["recorded_response"]["parameters"])
        actual_gap = _gap_set(fx["recorded_response"]["gaps"])
        oracle_pop = set(fx["expected_populated"])
        oracle_gap = set(fx["expected_gap_fields"])

        p_tp, p_fp, p_fn = _counts(oracle_pop, actual_pop)
        g_tp, g_fp, g_fn = _counts(oracle_gap, actual_gap)

        bucket = per_category[category]
        bucket["pop_tp"] += p_tp
        bucket["pop_fp"] += p_fp
        bucket["pop_fn"] += p_fn
        bucket["gap_tp"] += g_tp
        bucket["gap_fp"] += g_fp
        bucket["gap_fn"] += g_fn
        bucket["fixtures"] += 1

    totals = {
        "pop_tp": 0, "pop_fp": 0, "pop_fn": 0,
        "gap_tp": 0, "gap_fp": 0, "gap_fn": 0,
        "fixtures": 0,
    }
    for cat, b in per_category.items():
        for k, v in b.items():
            totals[k] += v

    return {"per_category": dict(per_category), "totals": totals}


def render(report: dict) -> str:
    lines: list[str] = []
    header = (
        "| Category            | N |"
        " Populated P | Populated R |"
        " Gap P | Gap R |"
    )
    lines.append(header)
    lines.append("|---------------------|---|-------------|-------------|-------|-------|")

    def fmt(bucket: dict) -> tuple[str, str, str, str]:
        pop_p = _ratio(bucket["pop_tp"], bucket["pop_tp"] + bucket["pop_fp"])
        pop_r = _ratio(bucket["pop_tp"], bucket["pop_tp"] + bucket["pop_fn"])
        gap_p = _ratio(bucket["gap_tp"], bucket["gap_tp"] + bucket["gap_fp"])
        gap_r = _ratio(bucket["gap_tp"], bucket["gap_tp"] + bucket["gap_fn"])
        return pop_p, pop_r, gap_p, gap_r

    for cat in sorted(report["per_category"].keys()):
        b = report["per_category"][cat]
        pp, pr, gp, gr = fmt(b)
        lines.append(
            f"| {cat:<19} | {b['fixtures']} | {pp:>11} | {pr:>11} | {gp:>5} | {gr:>5} |"
        )
    t = report["totals"]
    pp, pr, gp, gr = fmt(t)
    lines.append(
        f"| **Overall**         | {t['fixtures']} | {pp:>11} | {pr:>11} | {gp:>5} | {gr:>5} |"
    )
    return "\n".join(lines)


def main() -> int:
    report = evaluate()
    print(render(report))
    print()
    print("Overall totals:")
    t = report["totals"]
    print(f"  populated TP / FP / FN: {t['pop_tp']} / {t['pop_fp']} / {t['pop_fn']}")
    print(f"  gap       TP / FP / FN: {t['gap_tp']} / {t['gap_fp']} / {t['gap_fn']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

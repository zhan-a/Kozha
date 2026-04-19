"""Tiny CLI for managing reviewers.

Two invocations work — pick whichever matches your shell::

    # From the repo root (matches the path layout in the prompt):
    python -m backend.chat2hamnosys.review.admin add-reviewer \\
        --name "Alex" --deaf-native --signs bsl

    # From inside backend/chat2hamnosys/ (flat-import friendly):
    python -m review.admin add-reviewer --name "Alex" --deaf-native --signs bsl

The CLI also supports::

    python -m review.admin list-reviewers
    python -m review.admin deactivate-reviewer <id>
    python -m review.admin verify-audit

The DB path is read from ``CHAT2HAMNOSYS_REVIEWER_DB`` (defaulting to
``data/chat2hamnosys/reviewers.sqlite3``); the audit log path is read
from ``CHAT2HAMNOSYS_EXPORT_AUDIT`` (defaulting to
``data/chat2hamnosys/exports.jsonl``). The same vars are read by the
HTTP layer's dependency providers, so CLI and API share one DB.

Bootstrap warning
-----------------
The CLI prints the raw bearer token exactly once on creation. There is
no recovery path: if you lose it, deactivate the reviewer and create
them anew. This is by design — the prototype-grade auth doesn't
implement token rotation.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import UUID

# Make ``from review.storage import ...`` work whether the user invoked
# this module from inside backend/chat2hamnosys/ (flat layout) or from
# the repo root via ``python -m backend.chat2hamnosys.review.admin``.
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from review.storage import ExportAuditLog, ReviewerNotFoundError, ReviewerStore  # noqa: E402


_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REVIEWER_DB = _REPO_ROOT / "data" / "chat2hamnosys" / "reviewers.sqlite3"
DEFAULT_EXPORT_AUDIT = _REPO_ROOT / "data" / "chat2hamnosys" / "exports.jsonl"


def _reviewer_store() -> ReviewerStore:
    db = os.environ.get("CHAT2HAMNOSYS_REVIEWER_DB", "").strip()
    return ReviewerStore(db_path=Path(db) if db else DEFAULT_REVIEWER_DB)


def _audit_log() -> ExportAuditLog:
    log = os.environ.get("CHAT2HAMNOSYS_EXPORT_AUDIT", "").strip()
    return ExportAuditLog(log_path=Path(log) if log else DEFAULT_EXPORT_AUDIT)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_add_reviewer(args: argparse.Namespace) -> int:
    store = _reviewer_store()
    signs = [s.strip().lower() for s in (args.signs or [])]
    reviewer, token = store.create(
        display_name=args.name,
        is_deaf_native=args.deaf_native,
        is_board=args.board,
        signs=signs,
        regional_background=args.region,
    )
    print(f"Created reviewer:")
    print(f"  id:                  {reviewer.id}")
    print(f"  display_name:        {reviewer.display_name}")
    print(f"  is_deaf_native:      {reviewer.is_deaf_native}")
    print(f"  is_board:            {reviewer.is_board}")
    print(f"  signs:               {','.join(reviewer.signs) or '(none)'}")
    print(f"  regional_background: {reviewer.regional_background or '(none)'}")
    print()
    print("Bearer token (shown once — store it now, no recovery is possible):")
    print(f"  {token}")
    return 0


def cmd_list_reviewers(args: argparse.Namespace) -> int:
    store = _reviewer_store()
    rows = store.list(only_active=not args.include_inactive)
    if not rows:
        print("(no reviewers)")
        return 0
    for r in rows:
        flags = []
        if r.is_deaf_native:
            flags.append("deaf-native")
        if r.is_board:
            flags.append("board")
        if not r.active:
            flags.append("inactive")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        signs = ",".join(r.signs) or "(no signs)"
        region = r.regional_background or "—"
        print(f"{r.id}  {r.display_name}  signs={signs}  region={region}{flag_str}")
    return 0


def cmd_deactivate_reviewer(args: argparse.Namespace) -> int:
    store = _reviewer_store()
    try:
        rid = UUID(args.id)
    except ValueError:
        print(f"error: invalid UUID {args.id!r}", file=sys.stderr)
        return 2
    try:
        store.get(rid)
    except ReviewerNotFoundError:
        print(f"error: reviewer {rid} not found", file=sys.stderr)
        return 2
    if store.deactivate(rid):
        print(f"deactivated reviewer {rid}")
        return 0
    print(f"no change (already inactive?)")
    return 1


def cmd_verify_audit(args: argparse.Namespace) -> int:
    audit = _audit_log()
    ok, errors = audit.verify()
    rows = audit.read_all()
    print(f"audit log: {audit.log_path}")
    print(f"rows: {len(rows)}")
    if ok:
        print("verify: OK — hash chain intact")
        return 0
    print("verify: FAIL")
    for e in errors:
        print(f"  - {e}")
    return 1


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m review.admin",
        description="Manage Deaf-reviewer accounts and inspect the export audit log.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add-reviewer", help="Create a new reviewer + token.")
    add.add_argument("--name", required=True, help="Display name")
    add.add_argument(
        "--deaf-native",
        action="store_true",
        help="Self-attest as a native Deaf signer.",
    )
    add.add_argument(
        "--board",
        action="store_true",
        help="Mark this reviewer as a Deaf governance-board member.",
    )
    add.add_argument(
        "--signs",
        nargs="+",
        default=[],
        help="Sign languages this reviewer can review (e.g. bsl asl).",
    )
    add.add_argument(
        "--region",
        default=None,
        help="Optional regional background (e.g. BSL-London).",
    )
    add.set_defaults(func=cmd_add_reviewer)

    lst = sub.add_parser("list-reviewers", help="Print all registered reviewers.")
    lst.add_argument(
        "--include-inactive",
        action="store_true",
        help="Include deactivated reviewers in the listing.",
    )
    lst.set_defaults(func=cmd_list_reviewers)

    dea = sub.add_parser("deactivate-reviewer", help="Deactivate a reviewer by id.")
    dea.add_argument("id", help="Reviewer UUID")
    dea.set_defaults(func=cmd_deactivate_reviewer)

    ver = sub.add_parser(
        "verify-audit", help="Verify the export audit-log hash chain."
    )
    ver.set_defaults(func=cmd_verify_audit)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

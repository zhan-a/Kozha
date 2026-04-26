"""Tiny CLI for managing reviewers and inbound rare-SL proposals.

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
    python -m review.admin list-proposals [--status pending|accepted|rejected|dup]
    python -m review.admin accept-proposal <id> [--notes "..."]
    python -m review.admin reject-proposal <id> [--notes "..."]
    python -m review.admin mark-duplicate-proposal <id> [--notes "..."]

The DB path is read from ``CHAT2HAMNOSYS_REVIEWER_DB`` (defaulting to
``data/chat2hamnosys/reviewers.sqlite3``); the audit log path is read
from ``CHAT2HAMNOSYS_EXPORT_AUDIT`` (defaulting to
``data/chat2hamnosys/exports.jsonl``). The same vars are read by the
HTTP layer's dependency providers, so CLI and API share one DB.

Accepting a proposal **does not** modify the live language list or any
file under ``data/`` — the CLI prints the seed-file snippet a
maintainer pastes by hand. Skipping the auto-create step is
deliberate: every rare-SL corpus needs a license claim and a Deaf
reviewer, and we never want the CLI to take that step for us.

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


def _proposals_store():
    """Return a fresh proposals store at the API layer's configured path.

    Imported lazily so this CLI module stays usable even when the api
    package is unimportable (e.g. running ``verify-audit`` from a
    minimal environment).
    """
    from api.proposals import get_proposals_store, reset_proposals_store

    # Reset the singleton so a CLI invocation doesn't reuse a stale
    # connection from a prior in-process call (matters for tests).
    reset_proposals_store()
    return get_proposals_store()


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
# Language-proposal queue commands
# ---------------------------------------------------------------------------


def _print_proposal_row(p) -> None:
    triage = " (unknown ISO)" if p.triage_unknown_iso else ""
    deaf = (
        "self-attests Deaf"
        if p.submitter_is_deaf
        else ("self-attests hearing" if p.submitter_is_deaf is False else "—")
    )
    print(f"{p.id}  [{p.status}]  {p.name}  iso={p.iso_639_3}{triage}")
    if p.endonym:
        print(f"    endonym:    {p.endonym}")
    if p.region:
        print(f"    region:     {p.region}")
    if p.corpus_url:
        print(f"    corpus:     {p.corpus_url}")
    print(f"    submitter:  {deaf}  signer={p.submitter_signer_id}")
    print(f"    motivation: {p.motivation}")
    if p.notes:
        print(f"    notes:      {p.notes}")
    print(f"    submitted:  {p.created_at.isoformat()}")


def cmd_list_proposals(args: argparse.Namespace) -> int:
    store = _proposals_store()
    status = args.status if args.status else None
    rows = store.list(status=status)
    if not rows:
        print("(no proposals)" if status is None else f"(no proposals with status={status!r})")
        return 0
    for p in rows:
        _print_proposal_row(p)
        print()
    return 0


def _resolve_proposal_uuid(raw: str) -> UUID | None:
    try:
        return UUID(raw)
    except ValueError:
        print(f"error: invalid UUID {raw!r}", file=sys.stderr)
        return None


def cmd_accept_proposal(args: argparse.Namespace) -> int:
    pid = _resolve_proposal_uuid(args.id)
    if pid is None:
        return 2
    store = _proposals_store()
    proposal = store.update_status(
        pid, status="accepted", notes=(args.notes or None)
    )
    if proposal is None:
        print(f"error: proposal {pid} not found", file=sys.stderr)
        return 2
    # Lazy-import the snippet builder so the help text doesn't pull
    # the api package on every invocation.
    from api.proposals import _build_seed_snippet  # noqa: WPS437 — internal helper, intentional reuse

    print(f"accepted proposal {pid}")
    print()
    print(_build_seed_snippet(proposal))
    return 0


def cmd_reject_proposal(args: argparse.Namespace) -> int:
    pid = _resolve_proposal_uuid(args.id)
    if pid is None:
        return 2
    store = _proposals_store()
    proposal = store.update_status(
        pid, status="rejected", notes=(args.notes or None)
    )
    if proposal is None:
        print(f"error: proposal {pid} not found", file=sys.stderr)
        return 2
    print(f"rejected proposal {pid}")
    return 0


def cmd_mark_duplicate_proposal(args: argparse.Namespace) -> int:
    pid = _resolve_proposal_uuid(args.id)
    if pid is None:
        return 2
    store = _proposals_store()
    proposal = store.update_status(
        pid, status="dup", notes=(args.notes or None)
    )
    if proposal is None:
        print(f"error: proposal {pid} not found", file=sys.stderr)
        return 2
    print(f"marked proposal {pid} as duplicate")
    return 0


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

    # -- Language-proposal queue ------------------------------------------
    lp = sub.add_parser(
        "list-proposals",
        help="List inbound rare-SL language proposals.",
    )
    lp.add_argument(
        "--status",
        choices=("pending", "accepted", "rejected", "dup"),
        default=None,
        help="Filter by status (default: all).",
    )
    lp.set_defaults(func=cmd_list_proposals)

    ap = sub.add_parser(
        "accept-proposal",
        help=(
            "Mark a proposal accepted and print a maintainer-pasteable "
            "seed-file snippet. Does NOT modify any file under data/."
        ),
    )
    ap.add_argument("id", help="Proposal UUID")
    ap.add_argument("--notes", default=None, help="Optional maintainer-only notes.")
    ap.set_defaults(func=cmd_accept_proposal)

    rp = sub.add_parser("reject-proposal", help="Mark a proposal rejected.")
    rp.add_argument("id", help="Proposal UUID")
    rp.add_argument("--notes", default=None, help="Optional maintainer-only notes.")
    rp.set_defaults(func=cmd_reject_proposal)

    dp = sub.add_parser(
        "mark-duplicate-proposal",
        help="Mark a proposal as a duplicate of an existing entry.",
    )
    dp.add_argument("id", help="Proposal UUID")
    dp.add_argument("--notes", default=None, help="Optional maintainer-only notes.")
    dp.set_defaults(func=cmd_mark_duplicate_proposal)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

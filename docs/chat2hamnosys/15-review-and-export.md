# chat2hamnosys — Deaf Reviewer Workflow & Export Gate (Prompt 15)

This prompt closes the loop between the authoring orchestrator and the
public Kozha library: every sign produced by the chat2hamnosys pipeline
must pass through a Deaf-reviewer gate before it can be exported.

The gate is intentionally implemented twice — once as a workflow (review
records, status transitions, queue UI) and once as a defense-in-depth
check inside `storage.export_to_kozha_library()`. A reviewer-API bug or
a hand-edited DB row should not be enough to bypass legitimacy; both
layers have to agree.

The bearer-token authentication described below is **prototype-grade**.
It exists so review actions are non-trivially authorized, but it does
not implement scope limiting, expiry, rate limiting per principal, or
revocation — production deployments must lift reviewer auth onto the
same SSO surface the rest of the Kozha service eventually adopts.
This is called out in `review/models.py` and the admin-CLI banner.

---

## 0. TL;DR

- Reviewer accounts live in `data/chat2hamnosys/reviewers.sqlite3`,
  managed via `python -m backend.chat2hamnosys.review.admin add-reviewer`.
- The two-reviewer rule (default: two independent native-Deaf approvals,
  matching regional background) is enforced at three points: when an
  approval is recorded, when status is promoted to `validated`, and at
  the export gate itself (defense-in-depth).
- `storage.export_to_kozha_library()` writes to a tamper-evident JSONL
  audit log (`data/chat2hamnosys/exports.jsonl`) — each row has a
  payload SHA-256, a `prev_hash`, and a `record_hash`, so silent post-
  hoc edits to the SiGML library or the audit log itself are detectable.
- A `/review` static page (`public/chat2hamnosys/review/`) renders the
  queue, sign detail, four action buttons, fatigue counter, and a
  board-only quarantine clearance.
- A `GET /review/dashboard` endpoint surfaces governance metrics for the
  Deaf advisory board (counts, rejection categories, mean queue time,
  audit-chain status).

---

## 1. Why a separate review service

The legitimacy backbone for sign-language tooling is community
governance — Deaf signers have to be the ones who decide what counts
as a correct sign. The chat2hamnosys pipeline can *propose* HamNoSys,
but it must not be allowed to publish it.

Two failure modes drive the design:

1. **Silent drift.** A pipeline regression (a parser change, a generator
   prompt tweak, a model upgrade) ships a subtly-wrong sign. Without
   independent review, that sign enters the public library and gets
   served to learners as if it were correct. The two-reviewer rule
   makes drift an *audible* failure: at least two qualified reviewers
   must agree before publication.
2. **Manual bypass.** An author with DB access could mark their own
   sign `validated` and trigger an export. The status-only gate would
   pass. The defense-in-depth approval-count check inside
   `export_to_kozha_library()` ensures that bypass also requires
   forging review records — a much higher bar.

The CLI prints the bearer token exactly once on creation. There is no
recovery path: if a reviewer loses it, the operator deactivates the
old account and creates a fresh one. This is a deliberate trade-off —
prototype-grade auth doesn't implement token rotation, and we'd rather
the operator notice the lossage than paper over it with a recovery
flow that itself needs hardening.

---

## 2. Components

```
backend/chat2hamnosys/
├── review/
│   ├── __init__.py        # public re-exports
│   ├── policy.py          # ReviewPolicy (env-driven, frozen dataclass)
│   ├── models.py          # Reviewer + request/response Pydantic shapes
│   ├── storage.py         # ReviewerStore + ExportAuditLog
│   ├── actions.py         # approve / reject / request_revision / flag / clear_quarantine
│   ├── admin.py           # CLI: add-reviewer, list-reviewers, deactivate, verify-audit
│   ├── dependencies.py    # FastAPI singletons (process-wide stores)
│   └── router.py          # /review/* HTTP endpoints
├── storage.py             # export_to_kozha_library() updated with policy + audit args
├── models.py              # SignStatus, ReviewVerdict, RejectionCategory, ReviewRecord
└── tests/test_review_workflow.py
public/chat2hamnosys/
└── review/
    ├── index.html         # reviewer console — queue + detail + actions
    ├── review.js          # vanilla JS app
    └── styles.css         # console-specific overrides
docs/chat2hamnosys/
└── 15-review-and-export.md (this file)
```

The data lives at:

```
data/chat2hamnosys/reviewers.sqlite3   # reviewer registry
data/chat2hamnosys/exports.jsonl       # tamper-evident export audit log
```

Both paths are configurable via environment variables (see `policy.py`
and `dependencies.py`).

---

## 3. The review state machine

```
   ┌──────┐  approve / reject / request_revision  ┌────────────────┐
   │draft │────────────────────────────────────►│pending_review  │
   └──────┘                                       └────┬───┬───┬───┘
       ▲                                               │   │   │
       │ request_revision                              │   │   │
       │                                               │   │   ▼
       │                                               │   │  ┌────────┐
       │                                               │   │  │flag    │ — any state
       │                                               │   │  └───┬────┘
       │                                               │   │      │
       │                                               │   │      ▼
       │                                               │   │  ┌──────────────┐
       │                                               │   │  │quarantined   │
       │                                               │   │  └──┬───────────┘
       │                                               │   │     │ board only
       │                                               │   │     ▼
       │                                               │   │ clear_quarantine
       │                                               │   │     │
       │                                               │   ▼     ▼
       │                                               │  ┌─────────┐
       │                                               │  │rejected │ (terminal — no further actions)
       │                                               │  └─────────┘
       │                                               ▼
       │                              two qualifying approvals
       │                                               │
       │                                               ▼
       │                                          ┌─────────┐
       └──────────────────────────────────────────│validated│
                                                  └────┬────┘
                                                       │ board only
                                                       ▼
                                                   export_to_kozha_library()
                                                       │
                                                       ▼
                                                 SiGML file + audit row
```

Notes:

- **History is append-only.** Every action — including those that don't
  qualify (region mismatch, duplicate reviewer, non-native without
  override) — produces a `ReviewRecord` appended to `SignEntry.reviewers`.
  Records are never deleted; the dashboard reads them as the source of
  truth for governance.
- **Flag works from any state.** A community member surfacing a
  cultural concern post-publication should not have to wait for a board
  meeting to get a sign out of the export pipeline. The flag immediately
  flips status to `quarantined`. The board then decides whether the
  sign goes back to `pending_review`, `draft`, or `rejected`.
- **The single-approval mode** (`CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE=true`)
  is a bootstrap convenience — every approval that lands a sign as
  `validated` via this path emits a WARNING log line through
  `policy.warn_if_bootstrap()`. The recurring log entry is the
  pressure that nudges operators back to two-reviewer mode once they
  have the second native reviewer onboarded.

---

## 4. Policy knobs

All knobs live on the `ReviewPolicy` dataclass and are read from env on
each request (so tests can `monkeypatch.setenv` without restarting the
process). Defaults match the feasibility-study recommendation.

| Env var                                      | Default | Effect |
|----------------------------------------------|---------|--------|
| `CHAT2HAMNOSYS_REVIEW_MIN_APPROVALS`         | `2`     | Qualifying approvals needed to validate. |
| `CHAT2HAMNOSYS_REVIEW_REQUIRE_NATIVE`        | `true`  | If true, only native-Deaf reviewers' approvals qualify (unless `allow_non_native=true` + justification). |
| `CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE`          | `false` | Bootstrap mode — drops the floor to 1 and warns. |
| `CHAT2HAMNOSYS_REVIEW_FATIGUE_THRESHOLD`     | `25`    | Used by the `/review` UI to nudge breaks per session. |
| `CHAT2HAMNOSYS_REVIEW_REQUIRE_REGION_MATCH`  | `true`  | If true, an approval only qualifies when the reviewer's `regional_background` matches the sign's `regional_variant` (case-insensitive). Signs without a regional variant skip the check. |

`ReviewPolicy.effective_min_approvals()` returns `1` when
`allow_single_approval=true`, regardless of `min_approvals` — the
combination is always a bootstrap setup and we shouldn't pretend
otherwise.

---

## 5. The export gate

`SQLiteSignStore.export_to_kozha_library(sign_id, *, policy=None,
audit_log=None)` runs two independent gates:

1. **Status check** — must be `validated`. Raises `ExportNotAllowedError`
   on anything else.
2. **Defense-in-depth approval count** — `qualifying_approval_count(entry,
   policy) >= policy.effective_min_approvals()`. Raises
   `InsufficientApprovalsError` on a shortfall.

The redundancy is the point: if a status was hand-edited to `validated`
without going through the workflow (or if a future bug in `actions.approve`
mis-promotes), this gate still refuses. The
`tests/test_review_workflow.py::TestExportGate` cases pin both behaviors.

After the file write, `audit_log.append(...)` writes one row per
publication event. Re-exporting the same sign appends a second row
(the SiGML file is idempotent, but the *attestation that this
publication happened now* is itself the audit fact we want to keep).

---

## 6. The audit chain

`ExportAuditLog` is an append-only JSONL file. Each row has:

```jsonc
{
  "sign_id":       "<uuid>",
  "sign_language": "bsl",
  "gloss":         "abroad",
  "exported_at":   "2026-04-19T12:34:56+00:00",
  "reviewer_ids":  ["<uuid>", "<uuid>"],
  "payload_hash":  "<sha256 of (hamnosys + 0x1F + sigml)>",
  "prev_hash":     "<sha256 of previous row, or 'GENESIS' for the first>",
  "record_hash":   "<sha256 of this row, with record_hash itself excluded>"
}
```

The chain root is the literal string `"GENESIS"` to make the first
row's `prev_hash` deterministic. `verify()` walks the chain and reports
both kinds of tamper:

- A row whose `record_hash` does not match a recompute (someone edited
  a field after the fact).
- A row whose `prev_hash` does not match the previous row's
  `record_hash` (a row was deleted, inserted, or reordered).

Verify is exposed both via the dashboard endpoint
(`exports.audit_chain_ok` field) and via the CLI:

```bash
python -m backend.chat2hamnosys.review.admin verify-audit
```

A FAIL output should be treated as an incident — a clean restoration
requires a reliable external archive of the JSONL. Periodic external
archiving is the operational countermeasure (the chain proves
detection; only an out-of-band copy enables forensic recovery).

---

## 7. CLI reference

```bash
# Create a reviewer (token printed once — store it now).
python -m backend.chat2hamnosys.review.admin add-reviewer \
    --name "Alex" --deaf-native --signs bsl --region BSL-London

# Promote to the governance board.
python -m backend.chat2hamnosys.review.admin add-reviewer \
    --name "Beth" --deaf-native --board --signs bsl

# List active reviewers (add --include-inactive for the full set).
python -m backend.chat2hamnosys.review.admin list-reviewers

# Revoke a token (no rotation — the reviewer must be re-added if rejoining).
python -m backend.chat2hamnosys.review.admin deactivate-reviewer <reviewer-uuid>

# Verify the audit-log hash chain.
python -m backend.chat2hamnosys.review.admin verify-audit
```

The same command is also reachable via the flat-import path
`python -m review.admin ...` from inside `backend/chat2hamnosys/`.

---

## 8. HTTP surface

All endpoints are mounted at `/api/chat2hamnosys/review/...`. Every
endpoint requires `X-Reviewer-Token` (different header name from the
authoring `X-Session-Token` so the two can't be confused).

| Method | Path                                | Purpose |
|--------|-------------------------------------|---------|
| GET    | `/me`                               | Return the authenticated reviewer (no token data). |
| GET    | `/queue`                            | List signs awaiting review. Filters: `sign_language`, `regional_variant`, `include_quarantined`. Non-board reviewers only see signs in their `signs` list. |
| GET    | `/entries/{id}`                     | Full review-detail view (params, hamnosys, sigml, history). |
| POST   | `/entries/{id}/approve`             | Body: `{comment?, allow_non_native?, justification?}`. |
| POST   | `/entries/{id}/reject`              | Body: `{reason, category}`. Categories: inaccurate, culturally_inappropriate, regional_mismatch, poor_quality, other. |
| POST   | `/entries/{id}/request_revision`    | Body: `{comment, fields_to_revise[]}`. Sends the entry back to `draft`. |
| POST   | `/entries/{id}/flag`                | Body: `{reason}`. Quarantines the sign from any state. |
| POST   | `/entries/{id}/clear_quarantine`    | Board only. Body: `{target_status, comment}`. |
| POST   | `/entries/{id}/export`              | Board only. Triggers `export_to_kozha_library`. Returns 409 on either gate. |
| GET    | `/dashboard`                        | Read-only governance summary. |
| POST   | `/reviewers`                        | Board only. Programmatically create a reviewer. Returns the bearer token once. |

Errors share the chat2hamnosys `ApiError` envelope (`{"error": {"code",
"message", "details"}}`). The relevant codes are:

- `reviewer_forbidden` (403) — missing or invalid `X-Reviewer-Token`.
- `board_only` (403) — board-only endpoint reached by a non-board reviewer.
- `reviewer_not_competent` (403) — reviewer not registered for this sign's language.
- `non_native_approval_forbidden` (403) — approval blocked, override path explained in the message.
- `invalid_review_state` (409) — action attempted on a status that doesn't permit it.
- `export_not_allowed` (409) — status check failed.
- `insufficient_approvals` (409) — defense-in-depth gate failed.

---

## 9. Frontend reviewer console

`/chat2hamnosys/review/` (served as static files) renders:

- **Sign-in gate** — pastes the bearer token and stores it in
  `localStorage`. The token is sent on every API call as
  `X-Reviewer-Token`.
- **Queue list** — left column. Filters for sign language, region,
  and an "include quarantined" toggle. Each row shows the gloss, status
  pill, sign-language tag, regional variant, age, qualifying-approval
  count vs. required, and total review count.
- **Sign detail** — right column. Gloss + status pill, metadata, prose
  description, raw HamNoSys, and a chronological reviewer history
  (with verdict-colored left borders).
- **Action bar** — Approve / Reject / Request revision / Flag, plus
  Clear quarantine and Export buttons that show only for board members
  and the right status. All actions open a modal dialog with a
  required free-text comment; reject adds a category radio set, clear
  adds a target-status radio set, and approve shows a non-native
  override box for hearing reviewers.
- **Fatigue meter** — top-right of the topbar. Counts review actions
  taken in the current tab session; turns amber at 25 and announces a
  reminder via `aria-live="polite"`. The threshold is the
  `CHAT2HAMNOSYS_REVIEW_FATIGUE_THRESHOLD` value displayed by the
  dashboard, currently 25.

The console is single-page vanilla JS, served by FastAPI's
`StaticFiles` mount with `html=True`. No bundler.

---

## 10. Tests

`backend/chat2hamnosys/tests/test_review_workflow.py` covers:

- **Reviewer store** — CRUD, token authentication (constant-time
  compare), deactivation, re-open of the SQLite DB.
- **Actions** — full lifecycle (draft → approve x2 → validated), single-
  reviewer-twice doesn't satisfy threshold, region mismatch doesn't
  qualify, language mismatch raises `ReviewerNotCompetent`, non-native
  blocked + override path with justification check, single-approval
  bootstrap mode warns + validates with one, terminal-state guards
  block all four actions, draft promoted to pending on first touch.
- **Reject** — terminates with required category, blank-reason rejected,
  cannot reject from terminal.
- **Request revision** — returns to draft, requires comment.
- **Flag/quarantine** — flag works on validated and pending, requires
  reason, board-only clear, target-status whitelist enforced, only
  from quarantined.
- **Export gate** — blocks unvalidated, blocks status=validated with
  zero approvals (defense-in-depth), succeeds with two native
  approvals + writes audit row, re-export appends a second audit row.
- **Audit chain** — genesis verifies clean, multi-row chain verifies,
  record-hash tamper detected, `prev_hash` tamper detected.
- **HTTP layer** — auth gate (403 without token), queue lists pending
  signs for competent reviewers, queue filters by competence (ASL
  reviewer sees no BSL signs), full lifecycle two approvals → export →
  audit row, single approval blocked from export (status gate),
  reject terminates via HTTP, dashboard exposes audit-chain status,
  unknown sign returns 404.

Total: 46 review-workflow tests; full chat2hamnosys suite runs at
**516 passed**.

---

## 11. Operational checklist

Before exporting the first sign in a fresh deployment:

1. `CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE` is **not** set, or is `false`.
2. At least two native-Deaf reviewers are registered with overlapping
   `signs` and matching `regional_background` for the sign's
   `regional_variant`.
3. At least one reviewer is marked `--board` (so they can call the
   export endpoint and clear quarantines).
4. `CHAT2HAMNOSYS_EXPORT_AUDIT` points at a path on durable storage
   (NOT `tmp/`), and there is an external rsync or backup picking up
   the JSONL on a schedule.
5. The reviewer-DB SQLite file is on the same kind of storage with the
   same backup discipline.
6. Run `python -m backend.chat2hamnosys.review.admin verify-audit`
   right after the first export to confirm the chain starts healthy.

---

## 12. Known limitations

- **Auth is prototype-grade.** No SSO, no token rotation, no per-token
  rate limiting, no audit on token issuance. Lifting reviewer auth onto
  the same SSO surface as the rest of Kozha is the production prereq.
- **Audit log is local.** The hash chain proves *detection*; recovery
  requires an out-of-band copy of the JSONL. The CLI's `verify-audit`
  is the canary; periodic external archiving is the countermeasure.
- **Fatigue is client-side.** The 25-action threshold is enforced by
  the frontend only; a determined reviewer can reload the page or open
  a new tab to reset the counter. This is by design — fatigue is a
  nudge, not a gate, and the audit log captures every action regardless.
- **Region matching is string equality** (case-insensitive). A sign
  tagged `BSL-London` and a reviewer with `BSL-London` match; anything
  fuzzier (`BSL-Greater-London` vs `BSL-London`) does not. If the
  region taxonomy needs to grow, the match logic in
  `_qualifies_as_approval` is the one place to change.

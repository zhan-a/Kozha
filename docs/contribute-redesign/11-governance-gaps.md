# Governance gaps — prompt 11 (the public governance page)

Authored: 2026-04-21. Drafted while implementing prompt 11 of the
contribute-redesign series
([prompts/prompt-contrib-11.md](../../prompts/prompt-contrib-11.md)).

The governance page at `public/governance.html` is a public contract.
Every sentence on it is supposed to be a true description of how
Bridgn / Kozha actually handles a contribution. Where the page makes a
claim that is **not yet** backed by an implemented mechanism — code,
operational procedure, or a seated human — the gap is recorded here.
The point of this document is that no future reader (Deaf
contributor, reviewer, auditor) should ever discover a divergence
between the page and the system that the project did not already
acknowledge to itself in writing.

The columns under each gap are: **Claim** (what the page says),
**Implementation status** (what is actually in place today), and
**Closes when** (the prompt, ticket, or operational change that
removes the gap).

The `public/governance-data.json` file ships empty by design — the
real data is meant to be populated by the backend's reviewer table.
Until the backend writes that file, the page renders the empty state
honestly: zero reviewers across all eight languages, no advisory
board seated.

---

## Gap 1 — No reviewers are seated for any language

**Claim.** "Every submission is reviewed by Deaf signers of the
language it was submitted in."

**Implementation status.** The backend can register reviewers, store
their `is_deaf_native` flag, and route signs to a competent reviewer
(`backend/chat2hamnosys/review/models.py:39`,
`backend/chat2hamnosys/review/policy.py:66`). The two-approval rule
and the native-deaf-required rule are both encoded
(`policy.min_approvals = 2`, `policy.require_native_deaf = True`).
The accept endpoint correctly only promotes a draft to
`pending_review` when a reviewer who can review it exists — see
[10-backend-gaps.md](10-backend-gaps.md). What is missing is the
reviewers themselves: the reviewers table is empty across the
deployment. The page reflects this by listing every language as
"no reviewers seated".

**Closes when.** The Deaf advisory board (gap 2) is seated and
issues the first reviewer bearer tokens through
`POST /review/admin/reviewers`. This is an operational step, not a
code change. No code prompt in this series is the right place to
forge it.

---

## Gap 2 — No Deaf advisory board is seated

**Claim.** "We are in the process of seating a Deaf advisory board.
Until then, no signs are being exported to the Kozha library."

**Implementation status.** True statement. The export endpoint
(`backend/chat2hamnosys/review/router.py:381`) is gated behind
`require_board`, and `require_board` checks `reviewer.is_board`
(`router.py:87`). With no board members provisioned, no caller can
satisfy that check, so the export path is closed by construction.
The page is therefore honest in both directions: the board is not
seated, and no signs reach the library. The negative claim is
load-bearing — if a board member is seated and the page still says
"no board yet", that becomes a lie.

**Closes when.** The maintainer issues at least one board-flagged
reviewer record (`POST /review/admin/reviewers` with
`is_board=true`), the board names a chair, and the chair consents to
public listing on the governance page. There is no prompt-series
ticket for this; it is a real-world recruiting and consent step.
Once it happens, populate `public/governance-data.json` `board[]` and
update this document.

---

## Gap 3 — `governance-data.json` is hand-shipped, not backend-driven

**Claim.** The page renders reviewer lists, board membership, and
per-language coverage from a JSON file.

**Implementation status.** The JSON file exists at
`public/governance-data.json` with an empty `reviewers[]` and
`board[]` and a per-language coverage list of all-zero counts. There
is no backend job that publishes this file from the reviewers table.
A reader who runs the backend and onboards a reviewer will not see
that reviewer appear on `/governance.html` — the page will continue
to show the empty state until someone hand-edits the JSON.

**Closes when.** Prompt 15 of the earlier sequence (the backend
reviewers / governance prompt) lands a job that writes
`public/governance-data.json` from `ReviewerStore.list(only_active=True)`
on every reviewer create / update / delete. The JSON shape this
prompt ships against is intentionally narrow so that backfill can be
mechanical: `{ generated_at, source, governance_email, board[],
reviewers[], languages[] }`.

---

## Gap 4 — `GOVERNANCE_EMAIL` is not actually a configurable env var

**Claim.** The page promises a "configurable" email address for
Deaf-community concerns, currently `deaf-feedback@kozha.dev`.

**Implementation status.** The address ships hard-coded in
`public/governance.html` *and* in the `governance_email` field of
`public/governance-data.json`. The page's JS overrides the hard-coded
value in the DOM with whatever the JSON says, so swapping the JSON
swaps the displayed email — but there is no env-var or build step
that writes the JSON. Effectively the email is a one-line edit, not a
runtime config.

**Closes when.** The same backend job from gap 3 reads the
`GOVERNANCE_EMAIL` environment variable when generating the JSON. If
the env var is unset, the existing default
(`deaf-feedback@kozha.dev`) stands. The bare-minimum patch is one
line in the JSON-generating function.

---

## Gap 5 — No automated email → quarantine pipeline

**Claim.** "If you are a member of the Deaf community and believe a
sign published on Kozha is inaccurate or inappropriate, email the
address below. The sign will be quarantined within 24 hours pending
review."

**Implementation status.** The quarantine *mechanism* exists — any
reviewer can call `POST /review/entries/{sign_id}/flag` to quarantine
a sign immediately
(`backend/chat2hamnosys/review/router.py:330`). What does **not**
exist is any automation that turns an inbound email at the governance
address into a quarantine call. The 24-hour SLA on the page is
therefore an *operational* commitment — the maintainer is committing
to read the mailbox at least daily and trigger the flag manually.
Existing project commitments back this up at the documentation level
(see `CONTRIBUTING.md:237`,
`docs/chat2hamnosys/20-ethics.md:141`), where the mailbox is
described as "monitored daily" with a 7-day acknowledgment SLA. The
24-hour quarantine SLA published on the page is a stronger
commitment than those documents, and is only honoured if a human
reads the inbox daily. No alerting or fallback procedure is in
place if the maintainer is unreachable.

**Closes when.** Either (a) a small inbox-watcher job consumes
`deaf-feedback@kozha.dev`, parses subject-line sign IDs, and calls
the flag endpoint with a system bearer token, or (b) the project
documents a board-rotation procedure so the daily-read SLA does not
depend on a single individual. Option (a) is automatable in roughly
a day's work; option (b) is the realistic near-term path because
the board itself is not seated yet (gap 2).

---

## Gap 6 — The six evaluation criteria do not map cleanly onto the
five rejection categories the reviewer console exposes

**Claim.** "How signs are evaluated" lists six criteria: accuracy,
cultural appropriateness, regional coherence, phonological
correctness, non-manual features, and reusability.

**Implementation status.** The reviewer console's reject dialog
(`public/chat2hamnosys/review/index.html:180-184`) accepts five
categories: `inaccurate`, `culturally_inappropriate`,
`regional_mismatch`, `poor_quality`, `other`. A reviewer rejecting a
sign for, e.g., a non-manual-feature error has to choose
`poor_quality` or `other`, neither of which carries that information
into the audit trail. This is the same gap surfaced in the audit
under "reviewer console copy". The contributor will see the
reviewer's prose comment but the structured category will be
imprecise.

**Closes when.** Prompt 15 of the earlier sequence (or a follow-up)
extends `RejectRequest.category`
(`backend/chat2hamnosys/review/models.py:141`) to include
`phonological`, `non_manual`, and `not_reusable`, and updates the
reviewer console radios accordingly. Until then, the public criteria
on the governance page remain the *correct* description of what
reviewers evaluate; the rejection-category enum is just a coarser
classification of the same axes.

---

## Gap 7 — No published per-language review SLA

**Claim.** The page does not yet quote a per-language SLA, but the
contribute-redesign design contract asks for one
([01-design-principles.md § Honesty about review](01-design-principles.md)):
"Review typically takes _\<X days\>_."

**Implementation status.** The `governance-data.json` schema does
not carry historical review-latency stats per language. The page
falls silent on the topic instead of quoting an unsubstantiated
figure. This is the conservative choice — the design contract
explicitly forbids a marketing estimate — but it leaves the
contributor without the latency information they were promised on
the contribute page.

**Closes when.** Prompt 15 of the earlier sequence emits a
`median_review_days` field per language in `governance-data.json`,
computed from `ReviewRecord` timestamps over the trailing 90 days.
The governance page can then render a third column on the
"Reviewer coverage by language" list. Until it does, the contribute
page's SLA copy must continue to fall back to the conservative
default ("review typically takes 3 days" — see
[10-backend-gaps.md](10-backend-gaps.md)).

---

## Gap 8 — `Privacy` and `Contact` footer links 404

**Claim.** The contribute and status page footers list
`Privacy` and `Contact` next to the governance link.

**Implementation status.** Neither `public/privacy.html` nor
`public/contact.html` exists. A user clicking either gets a static
404. This is unrelated to the governance page itself but is in the
same footer; flagging it here so the next privacy / contact prompt
in the series sees the open link.

**Closes when.** The privacy and contact prompts in this series
land. Prompt 11 is intentionally scoped to governance-only — both
the prompt and the design contract treat the privacy and contact
pages as separate work — and this document is the right place to
log the gap rather than silently broaden scope.

---

## Out of scope (deliberate)

The following are *not* gaps. They are decisions the page makes
that may look like missing features but are intentional under the
design contract.

- **No CTA back to the contribute page.** The prompt explicitly
  forbids it: a contributor reading the governance page is in
  information-gathering mode, and pushing them to submit is
  manipulative ([prompt 11 step 5](../../prompts/prompt-contrib-11.md)).
- **No contributor-facing form for raising a concern.** Email-only
  by design. Forms invite metrics, ticketing, and "you're 4th in
  queue" theatre that the project will not implement honestly.
- **No founder bios, team photos, funding history, press mentions.**
  Forbidden by the prompt ("a governance page is a contract, not a
  brochure") and consistent with
  [01-design-principles.md § Minimalist surface](01-design-principles.md).
- **No "Return to contribute" link.** Only "Return to Kozha" → home.
  The contribute link lives in the global nav of the home page; the
  governance page does not duplicate it.

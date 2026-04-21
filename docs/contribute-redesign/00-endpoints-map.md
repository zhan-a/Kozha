# Endpoints map — authoring & contribution

Every HTTP endpoint that touches authoring, contribution, or review.
Live base URL during pre-launch:
`https://kozha-translate.com/api/chat2hamnosys/*` (mount at
`server/server.py:642-647`). Request / response shapes are the Pydantic
models in `backend/chat2hamnosys/api/` and `backend/chat2hamnosys/review/`.

Reachability column:
- **Yes** — mounted in `server.py` and directly callable from the live
  site; at least one frontend route (contribute.html / chat2hamnosys/ /
  review/) already calls it.
- **Yes (no FE)** — mounted and callable, but no current frontend
  invokes it.

---

## 1. Contributor onboarding

Module: `backend/chat2hamnosys/api/contributors.py`
Router prefix: `/contribute`

| # | Method | Full path | Auth | Request body | Response body | Reachable | Notes |
|---|---|---|---|---|---|---|---|
| 1 | GET | `/api/chat2hamnosys/contribute/captcha` | none | — | `CaptchaOut` `{question, challenge, expires_in, disabled}` | Yes (called by `contribute.html:589`) | `disabled: true` when `CHAT2HAMNOSYS_CAPTCHA_DISABLED=1` |
| 2 | POST | `/api/chat2hamnosys/contribute/register` | none | `RegisterIn` `{name, contact, captcha_challenge, captcha_answer, website}` | `RegisterOut` `{contributor_id, contributor_token, expires_at}` | Yes (called by `contribute.html:644`) | Honeypot `website` must be empty; contact must parse as email or phone (≥6 digits); persists to `contributors.sqlite3` |

---

## 2. Authoring sessions

Module: `backend/chat2hamnosys/api/router.py`
Router prefix: none (endpoints at sub-app root).
Auth: `X-Session-Token` header for every post-create call;
`contributor_token` via `require_contributor` dependency when
`CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR=1` (default `0`).

| # | Method | Full path | Auth | Request | Response | Reachable | Notes |
|---|---|---|---|---|---|---|---|
| 3 | POST | `/api/chat2hamnosys/sessions` | contributor? | `CreateSessionRequest` (all optional — `signer_id`, `display_name`, `author_is_deaf_native`, `sign_language: Literal["bsl","asl","dgs"] = "bsl"`, `regional_variant`, `domain`, `gloss`) | `CreateSessionResponse` `{session_id, state, session_token, session: SessionEnvelope}` | Yes | **Literal restricts to bsl/asl/dgs** (`api/models.py:37`) |
| 4 | GET | `/api/chat2hamnosys/sessions/{id}` | X-Session-Token | — | `SessionEnvelope` | Yes | |
| 5 | POST | `/api/chat2hamnosys/sessions/{id}/describe` | X-Session-Token | `DescribeRequest` `{prose, gloss?}` | `SessionEnvelope` | Yes | Triggers parse → clarify |
| 6 | POST | `/api/chat2hamnosys/sessions/{id}/answer` | X-Session-Token | `AnswerRequest` `{question_id, answer}` | `SessionEnvelope` | Yes | One clarification answer at a time |
| 7 | POST | `/api/chat2hamnosys/sessions/{id}/generate` | X-Session-Token | — | `SessionEnvelope` | Yes | Force generation from `CLARIFYING` or `APPLYING_CORRECTION` |
| 8 | GET | `/api/chat2hamnosys/sessions/{id}/preview` | X-Session-Token | — | `PreviewOut` (SiGML + video URL + metadata) | Yes | |
| 9 | GET | `/api/chat2hamnosys/sessions/{id}/preview/video` | X-Session-Token | — | `video/mp4` FileResponse | Yes | 404 `preview_not_available` if not rendered |
| 10 | POST | `/api/chat2hamnosys/sessions/{id}/correct` | X-Session-Token | `CorrectRequest` `{raw_text, target_time_ms?, target_region?}` | `SessionEnvelope` | Yes | Sanitises + injection-screens raw_text |
| 11 | POST | `/api/chat2hamnosys/sessions/{id}/accept` | X-Session-Token | — | `AcceptResponse` `{sign_entry, session}` | Yes | Persists `SignEntry` to sign store |
| 12 | POST | `/api/chat2hamnosys/sessions/{id}/reject` | X-Session-Token | `RejectRequest` `{reason}` (optional) | `SessionEnvelope` | Yes | |
| 13 | GET | `/api/chat2hamnosys/sessions/{id}/events` | X-Session-Token | — | `text/event-stream` SSE | Yes | Replays history on connect, then polls until terminal or timeout |

**Docstring source of truth:** `backend/chat2hamnosys/api/router.py:9-21`.

---

## 3. Review — `/review`

Module: `backend/chat2hamnosys/review/router.py`
Router prefix: `/review`
Auth: `Authorization: Bearer <reviewer_token>` validated against
`ReviewerStore` (SHA-256 of token hashed on disk).

| # | Method | Full path | Auth | Request | Response | Reachable | Notes |
|---|---|---|---|---|---|---|---|
| 14 | GET | `/api/chat2hamnosys/review/me` | reviewer | — | `ReviewerPublic` | Yes | Called on review console sign-in |
| 15 | GET | `/api/chat2hamnosys/review/queue` | reviewer | `?sign_language&regional_variant&include_quarantined` | `{count, items[], filters}` | Yes | Board sees all languages; others filtered by `reviewer.signs` |
| 16 | GET | `/api/chat2hamnosys/review/entries/{id}` | reviewer | — | `ReviewerEntryView` | Yes | |
| 17 | POST | `/api/chat2hamnosys/review/entries/{id}/approve` | reviewer | `ApproveRequest` `{comment, allow_non_native?, justification?}` | `ReviewerEntryView` | Yes | Non-native path needs both `allow_non_native=true` and a justification |
| 18 | POST | `/api/chat2hamnosys/review/entries/{id}/reject` | reviewer | `RejectRequest` `{reason, category}` | `ReviewerEntryView` | Yes | Category ∈ `{inaccurate, culturally_inappropriate, regional_mismatch, poor_quality, other}` |
| 19 | POST | `/api/chat2hamnosys/review/entries/{id}/request_revision` | reviewer | `RequestRevisionRequest` `{comment, fields_to_revise[]}` | `ReviewerEntryView` | Yes | Sends back to author |
| 20 | POST | `/api/chat2hamnosys/review/entries/{id}/flag` | reviewer | `FlagRequest` `{reason}` | `ReviewerEntryView` | Yes | Quarantines regardless of status |
| 21 | POST | `/api/chat2hamnosys/review/entries/{id}/clear_quarantine` | board | `ClearQuarantineRequest` `{comment, target_status}` | `ReviewerEntryView` | Yes (board) | `target_status` ∈ `{pending_review, draft, rejected}` |
| 22 | POST | `/api/chat2hamnosys/review/entries/{id}/export` | board | — | `ExportResult` | Yes (board) | Writes `data/hamnosys_<lang>_authored.sigml`, appends to export audit log |
| 23 | GET | `/api/chat2hamnosys/review/dashboard` | reviewer | — | governance snapshot `{policy, counts_by_status, flag_count, rejection_reasons, mean_time_in_queue_seconds, reviewers[], reviewer_activity, quarantined[], exports{total, audit_chain_ok, audit_errors, recent[]}}` | Yes (no FE) | Any reviewer can read; no HTML wrapper today |
| 24 | POST | `/api/chat2hamnosys/review/reviewers` | board | `CreateReviewerRequest` | `{reviewer, token}` (token shown once) | Yes (no FE) | Creates bearer token; only path to mint reviewers |

---

## 4. Admin

Module: `backend/chat2hamnosys/api/admin.py`

| # | Method | Full path | Auth | Response | Reachable | Notes |
|---|---|---|---|---|---|---|
| 25 | GET | `/api/chat2hamnosys/metrics` | none | Prometheus text (`text/plain`) | Yes (no FE) | Internal scrapers |
| 26 | GET | `/api/chat2hamnosys/admin/dashboard` | board | `text/html` | Yes (no FE) | Operations dashboard |
| 27 | GET | `/api/chat2hamnosys/admin/sessions/{id}` | board | `text/html` | Yes (no FE) | Per-session event trace |
| 28 | GET | `/api/chat2hamnosys/admin/cost` | board | `text/html` | Yes (no FE) | 30-day cost breakdown |
| 29 | GET | `/api/chat2hamnosys/health` | none | `{status: "ok", build_sha?}` | Yes (no FE) | Liveness |
| 30 | GET | `/api/chat2hamnosys/health/ready` | none | `{status, checks{session_store, sign_store, llm}}` | Yes (no FE) | Readiness — session DB, sign DB, `OPENAI_API_KEY` presence |

---

## 5. Translator root app (context, not authoring)

Module: `server/server.py`. Listed only to show the non-authoring
endpoints on the same FastAPI app, so the delineation is clear.

| # | Method | Path | Touches authoring? |
|---|---|---|---|
| ~ | GET | `/api/health` | no |
| ~ | POST | `/api/translate-text` | no |
| ~ | POST | `/api/translate` | no |
| ~ | POST | `/api/translate/batch` | no |
| ~ | POST | `/api/plan` | no |
| ~ | GET | `/data/*` | **reads from `data/`** — same directory the export writes into (`hamnosys_<lang>_authored.sigml` would be served here once produced) |
| ~ | GET | `/*` (static) | serves `public/` including `contribute.html` |

---

## 6. Not-yet-built endpoints (gaps)

Called out because the current redesign conversation will need to
pick some of these up:

- **Reviewer self-enrolment** — there is no `POST /review/apply` or
  similar; the only way to become a reviewer is for a board member
  to call `POST /review/reviewers` manually.
- **Public library browse** — no `GET /signs` that the translator
  site could call to show the contributor a preview of the corpus
  they're contributing to.
- **Contributor dashboard** — no `GET /me/signs` listing a user's
  own submitted drafts / their review status.
- **Delete-my-account** — the FAQ promises "You can ask us to
  delete it anytime via the GitHub issues link" (`contribute.html:481`);
  no programmatic endpoint exists.
- **Sign-language declaration at registration** — `RegisterIn` has
  no `signs` or similar field, so the queue filter cannot use
  contributor-declared competence.

---

## 7. Reachability summary

- All 30 endpoints above are mounted in the live app as of
  `main @ 95c9700`.
- 2 endpoints are called from `contribute.html`: #1 and #2.
- 11 endpoints are called from `chat2hamnosys/` authoring UI:
  #3–13.
- 8 endpoints are called from `chat2hamnosys/review/`: #14–22
  (with #21–22 board-only).
- 10 endpoints are mounted but have no existing FE caller: #23, #24,
  #25–30.
- Zero endpoints are called from the browser extension.

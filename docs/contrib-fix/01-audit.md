# 01 — Contribute pipeline audit

Read-only first pass for the seven-prompt contribute-pipeline fix run.
Every section below is grounded in file paths and line numbers. No
executable code is changed in this commit. Companion artefact:
[`01-audit-sse-log.txt`](./01-audit-sse-log.txt) — verbatim SSE
reproduction of the heartbeat-only failure.

Stack under audit:

- Backend: FastAPI sub-app at `backend/chat2hamnosys/`, mounted by
  `server/server.py:992-996` at `/api/chat2hamnosys`.
- Frontend: vanilla-JS IIFE modules under `public/contribute-*.js`
  served as static files. No build step.
- Storage: SQLite at `data/chat2hamnosys/sessions.sqlite3` (default,
  see `backend/chat2hamnosys/api/dependencies.py:37`).

## 1. Avatar elements inventory

Every place under `public/` where a CWASA avatar element or a stub
avatar appears. "Real" rows mount the licensed CWASA WebGL bundle via a
`<div class="CWASAAvatar av0">` element it scans for; "stub" rows are
decorative CSS-shape avatars or SVG region overlays unrelated to the
3-D engine.

| File | Line | Kind | Context |
|------|-----:|------|---------|
| `public/contribute.html` | 161-166 | stub | `c2-hero__avatar` — three `<span>` shapes in the marketing hero card next to the gloss "ELECTRON". CSS-only; never animated. |
| `public/contribute.html` | 342-350 | stub | `c2-viz-4__avatar` — same head/torso/arms shape pattern inside walkthrough step 4 ("Watch the avatar perform it"), with a static play-button glyph. |
| `public/contribute.html` | 750 | real | `<div class="CWASAAvatar av0" id="avatarCanvas">` inside `#avatarStage` — the live mount the CWASA bundle scans for. Section is `hidden` until generation lands. |
| `public/contribute.html` | 751 | real (hidden) | `<div class="CWASAGUI av0" style="display:none">` — CWASA's hidden control GUI; preview JS reads its speed slider from `public/contribute-preview.js:594`. |
| `public/contribute.html` | 752-770 | stub | `<svg class="avatar-regions">` — 11 polygon hit zones for click-to-correct on body parts. Decorative overlay over the canvas. |
| `public/contribute.html` | 749, 771 | stub | `avatar-backdrop` / `avatar-pulse` — paper-tone backdrop and idle pulse element, both decorative. |
| `public/contribute.html` | 772-782 | stub | `avatar-fallback` block — plain-text "Preview unavailable" message with regen / retry / diagnostics buttons; shown when CWASA fails to load. |
| `public/index.html` | 1116 | real | `<div class="CWASAAvatar av0" id="heroAvatar">` — landing-page hero. Initialised by `initCWASA` at `public/index.html:1282-1340`. |
| `public/app.html` | 1747 | real | `<div class="CWASAAvatar av0">` — translator stage. Initialised by `initCWASA` at `public/app.html:1895-1932`. |
| `public/chat2hamnosys/index.html` | 120 | real | `<div class="CWASAAvatar av0">` inside `#cwasaMount` — reference fallback used by the Prompt-1 demo at `public/chat2hamnosys/app.js:613-628`. |
| `public/contribute-status.html` | — | n/a | No avatar element. The page only renders status + reviewer comments. |
| `public/contribute-me.html` | — | n/a | No avatar element. The page is a per-browser submission list (sessionStorage-backed). |

Net inventory: one real avatar mount in the contribute pipeline
(`contribute.html:750`); two CSS-only stub avatars elsewhere on the
same page; the two status pages (`contribute-status.html`,
`contribute-me.html`) carry no avatar at all.

## 2. SSE event flow

End-to-end trace of a successful generation, from the browser's
``POST /sessions/<id>/describe`` to the client receiving a `generated`
SSE frame. Every numbered step cites file + line.

1. **Browser POSTs describe.** `public/contribute-context.js:458` —
   `fetchWithTimeout(API_BASE + '/sessions/' + ... + '/describe', ...)`.
   The `bridgn.openai_api_key` localStorage value (if any) is attached
   as `X-OpenAI-Api-Key` by the fetch wrapper at
   `public/contribute-byokey.js:58-85`.
2. **Server route entry.** `backend/chat2hamnosys/api/router.py:705-747`
   — `post_describe`. Loads + token-checks via `_load_session`
   (router.py:186-205), sanitises prose
   (api/security.py via `sanitize_user_input`), passes the prose into
   the orchestrator.
3. **Orchestrator transition.**
   `backend/chat2hamnosys/session/orchestrator.py:265-337` —
   `on_description`. Runs `parse_fn(prose)` (line 292), appends a
   `DescribedEvent` (line 310-312), updates the draft, then calls
   `_generate_and_ask` (line 322) which either appends a
   `ClarificationAskedEvent` (orchestrator.py:252) and stays in
   `CLARIFYING`, or returns the session in `GENERATING` (line 238).
4. **Synchronous generation.** Back in
   `backend/chat2hamnosys/api/router.py:739-745`, when the orchestrator
   returns `state == GENERATING`, the route handler runs `run_generation`
   inline (router.py:740-745). The `/answer` and `/correct` paths
   instead schedule it as a FastAPI `BackgroundTask`
   (`router.py:797-808` and `router.py:969-977`); `/describe` does not.
5. **GeneratedEvent appended.**
   `backend/chat2hamnosys/session/orchestrator.py:475-677` —
   `run_generation` calls `generate_fn`, `to_sigml_fn`, `render_fn`,
   then either appends `GeneratedEvent(success=True, hamnosys=…,
   sigml=…)` and transitions to `RENDERED` (orchestrator.py:656-677),
   or appends `GeneratedEvent(success=False, errors=[…])` and
   transitions to `AWAITING_DESCRIPTION` /
   `AWAITING_CORRECTION` (orchestrator.py:530-559, 587-608).
6. **Persist.** `backend/chat2hamnosys/api/router.py:746` —
   `session_store.save(session)`. The store writes one upsert row to
   `data/chat2hamnosys/sessions.sqlite3` via
   `backend/chat2hamnosys/session/storage.py:80-102`. Each call opens
   a fresh SQLite connection (`session/storage.py:73-76`); there is no
   in-process row cache, so the next reader sees the new history.
7. **SSE poll loop.**
   `backend/chat2hamnosys/api/router.py:1224-1278` —
   `_events_generator`, an `async` generator yielded by the
   `StreamingResponse` returned from `get_events`
   (router.py:1296-1327). On each poll (`asyncio.sleep(poll_interval)`,
   `EVENTS_POLL_INTERVAL = 1.0`, router.py:162) it calls
   `session_store.get(session_id)` (router.py:1247), then walks
   `session.history[seen:]` and yields one frame per new event.
8. **Frame serializer.** `backend/chat2hamnosys/api/router.py:1262-1266`
   — the single SSE-frame format string:

   ```
   id: <history_index>\n
   event: <event.type>\n
   data: <JSON model_dump>\n\n
   ```

   The `event.type` discriminator comes from the `Literal["generated"]`
   field on `GeneratedEvent`
   (`backend/chat2hamnosys/session/state.py:135`), via
   `event.model_dump(mode="json")` at router.py:1260. The keep-alive
   filler — emitted between polls when no new history has appeared —
   is the literal `b": keep-alive\n\n"` at router.py:1275.
9. **Client decoder.** `public/contribute-notation.js:932-950` —
   `consumeEventStream` reads the body via `ReadableStream`, splits on
   `\n\n`. `handleSseFrame` (notation.js:952-1003) parses `event:` /
   `data:` / `id:` lines, then on a `generated` /
   `correction_applied` / `clarification_answered` /
   `clarification_asked` event it triggers a full session refetch
   (`GET /sessions/{id}`) so the panel applies the authoritative
   envelope.

## 3. Why the SSE channel emits only heartbeats

**Reproduced.** See `01-audit-sse-log.txt`. With `OPENAI_API_KEY` unset
and no BYO key header sent, `POST /describe` returned `500
llm_config_error` in 60 ms; the SSE stream emitted only `: keep-alive`
comments for the full window; the SQLite session row stayed at
`state=awaiting_description` with `history=[]`. Cross-referenced
against the live store: `data/chat2hamnosys/sessions.sqlite3` currently
holds 110 sessions, all in `awaiting_description`, all with empty
`history` and empty `description_prose` — same fingerprint as the
reproduction. With `parse_fn` and `question_fn` stubbed (LLM bypassed)
the same flow advances to `RENDERED` and emits `described` +
`generated` SSE frames within 70 ms.

**Root cause, single line.**
`backend/chat2hamnosys/session/orchestrator.py:292` —

```python
parse_result = parse_fn(prose)
```

This call sits before any `session.append_event(...)`. When the
LLM-backed parser raises (most commonly `LLMConfigError` from
`backend/chat2hamnosys/llm/client.py:_resolve_api_key` when no project
key is provisioned and the contributor has not pasted a BYO key — the
exact configuration the deployed `kozha.env` ships with at
`deploy/systemd/kozha.env:7`), the exception propagates out of
`on_description`, past `post_describe`'s `session_store.save` at
`backend/chat2hamnosys/api/router.py:746`, and is caught by
`handle_llm_config_error` at `backend/chat2hamnosys/api/errors.py:138`.
The route returns a 500 envelope, but no `DescribedEvent` was ever
appended, so the session row is unchanged and the SSE poller has
nothing new to yield — only the 1-Hz keep-alive comment from
`router.py:1275`.

**Fix in one sentence (do not apply in this prompt).** Wrap the
`parse_fn(prose)` call in `on_description` with a `try` / `except` that,
on any parser exception, appends a `DescribedEvent` plus a
`GeneratedEvent(success=False, errors=[str(exc)])` to the session and
returns it in `AWAITING_DESCRIPTION`, so `post_describe` still saves
the new history and the SSE generator delivers a `generated` frame
the chat panel renders as an actionable error (BYO-key prompt,
budget-exceeded notice, parser-malformed notice — whichever maps).

## 4. Session state storage

- **SQLite path.** Default
  `data/chat2hamnosys/sessions.sqlite3`
  (`backend/chat2hamnosys/api/dependencies.py:37`,
  `DEFAULT_SESSION_DB`). Configurable via
  `CHAT2HAMNOSYS_SESSION_DB`
  (`backend/chat2hamnosys/api/dependencies.py:58-65`). The systemd
  unit ships
  `CHAT2HAMNOSYS_SESSION_DB=/var/lib/kozha/chat2hamnosys/sessions.sqlite3`
  (`deploy/systemd/kozha.env:12`).
- **Schema.**
  `backend/chat2hamnosys/session/storage.py:46-56`:

  ```sql
  CREATE TABLE IF NOT EXISTS sessions (
      id                TEXT PRIMARY KEY,
      state             TEXT NOT NULL,
      created_at        TEXT NOT NULL,
      last_activity_at  TEXT NOT NULL,
      payload           TEXT NOT NULL    -- AuthoringSession.model_dump_json
  );
  CREATE INDEX IF NOT EXISTS idx_sessions_state         ON sessions(state);
  CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity_at);
  ```

  Every session — draft, full event history, clarification turns,
  preview metadata — is JSON-encoded into the `payload` column via
  `AuthoringSession.model_dump_json()`
  (`session/storage.py:82`); only `state` and `last_activity_at` are
  denormalised for indexed lookup. Bound is one row per session
  (`session/storage.py:80-102`).
- **In-process cache.** None at the row level. The `_session_store`
  module-level singleton in
  `backend/chat2hamnosys/api/dependencies.py:53,58-65` caches the
  `SessionStore` *object* (so the SQLite path is resolved once per
  process), but every `SessionStore.get` /
  `SessionStore.save` opens a brand-new connection
  (`session/storage.py:73-76`) — there is no read-through or
  write-through cache, which is why the SSE generator at
  `backend/chat2hamnosys/api/router.py:1247` reliably picks up writes
  the route handler made on a different connection.
- **Token store.** Authoring tokens live in a separate SQLite file
  (`data/chat2hamnosys/session_tokens.sqlite3`,
  `api/dependencies.py:39`) and are checked by `_load_session` at
  `backend/chat2hamnosys/api/router.py:200-205`.
- **Mutating endpoints (every endpoint that calls `session_store.save`).**

  | Endpoint | Handler | Save call |
  |---|---|---|
  | `POST   /sessions` | `create_session` (router.py:633-681) | router.py:673 |
  | `POST   /sessions/{id}/describe` | `post_describe` (router.py:705-747) | router.py:746 |
  | `POST   /sessions/{id}/answer` | `post_answer` (router.py:750-811) | router.py:798, 810 |
  | `POST   /sessions/{id}/generate` | `post_generate` (router.py:814-873) | router.py:872 |
  | `POST   /sessions/{id}/correct` | `post_correct` (router.py:925-979) | router.py:967 |
  | `POST   /sessions/{id}/accept` | `post_accept` (router.py:982-1100) | router.py:1070 |
  | `POST   /sessions/{id}/reject` | `post_reject` (router.py:1194-1216) | router.py:1215 |
  | (background) `_run_correction_async` | router.py:465-529 | router.py:527 |
  | (background) `_run_generation_async` | router.py:532-579 | router.py:577 |
  | (cleanup) `SessionStore.resume` (timeout-driven abandon) | session/storage.py:137-165 | session/storage.py:164 |
  | (cleanup) `SessionStore.delete_older_than` (retention sweep) | session/storage.py:169-186 | n/a — DELETE only |

  The SSE endpoint `GET /sessions/{id}/events` at router.py:1296-1327
  is read-only.

## 5. Public API surface of the contribute pipeline

Routes mounted under `/api/chat2hamnosys` (parent mount at
`server/server.py:994`). Authoring endpoints additionally require an
`X-Session-Token` header that was minted on `POST /sessions` and
verified by `_load_session` at `router.py:200-205`; the SSE endpoint
also accepts `?token=…` because `EventSource` cannot send custom
headers (router.py:1311). Request / response bodies cite
`backend/chat2hamnosys/api/models.py`.

| Method | Path | Request shape | Response shape | Side effects |
|---|---|---|---|---|
| `POST`  | `/sessions` | `CreateSessionRequest` (models.py:29-40): optional `signer_id`, `display_name`, `author_is_deaf_native`, `sign_language` (`bsl`/`asl`/`dgs`, default `bsl`), `regional_variant`, `domain`, `gloss`. | `CreateSessionResponse` — `session_id`, `state`, `session_token`, full `SessionEnvelope`. 201. | Inserts session row + token row; emits `SESSION_CREATED` obs event; rate-limited 2/min/IP (router.py:138-149). |
| `GET`   | `/sessions/{id}` | header `X-Session-Token`. | `SessionEnvelope` (`models.py`: `SessionEnvelope`). | None; 404 if missing, 403 if token mismatches. |
| `POST`  | `/sessions/{id}/describe` | `DescribeRequest` (models.py:43-49) — `prose` (≥1 char), optional `gloss`. | `SessionEnvelope`. | Calls parser + clarifier + (synchronously) generator + renderer; appends `DescribedEvent`, `ClarificationAskedEvent` or `GeneratedEvent`; saves session. |
| `POST`  | `/sessions/{id}/answer` | `AnswerRequest` (models.py:52-63) — `question_id`, `answer` (string or int). | `SessionEnvelope` (mid-transition; clients refetch on the SSE `generated` frame). | Applies the answer; if remaining gaps run out, schedules `_run_generation_async` (router.py:797-808). |
| `POST`  | `/sessions/{id}/generate` | empty body. | `SessionEnvelope`. | Forces `run_generation` from `CLARIFYING` / `AWAITING_DESCRIPTION` / `AWAITING_CORRECTION` if a `parameters_partial` exists; appends `GeneratedEvent`; saves. |
| `GET`   | `/sessions/{id}/preview` | header `X-Session-Token`. | `PreviewOut` (models.py:119+) — `status`, `message`, `video_url`, `sigml`, `hamnosys`. | None. |
| `GET`   | `/sessions/{id}/preview/video` | header `X-Session-Token`. | `video/mp4` `FileResponse`. | None; 404 if no rendered video. |
| `POST`  | `/sessions/{id}/correct` | `CorrectRequest` (models.py:66-73) — `raw_text`, optional `target_time_ms`, `target_region`. | `SessionEnvelope` (mid-transition). | Appends `CorrectionRequestedEvent`; transitions to `APPLYING_CORRECTION`; schedules `_run_correction_async` background task (router.py:969-977). |
| `POST`  | `/sessions/{id}/accept` | query `force=bool` (default false), header `X-Session-Token`. | `AcceptResponse` — `sign_entry: SignEntryOut`, `session: SessionEnvelope`. | Promotes draft → `SignEntry` via `on_accept` (`session/orchestrator.py:752-779`); writes the row to `data/authored_signs.sqlite3` (`backend/chat2hamnosys/storage.py`); promotes `draft` → `pending_review` if a competent reviewer is on the roster (router.py:1085-1096). |
| `POST`  | `/sessions/{id}/reject` | optional `RejectRequest` — `reason: str`. | `SessionEnvelope`. | Appends `RejectedEvent`; transitions to `ABANDONED`. |
| `GET`   | `/sessions/{id}/status` | optional header `X-Session-Token`. | `StatusResponse` (models.py: `StatusResponse`) — public envelope; `description_prose` + `reviewer_comments` only when token verifies. | None. Backs the `/contribute/status/<id>` HTML page. |
| `GET`   | `/sessions/{id}/events` | header `X-Session-Token` *or* query `?token=…`; optional `Last-Event-ID`. | `text/event-stream`; one `id: …\nevent: <type>\ndata: <json>\n\n` frame per new history entry, plus periodic `: keep-alive`, plus terminal `event: closed` or `event: timeout`. | None — read-only poll loop. |
| `GET`   | `/hamnosys/symbols` | optional `If-None-Match`. | `application/json` symbol table; `Cache-Control: immutable`. | None. |
| `GET`   | `/reference/sigml` | optional `If-None-Match`. | `application/json` SiGML tag catalog; `Cache-Control: immutable`. | None. |
| `GET`   | `/contribute/captcha` | none. | `CaptchaOut` — signed challenge + question. | None. (`backend/chat2hamnosys/api/contributors.py:360`.) |
| `POST`  | `/contribute/register` | `RegisterIn` — captcha + name + contact + honeypot. | `RegisterOut` — `contributor_token`. | Persists contributor row to `data/chat2hamnosys/contributors.sqlite3`. (`api/contributors.py:387`.) |

Outside `/api/chat2hamnosys`, two static-shell HTML routes round out
the contribute pipeline:

| Method | Path | Handler | Response |
|---|---|---|---|
| `GET` | `/contribute/status/{session_id}` | `serve_contribute_status` (`server/server.py:1012-1015`) | static `public/contribute-status.html` |
| `GET` | `/contribute/me` | `serve_contribute_me` (`server/server.py:1065-1068`) | static `public/contribute-me.html` |

Both pages then call the `/api/chat2hamnosys/sessions/{id}/status`
endpoint above — see the comment at `server/server.py:1009`.

## 6. Recommendation: Option A vs Option B

**Verdict: Option A — strip the server-side session module and move
resume-token storage to the client.** Option B keeps a stateful
authoring loop on the server, and that's the surface the bug hides on:
the failure documented in §3 only exists because the orchestrator owns
a 9-state transition matrix and the parser call sits in the one slot
that doesn't write to history. Option A collapses the contribute API
to a small set of stateless endpoints — `parse-and-clarify`,
`generate`, `correct`, `submit` — each of which returns a complete
result envelope; the browser persists the in-flight draft + token in
`localStorage`, and `contribute-status.html` /
`contribute-me.html` resolve a draft id by hitting a stateless read
endpoint that already exists in spirit (`GET .../status` at
router.py:1103). Deploy complexity collapses too: no SQLite session DB
to back-up, no retention job
(`SessionStore.delete_older_than`, storage.py:169), no token store —
the live ops doc currently lists three SQLite files in
`deploy/systemd/kozha.env:12-15` and Option A removes two of them,
which directly improves the recovery story flagged in
`feedback_deploy_secrets.md`. Option B's "fall back to SQLite when
Supabase env vars are unset" branch is exactly the conditional
complexity the codebase has been bitten by before — Supabase keys
*will* be unset when this work runs (per the prompt), so Option B
ships as SQLite-only first and Supabase-second, doubling the surface
that has to be regression-tested before the pre-launch freeze. Blast
radius: under Option A a contribute outage takes down a few stateless
HTTP endpoints with no row-level state to corrupt; under Option B an
outage of either SQLite or Supabase strands every in-flight draft.
**Biggest risk of Option B (the path not taken): the SQLite ↔ Supabase
fallback ships behind a single env-var check, and any path where
Supabase becomes partially reachable — an auth-only outage, a
Realtime-only outage, a region-failover slow-path — silently degrades
half the contributors to one backend and half to the other, producing
a class of "my draft disappeared" bug reports that no test in this
repo can catch before launch.**

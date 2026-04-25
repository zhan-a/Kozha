ultrathink

# Prompt 06 — Session strip (Option A) or Supabase integration (Option B)

You are Worker 1, prompt 6 of 7. Generation and corrections work end-to-end
as of prompts 04 and 05. The audit doc at `docs/contrib-fix/01-audit.md`
made a recommendation between two storage options. Read its
"Recommendation" section first; that section names Option A or Option B
and you implement the named one. Do not second-guess the verdict.

## Context you need (Worker 1 has no memory between prompts)

- Repo: `https://github.com/zhan-a/Kozha`. Working dir
  `/Users/natalalefler/Kozha`. Backend in `backend/chat2hamnosys/`,
  frontend in `public/`. Production at `https://kozha-translate.com`.
- The contribute pipeline currently keeps server-side session state in
  SQLite (`backend/chat2hamnosys/session/storage.py`). Every contribute
  request goes through that store. The contributor's resume URL is
  `/contribute.html#s/<uuid>` plus a session token in `sessionStorage`.
  Two status pages (`public/contribute-status.html`,
  `public/contribute-me.html`) read from that store.
- Two paths the audit chose between:
  - **Option A — server-side session strip.** Move resume-token
    storage to the browser. Mint a UUID client-side, persist it +
    a token in `localStorage`, drop the SQLite session module.
    Status pages resolve the token by hitting a stateless read
    endpoint that returns the latest sign entry by token.
  - **Option B — Supabase backing store.** Keep the session module
    interface; swap the SQLite implementation for Supabase Postgres
    when env vars are set, fall back to SQLite when they are not.
- Supabase env vars (`SUPABASE_URL`, `SUPABASE_ANON_KEY`) will not be
  set when this prompt runs and may not be set in CI either. Code
  must run cleanly with those vars unset.
- Reviewer roster framing, Deaf-led copy, and visual layout are
  off-limits in this prompt.

## Objective

Implement the option named in `docs/contrib-fix/01-audit.md` §
"Recommendation". One commit. The full test suite still passes with all
Supabase env vars unset.

## Branch A — if the audit picked Option A (server-side session strip)

Steps:

1. Switch the URL fragment routing in `public/contribute-context.js`
   from `#s/<server-uuid>` to `#s/<client-uuid>`. Mint the UUID with
   `crypto.randomUUID()` on the client. Persist
   `{ uuid, token, lastUpdated }` in `localStorage` keyed
   `kozha.contrib.session`.
2. Delete the SQLite-backed session module
   (`backend/chat2hamnosys/session/storage.py`,
   `backend/chat2hamnosys/session/store_sqlite.py`, the dependency
   injection in `backend/chat2hamnosys/api/app.py`). Keep the in-
   process `SignEntryStore` (sign entries are still server-owned
   data; only the conversational session is going away).
3. Add a stateless read endpoint
   `GET /signs/by-token/<token>` that returns the sign entry the
   token last produced (or 404). Status pages query this endpoint.
4. Update `public/contribute-status.html` and
   `public/contribute-me.html` to resolve the token via the new
   endpoint.
5. Run the smoke tests from prompts 04 and 05; both must pass after
   the change.

Constraints:
- Do not break the on-disk sign-entry store. Sign entries (`SignEntry`)
  remain in SQLite; only the session conversation goes away.
- The 404 path in `/signs/by-token/<token>` must return a JSON error
  body matching the existing `error.code` shape.

## Branch B — if the audit picked Option B (Supabase backing store)

Steps:

1. Write `server/contribute/migrations/001_contributions.sql` with the
   schema. At minimum: a `contributions` table mirroring the SQLite
   `sessions` columns (uuid PK, draft JSONB, status text,
   created_at timestamptz, updated_at timestamptz). Include indexes
   on `(status, updated_at)` and `(token)` if the existing schema
   uses a token column.
2. Add a Supabase-backed `SessionStore` implementation behind the
   existing storage interface in `backend/chat2hamnosys/session/
   store_supabase.py`. Use the official `supabase-py` client.
3. Gate the implementation on env vars: read `SUPABASE_URL` and
   `SUPABASE_ANON_KEY` (or `SUPABASE_SERVICE_ROLE_KEY` if writes
   need it). When either is unset, fall back to the existing
   SQLite store and log exactly once at startup:
   `WARN  contribute.session: SUPABASE_URL unset; using SQLite store`.
4. Write a tiny adapter test
   `backend/chat2hamnosys/tests/test_session_store_selection.py`
   asserting:
   - With env vars unset, `get_session_store()` returns the SQLite
     impl.
   - With env vars set (monkeypatched), `get_session_store()` would
     return the Supabase impl (mocked client; do not require a live
     Supabase to test this).
5. Run the full backend test suite. It must pass with no Supabase
   env vars set.
6. Add `pip install supabase` to
   `backend/chat2hamnosys/requirements.txt`.

Constraints:
- Never commit a `.env` file, a key, or a placeholder key. Document
  the required env vars in the migration file's header comment.
- The fallback path must be the default. The module must import
  cleanly without `supabase-py` installed if `SUPABASE_URL` is
  unset (use a deferred import inside the Supabase code path).
- The migration file is SQL only; it does not run automatically.
  Operators run it via `psql` or the Supabase dashboard.

## Acceptance (whichever branch you took)

- [ ] Full backend test suite passes with `SUPABASE_URL` and
      `SUPABASE_ANON_KEY` unset:
      `cd backend/chat2hamnosys && python -m pytest -q`.
- [ ] Smoke tests from prompts 04 and 05 still pass against local
      `uvicorn`.
- [ ] Frontend smoke: open `/contribute.html`, complete one
      generation + correction round-trip, hit Submit. The status
      URL it returns resolves to a valid sign entry.

## Verification commands

```bash
# Backend tests, no Supabase env
cd backend/chat2hamnosys && python -m pytest -q && cd ../..

# Smoke tests still green
uvicorn backend.chat2hamnosys.api.app:app --host 127.0.0.1 --port 8000 &
SERVER=$!
sleep 2
pytest tests/contrib_generate_smoke.py tests/contrib_correct_smoke.py -v
kill $SERVER
```

## Commit + push

```bash
git add -A
git commit -m "feat(contrib): implement audit's session storage decision (A or B)"
git push origin main
```

ultrathink

# Prompt 04 — Fix the generation API end-to-end

You are Worker 1, prompt 4 of 7. The audit at `docs/contrib-fix/01-audit.md`
already identified why the SSE channel only emits heartbeats and the session
never advances past `awaiting_description`. This prompt applies the fix and
locks it in with a smoke test.

## Context you need (Worker 1 has no memory between prompts)

- Repo: `https://github.com/zhan-a/Kozha`. Working dir
  `/Users/natalalefler/Kozha`. Backend lives in `backend/chat2hamnosys/`,
  served via `uvicorn`. Production domain is
  `https://kozha-translate.com`. The deploy runs automatically on push
  to `main` via `.github/workflows/deploy.yml` (the a11y gate is
  currently in TEMP DEV MODE — search for that string to confirm).
- The contribute pipeline flow: client POSTs `/sessions/<id>/describe`
  with prose; backend acknowledges; backend should run the LLM-driven
  generator (`backend/chat2hamnosys/generator/sigml_direct.py`) in a
  background task; backend should emit a `generation` event over the
  per-session SSE channel when the generator finishes; client receives
  that event and renders the avatar.
- The bug: SSE delivers only the periodic heartbeat. The session state
  stays `awaiting_description` indefinitely. The audit doc names the
  exact line responsible — read it first.
- LLM access: `OPENAI_API_KEY` is set on the deployed backend. Locally
  it may be unset, in which case the deterministic stub generator runs
  and still produces a HamNoSys string. Both paths must succeed in the
  smoke test.
- Reviewer roster framing, Deaf-led copy, and visual layout are
  off-limits in this prompt.

## Objective

Apply the audit's named fix to the SSE event chain so that a
`POST /sessions/<id>/describe` reliably produces a `generation` event on
the SSE stream within 30 seconds. Lock the fix in with a single smoke
test that runs both locally and against production.

## Steps

1. Read `docs/contrib-fix/01-audit.md` § 3 ("Why the SSE channel emits
   only heartbeats"). Apply the named fix. If the fix touches the
   FastAPI background-task wiring or the event emitter, also re-check
   the surrounding error paths so a generator failure produces a
   `generation` event with `success=false` rather than swallowing
   silently (the chat panel already handles both shapes).
2. Add `tests/contrib_generate_smoke.py`. Single test function:
   ```
   POST  /sessions               (create a session)
   POST  /sessions/<id>/describe with prose "the sign for hello"
   OPEN  GET  /sessions/<id>/events  (SSE)
   ASSERT a frame whose event type contains "generat" arrives
          within 30 seconds.
   ON FAILURE, print every SSE line captured up to the timeout to
                stderr so the failure is diagnosable.
   ```
3. Drive the test with `httpx` (sync or async) and `httpx-sse`. The
   test must accept a `KOZHA_BACKEND` env var; default
   `http://127.0.0.1:8000`. When `KOZHA_BACKEND` is the deployed
   domain, the test must hit `/api/chat2hamnosys/...` paths (or
   whatever prefix the production proxy uses — check
   `deploy/nginx/kozha.conf`).
4. Run the test against local `uvicorn` and against
   `https://kozha-translate.com` after the auto-deploy completes.
   Both must pass.

## Constraints

- One smoke test, one commit. Do not add additional integration tests
  in this prompt; prompt 05 covers corrections, prompt 07 covers full
  e2e.
- The smoke test must NOT depend on `OPENAI_API_KEY` being set. The
  stub path produces a usable result; the real LLM path produces a
  better one. Either is a pass for "the SSE chain delivered the
  event."
- Do not introduce new server endpoints. The fix lives in existing
  code paths.
- Do not regress the a11y baseline; this prompt should not touch
  frontend at all unless the audit says the SSE bug is on the client
  side (in which case touch only what's needed and rerun
  `npm run --silent a11y:ci`).

## Acceptance

- [ ] `pytest tests/contrib_generate_smoke.py -v` passes against a
      local `uvicorn` started from `backend/chat2hamnosys`.
- [ ] `KOZHA_BACKEND=https://kozha-translate.com pytest
      tests/contrib_generate_smoke.py -v` passes (run after the
      auto-deploy completes — wait until `git log -1 origin/main`
      matches the SHA on `https://kozha-translate.com/data/last_deploy.json`).
- [ ] Manual: open `/contribute.html` in a real browser, type a
      description, see a `generation` event in DevTools network tab
      within 30 s and the avatar render.

## Verification commands

```bash
# Local
uvicorn backend.chat2hamnosys.api.app:app --host 127.0.0.1 --port 8000 &
SERVER=$!
sleep 2
pytest tests/contrib_generate_smoke.py -v
kill $SERVER

# Deployed (after auto-deploy completes)
KOZHA_BACKEND=https://kozha-translate.com pytest tests/contrib_generate_smoke.py -v
```

## Commit + push

```bash
git add -A
git commit -m "fix(contrib): SSE generation event chain + smoke test"
git push origin main
```

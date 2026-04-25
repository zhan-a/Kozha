ultrathink

# Prompt 08 — Cleanup (optional)

You are Worker 1, prompt 8 of 7 (the optional one). All seven previous
prompts have run. This prompt removes dead code that prompts 02–06
left behind, scrubs stale documentation, and clears any TODOs the
fixes addressed.

## Context you need (Worker 1 has no memory between prompts)

- Repo: `https://github.com/zhan-a/Kozha`. Working dir
  `/Users/natalalefler/Kozha`. Backend in `backend/chat2hamnosys/`,
  frontend in `public/`. Production at `https://kozha-translate.com`.
- The audit doc (`docs/contrib-fix/01-audit.md`) and prompts 02–07
  have completed. Read prompt 06's commit message to know which
  storage option was implemented (Option A or Option B); that
  determines which dead code lives in this commit.
- Reviewer roster framing, Deaf-led copy, and visual layout are
  off-limits in this prompt.

## Objective

Delete dead code surfaced by the previous six prompts, update any
docs referencing the old `#s/<server-uuid>` URL pattern (if Option A
was implemented), and scrub TODO comments addressed by prompts 02–07.
One commit. Zero behaviour change.

## Steps

1. Identify dead code. Two candidates depending on which storage
   option was chosen in prompt 06:
   - **If Option A** (server-side session strip): the SQLite session
     module's old import sites, the `SessionStore` dependency
     injection that's now unused, the API routes that mutated server-
     side session state and have been replaced by a stateless read,
     and any client-side fragment-routing code that still expects
     the old URL shape.
   - **If Option B** (Supabase backing store): old in-memory session
     paths that are now strictly less reliable than the
     SQLite/Supabase alternative, any debugging branches that
     bypassed the storage layer, dead helper functions that only
     existed for the SQLite-only path.
2. Identify stale doc references. `grep -rn "#s/<uuid>" docs/
   public/` and friends. If Option A was implemented, the
   client-side UUID is now what fills the fragment; update doc
   prose accordingly. If Option B was implemented, the URL pattern
   is unchanged but storage backend docs need to mention the
   Supabase fallback.
3. Identify TODO comments addressed by 02–07. `grep -rn "TODO\|FIXME"
   public/ backend/ | grep -iE "ssE|generation|correct|chip|avatar|
   session"`. Remove the comments whose subject prompts 02–07 fixed.
4. Verify nothing else broke. Run the full backend test suite, the
   smoke tests from prompts 04 and 05, and the a11y baseline.

## Constraints

- Behaviour change must be zero. If you find something that looks
  buggy but the previous prompts didn't address it, leave it for a
  future task and note it in the commit body — do not silently fix
  it here.
- Do not delete the audit doc. `docs/contrib-fix/01-audit.md` stays
  as a historical record of why the fix sequence ran.
- Do not delete the prompts themselves
  (`docs/contrib-fix/prompts/*.md`); they document the work for
  future maintainers.
- Do not run a `node_modules` reinstall or change `package.json` /
  `requirements.txt` unless prompts 02–07 left an obviously
  unused dependency.

## Acceptance

- [ ] `cd backend/chat2hamnosys && python -m pytest -q` passes,
      same number of tests as before this commit (or more, if you
      also moved a test file).
- [ ] Smoke tests still green:
      `pytest tests/contrib_generate_smoke.py
      tests/contrib_correct_smoke.py -v`.
- [ ] `grep -rn "TODO\|FIXME" public/ backend/ | wc -l` is strictly
      smaller than it was before this commit (record both numbers
      in the commit body).

## Verification commands

```bash
# Before the cleanup, capture baseline counts
TODO_BEFORE=$(grep -rn "TODO\|FIXME" public/ backend/ | wc -l)
TEST_BEFORE=$(cd backend/chat2hamnosys && python -m pytest --collect-only -q 2>&1 | tail -1)
echo "before: TODO=$TODO_BEFORE TESTS=$TEST_BEFORE"

# (do the cleanup)

# After
TODO_AFTER=$(grep -rn "TODO\|FIXME" public/ backend/ | wc -l)
echo "after: TODO=$TODO_AFTER (was $TODO_BEFORE)"

# Run the full smoke battery
cd backend/chat2hamnosys && python -m pytest -q && cd ../..
uvicorn backend.chat2hamnosys.api.app:app --host 127.0.0.1 --port 8000 &
SERVER=$!
sleep 2
pytest tests/contrib_generate_smoke.py tests/contrib_correct_smoke.py -v
kill $SERVER

# a11y baseline
npm run --silent a11y:ci 2>&1 | tail -3
```

## Commit + push

```bash
git add -A
git commit -m "chore(contrib): drop dead code, refresh docs, scrub TODOs"
git push origin main
```

ultrathink

# Prompt 01 — Contribute pipeline audit

You are Worker 1 in a sequence of seven prompts that fix the Kozha contribute
pipeline. This is the read-only first pass. You produce one document and one
commit; you change no executable code.

## Context you need (Worker 1 has no memory between prompts)

- Repo: `https://github.com/zhan-a/Kozha`. Stack: Python + FastAPI backend
  under `backend/chat2hamnosys/`, plain static HTML/CSS + vanilla-JS IIFE
  modules under `public/`. No build step. Working dir locally:
  `/Users/natalalefler/Kozha`.
- The contribute pipeline lets a contributor type a description of a sign,
  the backend generates SiGML via an LLM (`backend/chat2hamnosys/generator/
  sigml_direct.py`), and a 3D avatar (CWASA) plays it. Frontend modules
  start with `public/contribute-*.js`.
- The pipeline is currently broken: contributors describe a sign, the
  backend acknowledges, but the SSE channel sends only heartbeats and the
  session never advances past `awaiting_description`. No avatar plays.
- A SQLite session store backs the API today. There is interest in moving
  to Supabase later, but Supabase env vars (`SUPABASE_URL`,
  `SUPABASE_ANON_KEY`) will not be set when this work runs. Anything you
  recommend must run cleanly with those variables unset.
- Reviewer roster, Deaf advisory board, and the "Deaf-led / Deaf-reviewed"
  framing are correct as written; do not flag those as audit findings.

## Objective

Produce a single audit document at `docs/contrib-fix/01-audit.md` that the
remaining six prompts will read. Make exactly one commit, push it.

## What the document must contain

Use these section headings verbatim, in this order. Each section must be
specific (file paths + line numbers, not paraphrase).

1. **Avatar elements inventory.** Every place a CWASA avatar element or a
   placeholder/stub avatar appears in `public/`. Distinguish real
   `<div class="CWASAAvatar">` mounts from stubs (placeholder `<div>`s,
   static SVGs, "ready when you are" markers). One row per occurrence:
   file, line, kind (real / stub), context.

2. **SSE event flow.** Trace a successful generation from
   `POST /sessions/<id>/describe` through to the client receiving a
   `generation` event. Name every function call, every event-emitter
   invocation, every queue or store the event passes through, and the
   exact handler in the SSE serializer that turns it into an SSE frame.
   File paths and line numbers throughout.

3. **Why the SSE channel emits only heartbeats.** Reproduce the failure
   first (run the backend locally, open SSE, POST a description, capture
   the SSE log to `docs/contrib-fix/01-audit-sse-log.txt` — this single
   capture file is allowed). Then identify the root cause to a single
   line of code. State the fix in one sentence; do not apply it.

4. **Session state storage.** Where session state is read from and
   written to today. SQLite path, schema, the in-process cache (if any),
   and every endpoint that mutates it.

5. **Public API surface of the contribute pipeline.** Every HTTP route
   under the contribute prefix: method, path, request shape, response
   shape, side effects. Include SSE endpoints.

6. **Recommendation: Option A vs Option B.**
   - Option A: rip out the server-side session module entirely; move
     resume-token storage to `localStorage` keyed by a UUID minted
     client-side; status pages (`contribute-status.html`,
     `contribute-me.html`) resolve the token by hitting a stateless
     read endpoint.
   - Option B: keep the server-side session module, swap the SQLite
     backing store for Supabase (Postgres + Realtime) when env vars
     are set; fall back to SQLite when they are not.

   One paragraph (5–10 sentences) recommending A or B. Lead with the
   verdict. Justify with: deploy complexity, blast radius of an outage,
   the "Supabase keys may be unset" constraint, and the pre-launch
   freeze in the project. End with the single biggest risk of the
   option you did NOT pick.

## Constraints

- No executable code changes. Only `docs/contrib-fix/01-audit.md` and the
  one log file may be created.
- Do not skip the SSE reproduction. If you cannot reproduce, the audit
  document must say so plainly and propose what additional access (env
  vars, log levels) would let you reproduce.
- Do not propose Option C or hedge between A and B. Pick one.
- No emoji. No conversational tone in the document.

## Verification

```bash
test -f docs/contrib-fix/01-audit.md
test -f docs/contrib-fix/01-audit-sse-log.txt
grep -E "^## " docs/contrib-fix/01-audit.md | wc -l   # >= 6 sections
```

## Commit + push

This is non-negotiable. The runner detects success by `git log` advancing
on origin/main. No "if everything looks good"; just push.

```bash
git add -A
git commit -m "docs(contrib-fix): pipeline audit, SSE root cause, Option A/B verdict"
git push origin main
```

ultrathink

# Prompt 05 — Fix the corrections API end-to-end

You are Worker 1, prompt 5 of 7. Generation works again as of prompt 04
(`tests/contrib_generate_smoke.py` is green locally and against
production). Your job: make the three correction modes work, and lock
each one in with a smoke test.

## Context you need (Worker 1 has no memory between prompts)

- Repo: `https://github.com/zhan-a/Kozha`. Working dir
  `/Users/natalalefler/Kozha`. Backend in `backend/chat2hamnosys/`,
  frontend in `public/`. Production at `https://kozha-translate.com`.
- The contribute pipeline at this point: a contributor types a
  description, the avatar renders a generated SiGML, and they see a
  chip strip below the avatar (one chip per `<ham*/>` tag) plus a chat
  box for free-text follow-ups.
- Three correction modes the user can perform:
  - **Chip click — handshape swap.** Click a handshape chip
    (`<hamfist/>`, `<hamflathand/>`, etc.), the picker offers other
    handshapes from the same category, click one, the SiGML updates
    in-place and the avatar re-renders.
  - **Chip click — orientation swap.** Same flow, different category
    (`hampalmu`, `hampalmd`, etc.).
  - **Free-text correction in chat.** Contributor types
    "make the palm face downward instead", the backend interprets via
    the correction interpreter
    (`backend/chat2hamnosys/correct/correction_interpreter.py`),
    updates the SiGML (the `palmor` tag changes to `hampalmd`), and
    the avatar re-renders.
- The chip-swap UI exists in `public/contribute-sigml-edit.js`. The
  free-text correction goes through `POST /sessions/<id>/correct` and
  fires the same SSE channel as generation (a `generation` event with
  the new SiGML).
- Reviewer roster framing, Deaf-led copy, and visual layout are
  off-limits in this prompt.

## Objective

Make all three correction modes update the SiGML and re-render the
avatar. Lock the work in with a single smoke test
`tests/contrib_correct_smoke.py` containing one test per mode.

## Steps

1. Drive each correction mode by hand in the browser to confirm the
   current failure mode (chip click silently no-ops? chat 500s? SSE
   doesn't fire?). Document the failure mode in the commit body.
2. Fix each path. Likely-but-not-certain fixes: the chip handler may
   POST to `/correct` with the wrong payload shape; the correction
   interpreter may not recognise the swap intent; the SSE emitter
   may use a different event name on the correct path than on the
   generate path. The audit doc may have hints in § 2 and § 5.
3. Add `tests/contrib_correct_smoke.py` with three test functions,
   each one full round-trip:
   - `test_chip_handshape_swap`: create session, describe, generate,
     POST `/correct` with a chip-swap payload that swaps `hamfist` →
     `hamflathand`, assert the resulting SiGML contains
     `<hamflathand/>` and not `<hamfist/>`.
   - `test_chip_orientation_swap`: same shape, swap `hampalmu` →
     `hampalmd`, assert.
   - `test_chat_correction`: send raw text "make the palm face
     downward instead", assert the SiGML diff includes a `palmor`
     change to `hampalmd` (or the closest available downward palm
     tag).
4. Each test must accept the same `KOZHA_BACKEND` env var as
   `tests/contrib_generate_smoke.py`. Each must complete in under
   60 s including the LLM round-trip; tighten the timeout if the
   stub generator is in use.

## Constraints

- Do not regress the generation smoke test from prompt 04. Run it
  before pushing.
- The chat correction test asserts on the SiGML diff, not on a
  natural-language LLM response. The interpreter is allowed to
  improve over time; the structural expectation (a tag in the
  `palm_direction` category changes toward "down") must hold.
- Do not introduce new endpoints. The corrections flow uses the
  existing `/correct` route.
- One commit. Three test functions, one fix patch (which may touch
  multiple files).
- No copy changes. No layout changes.

## Acceptance

- [ ] `pytest tests/contrib_correct_smoke.py -v` passes against
      local `uvicorn`. All three sub-tests green.
- [ ] `KOZHA_BACKEND=https://kozha-translate.com pytest
      tests/contrib_correct_smoke.py -v` passes (after the
      auto-deploy completes).
- [ ] Manual: open `/contribute.html`, generate a sign, click a
      handshape chip and pick an alternative — the avatar
      re-renders with the new shape; type "make the palm face
      downward instead" in chat and the avatar re-renders again.

## Verification commands

```bash
# Local
uvicorn backend.chat2hamnosys.api.app:app --host 127.0.0.1 --port 8000 &
SERVER=$!
sleep 2
pytest tests/contrib_generate_smoke.py tests/contrib_correct_smoke.py -v
kill $SERVER

# Deployed
KOZHA_BACKEND=https://kozha-translate.com \
  pytest tests/contrib_correct_smoke.py -v
```

## Commit + push

```bash
git add -A
git commit -m "fix(contrib): chip swap + chat correction → SiGML update + smoke tests"
git push origin main
```

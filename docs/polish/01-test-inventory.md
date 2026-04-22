# Test inventory

Every test file under the repo, grouped by runner. Numbers are file counts, not test case counts.

## server/tests/ — 4 files, FastAPI/pytest

```
server/tests/
├── __init__.py
├── test_contribute_notation_markup.py      4.9 KB
├── test_contribute_page_loads.py           3.7 KB
├── test_contribute_preview_markup.py       6.8 KB
└── test_contribute_preview_wiring.py       3.9 KB
```

All four are markup/integration tests for the contribute flow. They build a minimal `FastAPI` app mounting `public/` via `StaticFiles` (not the full `server:app`, to avoid spaCy/argos load), serve the static HTML, and assert selectors + forbidden copy are present/absent. See e.g. `test_contribute_page_loads.py:79-108` — a blocklist of every phrase from the pre-redesign landing that must not survive in the new contribute page.

**Runner.** `pytest` invoked from repo root: `pytest server/tests/ -q`. Requires `fastapi` + `httpx` (already in `server/requirements.txt`).

**CI coverage.** Not explicit. The only workflow that runs pytest is `chat2hamnosys.yml`, and its `pytest tests -q` step is inside `backend/chat2hamnosys/`. These `server/tests/` are **not invoked by any existing workflow**. They execute only locally — or they are silent regressions. Worth surfacing in the synthesis as a latent risk.

## backend/chat2hamnosys/tests/ — 39 Python files + Playwright subdir

```
backend/chat2hamnosys/tests/
├── conftest.py
├── fixtures/
├── playwright/
│   ├── conftest.py
│   ├── _test_server.py              (helper, not a test)
│   ├── test_contribute_chat.py
│   ├── test_contribute_form.py
│   ├── test_contribute_notation.py
│   ├── test_contribute_session.py
│   ├── test_contribute_submit.py
│   ├── test_region_click_correction.py
│   └── test_smoke.py
├── test_answer_parser.py
├── test_api_contributors.py
├── test_api_endpoints_reachable.py
├── test_api_hamnosys_symbols.py
├── test_api_integration.py
├── test_api_router.py
├── test_clarify_templates.py
├── test_correction_interpreter.py
├── test_description_parser.py
├── test_eval_human_eval.py
├── test_eval_metrics.py
├── test_eval_regression.py
├── test_eval_runner.py
├── test_eval_simulator.py
├── test_llm_client.py
├── test_models.py
├── test_obs_alerts.py
├── test_obs_dashboard.py
├── test_obs_integration.py
├── test_obs_logger.py
├── test_obs_metrics.py
├── test_params_to_hamnosys.py
├── test_prompts_library.py
├── test_question_generator.py
├── test_rendering.py
├── test_review_workflow.py
├── test_security_gitleaks.py
├── test_security_injection.py
├── test_security_pii.py
├── test_security_rate_limit.py
├── test_security_sanitize.py
├── test_session_orchestrator.py
├── test_session_state.py
├── test_session_storage.py
└── test_storage.py
```

**Runner.** `pytest` with `PYTHONPATH=.` from `backend/chat2hamnosys/`:
```
cd backend/chat2hamnosys
PYTHONPATH=. pytest tests -q --ignore=tests/playwright
```

The `--ignore=tests/playwright` in CI is because playwright tests need a headless Chromium launch and a running server stub — heavier than the unit suite. CI runs the non-playwright subset; playwright is local-only today.

**CI coverage.** `chat2hamnosys.yml:pr-checks` runs the above (non-playwright subset) with `OPENAI_API_KEY=""` and expects the deterministic stub LLM client to kick in. Also runs a 10-fixture smoke eval (`python -m eval smoke --suite golden_signs --stub`).

**Security subset.** `security.yml` tries to run `tests/test_security_sanitize.py + test_security_injection.py + test_security_rate_limit.py + test_security_pii.py + test_security_gitleaks.py` — but this workflow is broken (memory). The tests themselves run fine locally.

## public/ — no JS unit tests

No `.test.js` / `.spec.js` files, no Jest / Vitest / Mocha config. Static-script architecture: frontend logic lives in `public/contribute-*.js` modules, tested indirectly via the Python+Playwright tests listed above.

## extension/ — no tests

The Chrome extension ships as four JS files (`background.js`, `content-shared.js`, `content-universal.js`, `content-youtube.js`) plus popup/panel assets. No tests. The `grep "test|describe|it("` hit on these files was false-positive (the words appear in comments, not test infrastructure).

## a11y — axe-core + pa11y

Not technically "tests" but enforced via CI. `scripts/a11y/run.mjs` runs axe-core + pa11y across 11 scenarios defined in `docs/contribute-redesign/12-a11y-baseline.md`. Workflow: `.github/workflows/a11y.yml`. Scripts declared in `package.json`:

```json
"scripts": {
  "a11y":    "node scripts/a11y/run.mjs",
  "a11y:ci": "node scripts/a11y/run.mjs --ci"
}
```

devDependencies: `@axe-core/puppeteer`, `axe-core`, `pa11y`, `puppeteer`.

**Local invocation.** `npm ci && npm run a11y`. Outputs an updated baseline markdown and per-scenario JSON under `docs/contribute-redesign/12-a11y-raw/`.

**CI invocation.** `npm run a11y:ci` — same script, strict exit codes on critical/serious axe violations. Passes today (baseline is zero critical + zero serious).

## Other scripted checks

- `.pre-commit-config.yaml` at root — not enumerated here, typically runs ruff, black, gitleaks on staged files.
- `scripts/loadtest.py` — python load-testing script, not a pytest test.
- `scripts/restore.sh` — shell restoration script, not a test.

## Local invocation cheat-sheet

```bash
# Python: server-side markup tests (not in CI today)
pytest server/tests/ -q

# Python: chat2hamnosys backend (full, excluding playwright)
cd backend/chat2hamnosys && PYTHONPATH=. pytest tests -q --ignore=tests/playwright

# Python: chat2hamnosys with playwright (requires chromium, runs local server)
cd backend/chat2hamnosys && PYTHONPATH=. pytest tests/playwright -q

# a11y axe+pa11y (uses puppeteer's bundled chromium)
npm ci
npm run a11y
```

## Gaps

1. `server/tests/` is not wired into any CI workflow. If `public/contribute.html` regresses, these tests only catch it locally. Either add a `server-tests` job to `chat2hamnosys.yml` or teach `a11y.yml` to also run them — a low-cost hardening.
2. Extension has no tests at all. Low priority; the extension's surface area is small and it shares JS with the main app via `content-shared.js`.
3. No integration test exercises the `/api/plan` → `/api/translate-text` → CWASA rendering chain end-to-end. The translator bug documented in `01-translator-bug-repro.md` would be caught today only by manual reproduction. Likely needs a playwright scenario added under `tests/playwright/` in a later prompt.

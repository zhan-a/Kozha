# Contributing to Kozha

Kozha is a speech-to-sign-language pipeline. The repository contains two distinct codebases with
different contribution rules:

- **Core translator** (`server/`, `extension/`, `data/`, `public/`) — text → glosses → SiGML →
  CWASA avatar. Hearing engineers can contribute freely; standard PR review applies.
- **chat2hamnosys** (`backend/chat2hamnosys/`) — LLM-mediated sign-authoring with a Deaf-reviewer
  gate. **Special rules apply.** Read [docs/chat2hamnosys/20-ethics.md](docs/chat2hamnosys/20-ethics.md)
  before sending a PR that touches generation output.

Most of this document is about chat2hamnosys, which has the higher governance burden.

---

## 1. Dev environment

### Prerequisites

- Python 3.11 or 3.12
- Docker (for the backend stack and observability profile)
- A modern browser with WebGL (Chrome, Firefox, Safari) for avatar rendering

### Set up

```bash
git clone https://github.com/your-username/Kozha.git
cd Kozha

# Python dependencies for the chat2hamnosys backend
python -m venv venv
source venv/bin/activate
pip install -r backend/chat2hamnosys/requirements.txt
pip install -r server/requirements.txt   # optional: only if working on the core translator

# Pre-commit hooks: gitleaks, ruff, etc.
pip install pre-commit
pre-commit install

# Local config
cp .env.example .env
# Open .env and set:
#   OPENAI_API_KEY=sk-...
#   CHAT2HAMNOSYS_SIGNER_ID_SALT=$(openssl rand -hex 48)
```

### Run

```bash
# Stack (chat2hamnosys backend + volumes)
docker compose up

# With Prometheus / dashboards
docker compose --profile obs up

# Tests
pytest backend/chat2hamnosys/tests
```

The authoring UI lives at `http://localhost:8000/chat2hamnosys/` once the backend is up.

---

## 2. PR rules

### Standard rules (all PRs)

- **Tests are required.** Every new function or behavior needs at least one test under
  `backend/chat2hamnosys/tests/`. We use pytest. PRs without tests are returned unread.
- **Every LLM call goes through `llm/client.py`.** Do not import the OpenAI SDK directly from any
  other module. The wrapper enforces retries, budget caps, and telemetry.
- **Every user string is sanitized at the API layer.** Any new endpoint that accepts user input
  must call `sanitize_user_input` from `api/security.py` before passing the input downstream. PRs
  that bypass this are returned.
- **Pre-commit must pass.** This includes gitleaks (no secrets in commits), ruff (lint), and the
  Jinja2 contract test (no user vars in template context).
- **Don't widen the scope of an LLM prompt without versioning it.** See § 4 below.

### Special rule for `chat2hamnosys` generation output

> **No PR that changes generation output may merge without a Deaf reviewer's sign-off.**

A PR "changes generation output" if it touches any of:

- `backend/chat2hamnosys/parser/` (changes how prose decomposes into slots)
- `backend/chat2hamnosys/clarify/` (changes which questions are asked)
- `backend/chat2hamnosys/generator/` (changes the produced HamNoSys, including `vocab_map.yaml`)
- `backend/chat2hamnosys/correct/` (changes how corrections are interpreted)
- `backend/chat2hamnosys/prompts/` (any `.md.j2` template change)
- `backend/chat2hamnosys/hamnosys/` (changes the validator's verdicts)
- `backend/chat2hamnosys/rendering/` (changes the SiGML output format)

The Deaf reviewer's sign-off is recorded as a GitHub review approval from a member of the
`@kozha/deaf-reviewers` team. If you do not see this team in your repo configuration, the
reviewer board has not yet been seated; in that interim period, the project maintainer (Bogdan)
must explicitly delegate sign-off authority and document it in the PR description, citing the
reason from the [ethics statement § 1](docs/chat2hamnosys/20-ethics.md#1-deaf-community-leadership).

The branch protection rule that enforces this is configured in `.github/CODEOWNERS` (currently
manual; automated CODEOWNERS once the team is seated).

---

## 3. How to add a new prompt version

Prompts are versioned to make A/B comparison and regression debugging tractable. Never edit a
prompt template in place — always create a new versioned file.

```bash
# Existing: backend/chat2hamnosys/prompts/parse_description_v1.md.j2
# Adding: parse_description_v2.md.j2

cd backend/chat2hamnosys/prompts
cp parse_description_v1.md.j2 parse_description_v2.md.j2
# Edit v2 with your changes.
```

Then:

1. **Update the loader.** Add the new version to the lookup in `prompts/loader.py`.
2. **Add a switch.** Plumb `prompt_version` through the call site (e.g. `parser.parse_description(prose, prompt_version="v2")`).
3. **A/B in eval.** Run the eval harness with both versions:
   ```bash
   python -m eval run --prompt-version v1 --out runs/parser-v1.json
   python -m eval run --prompt-version v2 --out runs/parser-v2.json
   python -m eval compare runs/parser-v1.json runs/parser-v2.json
   ```
4. **Record in the research log.** Add a dated entry to
   [docs/chat2hamnosys/20-research-log.md](docs/chat2hamnosys/20-research-log.md) with the
   hypothesis, the metric deltas, and the decision (promote / keep both / discard).
5. **Promote the default only after Deaf-reviewer sign-off.** Changing the default prompt is a
   "changes generation output" change (see § 2 above).

---

## 4. How to add a new handshape to the vocab map

`generator/vocab_map.yaml` is the deterministic English-to-HamNoSys lookup. Adding a new
handshape extends the deterministic path and reduces LLM-fallback usage.

```yaml
# generator/vocab_map.yaml — partial, illustrative
handshape:
  flat:           "\uE001"
  flat_thumb_out: "\uE001\uE0F4"
  index:          "\uE002"
  # ... add new entry:
  round_o:        "\uE013"        # the O-shape used in ELECTRON, etc.
```

Procedure:

1. **Verify the codepoint** belongs to the handshape class. Check `hamnosys/symbols.py`:
   `classify(0xE013)` should return `SymClass.HANDSHAPE_BASE`.
2. **Add a fixture.** Add at least one entry to `eval/fixtures/golden_signs.jsonl` whose expected
   `handshape_dominant` is the new value, sourced from a real sign that uses it.
3. **Run the generator eval.**
   ```bash
   python -m eval run --stage generator --out runs/generator-with-round-o.json
   ```
   Confirm the new fixture passes and no old fixtures regress.
4. **Update the parser's term map.** The parser's English vocabulary in `parser/description_parser.py`
   needs to recognise the surface form ("round-O", "round o", "O-handshape") and route it to the
   new YAML key.
5. **Test.** `pytest backend/chat2hamnosys/tests/test_params_to_hamnosys.py` and
   `test_description_parser.py` must pass.
6. **Deaf-reviewer sign-off** (changes generation output → § 2).

---

## 5. How to add a new reviewer

Reviewers are added by an existing board member via the admin CLI. Hearing maintainers cannot
add reviewers unilaterally.

```bash
# Bootstrap (first reviewer ever — only run by the board chair):
python -m review.admin bootstrap-board \
    --name "Alice Smith" \
    --email alice@example.org \
    --languages bsl,asl \
    --is-deaf-native true \
    --regional-background "BSL-Manchester" \
    --is-board-member true

# Add a regular reviewer:
python -m review.admin add-reviewer \
    --name "Bob Lee" \
    --email bob@example.org \
    --languages dgs \
    --is-deaf-native true \
    --regional-background "DGS-Berlin"

# Promote a reviewer to the board:
python -m review.admin promote-to-board --reviewer-id <uuid>

# Rotate a reviewer off (24-month term limit):
python -m review.admin retire-reviewer --reviewer-id <uuid> --reason "term ended 2026-04-19"
```

Each command emits a row in the audit log. The reviewer receives their bearer token via the email
address provided; the token rotates on each board action against their record.

See [docs/chat2hamnosys/15-review-and-export.md § admin commands](docs/chat2hamnosys/15-review-and-export.md)
for the full CLI surface.

---

## 6. Code-review standards (checklist)

Before requesting review, confirm each item:

- [ ] Tests added under `backend/chat2hamnosys/tests/` and pass locally.
- [ ] Every LLM call goes through `llm/client.py`.
- [ ] Every new user input is sanitized at the API layer.
- [ ] Pre-commit passes (gitleaks, ruff, Jinja2 contract test).
- [ ] If a prompt template was changed: a new version was created (not edited in place).
- [ ] If `vocab_map.yaml` was changed: at least one fixture exercises the new entry.
- [ ] If generation output is affected: a Deaf-reviewer approval is on the PR (or, in interim,
      explicit maintainer delegation per § 2).
- [ ] If env vars were added: documented in `.env.example` and `docs/chat2hamnosys/19-env-vars.md`
      in the same change.
- [ ] If new behavior is observable: a metric or event was added (`obs/metrics.py`,
      `obs/events.py`).
- [ ] No commented-out code, no dead imports, no debug prints.

---

## 7. Reporting security issues

Do not file public GitHub issues for security vulnerabilities. Email **security@kozha.dev** with
the details. We follow the disclosure protocol in
[docs/chat2hamnosys/17-security.md](docs/chat2hamnosys/17-security.md).

## 8. Reporting Deaf-community concerns

Email **deaf-feedback@kozha.dev**. Monitored daily. The board chair acknowledges within 7
business days. See [docs/chat2hamnosys/20-ethics.md § 5](docs/chat2hamnosys/20-ethics.md#5-data-policy).

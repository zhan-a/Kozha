# Security, Safety & Abuse-Response — chat2hamnosys

Accompanies Prompt 17's hardening work. Three sections:

1. **Secret hygiene** — how keys are handled, rotated, and kept out of
   the repo.
2. **Dependency-upgrade policy** — when and how the team responds to
   vulnerability reports from `pip-audit` / `npm audit`.
3. **Abuse-response plan** — kill-switch procedure, rollback workflow,
   apology template, and Deaf-community contact channel.

The adjacent `17-injection-surface.md` catalogs every user-controlled
string that reaches an LLM prompt. The adjacent security test files
under `backend/chat2hamnosys/tests/test_security_*.py` lock in the
guarantees.

---

## 1. Secret hygiene

### Where keys live

- **`OPENAI_API_KEY`** is the single secret the application requires.
  It is loaded only from the process environment — see
  `backend/chat2hamnosys/llm/client.py:239` (`os.environ.get
  ("OPENAI_API_KEY")`). No config file, no fallback default.
- **`CHAT2HAMNOSYS_SIGNER_ID_SALT`** is the HMAC salt used when
  `pii_policy="hashed"`. Treat it as a secret (rotating invalidates
  pseudonym linkage across the before/after boundary, but does not
  decrypt anything — it's a one-way salt). Store in the same secret
  manager as the OpenAI key.
- Deploy-time: the EC2 host reads both from `/etc/kozha/env` via
  `systemd` EnvironmentFile. `.env` files are in `.gitignore` and
  never committed.

### Preventing accidental commits

- Local: `.pre-commit-config.yaml` wires **gitleaks 8.18** into
  `pre-commit`. Run `pre-commit install` once per clone; the hook
  scans every staged diff against `.gitleaks.toml`, which layers
  strict OpenAI/Anthropic patterns on top of the gitleaks defaults.
- CI: `.github/workflows/security.yml:gitleaks` re-runs the scan on
  the full history. A PR that sneaks past the local hook still fails
  the check.
- Regression test: `tests/test_security_gitleaks.py::test_fake_openai_key_is_caught`
  shells out to `gitleaks` against a temporary repo containing a
  synthetic `sk-…` blob and asserts a non-zero exit code. The test
  is skipped if `gitleaks` is not on PATH.

### Rotation

Before any public deployment:

1. **Rotate the dev key.** Any `OPENAI_API_KEY` that was loaded into a
   developer machine during the 20-prompt build cycle must be
   considered burned; rotate in the OpenAI dashboard.
2. **Audit for leakage.** Grep the following for the dev key fragment
   before the rotation:
   - CI logs for the `Deploy` workflow.
   - `logs/llm/*.jsonl` telemetry files — these record metadata only
     (no content), so the key should never appear. Grep anyway.
   - Systemd journal on the EC2 host: `journalctl -u kozha.service | grep sk-`.
   - Any Slack thread where someone pasted a debugging traceback.
3. **Re-provision secrets** through the secret manager, restart
   `kozha.service`, and revoke the old key.

Rotate the signer-id salt on the same cadence as the API key.

---

## 2. Dependency-upgrade policy

### Lockfiles

- `backend/chat2hamnosys/requirements.txt` currently uses semver
  range pins (e.g. `fastapi>=0.110,<1`). That is enough for the
  develop-ment phase but **not** for production. Before the first
  public deploy, freeze a lock file via `pip-compile` / `uv pip
  compile` into `backend/chat2hamnosys/requirements.lock` and have
  the deploy workflow install from the lock. The CI audit runs
  against the lock once it exists.
- JavaScript: no `package.json` exists today; when one is added, a
  committed `package-lock.json` is mandatory and `npm ci` (not
  `npm install`) is used in CI and deploys.

### Audit frequency and response SLA

| Severity | Action | SLA |
|---|---|---|
| Critical | Treat as an incident — block merges, patch on the day of disclosure, emergency deploy | 24 h |
| High | Open a ticket, bump within the next scheduled release | 7 days |
| Medium | Log in the backlog, bundle with the next minor-version sweep | 30 days |
| Low | Ignore unless a higher-severity advisory arrives for the same package | n/a |

`pip-audit` runs in `.github/workflows/security.yml` on every PR and
nightly. Findings above HIGH fail the CI check. MEDIUM/LOW findings
are informational (surfaced in the job log but do not fail the job).

### Upgrade mechanics

- One PR per package, not a wall of bumps.
- Always re-run the full test suite (`pytest` + Playwright) after an
  upgrade. The `feat(chat2hamnosys): security hardening` commit adds
  a `test_security_*` suite specifically to catch regressions in
  sanitation / injection / rate-limit behavior during future
  upgrades.

---

## 3. Abuse-response plan

### Scope

Offensive or culturally inappropriate output can reach end users
through three failure paths, in decreasing order of severity:

1. A validated sign is exported into the live Kozha library — already
   reviewed and approved, now visible to end users.
2. A pending-review session shows offensive clarification questions
   or correction interpretations to the signer during authoring (the
   moderation filter from Prompt 17 §4 covers this).
3. A malicious user crafts a prompt-injection that coerces the LLM
   into producing something useful for them (leak system prompt,
   generate offensive output, enumerate private data).

### Kill switch

The server exposes a single environment-variable kill switch:

- Set `CHAT2HAMNOSYS_KILL=1` on the EC2 host (in
  `/etc/kozha/env`) and `systemctl restart kozha.service`. The
  backend then rejects every POST under `/api/chat2hamnosys/*` with
  `503 service_unavailable`. Read-only endpoints remain online so
  the extension UI degrades gracefully.
- Intended use: sub-five-minute response to an unfolding incident
  while a proper fix is being prepared.

### Rollback from the audit log

Prompt 15's export-audit log (`review/storage.py:ExportAuditLog`) is
the authoritative record of what was written to
`data/hamnosys_<lang>_authored.sigml`. To roll back:

1. Identify the offending sign id(s) — either from a user report or
   from moderation flags attached to the review record.
2. `from review.storage import ExportAuditLog; log = ExportAuditLog(...);
   log.list(sign_id=<id>)` to enumerate every export row.
3. `SignStore.update_status(sign_id, "quarantined")` on the canonical
   store, which makes future export attempts fail the status gate.
4. Re-run the export step with the updated store to regenerate
   `hamnosys_<lang>_authored.sigml` **without** the quarantined
   entries. The file is overwritten in place; the extension picks up
   the change on its next fetch.
5. Append a `rollback_rationale` record to the audit log so downstream
   consumers can correlate the disappearance of the entry with the
   incident.

### Public apology template

Kept short; detail belongs in the post-mortem, not the apology.

> We became aware on [DATE] of a sign in Kozha that was culturally
> harmful to the [COMMUNITY] community. The sign was authored by
> [AUTHORED_BY_MECHANISM, e.g. a community contributor on YYYY-MM-DD]
> and, despite our review process, made it into the public library.
> We have removed the sign, are conducting a post-mortem on how our
> review process failed, and will share the findings publicly. We
> apologize to the [COMMUNITY] community, and we are grateful to
> [REPORTER, if they consent to naming] for reporting it. If you have
> seen a similar problem, please contact us at deaf-feedback@kozha.dev.

### Community contact channel

- **Email**: `deaf-feedback@kozha.dev` (a real mailbox, monitored by
  the project lead on weekdays). Intentionally **not** a web form —
  forms create friction and signal that the reporter will be
  corralled into a ticket queue.
- **Response target**: acknowledge within 48 h; substantive response
  and kill-switch action within 5 business days at the outside. For
  confirmed cultural-appropriateness issues, kill-switch immediately
  and apologize first, investigate second.
- **Moderator list**: at least two Deaf-native reviewers on the
  contact distribution list at all times. Named reviewers rotate
  quarterly so no single reviewer becomes a bottleneck.

### Prompt-injection incidents

A `verdict != DESCRIPTION` outcome from the injection screen is
logged but does not page anyone by default. Operators should set up a
Slack alert on a rolling count of ≥10 `INSTRUCTIONS` verdicts from
the same IP in an hour; the `security.rate_limit.anomaly_detector`
hook from Prompt 17 §7 delivers that signal.

If a prompt-injection succeeds (i.e. a sign enters the system whose
description shows clear model-directed instructions), treat it as an
incident in the same tier as a malformed export: kill-switch, audit
the in-flight sessions from that IP, and update the regex
bank in `security.injection._INJECTION_PATTERNS` to catch the shape
next time.

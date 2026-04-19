# chat2hamnosys — Production readiness checklist (Prompt 19)

**Every box must be checked before any deployment that handles real
Deaf-community submissions.** Copy this file into the deploy ticket,
record the verifier's name + date alongside each item, and link the
evidence (PR, runbook entry, screenshot, signed memo).

If you skip a box, write down *why* on the ticket — silent omissions
are how avoidable harms ship.

---

## A. Governance and review

- [ ] **Deaf advisory board seated and signed off.** Names + roles
      recorded; charter linked. Verifier: ____________  Date: ________
- [ ] **Two-reviewer rule enabled.** `CHAT2HAMNOSYS_REVIEW_MIN_APPROVALS=2`
      on the production env.
- [ ] **Single-approval mode disabled.** `CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE=false`
      on the production env (no `true` overrides anywhere).
- [ ] **Native-Deaf approver requirement on.**
      `CHAT2HAMNOSYS_REVIEW_REQUIRE_NATIVE=true`.
- [ ] **Export gate tested end-to-end.** Manually reject one sign,
      confirm export blocks; approve another, confirm export releases.
      Evidence: session ids + audit-log entries.
- [ ] **Audit log writing to append-only storage.** Pointed at
      `CHAT2HAMNOSYS_EXPORT_AUDIT`; the host filesystem makes the file
      append-only (`chattr +a` on ext4, equivalent on ZFS) so a
      compromised process can't rewrite history.

## B. Identity and credentials

- [ ] **Production OpenAI key in use** (not a demo / shared key).
- [ ] **Key rotated within last 30 days.** Record rotation date in the
      secret manager.
- [ ] **`OPENAI_API_KEY` lives in the secret manager**, not in `.env`
      files on the host. Verify with `grep -R OPENAI_API_KEY /etc /opt`
      and the deployment-target's secrets list.
- [ ] **`CHAT2HAMNOSYS_SIGNER_ID_SALT` set per-deployment.** Boot logs
      do **not** contain the "using built-in default" warning.
- [ ] **`.env` files removed from production hosts** once the secret
      manager is the source of truth (defense in depth).

## C. Network and TLS

- [ ] **TLS terminates with a real certificate** issued by a public CA
      (Let's Encrypt, ACM, etc.). Self-signed certs are rejected.
- [ ] **HSTS enabled** with a multi-year max-age. (See
      `deploy/nginx/kozha.conf` for the canonical header.)
- [ ] **`CHAT2HAMNOSYS_CORS_ORIGINS` locked to the production frontend
      domain.** No `*`, no localhost.
- [ ] **`/api/chat2hamnosys/metrics` is not publicly reachable.** Either
      bound to localhost (systemd) or restricted via ingress ACL
      (nginx allow / fly internal_port firewall / VPC).

## D. Cost and abuse controls

- [ ] **Daily budget cap set** (`CHAT2HAMNOSYS_DAILY_BUDGET_USD`) at
      production-realistic level.
- [ ] **Alert webhook wired up** (`CHAT2HAMNOSYS_ALERT_WEBHOOK_URL`)
      and a **test alert fired** in the last 24 hours.
- [ ] **Rate limits tuned to expected load.**
      `CHAT2HAMNOSYS_RATE_LIMIT` set; load-test (item F) confirms
      10 % headroom over peak.
- [ ] **Per-IP and global daily caps set**
      (`CHAT2HAMNOSYS_PER_IP_DAILY_CAP_USD`,
      `CHAT2HAMNOSYS_GLOBAL_DAILY_CAP_USD`).
- [ ] **Per-session budget cap set** (`CHAT2HAMNOSYS_SESSION_BUDGET_USD`)
      and a `BudgetExceeded` path tested via integration test.

## E. Code, dependencies, and CI

- [ ] **Pre-commit hooks for secret scanning active.**
      `.pre-commit-config.yaml` checked in; `pre-commit install` run on
      every contributor's machine.
- [ ] **Dependency audit green.** `pip-audit -r backend/chat2hamnosys/requirements.txt`
      reports zero CRITICAL/HIGH; `npm audit --audit-level=high` green
      (or no `package.json`).
- [ ] **Latest CI run on `main` passes** including the eval smoke
      suite. Link the GitHub run.
- [ ] **`BUILD_SHA` returned by `/health` matches the deployed
      commit.** Smoke check: `curl https://prod/api/chat2hamnosys/health`.

## F. Performance and capacity

- [ ] **Load test run at expected peak + 2× margin.** Record p50 / p95
      / p99 per endpoint, error rate, and total $ spent. Numbers + raw
      output linked from this ticket. (See
      [`scripts/loadtest.py`](../../scripts/loadtest.py).)
- [ ] **No errors above 0.5 % at peak load.**
- [ ] **p95 generate latency < 8 s** at peak (LLM-dominated; revisit
      after model swap).

## G. Backup, restore, and disaster recovery

- [ ] **Daily backup job scheduled.** `scripts/backup.sh` runs via
      cron / GitHub Actions / fly cron and writes to off-host storage.
- [ ] **Restore procedure tested.** `scripts/restore.sh` executed
      against a dry-run instance in the last 30 days; recovery time
      recorded.
- [ ] **Backup target is off-host.** S3, R2, fly volume snapshot to
      another region — *not* the same disk as the primary.

## H. Operations

- [ ] **Runbook up to date.** [`19-runbook.md`](19-runbook.md)
      reviewed within the last quarter; on-call name + contact at the
      top.
- [ ] **Contact form / email for Deaf-community feedback live and
      monitored.** Address documented; SLA for first reply documented.
- [ ] **Quarantine procedure rehearsed.** The "offensive content
      flagged" scenario in the runbook executed end-to-end at least
      once.

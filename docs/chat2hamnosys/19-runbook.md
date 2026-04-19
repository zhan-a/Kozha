# chat2hamnosys — Runbook (Prompt 19)

For the on-call engineer. Each scenario starts with the **symptom**
you'd see, the **first command** to run, the **likely causes** ranked
by frequency, and the **remediation** steps. Cross-link to the security
doc when an incident has user-safety implications.

> **On-call:** _____________  •  **Pager:** _____________
> **Last reviewed:** 2026-04-19  •  **Next review:** 2026-07-19

Quick links:

- [Backup and restore](#backup-and-restore)
- [Security incidents → 17-security.md](17-security.md)
- [Env var reference → 19-env-vars.md](19-env-vars.md)

---

## 1. Service down (`/health` returns non-200)

**Symptoms.** External monitor pages; users see 502 from the proxy.

**First command.**

```bash
curl -i https://kozha.example/api/chat2hamnosys/health
# Then on the host:
journalctl -u kozha.service -n 200 --no-pager      # systemd
fly logs --app kozha                               # fly
railway logs                                       # railway
```

**Likely causes**, ranked:

1. **Process crashed** (OOM, unhandled exception on startup). `journalctl`
   shows a Python traceback or `Killed`.
2. **Volume not mounted** (sqlite path 404). Logs mention
   `OperationalError: unable to open database file`.
3. **TLS / proxy down** (502 but the app is up). `curl
   http://127.0.0.1:8000/api/chat2hamnosys/health` from the host
   succeeds.
4. **Disk full.** `df -h /var/lib/kozha` near 100 %.

**Remediation.**

| Cause | Fix |
|---|---|
| 1 OOM | Bump VM memory; check the `/admin/cost` page for an outlier session that ballooned. |
| 1 Crash | Capture the traceback; if a recent deploy, roll back: `fly releases` → `fly deploy --image registry.fly.io/kozha:<previous-sha>`. |
| 2 Volume | Confirm mount: `mount | grep /app/data`. On fly: `fly volumes list`. Re-attach and restart. |
| 3 Proxy | `nginx -t && systemctl reload nginx`. Check certbot expiry: `certbot certificates`. |
| 4 Disk | Rotate logs (`/var/log/kozha`), prune `data/preview_cache/`, then run `scripts/backup.sh` and consider growing the volume. |

---

## 2. LLM calls failing

**Symptoms.** `/health/ready` reports `llm_config: ok=false`, or
session histories show `BudgetExceeded` / `LLMConfigError`. Users see
"generation failed".

**First command.**

```bash
# Check key configuration without leaking the value:
journalctl -u kozha.service | grep -E "openai|LLMConfig|429|503" | tail -50
# Probe OpenAI directly from the host (uses curl, not the app):
curl -s -o /dev/null -w "%{http_code}\n" https://api.openai.com/v1/models \
    -H "Authorization: Bearer $OPENAI_API_KEY"
```

**Likely causes.**

1. **Key missing or rotated out from under us.** `/health/ready` →
   `llm_config: ok=false`.
2. **OpenAI outage.** [status.openai.com](https://status.openai.com)
   shows degraded.
3. **Daily budget hit.** Logs include
   `BudgetExceeded: session budget exceeded` or the alerting daemon
   posted "daily budget projection exceeded". `/admin/cost` shows the
   spend curve.
4. **Rate limit (429) from OpenAI.** Logs show retries exhausted.

**Remediation.**

| Cause | Fix |
|---|---|
| 1 | Re-set the secret (`fly secrets set OPENAI_API_KEY=...` etc.); restart. |
| 2 | Status banner the frontend; pause non-critical batches. Wait it out. |
| 3 | If unexpected, treat as cost-spike (§5). If legitimate, raise `CHAT2HAMNOSYS_DAILY_BUDGET_USD` after sign-off. |
| 4 | Lower `CHAT2HAMNOSYS_RATE_LIMIT` so we throttle ourselves before OpenAI does. |

---

## 3. Renderer hung or crashed

**Symptoms.** Sessions stuck in state `GENERATING` or `RENDERED` with
no preview URL; `/admin/dashboard` shows a tail of "preview pending".

**First command.**

```bash
# Renderer is a separate container in compose; otherwise it runs
# in-process (or as the subprocess template KOZHA_RENDERER_CMD).
docker compose ps renderer
docker compose logs --tail=200 renderer
# In-process / subprocess case:
journalctl -u kozha.service | grep -Ei "render|preview|jasigning" | tail -50
```

**Likely causes.**

1. **Renderer container exited.** `docker compose ps` shows `Exited`.
2. **Subprocess template misconfigured.** `KOZHA_RENDERER_CMD` points
   at a binary that no longer exists.
3. **Preview cache disk full.** `du -sh /app/data/preview_cache`
   approaches volume size.

**Remediation.**

| Cause | Fix |
|---|---|
| 1 | `docker compose up -d renderer`. Investigate exit reason from logs. |
| 2 | `unset KOZHA_RENDERER_CMD` to fall back to metadata-only previews; fix the template; restart. |
| 3 | Prune older entries: `find /app/data/preview_cache -mtime +7 -delete`. |

---

## 4. Review queue backing up

**Symptoms.** Alerts fire (`pending review stale > N hours`); reviewer
dashboard shows a long list.

**First command.**

```bash
# Counts per state in the production sqlite store
sqlite3 /app/data/authored_signs.sqlite3 \
    "SELECT status, COUNT(*) FROM signs GROUP BY status;"
```

**Likely causes.**

1. **Reviewer fatigue threshold hit.** Active reviewers all over
   `CHAT2HAMNOSYS_REVIEW_FATIGUE_THRESHOLD`.
2. **Review board lost a member** without a replacement; not enough
   native-Deaf reviewers for the queue.
3. **Spam wave.** A single signer or IP submitted hundreds of items.
4. **Region-match requirement** stalls items whose variant has no
   matching reviewer.

**Remediation.**

| Cause | Fix |
|---|---|
| 1 | Recruit; *don't* lower the fatigue threshold without board sign-off. |
| 2 | Pause intake (frontend banner) until coverage restored. |
| 3 | Tighten `CHAT2HAMNOSYS_RATE_LIMIT` and `CHAT2HAMNOSYS_PER_IP_DAILY_CAP_USD`. Quarantine the signer ID. |
| 4 | Temporarily set `CHAT2HAMNOSYS_REVIEW_REQUIRE_REGION_MATCH=false` only after written board approval; document on this ticket. |

---

## 5. Cost spiking unexpectedly

**Symptoms.** Slack webhook posts "daily budget projection exceeded";
`/admin/cost` curve bends upward.

**First command.**

```bash
# Top 10 sessions by cost in the last 24 h
sqlite3 /app/data/chat2hamnosys/sessions.sqlite3 \
    "SELECT id, signer_id, cost_usd FROM sessions \
     WHERE created_at > datetime('now','-1 day') \
     ORDER BY cost_usd DESC LIMIT 10;"
```

**Likely causes.**

1. **Single signer / IP abusing the API.** One session id dominates
   the total.
2. **Long correction loops** (one user sending many corrections per
   session). Visible as high `corrections_count`.
3. **Stale model id with worse pricing.** Recent SDK swap regressed
   `llm/pricing.py`.
4. **Genuine traffic surge** from a launch.

**Remediation.**

| Cause | Fix |
|---|---|
| 1 | Quarantine the signer (revoke their tokens) and lower per-IP cap. |
| 2 | Open the session in `/admin/sessions/<id>` and force-reject if abusive. |
| 3 | Pin model id; verify pricing entry in `llm/pricing.py`. |
| 4 | Raise `CHAT2HAMNOSYS_DAILY_BUDGET_USD` after written approval; communicate to finance. |

---

## 6. Database full

**Symptoms.** Writes fail with `database or disk is full`; sessions
intermittently 503.

**First command.**

```bash
df -h /app/data
sqlite3 /app/data/authored_signs.sqlite3 "PRAGMA integrity_check;"
```

**Remediation.**

1. Run `scripts/backup.sh` (off-host snapshot) **before** doing
   anything else.
2. `sqlite3 /app/data/authored_signs.sqlite3 "VACUUM;"` to reclaim
   space if the file is fragmented.
3. Prune `data/preview_cache/` of files older than 30 days.
4. If still tight, grow the volume (fly: `fly volumes extend`; VM:
   `lvextend` / cloud snapshot grow).
5. If integrity check fails, restore from the latest off-host backup
   (see [Backup and restore](#backup-and-restore)).

---

## 7. A sign was approved but shouldn't have been

**Treat this as user-safety urgent.** Cross-reference
[`17-security.md`](17-security.md) for the full incident playbook.

**Symptoms.** Reviewer or community member reports an exported sign is
incorrect, offensive, or wasn't theirs to approve.

**Emergency un-export procedure:**

```bash
# 1. Identify the sign + audit-log row
SIGN_ID="..."   # from the report
sqlite3 /app/data/authored_signs.sqlite3 \
    "SELECT id, gloss, status, sign_language, regional_variant \
     FROM signs WHERE id = '$SIGN_ID';"
grep "$SIGN_ID" /app/data/chat2hamnosys/export_audit.jsonl

# 2. Quarantine — flip status back to pending review, mark blocked
sqlite3 /app/data/authored_signs.sqlite3 <<SQL
BEGIN;
UPDATE signs
   SET status = 'quarantined',
       updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now')
 WHERE id = '$SIGN_ID';
COMMIT;
SQL

# 3. Append a tamper-evident audit row
echo "{\"ts\":\"$(date -u +%FT%TZ)\",\"event\":\"emergency_unexport\",\"sign_id\":\"$SIGN_ID\",\"by\":\"$USER\",\"reason\":\"<short reason>\"}" \
    >> /app/data/chat2hamnosys/export_audit.jsonl

# 4. Force frontend cache invalidation if applicable
#    (the frontend re-fetches SiGML on hard reload; communicate to ops)
```

After quarantine: open a board ticket; the sign re-enters review with
its history attached. Do **not** delete the row — preserve the chain
of custody.

---

## 8. Deaf-community member flagged offensive content

This is a security-relevant incident. The full procedure lives in
[`17-security.md` → "Offensive content reported"](17-security.md). This
runbook only covers the immediate-response steps.

1. **Acknowledge within 4 hours.** A canned reply ("we received your
   report, the sign is being quarantined while we investigate") is
   fine and required.
2. **Quarantine the sign** using the procedure in §7.
3. **Quarantine adjacent items by the same signer / reviewers** if the
   reporter says the issue spans more than one entry.
4. **Notify the Deaf advisory board** with the report and the quarantine
   list.
5. **Update the security doc** with what was found and how it was
   resolved (closes the loop for future reviewers).

Never reply to the reporter with anything that could identify the
signer or the reviewers — that breaches their privacy as much as the
original offense breached the community's trust.

---

## Backup and restore

```bash
# Daily — wire into cron / fly cron / GitHub Actions
scripts/backup.sh /backups/$(date +%Y-%m-%d)

# Restore (interactive — confirms before clobbering live data)
scripts/restore.sh /backups/2026-04-18
```

Both scripts are documented inline. After any restore, run a smoke
session via `scripts/loadtest.py --sessions 1 --rate 1` to confirm the
restored state is functional before reopening to traffic.

---

## Scaling out

Sqlite is single-writer. When you outgrow a single VM:

1. **First:** add gunicorn workers (`--workers 2`) and set `WAL` mode
   on the sqlite stores. This bought us ~3× throughput in load tests.
2. **Then:** migrate the sign / session / token stores to Postgres.
   The schema is dataclass-driven; the migration is non-trivial but
   tractable.
3. **Only then:** horizontal scale with multiple VMs behind a sticky
   load balancer.

Don't skip steps 1 → 2 — multi-writer sqlite under load corrupts
silently.

# Polish-13 restore runbook

Procedures for recovering from the classes of failure that polish-13's
observability is meant to detect. Each section names the signal, the
rule that fires, and the steps to run. If a section is silent about
who pages: this is an internal tool — the on-call operator (currently
the solo maintainer) is who reads this page.

---

## Data/ backup and restore

### What gets backed up

The `Snapshot & Backup` workflow
(`.github/workflows/snapshot.yml`) zips the `data/` directory daily
and uploads it as an Actions artifact with 90-day retention. Excluded
from the zip (already in gitignore or runtime-only):

- `data/chat2hamnosys/` — chat2hamnosys session SQLite stores.
- `data/authored_signs.sqlite3*` — authored-signs runtime DB.
- `data/alerts_state.json` — rolling state for the alerter.

Everything else — `.sigml`, `.meta.json`, quarantine sidecars,
`progress_log.jsonl` — is in the zip.

### Restore from an artifact

1. Go to Actions → `Snapshot & Backup` on GitHub.
2. Pick the run closest to the target date (artifact names are
   `bridgn-data-YYYYMMDD.zip`).
3. Download and unzip:
   ```
   gh run download <RUN_ID> --name bridgn-data-20260422.zip
   unzip bridgn-data-20260422.zip
   ```
4. Inspect the diff against HEAD:
   ```
   diff -r data/ /tmp/restore/data/ | head
   ```
5. Copy only the files you need. Never `rm -rf data/` and paste the
   zip over it — the runtime subdirs we excluded are not in the zip
   and would vanish.
6. On the EC2 host, `sudo systemctl restart kozha.service` so the
   review-metadata loader and snapshot pick up the replacements.
7. Re-run the snapshot:
   ```
   cd /home/ubuntu/Kozha && /home/ubuntu/kozha-venv/bin/python3 -m server.progress_snapshot
   ```

### Restore a single `.meta.json`

Most common case: someone overwrote a meta file and the diff in git
is unhelpful because the edit was committed. Because `git log` is
authoritative for anything under version control, prefer:

```
git log -p -- data/Dutch_SL_NGT.sigml.meta.json | head -80
git show <SHA>:data/Dutch_SL_NGT.sigml.meta.json > data/Dutch_SL_NGT.sigml.meta.json
```

Only reach for the zipped artifact if the file never made it into a
commit (community ingest with a short-lived branch, say).

---

## Snapshot staleness

**Signal:** `server/alerts.py` emits `{"rule": "snapshot_stale"}` or
`{"rule": "snapshot_missing"}`. Dashboard "generated at" timestamp is
older than today's date.

**Root cause:** The `Snapshot & Backup` workflow failed or is
disabled.

**Recovery:**

1. Check the latest run: Actions → `Snapshot & Backup`.
2. If the last run is older than 36 h, run the workflow manually:
   Actions → `Snapshot & Backup` → `Run workflow`.
3. If the workflow itself is failing (not just missed scheduling),
   inspect the job log. Common failures:
   - `git push` rejected because the bot lacks `contents: write`.
     Check the workflow's `permissions:` block.
   - Python import error because
     `scripts/database_health_audit.py` moved.
   - Disk full on the runner — unlikely, but the `zip` step fails
     loudly if so.
4. If nothing has changed for a while (weekend), the workflow silently
   commits nothing — this is expected. The `progress_log.jsonl` will
   still get a new row each day.

---

## Error-rate spike

**Signal:** `server/alerts.py` emits `{"rule": "error_rate", "level":
"page"}`. 5xx rate > 5 % over the last window.

**Recovery:**

1. Pull recent request logs on the host:
   ```
   sudo journalctl -u kozha.service --since '15 min ago' \
     | jq 'select(.outcome == "server_error") | {ts, request_id, path, ip_hash}'
   ```
2. Group by `path` to see whether this is a specific route regressing
   (most useful when the 5xx is localized, e.g. `/api/translate-text`
   because argostranslate died again).
3. If the traceback is absent from logs (deploy-13's privacy pass
   strips stack traces from responses), re-run the failing request
   locally with `KOZHA_LOG_VERBOSE=1` and inspect stdout. Never set
   verbose on the production service — that flag is for
   developer-box debugging.
4. If the cause is a deploy regression: `git log -n 5 --oneline` and
   revert if needed. The deploy workflow is idempotent — `git reset
   --hard` on the host comes along for the ride.

---

## Unknown-word rate spike

**Signal:** `{"rule": "unknown_word_spike", "level": "warn"}`. Delta
is 3× the rolling baseline.

**Recovery:**

This is usually _informational_, not an incident. It means either:

- A new crop of users is searching for vocabulary we don't cover.
- A text source upstream of the translator changed and is producing
  tokens that now miss our normalization (e.g. a new emoji in the
  stream).

Steps:

1. Pull the top unknown words from the last day's logs:
   ```
   sudo journalctl -u kozha.service --since '24 hours ago' \
     | jq 'select(.outcome == "success") | .target_sign_lang' \
     | sort | uniq -c | sort -rn
   ```
   (Token-level unknowns are _not_ logged per privacy pass §8 — only
   per-request totals show up. To get the actual tokens, you need
   `KOZHA_LOG_VERBOSE=1` on a separate debugging replica.)
2. Hand the top words to the contribute pipeline. Prompt 8 already
   has a "help wanted" surface; updating `data/<language>.meta.json`
   with the new community submissions closes the loop.
3. Silence the alert for the day if it's expected churn: set
   `KOZHA_ALERT_UNKNOWN_SPIKE=6.0` temporarily.

---

## Deploy failed

**Signal:** `{"rule": "deploy_failed"}` from the alerter, or a red
`Deploy` workflow in Actions.

**Recovery:**

1. Open the failed run. Most common failure: the SSH step timed out
   (EC2 instance was reprovisioned, host key changed). Re-run the
   workflow; the `ssh-keyscan` step refreshes the known-hosts entry.
2. If `pip install` fails, a package's hash changed on PyPI; either
   bump the pin or retry. The venv is reused across runs
   (`/home/ubuntu/kozha-venv`), so transient PyPI flakes don't wipe
   prior state.
3. If `systemctl restart kozha.service` fails, SSH into the host and
   `journalctl -u kozha.service -n 100` to get the Python traceback.
   The deploy workflow does not block on `status` output, so a
   restart that returns 0 but logs a crash still "succeeds" from
   the runner's perspective — the `/health/ready` probe is what
   catches that, not this workflow.
4. After recovery, overwrite `data/last_deploy.json` on the host so
   the alerter clears:
   ```
   cat > /home/ubuntu/Kozha/data/last_deploy.json <<JSON
   {"status": "success", "sha": "$(git -C /home/ubuntu/Kozha rev-parse HEAD)", "at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
   JSON
   ```

---

## Alerts

### Running the alerter on the host (recommended)

The GitHub-hosted alerter
(`.github/workflows/alerts.yml`) has the documented 5–20 min GHA cron
delay. For tighter SLOs, run the alerter under a systemd timer:

```
# /etc/systemd/system/kozha-alerts.service
[Unit]
Description=Kozha rule-based alerter
After=network-online.target

[Service]
Type=oneshot
User=ubuntu
WorkingDirectory=/home/ubuntu/Kozha
EnvironmentFile=/etc/kozha.env
ExecStart=/home/ubuntu/kozha-venv/bin/python3 -m server.alerts \
  --metrics-url http://127.0.0.1:8000/metrics

# /etc/systemd/system/kozha-alerts.timer
[Unit]
Description=Run kozha-alerts every 5 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=5min
AccuracySec=30s

[Install]
WantedBy=timers.target
```

`sudo systemctl enable --now kozha-alerts.timer` to activate.

### Wiring up a webhook

`KOZHA_ALERT_WEBHOOK_URL` in `/etc/kozha.env` (or the GitHub repo
secret `ALERT_WEBHOOK_URL` for the Actions-based alerter).
Slack/Discord/PagerDuty incoming-webhook URLs all accept the simple
JSON body the alerter posts (`{text, rule, level, message, detail,
at}`).

---

## Observability kill-switch

Something is misbehaving and you need the instrumentation out of the
way to isolate the problem:

- `KOZHA_LOG_LEVEL=ERROR` drops request-log noise.
- Unset `KOZHA_ADMIN_TOKEN` to 404 the `/admin/ops` page.
- `rm public/progress_snapshot.json` forces the `/progress_snapshot.json`
  endpoint to 503 — useful when the snapshot itself is suspected of
  producing bad JSON.

Removing the observability module from the import graph is _not_
supported — the route handlers reference `_obs.record_translation`
directly. Use the env-var toggles instead.

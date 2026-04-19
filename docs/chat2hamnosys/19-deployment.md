# chat2hamnosys — Deployment guide (Prompt 19)

Three deploy targets are documented; pick whichever matches your
constraints. Each section is a copy-paste runbook from "I have a clean
host" to "service is up and `/health` returns 200".

> **Before any deployment** that handles real Deaf-community
> submissions, complete every box in
> [`19-prod-checklist.md`](19-prod-checklist.md).

---

## 0. Common prerequisites

- A populated `.env` (see [`.env.example`](../../.env.example) and the
  full manifest in [`19-env-vars.md`](19-env-vars.md)).
- A real `OPENAI_API_KEY` (the readiness probe degrades to 503 without
  it). The contribution flow on `kozha-translate.com/contribute.html`
  makes real OpenAI calls at project expense — the daily caps in
  `fly.toml` (`$20/day` global, `$5/day per-IP`) bound the spend.
- A per-deployment `CHAT2HAMNOSYS_SIGNER_ID_SALT` — generate with
  `openssl rand -hex 32`. The default triggers a startup warning.
- A per-deployment `CHAT2HAMNOSYS_CAPTCHA_SECRET` — generate with
  `openssl rand -hex 32`. Rotating invalidates in-flight captchas
  (users refresh the page); rotate whenever you'd rotate the signer
  salt.
- `CHAT2HAMNOSYS_REQUIRE_CONTRIBUTOR=1` in the env (already set in
  `fly.toml`). Toggling it off re-opens the authoring API to
  unregistered users and is **not** recommended for public deploys.
- A backup target. Run `scripts/backup.sh` once before first traffic to
  prove the procedure works (see
  [`19-runbook.md` → Backup](19-runbook.md#backup-and-restore)).

### Image size budget

The multi-stage `Dockerfile` is sized to stay **under 500 MB**. Verify
after each build:

```bash
docker images kozha:local --format '{{.Repository}}:{{.Tag}}\t{{.Size}}'
```

The number is **not yet measured** in CI — the
`build-and-publish` job in `.github/workflows/chat2hamnosys.yml` fails
the build if the image exceeds 500 MB. Record the first real
measurement alongside the deploy ticket. Per-component design
estimates:

| Component                 | Approx. size |
| ------------------------- | -----------: |
| `python:3.12-slim` base   |       ~50 MB |
| chat2hamnosys deps        |      ~120 MB |
| FastAPI / uvicorn / spaCy |      ~150 MB |
| en_core_web_sm model      |       ~50 MB |
| App source + venv overlay |       ~40 MB |

If you need the seven additional spaCy language models or
`argostranslate` for the legacy translator, install them at deploy time
into a derived image (see *Translation extras* below).

### Translation extras (optional)

```dockerfile
FROM kozha:local
USER root
RUN pip install --no-cache-dir -r /app/server/requirements.txt
USER kozha
```

Adds ~350 MB. Only do this if your deployment needs the multilingual
spaCy / argos pass-through (chat2hamnosys does not).

---

## 1. Fly.io

`fly.toml` is committed at the repo root.

```bash
# One-time setup
fly auth login
fly launch --no-deploy --copy-config           # accept the existing fly.toml
fly volumes create kozha_data --region iad --size 10
fly secrets set \
    OPENAI_API_KEY="sk-..." \
    CHAT2HAMNOSYS_SIGNER_ID_SALT="$(openssl rand -hex 32)" \
    CHAT2HAMNOSYS_CAPTCHA_SECRET="$(openssl rand -hex 32)" \
    CHAT2HAMNOSYS_ALERT_WEBHOOK_URL="https://hooks.slack.com/services/..."

# Deploy
fly deploy --build-arg BUILD_SHA="$(git rev-parse --short HEAD)"

# Verify
fly status
curl https://kozha.fly.dev/api/chat2hamnosys/health
```

The `kozha_data` volume is mounted at `/app/data`; sqlite stores live
under `/app/data/chat2hamnosys/`. Snapshots: `fly volumes snapshots
list kozha_data`.

## 2. Railway

`railway.toml` is committed at the repo root. Railway's volume UI is
the source of truth for storage — **create a Volume named `kozha_data`
and mount it at `/app/data`** before deploying.

```bash
railway login
railway link                                   # attach to project
railway variables set \
    OPENAI_API_KEY=sk-... \
    CHAT2HAMNOSYS_SIGNER_ID_SALT="$(openssl rand -hex 32)" \
    CHAT2HAMNOSYS_ALERT_WEBHOOK_URL="https://hooks.slack.com/..."
railway up                                     # build + deploy
railway open                                   # opens the live URL
```

Railway exposes the deploy SHA as `RAILWAY_GIT_COMMIT_SHA`; the
`railway.toml` build args wire it through to `BUILD_SHA` so `/health`
returns the right id.

## 3. Self-hosted Ubuntu 24.04 VM

Files in [`deploy/`](../../deploy):
- `deploy/systemd/kozha.service` — service unit
- `deploy/systemd/kozha.env` — env file template
- `deploy/nginx/kozha.conf` — TLS-terminating reverse proxy

```bash
# As root on the VM:

# 1. System packages
apt-get update
apt-get install -y python3.12 python3.12-venv python3-pip nginx \
    certbot python3-certbot-nginx git curl rsync

# 2. App user + dirs
useradd --system --create-home --home-dir /opt/kozha --shell /bin/bash kozha
install -d -o kozha -g kozha /var/lib/kozha/chat2hamnosys
install -d -o kozha -g kozha /var/log/kozha
install -d -o kozha -g kozha /opt/kozha

# 3. Source + venv
sudo -u kozha git clone https://github.com/<org>/Kozha.git /opt/kozha
cd /opt/kozha
sudo -u kozha python3.12 -m venv /opt/kozha/venv
sudo -u kozha /opt/kozha/venv/bin/pip install --no-cache-dir \
    -r backend/chat2hamnosys/requirements.txt
sudo -u kozha /opt/kozha/venv/bin/pip install --no-cache-dir \
    fastapi uvicorn[standard] gunicorn "spacy>=3.8.0" \
    "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0.tar.gz"

# 4. Service unit + env file
cp deploy/systemd/kozha.service /etc/systemd/system/kozha.service
cp deploy/systemd/kozha.env     /etc/kozha.env
chmod 600 /etc/kozha.env
chown root:kozha /etc/kozha.env
$EDITOR /etc/kozha.env                         # fill in OPENAI_API_KEY + salt + webhook
systemctl daemon-reload
systemctl enable --now kozha.service
journalctl -u kozha.service -e                 # confirm startup

# 5. nginx + Let's Encrypt
cp deploy/nginx/kozha.conf /etc/nginx/sites-available/kozha
ln -s /etc/nginx/sites-available/kozha /etc/nginx/sites-enabled/kozha
nginx -t && systemctl reload nginx
certbot --nginx -d kozha.example               # populates the 443 server block
systemctl reload nginx

# 6. Verify
curl https://kozha.example/api/chat2hamnosys/health
```

Logs: `journalctl -u kozha.service -f`. Restart after a config edit:
`systemctl restart kozha.service`.

---

## 4. Staging environment

Pattern: a second deployment of the same image, pointed at a separate
OpenAI billing project and a separate sqlite path. Cost isolation +
data isolation.

```bash
# Fly example — second app, same Docker image, hard-capped budget.
fly apps create kozha-staging
fly volumes create kozha_data_staging --region iad --size 5 \
    --app kozha-staging
fly secrets set \
    OPENAI_API_KEY="sk-staging-key-from-its-own-billing-project" \
    CHAT2HAMNOSYS_SIGNER_ID_SALT="$(openssl rand -hex 32)" \
    CHAT2HAMNOSYS_DAILY_BUDGET_USD=20.0 \
    CHAT2HAMNOSYS_GLOBAL_DAILY_CAP_USD=20.0 \
    CHAT2HAMNOSYS_PER_IP_DAILY_CAP_USD=2.0 \
    --app kozha-staging
fly deploy --app kozha-staging \
    --build-arg BUILD_SHA="$(git rev-parse --short HEAD)-staging"
```

Hard rules for any staging environment:

1. **`CHAT2HAMNOSYS_DAILY_BUDGET_USD=20.0` and
   `CHAT2HAMNOSYS_GLOBAL_DAILY_CAP_USD=20.0`.** Both. The first feeds
   the alert; the second is the hard ceiling.
2. **Distinct OpenAI billing project.** The cap above is defense in
   depth; the billing-side limit is the real backstop.
3. **Distinct storage paths.** Use a different volume / sqlite file so
   experiments never collide with production sign data.
4. **Distinct `CHAT2HAMNOSYS_SIGNER_ID_SALT`.** Hashes minted in
   staging must not be join-able to production hashes.
5. **`CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE=true` is OK in staging only.**
   Production must keep the two-reviewer rule on.

A staging URL like `kozha-staging.fly.dev` should never be on the same
DNS / TLS cert as production.

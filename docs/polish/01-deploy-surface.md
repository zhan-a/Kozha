# Deployment surface

Facts extracted from `.github/workflows/*.yml` and the EC2 host layout captured in memory. Every later prompt must not block deploys unless scoped to do so.

## Workflows that run

| workflow | file | trigger | blocks `main`? |
|---|---|---|---|
| Deploy | `.github/workflows/deploy.yml` | push to main, manual dispatch | **yes** — SSHes to EC2, resets repo, pip-installs, restarts systemd |
| a11y baseline | `.github/workflows/a11y.yml` | PR + push to main when `public/**` touched | no (reports only; workflow can succeed or fail but is not required) |
| Security | `.github/workflows/security.yml` | push to main, PR, manual | no — currently broken, see memory `project_security_workflow_broken.md` |
| chat2hamnosys CI | `.github/workflows/chat2hamnosys.yml` | PR + push to main when backend touched | PR lane gates merges; push-to-main runs image build |
| chat2hamnosys Deploy | `.github/workflows/chat2hamnosys-deploy.yml` | manual only | separate environment; doesn't affect main site |

Files that trigger the primary `deploy.yml` are *all* file changes on the main branch — there are no path filters. Any commit to main triggers the deploy, including doc-only commits. Practical effect: `docs/polish/**` writes will re-run the EC2 deploy. That is fine and expected; the deploy script is idempotent (git reset --hard to origin/main, pip install, systemd restart).

The a11y workflow *does* have path filters and only fires when `public/**` or the a11y scripts change. `docs/polish/**` changes do not trigger it. Non-issue.

## The deploy job (detailed)

`deploy.yml` runs on `ubuntu-latest` and performs these steps:

1. `actions/checkout@v4` — checks out the repo on the runner (optional since the actual deploy is SSH-based).
2. `webfactory/ssh-agent@v0.9.0` — starts ssh-agent with `secrets.EC2_SSH_KEY`.
3. `ssh-keyscan` the host key into `~/.ssh/known_hosts` using `secrets.EC2_HOST`.
4. Opens an SSH session to `secrets.EC2_USER@secrets.EC2_HOST`, streams a heredoc with `bash -l -s`:
   - `set -euo pipefail`
   - `cd /home/ubuntu/Kozha && git fetch --prune origin && git reset --hard origin/main && git clean -fd -e ./data/` (preserves the data directory from cleanup)
   - `cd server && source /home/ubuntu/kozha-venv/bin/activate`
   - `pip install --no-cache-dir -r requirements.txt`
   - Verifies `../backend/chat2hamnosys/requirements.txt` exists; `pip install` that too
   - If `OPENAI_API_KEY` secret is non-empty, it updates `/etc/kozha.env` with the fresh key. Ownership is set to `root:kozha` if the `kozha` group exists; otherwise falls back to `root:root`.
   - **If the secret is empty, the env file is left as-is.** This gates the write behind the secret's presence. The `contribute.html` BYO-key field is the graceful-degradation path. This behavior was introduced per memory `feedback_deploy_secrets`.
   - `sudo systemctl daemon-reload && sudo systemctl restart kozha.service && sudo systemctl --no-pager status kozha.service`

**Secrets consumed.** `EC2_SSH_KEY`, `EC2_HOST`, `EC2_USER`, `OPENAI_API_KEY`. None of these are present in the repo; they are configured under Settings → Secrets → Actions.

**Failure modes that would block main.**
- SSH handshake fails (wrong key, host down, key mismatch after rebuild).
- `pip install` fails (upstream PyPI flake, or a removed package).
- spaCy model download failures (server.py expects `en_core_web_sm`; pip install should include it via the requirements).
- systemd restart fails (the service has a broken python import; happens e.g. when a new code path references an uninstalled dep).

**Failure modes that are graceful.**
- Missing `OPENAI_API_KEY` secret — env-file left alone; BYO-key handles it.
- Missing `/etc/kozha.env` entirely with a template available — bootstrapped from `deploy/systemd/kozha.env`.
- `kozha` group missing — falls back to `root:root` group ownership.

## Environment layout on the host (per memory `project_ec2_host_layout`)

- Repo checked out at `/home/ubuntu/Kozha`.
- Python venv at `/home/ubuntu/kozha-venv`.
- Service unit is `kozha.service`; logs go to journalctl.
- There is no `kozha` system user/group despite the workflow attempting to use it — the service runs under `ubuntu`. The deploy script's group fallback handles this correctly.

Any change that hardcodes the `kozha` user or group elsewhere in the codebase will break on this host. Memory already records this drift.

## a11y workflow as a signal (not a gate)

`a11y.yml` runs axe-core + pa11y against 11 scenarios listed in `docs/contribute-redesign/12-a11y-baseline.md`. It uploads a baseline artifact each run. The path filter means it only fires on PRs that touch `public/**` or scripts. A push that only touches `docs/polish/**` will not trigger it.

Because the gitStatus at session start showed `M docs/contribute-redesign/12-a11y-baseline.md` (a tracked modification), the baseline may already be off — a later prompt that touches `public/**` will have to reconcile. The audit itself touches no `public/**` file.

## Security workflow (broken)

Per memory `project_security_workflow_broken`, `security.yml` currently fails in 0 seconds on every push — pre-existing and not caused by this audit. Do not attempt to fix it from this audit; if a later polish prompt needs to lean on security.yml, plan to fix it as scoped work.

Reading the file: the job structure looks fine on paper (gitleaks → pip-audit → npm-audit gated by `hashFiles('package.json')` → tests). The zero-second failure is likely an auth or permissions issue with gitleaks or a runner startup problem, not a code-level bug in the YAML. Actual debugging is out of scope here.

## Node / npm status

There *is* a `package.json` and `package-lock.json` at repo root. `security.yml`'s npm-audit job is gated behind `hashFiles('package.json')`, so it runs. `chat2hamnosys.yml` also has a conditional prettier job gated on `package.json`. The dependencies in package.json are the a11y toolchain (axe, pa11y, puppeteer) — no production JS bundle.

Practical effect: frontend JS is *not* built or bundled. All `public/*.js` files are shipped verbatim, statically served. No npm step gates the deploy.

## What a later polish prompt is allowed to do

- Touch `public/*.{html,css,js}` — will trigger a11y.yml (non-blocking), deploy.yml (blocking, but a failure is usually not caused by frontend-only edits).
- Touch `server/**` or `backend/**` — triggers the same plus chat2hamnosys.yml PR lane. pytest runs. Keep server changes minimal.
- Add a new workflow or modify an existing one — risky; only do if the prompt is scoped for infrastructure.
- Change `.github/workflows/deploy.yml` — **do not** without explicit scoping. The remote deploy script is sensitive to exact shell quoting.

## What this prompt did

Nothing that touches any of the above. Only `docs/polish/**` is written. The a11y path filter won't fire; the deploy workflow will run on push but only pulls/reinstalls/restarts — no behavior change. Safe.

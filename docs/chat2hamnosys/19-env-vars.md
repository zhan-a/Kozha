# chat2hamnosys — Environment variable manifest (Prompt 19)

Single source of truth for every env var the system reads. **Rule: if a
variable isn't in this table, it should not exist in code.** Adding a
new var requires updating `.env.example`, this file, and the consumer
in the same change.

Columns:

| Field | Meaning |
|---|---|
| `Name` | Env var name. |
| `Purpose` | What the runtime does with it. |
| `Example` | A safe placeholder value. |
| `Required` | `yes` (boot fails / endpoint 503s without it), `no` (has a default). |
| `Reader` | The module that reads it (file:line where practical). |

---

## 1. Identity / build

| Name | Purpose | Example | Required | Reader |
|---|---|---|---|---|
| `BUILD_SHA` | Image / deploy identifier. Surfaced in `GET /api/chat2hamnosys/health` so callers can confirm which build they hit. Injected by the Docker build arg. | `8b1c4e2` | no (defaults to `unknown` in image, omitted from response if blank) | `backend/chat2hamnosys/api/admin.py` (`get_health`) |
| `KOZHA_HOME` | Container working directory. Used by ops scripts to locate the app root. | `/app` | no (default `/app`) | Dockerfile + scripts |
| `KOZHA_HTTP_PORT` | Host port the compose stack publishes. | `8000` | no (default `8000`) | `docker-compose.yml` |
| `PROMETHEUS_HTTP_PORT` | Host port for the optional Prometheus container under `--profile obs`. | `9090` | no (default `9090`) | `docker-compose.yml` |

## 2. LLM / OpenAI

| Name | Purpose | Example | Required | Reader |
|---|---|---|---|---|
| `OPENAI_API_KEY` | Auth credential for every LLM call. Without it the readiness probe degrades to 503 and any endpoint that needs the model raises `LLMConfigError`. | `sk-...` | **yes** | `backend/chat2hamnosys/llm/client.py:243`, `api/admin.py:215` |

## 3. Budget / cost guardrails

| Name | Purpose | Example | Required | Reader |
|---|---|---|---|---|
| `CHAT2HAMNOSYS_SESSION_BUDGET_USD` | Hard cap on cumulative spend for one authoring session. Pre-flight check raises `BudgetExceeded` before the network call. | `2.0` | no (default `2.0`) | `backend/chat2hamnosys/llm/budget.py:49` |
| `CHAT2HAMNOSYS_DAILY_BUDGET_USD` | Daily ceiling used by the cost-projection alert. | `200.0` | no (default unset → projection skipped) | `backend/chat2hamnosys/obs/alerts.py:494` |
| `CHAT2HAMNOSYS_PER_IP_DAILY_CAP_USD` | Per-IP daily spend ceiling enforced by the security middleware. | `10.0` | no (default `10.0`) | `backend/chat2hamnosys/security/config.py:131` |
| `CHAT2HAMNOSYS_GLOBAL_DAILY_CAP_USD` | Process-wide hard ceiling. When breached, all LLM calls 503 until midnight UTC. | `200.0` | no (default `200.0`) | `backend/chat2hamnosys/security/config.py:134` |

## 4. Storage backend

| Name | Purpose | Example | Required | Reader |
|---|---|---|---|---|
| `CHAT2HAMNOSYS_STORAGE_BACKEND` | Selects the sign-store driver. `sqlite` is the only implementation today; `json` is reserved for the per-language SiGML file driver described in Prompt 1 §9e and is a no-op until added. | `sqlite` | no (default `sqlite`) | reserved — not yet branched on (see Prompt 1 §9e) |
| `CHAT2HAMNOSYS_STORAGE_PATH` | Filesystem path for the active sign store. For `sqlite`, the SQLite file. For `json`, the per-language SiGML directory. | `/app/data/authored_signs.sqlite3` | no (default uses `CHAT2HAMNOSYS_SIGN_DB`) | reserved — see `CHAT2HAMNOSYS_SIGN_DB` for the live setting |
| `CHAT2HAMNOSYS_DATA_DIR` | Root of the Kozha data directory the API resolves relative paths against (e.g. exports). | `/app/data` | no (defaults to repo `data/`) | `backend/chat2hamnosys/api/dependencies.py:74` |
| `CHAT2HAMNOSYS_SESSION_DB` | SQLite path for the live `SessionStore`. | `/app/data/chat2hamnosys/sessions.sqlite3` | no | `backend/chat2hamnosys/api/dependencies.py:63` |
| `CHAT2HAMNOSYS_SIGN_DB` | SQLite path for the live `SQLiteSignStore`. | `/app/data/authored_signs.sqlite3` | no | `backend/chat2hamnosys/api/dependencies.py:73` |
| `CHAT2HAMNOSYS_TOKEN_DB` | SQLite path for the per-session token store. | `/app/data/chat2hamnosys/session_tokens.sqlite3` | no | `backend/chat2hamnosys/api/dependencies.py:84` |
| `CHAT2HAMNOSYS_REVIEWER_DB` | SQLite path for the reviewer / approval store. | `/app/data/chat2hamnosys/reviewers.sqlite3` | no | `backend/chat2hamnosys/review/dependencies.py:50` |
| `CHAT2HAMNOSYS_EXPORT_AUDIT` | Append-only JSONL file recording every export decision. | `/app/data/chat2hamnosys/export_audit.jsonl` | no | `backend/chat2hamnosys/review/dependencies.py:59` |

## 5. HTTP surface

| Name | Purpose | Example | Required | Reader |
|---|---|---|---|---|
| `CHAT2HAMNOSYS_CORS_ORIGINS` | Comma-separated allow-list of origins. `*` is permissive (dev only); production locks to the frontend host. | `https://kozha.example` | no (default `*`) | `backend/chat2hamnosys/api/app.py:78` |
| `CHAT2HAMNOSYS_RATE_LIMIT` | slowapi-format rate string applied per-IP across the chat2hamnosys mount. | `30/minute` | no (default `30/minute`) | `backend/chat2hamnosys/api/router.py:124` |
| `CHAT2HAMNOSYS_RATE_LIMIT_PER_MIN` | Documented integer-only alias of `CHAT2HAMNOSYS_RATE_LIMIT`. Operators who set this should also set the slowapi-format string to keep the limiter reading the same value. Tracked in this manifest so deployment templates can stay numeric. | `30` | no | docs only — surfaced in `19-deployment.md` |
| `CHAT2HAMNOSYS_MAX_INPUT_LEN` | Hard cap on prose / answer / correction length (characters). | `2000` | no (default `2000`) | `backend/chat2hamnosys/security/config.py:129` |

## 6. Logging / observability

| Name | Purpose | Example | Required | Reader |
|---|---|---|---|---|
| `CHAT2HAMNOSYS_LOG_LEVEL` | Root Python logger level. `DEBUG` enables per-request traces; production uses `INFO` or `WARNING`. | `INFO` | no (default `INFO`) | uvicorn / process bootstrap (set via `--log-level` in deployment templates) |
| `CHAT2HAMNOSYS_LOG_CONTENT` | When `true`, prose, answers, and corrections are logged verbatim. **Default `false`** — keep it off in production unless you have a documented research-data-handling agreement. | `false` | no (default `false`) | observed by the structured logger; off by default in `obs/logger.py` |
| `CHAT2HAMNOSYS_LOG_SINK` | Where the structured event logger writes (`file` or `stdout`). `file` rotates daily under `CHAT2HAMNOSYS_LOG_DIR`. | `file` | no (default `file`) | `backend/chat2hamnosys/obs/logger.py:130` |
| `CHAT2HAMNOSYS_LOG_DIR` | Directory the file sink rotates into. | `/app/logs` | no (default `<repo>/logs`) | `backend/chat2hamnosys/obs/logger.py:349` |
| `CHAT2HAMNOSYS_LOG_RETENTION_DAYS` | Days of rotated logs kept; older files are deleted on next boot / rotate. | `30` | no (default `30`) | `backend/chat2hamnosys/obs/logger.py:354` |
| `CHAT2HAMNOSYS_ALERT_WEBHOOK_URL` | Slack / Discord webhook target for the alert daemon. Unset → log-only. | `https://hooks.slack.com/...` | no | `backend/chat2hamnosys/obs/alerts.py:491` |
| `CHAT2HAMNOSYS_ALERT_INTERVAL_S` | Alert daemon poll period (seconds). | `300` | no (default `300`) | `backend/chat2hamnosys/obs/alerts.py:525` |
| `CHAT2HAMNOSYS_ALERT_SILENCE_S` | Suppression window so identical alerts don't fire repeatedly. | `1800` | no (default `1800`) | `backend/chat2hamnosys/obs/alerts.py:500` |
| `CHAT2HAMNOSYS_PENDING_REVIEW_STALE_H` | Hours a pending review may sit before the queue-stalled alert trips. | `24` | no | `backend/chat2hamnosys/obs/alerts.py:497` |
| `CHAT2HAMNOSYS_ANOMALY_ALERT_URL` | Distinct webhook for anomaly events (e.g. injection screen hits). Falls back to `CHAT2HAMNOSYS_ALERT_WEBHOOK_URL`. | `https://hooks.slack.com/...` | no | `backend/chat2hamnosys/security/config.py:144` |

## 7. Renderer integration

| Name | Purpose | Example | Required | Reader |
|---|---|---|---|---|
| `CHAT2HAMNOSYS_RENDERER_URL` | HTTP endpoint of the (optional) server-side renderer service. Today the canonical avatar pipeline runs in the browser via CWASA; this var is documented as the wire-protocol for a future renderer container (see `docker-compose.yml`). | `http://renderer:9000` | no (default unset → fallback to subprocess template) | reserved — wired by deployment template, not yet read in code |
| `KOZHA_RENDERER_CMD` | Subprocess template (e.g. headless JASigning) used by the rendering layer when the URL form is unset. `{sigml}` is replaced with a temp-file path. | `jasigning --sigml {sigml} --out {out}` | no (default unset → preview is metadata-only) | `backend/chat2hamnosys/rendering/preview.py:116` |

## 8. Reviewer / governance

| Name | Purpose | Example | Required | Reader |
|---|---|---|---|---|
| `CHAT2HAMNOSYS_REVIEWER_APPROVAL_COUNT` | Documented public name for the two-reviewer rule; mirrored to `CHAT2HAMNOSYS_REVIEW_MIN_APPROVALS` when populating an environment. | `2` | no (default `2`) | docs / templates only |
| `CHAT2HAMNOSYS_REVIEW_MIN_APPROVALS` | Minimum number of independent reviewer approvals before export is unlocked. | `2` | no (default `2`) | `backend/chat2hamnosys/review/policy.py:76` |
| `CHAT2HAMNOSYS_REVIEW_REQUIRE_NATIVE` | When `true`, at least one approver must be a native Deaf reviewer. | `true` | no (default `true`) | `backend/chat2hamnosys/review/policy.py:77` |
| `CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE` | Bypass the two-reviewer rule. **Disable in production** — only used for solo-author research mode. | `false` | no (default `false`) | `backend/chat2hamnosys/review/policy.py:79` |
| `CHAT2HAMNOSYS_REVIEW_FATIGUE_THRESHOLD` | Maximum reviews per reviewer per day before the queue auto-skips them. | `25` | no (default `25`) | `backend/chat2hamnosys/review/policy.py:82` |
| `CHAT2HAMNOSYS_REVIEW_REQUIRE_REGION_MATCH` | When `true`, at least one approver's regional variant must match the sign's variant. | `true` | no (default `true`) | `backend/chat2hamnosys/review/policy.py:85` |

## 9. Security

| Name | Purpose | Example | Required | Reader |
|---|---|---|---|---|
| `CHAT2HAMNOSYS_PII_POLICY` | `hashed` (HMAC signer ids) or `plaintext` (raw — IRB-only). | `hashed` | no (default `hashed`) | `backend/chat2hamnosys/security/config.py:120` |
| `CHAT2HAMNOSYS_SIGNER_ID_SALT` | HMAC salt for hashed signer ids. **Set per-deployment**; the default triggers a startup warning. | `48-byte-random-hex` | no (default unsafe placeholder) | `backend/chat2hamnosys/security/config.py:114` |
| `CHAT2HAMNOSYS_ENABLE_INJECTION_CLASSIFIER` | `0` disables the LLM-backed injection classifier (regex screen still runs). | `1` | no (default `1`) | `backend/chat2hamnosys/security/config.py:139` |
| `CHAT2HAMNOSYS_ENABLE_OUTPUT_MODERATION` | `0` disables the OpenAI moderation post-check on outgoing content. | `1` | no (default `1`) | `backend/chat2hamnosys/security/config.py:142` |

---

## Naming conventions

- **Prefix everything with `CHAT2HAMNOSYS_`** unless the variable is shared with the wider Kozha host (`KOZHA_*`) or with industry standards (`OPENAI_API_KEY`, `BUILD_SHA`).
- **Document the canonical reader.** When two names exist for the same setting (e.g. `CHAT2HAMNOSYS_REVIEWER_APPROVAL_COUNT` ↔ `CHAT2HAMNOSYS_REVIEW_MIN_APPROVALS`), the table calls out the canonical reader. Templates set both so a future consolidation is non-breaking.
- **Never log a value** — only the variable name, and only at WARNING+ when the value is missing or unsafe (see `security/config.py:115`).

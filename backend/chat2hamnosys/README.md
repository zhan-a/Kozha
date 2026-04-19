# chat2hamnosys

LLM-mediated authoring of new sign-language lexicon entries for [Kozha](../../README.md). A domain
expert types a prose description of one sign in plain English; the system parses the description
into phonological slots, asks at most three clarifying questions, generates a HamNoSys 4.0 string,
validates it against a Lark grammar, renders a SiGML preview, accepts pointed corrections, and
routes the result through a two-Deaf-reviewer gate before exporting it into the live per-language
sign databases under `data/`.

## Scope statement

This system authors **citation-form lexical signs with Deaf-reviewer validation**. It does not
generate connected discourse, classifier constructions, role shift, depicting verbs, simultaneous
two-sign productions, or non-manual prosody beyond a single mouth picture and eyebrow flag. See the
[feasibility study](../../docs/chat2hamnosys/00-repo-audit.md) for the linguistic boundaries and
[20-architecture.md](../../docs/chat2hamnosys/20-architecture.md) for the precise list of "what each
component explicitly does not do."

It is **not a replacement for human interpreters**, not a sign-to-text translator, not an ASR
front-end, and not a teaching tool. It is an **offline lexicographer's assistant** with a Deaf-led
governance gate, intended to grow Kozha's vocabulary safely.

## Quickstart (target: under 15 minutes)

```bash
# 1. Clone and configure
git clone https://github.com/your-username/Kozha.git
cd Kozha
cp .env.example .env
# Edit .env: paste your OPENAI_API_KEY and a 48-byte CHAT2HAMNOSYS_SIGNER_ID_SALT.
#   openssl rand -hex 48

# 2. Build and run
docker compose up

# 3. Browse to the authoring UI
open http://localhost:8000/chat2hamnosys/

# 4. Run a demo session
#    a. Click "New session"
#    b. Pick "BSL"
#    c. Describe a sign you know — e.g. "HELLO is signed with a flat handshape at the
#       temple, palm forward, fingertips up. The hand moves in a small arc away from the head."
#    d. Answer the clarifying question if any
#    e. Click "Generate" → preview plays in the avatar panel
#    f. Click a hand region to flag a correction, or accept

# 5. Replay a known-good session for sanity
python -m examples.replay --example electron
```

Quickstart troubleshooting and timing notes live in [20-handoff.md](../../docs/chat2hamnosys/20-handoff.md).

## Architecture sketch

```
                                   ┌──────────────────────────────────────────┐
                                   │  Author (lexicographer, domain expert)   │
                                   └────────────────────┬─────────────────────┘
                                                        │ prose description
                                                        ▼
                              ┌──────────────────────────────────────┐
                              │  api/security.py                     │  injection screen
                              │   ├─ regex / LLM injection screen    │  + sanitize + tag-wrap
                              │   └─ output moderation               │
                              └──────────────────┬───────────────────┘
                                                 ▼
                              ┌──────────────────────────────────────┐
                              │  parser/  — prose → PartialParameters│
                              │  + Gap[]                             │
                              └──────────────────┬───────────────────┘
                                                 ▼
                          ┌────────────[ gaps found? ]───────┐
                          │ no                            yes│
                          │                                  ▼
                          │      ┌──────────────────────────────────────┐
                          │      │  clarify/ — Gap → Question →         │
                          │      │  PartialParameters (loop ≤3 turns)   │
                          │      └──────────────────┬───────────────────┘
                          ▼                         │
                    ┌─────┴─────────────────────────┘
                    ▼
        ┌──────────────────────────────────────┐
        │  generator/  — Parameters → HamNoSys │  vocab lookup
        │   ├─ vocab_map.yaml deterministic    │  + LLM slot fallback
        │   └─ LLM fallback for OOV slots      │  + retry on invalid
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  hamnosys/  — Lark grammar validate  │  loops back to generator
        └──────────────────┬───────────────────┘  on retry-able errors
                           ▼
        ┌──────────────────────────────────────┐
        │  rendering/  — HamNoSys → SiGML XML  │
        │  + (optional) preview cache → MP4    │
        └──────────────────┬───────────────────┘
                           ▼
                  [ author reviews preview ]
                           ▼
        ┌──────────────────────────────────────┐
        │  correct/ — freeform correction →    │  loops back to generator
        │  param diffs → re-generate           │  with revised slots
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  session.on_accept → SignEntry       │  status: draft
        │  storage/  — SQLite persist          │
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  review/ — Deaf reviewer console     │  status: pending_review
        │   ├─ two-reviewer rule               │     ↓ (≥2 qualifying approvals)
        │   ├─ ≥1 native-Deaf approval         │  status: validated
        │   └─ board export gate               │     ↓
        └──────────────────┬───────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │  storage.export_to_kozha_library     │  → data/hamnosys_<lang>_authored.sigml
        │  + ExportAuditLog (hash chain)       │  + JSONL audit row
        └──────────────────────────────────────┘
```

Throughout: every LLM call is wrapped by `llm/client.py` (retry + per-session budget cap +
fallback model). Every event lands in `obs/logger.py` (PII-hashed JSONL) and is counted in
`obs/metrics.py` (Prometheus text format at `/api/chat2hamnosys/metrics`).

## Component map

| Module | Responsibility |
|---|---|
| `api/` | HTTP surface, session token auth, slowapi rate limit, security middleware |
| `parser/` | Prose → `PartialSignParameters` + `Gap[]` |
| `clarify/` | Gap → multiple-choice or freeform `Question`; answers → updated parameters |
| `generator/` | Parameters → HamNoSys (vocab lookup + LLM slot fallback + repair loop) |
| `correct/` | Freeform correction text → minimal parameter diffs |
| `hamnosys/` | Lark grammar, 231 HamNoSys 4.0 codepoints, validator + normalizer |
| `rendering/` | HamNoSys → SiGML XML; optional subprocess/HTTP preview renderer |
| `session/` | State machine, append-only event log, SQLite session store |
| `review/` | Two-reviewer rule, approval policy, ExportAuditLog, board CLI |
| `llm/` | OpenAI SDK wrapper: retries, budget guard, model fallback, telemetry |
| `obs/` | Structured events, Prometheus metrics, dashboard, alert rules |
| `security/` | Input sanitizer, regex+LLM injection screen, rate limit, output moderation |
| `prompts/` | Versioned Jinja2 templates for parser, clarifier, generator, correction |
| `eval/` | E2E harness: 50 fixtures, parser/generator/e2e metrics, ablations, human-eval bridge |
| `examples/` | Five recorded sessions (replayable; double as golden tests) |
| `models.py` | `SignEntry`, `SignParameters`, `ClarificationTurn`, `ReviewRecord` Pydantic contracts |
| `storage.py` | `SQLiteSignStore` + export gate (status check + approval-count check) |

## Documentation index

Every prompt-numbered doc under [`docs/chat2hamnosys/`](../../docs/chat2hamnosys/):

| Doc | Topic |
|---|---|
| [00-repo-audit.md](../../docs/chat2hamnosys/00-repo-audit.md) | Baseline audit; integration plan; linguistic scope |
| [05-description-parser-eval.md](../../docs/chat2hamnosys/05-description-parser-eval.md) | Parser eval: per-field accuracy, gap precision/recall |
| [07-generator-eval.md](../../docs/chat2hamnosys/07-generator-eval.md) | Generator eval: 41-fixture gold set, symbol F1 |
| [15-review-and-export.md](../../docs/chat2hamnosys/15-review-and-export.md) | Deaf-reviewer workflow, two-reviewer rule, audit chain |
| [16-eval-results-readme.md](../../docs/chat2hamnosys/16-eval-results-readme.md) | E2E harness, ablations, regression guard, human-eval bridge |
| [17-injection-surface.md](../../docs/chat2hamnosys/17-injection-surface.md) | Every user-controlled string traced to LLM prompt |
| [17-security.md](../../docs/chat2hamnosys/17-security.md) | Secrets, kill switch, rollback, abuse response |
| [18-observability.md](../../docs/chat2hamnosys/18-observability.md) | Events, metrics, dashboard, 5 alerts, runbooks |
| [19-deployment.md](../../docs/chat2hamnosys/19-deployment.md) | Fly.io / Railway / EC2 deploy targets |
| [19-env-vars.md](../../docs/chat2hamnosys/19-env-vars.md) | Full env-var manifest |
| [19-prod-checklist.md](../../docs/chat2hamnosys/19-prod-checklist.md) | Pre-deploy audit checklist |
| [19-runbook.md](../../docs/chat2hamnosys/19-runbook.md) | On-call incidents and remediation |
| [20-architecture.md](../../docs/chat2hamnosys/20-architecture.md) | Component diagram, data flow, threat model |
| [20-ethics.md](../../docs/chat2hamnosys/20-ethics.md) | Deaf community leadership, governance, data policy |
| [20-demo-script.md](../../docs/chat2hamnosys/20-demo-script.md) | 90s / 5min / 15min walkthrough scripts |
| [20-glossary.md](../../docs/chat2hamnosys/20-glossary.md) | HamNoSys, SiGML, BSL, ASL, gloss, citation form, etc. |
| [20-press/](../../docs/chat2hamnosys/20-press/) | One-pager, screenshots, video, social-media variants |
| [20-research-log.md](../../docs/chat2hamnosys/20-research-log.md) | Dated experiment log, ablation results, hypotheses |
| [20-handoff.md](../../docs/chat2hamnosys/20-handoff.md) | Current state, open questions, contacts |

## Development

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for dev setup, the prompt-versioning rule, the vocab-map
extension procedure, the reviewer onboarding workflow, and the Deaf-reviewer sign-off rule for any
PR that changes generation output.

## Contact

Concerns from the Deaf community: **deaf-feedback@kozha.dev** (monitored daily).
Technical questions: open an issue on GitHub.

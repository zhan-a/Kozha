# Handoff

This document is for the next maintainer of the chat2hamnosys subsystem (or future-Bogdan after a
6-month break). It captures the *state of the world as of the closeout commit* (2026-04-19), the
decisions still open, the things that are shakier than they look, and the one piece of advice the
author wishes someone had given him at the start.

If you are picking this up cold, read in this order: this document → the [README](../../backend/chat2hamnosys/README.md) →
[20-architecture.md](20-architecture.md) → [20-ethics.md](20-ethics.md). Then run the quickstart.

---

## 1. State as of 2026-04-19

### What works (production-ready)

| Area | Status | Notes |
|---|---|---|
| Parser (prose → 8 phonological slots + Gap[]) | Working | 0.81 mean per-field accuracy on 50-fixture set |
| Clarifier (Gap → Question → resolved value) | Working | ≤3 turns capped; multi-choice + freeform |
| Generator (Parameters → HamNoSys) | Working | Vocab map + LLM fallback + repair loop; 0.62 exact-match |
| HamNoSys validator | Working | Lark grammar, 231 codepoints, semantic checks |
| Renderer (HamNoSys → SiGML) | Working | XML composition; preview cache content-addressed |
| Correction interpreter | Working | Click-targeted region + freeform text → minimal diffs |
| Session state machine | Working | Append-only event log, SQLite persistence |
| Two-reviewer rule + native-Deaf signoff | Working | Code is correct; **board not yet seated** |
| Export gate (defense-in-depth) | Working | Status check + approval count check, redundantly |
| ExportAuditLog (hash chain) | Working | Tamper-evident; periodic external archiving recommended |
| Reviewer console UI | Working | Bearer-token auth; queue + detail + action bar + fatigue meter |
| Eval harness | Working | 50 fixtures, 4 metric layers, 3 ablations, regression guard |
| LLM client wrapper | Working | Retry, per-session budget, model fallback, telemetry JSONL |
| Observability: events, metrics, dashboard, alerts | Working | 5 alert rules, 3 runbook playbooks |
| Security: injection, sanitize, rate limit, PII, moderation | Working | All layers shipped; gitleaks pre-commit + CI |
| Docker compose (local + obs profile) | Working | Image 412 MB |
| Fly.io / Railway deployment | Working | Both verified end-to-end |

### What's shaky (works but I would not bet money on it)

| Area | Why it's shaky | What to do |
|---|---|---|
| **Board seating** | The two-reviewer rule is enforced in code, but the board does not exist yet. Currently runs in `CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE=true` bootstrap mode for the maintainer. | This is **the single most important open task.** See § 3. |
| **Eval gold set** | 50 fixtures exist but only the DGS subset (~22 entries from Prompt 7) is Deaf-native-verified. The BSL/ASL fixtures were authored from corpus descriptions by hearing developers. | Get the seated reviewer board to validate the BSL/ASL fixtures. Until then, treat the BSL/ASL exact-match numbers as upper bounds on real performance, not point estimates. |
| **Human-eval results** | n=12 (4 reviewers × 3 signs each in BSL only). Confidence intervals are wide. | Run a proper n≥30 per-language Huenerfauth study after board seating. |
| **EC2 deployment path** | systemd unit and nginx config are written but the path was never fully verified end-to-end on a real EC2 box. | If you need EC2, budget half a day for the first deployment; expect to discover at least one config gap. |
| **Renderer subprocess path** | The legacy `KOZHA_RENDERER_CMD` subprocess template still works but is not used in the canonical Docker setup. The HTTP renderer path (`CHAT2HAMNOSYS_RENDERER_URL`) is documented but the Docker image does not yet ship a renderer service. | Today previews come from the browser CWASA. The server-side renderer block in `docker-compose.yml` is commented out. If you need pre-rendered MP4s, wire it up. |
| **Two-handed sign asymmetry** | Vocab map has flat handshape lookup for both hands; no Battison constraints. Two-handed signs work for symmetric cases (CATALYST in the examples), break silently for asymmetric ones. | A linguistic problem more than a code problem. See [20-research-log.md § rejected approaches](20-research-log.md#rejected-approaches). |
| **Mouth-picture coverage** | We encode mouth pictures as freeform SAMPA strings; CWASA renders them, but we have no validator for the SAMPA inventory. Bad strings produce visual glitches, not errors. | Either add a SAMPA validator or remove the field until we do. Probably the former. |

### Quickstart sanity check (2026-04-19, closeout)

Verified at closeout from the project root, on the maintainer's macOS dev machine:

| Step | Duration | Notes |
|---|---|---|
| `python3` import path bootstrap (`backend/chat2hamnosys/__init__.py`) | <1 ms | OK |
| All 5 example fixtures resolve and parse as JSON | 7 ms | OK |
| Replay CLI imports cleanly (`from examples.replay import main`) | 36 ms | OK |
| `python -m examples.replay --list` exits 0 | <100 ms | OK |
| `docker-compose.yml` is valid YAML, 2 services, 5 volumes, 2 networks | n/a | OK |
| All 56 internal Markdown links across the doc set resolve | n/a | OK |
| **`docker compose up` end-to-end on a fresh box** | **not verified at closeout** | Docker not installed in the closeout environment. The Fly.io and Railway deployment paths *were* verified previously (see [19-deployment.md](19-deployment.md)). Adding this to § open questions for first-deploy validation by the next maintainer. |
| **Browser walkthrough at `localhost:8000/chat2hamnosys/`** | **not verified at closeout** | Same reason. This is the most important thing for the next maintainer to time end-to-end on a clean machine. |

**Honest assessment:** the Python-side machinery and fixtures are sound. The full quickstart
including Docker, browser, and a real OPENAI_API_KEY was last end-to-end verified on the
maintainer's machine on 2026-04-15 during Prompt 19 deployment validation, not at this closeout.
Re-verify before the first external demo.

### What's planned but not built

| Item | Why not done | Difficulty |
|---|---|---|
| LSF and NGT support | No reviewer board recruited for these languages; no fixture set | Add a vocab map + reviewers + fixtures. ~2 weeks per language. |
| Server-side avatar renderer | Subprocess path exists; HTTP path documented; no production renderer service | Medium — JASigning headless under xvfb. ~1 week. |
| SAMPA mouth-picture validator | Realised late; not blocking but noisy | Small — ~1 day to add to `hamnosys/`. |
| Deaf co-PI recruitment formal process | Discussed; no written rubric | Governance, not engineering. ~2 weeks of conversations. |
| Public landing page for the Deaf community | The `kozha-translate.com` site is the core translator only | Medium — content + design. ~1 week. |
| Full IRB-aware logging mode | Plaintext logging policy exists; IRB attestation procedure not written | Small but legal — coordinate with university IRB. |
| OpenAI-provider abstraction | Provider-agnostic in design; only OpenAI in code | Small — single-file swap in `llm/client.py`. ~2 days. |

---

## 2. Open questions for the next researcher

These are decisions that need to be made, not bugs to be fixed. Each has a date stamp from when it
was raised.

### 2.1 Data licensing for the validated lexicon

**Raised:** 2026-03-30. **Status:** open.

The validated lexicon (entries that pass the two-reviewer gate and are exported to `data/`) is
*derived from* the LLM authoring process and *vouched for by* Deaf reviewers. Who owns it? What
license is it released under?

Options:

- **CC BY-SA 4.0:** matches the parent Kozha project conventions. Requires attribution +
  share-alike.
- **CC BY-NC-SA 4.0:** stricter (non-commercial). Closer to the source corpora's licenses (e.g.
  Universität Hamburg's DictaSign).
- **Custom Deaf-community license:** modeled on the [Lexicon Project](https://www.lexiconproject.org/)
  licenses with a Deaf-community-veto clause.

**My recommendation, not a decision:** custom license, drafted with input from a Deaf-community
lawyer. Until the license is settled, the validated lexicon is internal-only.

### 2.2 Funding trajectory

**Raised:** 2026-04-15. **Status:** open.

The project is unfunded as of today. To staff the Deaf reviewer board appropriately, we need
~$60k/year in honoraria (estimated $1k/board member/month × ~5 active members). Without funding,
the board is volunteer-only, which (a) limits who can serve and (b) is ethically uncomfortable
because we are asking Deaf reviewers to do uncompensated labour for a hearing-led project.

Options:

- **Academic grant** through the MIT advisor (Gómez-Bombarelli). Closest fit. Materials in
  preparation.
- **Accessibility-tech foundation grant** (e.g. Mozilla, Ford). Possible, slower.
- **Industry sponsorship.** Risk of conflict of interest with the governance model. Would
  require unusually strong contractual protections.

**Decision required by:** before the board is formally seated. Putting unpaid Deaf reviewers in
charge of a system that hearing engineers built is the wrong order of operations.

### 2.3 Scope expansion: ASL classifiers

**Raised:** every meeting. **Status:** consistently rejected, possibly worth revisiting once.

The architecture explicitly does not generate classifiers. The reasons (linguistic complexity,
context-bound grammar, would require re-architecting the parser) are documented in
[20-research-log.md § rejected approaches](20-research-log.md#rejected-approaches).

The next researcher should *not* take this on as an extension. If a future user community needs
classifier authoring, build a separate system optimised for that linguistic problem. Don't
contort this one.

### 2.4 Deaf co-PI recruitment

**Raised:** 2026-04-01. **Status:** in progress, no formal candidate yet.

The maintainer (Bogdan) is hearing. Long-term project leadership should include a Deaf co-PI with
authority equal to the maintainer's, not just a board member among many. Ideally a Deaf
linguist or accessibility-tech researcher with experience in community-governed projects.

**What's been done:** informal outreach to two academic groups. No commitments.

**What needs doing:** formalize the role description (responsibilities, authority, compensation),
distribute through Deaf-academic networks, conduct interviews with the existing board (when
seated). This is a governance milestone, not a technical one.

### 2.5 Scope expansion: support for emerging signs / community-coined neologisms

**Raised:** 2026-04-10. **Status:** open.

Kozha currently supports established lexicon entries. New community-coined signs (e.g. for new
technologies, social movements) emerge constantly and may not fit the citation-form mold cleanly.

Question: should this system have a separate workflow for explicitly-marked "emerging sign"
entries (perhaps requiring a different reviewer composition, or a time-bounded review window)?

**My recommendation, not a decision:** wait until the board exists and let the board decide.
Community-coined signs are exactly the kind of decision that should not be made by hearing
engineers ahead of the board.

---

## 3. The single most important task

**Recruit and seat the Deaf reviewer board.** Three native-Deaf reviewers per supported language,
formal terms, audited bearer tokens. Without this, the system is in bootstrap mode forever and
every export is a single-reviewer override that violates the project's own ethics statement.

Until the board exists, the project should not promote itself as a community-governed system. It
is honestly one engineer's prototype with governance machinery wired up but not exercised. That's
fine for a research project; it is not fine for a public-facing tool.

If you take one thing from this handoff: **do this first.**

---

## 4. Things I wish someone had told me at the start

> **Build the governance machinery before building the model. The reverse order means you ship a
> system that works technically but cannot be deployed ethically, and then you have to slow down
> at exactly the moment momentum should be picking up.**

Concretely: I built the parser, the generator, the validator, the renderer, the correction loop,
the eval harness, the security layer, the observability layer, the deployment scripts — and only
then started writing [20-ethics.md](20-ethics.md) and the reviewer console. The right order would
have been:

1. Write the ethics statement first. Talk to Deaf community organisations *before* writing code.
2. Recruit the reviewer board second.
3. Build the reviewer console third.
4. Build the authoring pipeline fourth.

That order forces you to accept governance constraints as design constraints, not as
post-hoc additions. It's slower in week one and faster every week after.

Other smaller things:

- **Version your prompts from day one.** I built three prompts inline before realising I needed
  a versioning system. The migration to `prompts/*.md.j2` was a one-day churn that should have
  been a 30-minute setup.
- **Decide your eval gold-set construction protocol before you build the gold set.** I built one
  gold set from the vocab map (Prompt 7), then realised every match was a tautology, then built a
  second one from corpus descriptions. The first set still exists in the repo and is misleading
  if you don't read [20-research-log.md § Prompt 7](20-research-log.md#2026-03-15--prompt-7-generator-first-version).
- **Don't over-design the state machine.** The 8-state SessionState was right for the API
  surface but I spent two days adding a 4-state SignStatus before I needed it. Both exist now;
  it's fine. But the SignStatus could have been added when the second reviewer was needed
  (Prompt 15) without rework, and that would have saved the upfront cognitive overhead.
- **The audit log's hash chain is more useful than I expected.** Two months in, I was glad to
  have it. Don't skip it.

---

## 5. Contacts

### Project

- **Bogdan Mironov** (maintainer, hearing) — bogdan@kozha.dev / mironovbogdan825@gmail.com
- **Zhan** (Kozha core developer)
- **Advisor:** Askhat Zhumabekov

### Academic

- **Gómez-Bombarelli group**, MIT (academic supervision for the chat2hamnosys subsystem)
  — see [gomezbombarelli.mit.edu](https://gomezbombarelli.mit.edu/) for current group members
- **Universität Hamburg, Institute of German Sign Language** (HamNoSys + DGS-Korpus authority)
- **Virtual Humans Group, UEA** (CWASA / JASigning licensing and integration)

### Community

- **deaf-feedback@kozha.dev** — Deaf community feedback (monitored daily)
- **research@kozha.dev** — academic enquiries
- **security@kozha.dev** — security disclosures
- **Reviewer board chair** — TBD, pending board seating

### If the project is being handed off entirely

If someone other than Bogdan is taking over maintainership:

1. **Transfer the GitHub admin role** + secrets (Fly.io, Railway, OpenAI API key, SIGNER_ID_SALT).
2. **Update the contact emails** in `.env.example`, `20-ethics.md`, `20-press/one-pager.md`,
   `CONTRIBUTING.md`, this document. The `deaf-feedback@` address must continue to be monitored
   without interruption — if the new maintainer cannot guarantee that, pause the system per
   [20-ethics.md § 8](20-ethics.md#8-contact).
3. **Notify the reviewer board** (when it exists) in writing. Their continued participation is
   contingent on knowing who is on the other end of the system.
4. **Update [20-research-log.md](20-research-log.md)** with a handover entry dated to the
   transition, summarising the state at handover.

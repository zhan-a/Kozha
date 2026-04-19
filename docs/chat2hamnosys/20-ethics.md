# Ethics statement

This document is binding on every contributor to the `chat2hamnosys` subsystem. It is enforced
at PR-review time (see [CONTRIBUTING.md](../../CONTRIBUTING.md)) and at deployment time (see
[19-prod-checklist.md](19-prod-checklist.md)).

## 1. Deaf community leadership

We hold the position that **sign languages are the property of their Deaf communities**. They
are not source material for hearing engineers to improve, optimize, or extend on their own
authority. Every linguistic decision encoded in this system — what counts as a valid sign, what
counts as a regional variant, what counts as culturally inappropriate output — is a decision
that belongs to native Deaf signers from the community of the language in question.

Concretely, this means:

- **Generation output is provisional until a Deaf reviewer signs off.** No sign produced by the
  LLM appears in the live Kozha library without ≥2 reviewer approvals, of which ≥1 must be
  from a native-Deaf reviewer registered for that sign language. This is enforced in
  `review/policy.py` and double-checked at the storage gate (`storage.py:export_to_kozha_library`).
- **Hearing engineers do not vote on linguistic correctness.** Hearing collaborators may
  flag obvious software bugs (the renderer crashed, the validator misclassified a known-good
  symbol) but cannot mark a sign as accurate.
- **Deaf governance is structural, not advisory.** The reviewer board has unilateral authority
  to: pause the system, retract any exported sign, change the approval policy, refuse
  funding sources, and require this document to be amended. There is no "engineering override".

This is the position of the [European Union of the Deaf](https://www.eud.eu/) and the
[World Federation of the Deaf](https://wfdeaf.org/). We adopt it explicitly. See § Alignment
with Deaf-led organisations below.

## 2. What this tool is for

It is for:

- **Lexicographers and linguists** — both Deaf and hearing — who want to expand a sign language's
  documented citation-form vocabulary.
- **Deaf community contributors** who want to add regional or domain-specific signs to a public
  resource without learning HamNoSys notation directly.
- **Researchers** investigating sign-language phonology, who can use the parser/generator pipeline
  as an instrument and the audit log as a corpus.
- **The Kozha translation pipeline**, which consumes the validated entries via the per-language
  SiGML databases.

## 3. What this tool is *not* for

It is not for:

- **Replacing human interpreters.** Sign languages are full languages and interpretation is a
  professional skill. This tool authors *static lexicon entries*; it cannot interpret a meeting,
  translate a courtroom proceeding, or substitute for a qualified interpreter in any setting where
  one is required.
- **Generating connected discourse.** The system produces citation forms (one sign at a time, in
  isolation). It does not model classifier constructions, role shift, depicting verbs,
  simultaneous productions, signing space, or the prosody of connected signing.
- **Bypassing the Deaf community.** The tool will refuse to export a sign without ≥2 reviewer
  approvals (enforced in two layers; see [20-architecture.md § threat model](20-architecture.md)).
  A "bootstrap mode" with a single reviewer exists for initial seeding only; it logs a WARNING
  on every validation and must be disabled in production
  (`CHAT2HAMNOSYS_REVIEW_ALLOW_SINGLE=false`).
- **Surveillance, profiling, or non-consensual research on Deaf signers.** The PII-hashed logging
  default (`CHAT2HAMNOSYS_PII_POLICY=hashed`) anonymizes signer identifiers via HMAC. Plaintext
  logging requires explicit IRB approval, recorded in the deployment's `19-prod-checklist.md`
  attestation.
- **A teaching tool.** The avatar renders a *phonological reconstruction* of a sign, not a fluent
  Deaf signer's natural production. It is not suitable for sign-language learners as a model of
  natural signing.

## 4. Governance

### Advisory board composition

The reviewer board has **at least three native-Deaf reviewers per sign language** the system
supports. As of 2026-04-19 the system supports `bsl`, `asl`, and `dgs`; that means a minimum of
nine board members across the three languages. Composition requirements:

- Native-Deaf signers (deafness and language acquisition before age 5).
- Demonstrated community standing — affiliated with at least one Deaf-led linguistic, cultural,
  or accessibility organisation (e.g. BDA for BSL, NAD for ASL, DGB for DGS).
- Term-bounded: 24 months, renewable once.
- At least one board member designated as **chair** with authority to pause the system.

Hearing reviewers may participate but cannot count toward the ≥1 native-Deaf approval requirement
without an explicit `allow_non_native=true` flag plus written justification, which is logged in
the audit chain.

### Reviewer requirements

A reviewer must have:

- A registered language competence list (e.g. `["bsl"]`, `["asl", "dgs"]`).
- An optional `regional_background` tag (e.g. `"BSL-Manchester"`, `"ASL-Black"`) so the policy can
  match regional variants when `CHAT2HAMNOSYS_REVIEW_REQUIRE_REGION_MATCH=true`.
- A confirmed deafness self-identification (`is_deaf_native=true|false|unknown`).

A reviewer cannot review their own authored entries. This is enforced at the API layer.

### Refusal mechanism

The board chair (or any two board members acting jointly) may at any time:

1. **Pause new generation** by setting the global daily cost cap to `$0`
   (`CHAT2HAMNOSYS_GLOBAL_DAILY_CAP_USD=0`); the next LLM call returns 503 until reversed.
2. **Retract any exported sign** by reverting its status from `validated` to `quarantined`. The
   sign disappears from the next library export. The audit log retains the original export row;
   a new audit row records the retraction.
3. **Demand a public apology** following the template in [17-security.md § abuse response](17-security.md#abuse-response).

These actions require no engineering escalation, no PR, and no technical knowledge beyond the
admin CLI documented in [15-review-and-export.md § admin commands](15-review-and-export.md).

## 5. Data policy

### What we store

| Class | Stored? | Where | Retention |
|---|---|---|---|
| Author prose descriptions | Yes | `sessions.sqlite3` (in `events` blob) | 30 days from session end |
| Resolved phonological parameters | Yes | `authored_signs.sqlite3` | Indefinite (it's the lexicon) |
| HamNoSys + SiGML output | Yes | `authored_signs.sqlite3` + `data/*.sigml` | Indefinite |
| Reviewer verdicts and comments | Yes | `authored_signs.sqlite3` (in `reviewers[]`) | Indefinite |
| Signer identifier (HMAC-hashed) | Yes | `sessions.sqlite3`, log files | Hashed; rotation invalidates |
| Raw IP address | Logged transiently for rate limiting | not persisted | not persisted |
| OpenAI request/response bodies | **No** | logs record `model`, `latency`, `cost`, `tokens` only | n/a |
| Voice / video of signer | **No** | system has no audio/video input | n/a |

### Who sees what

- **Author of a sign:** sees their own session history while the session is active.
- **Reviewers (any registered):** see the queue of pending entries in their language(s).
- **Reviewers (board):** plus governance dashboard, audit chain, all sessions, all sign records.
- **Hearing engineers (Bogdan, future maintainers):** see the deployment, code, and aggregated
  dashboard; should not access individual signer identity unless responding to a deletion request
  or a legal compulsion.
- **Public:** the live SiGML library (`data/hamnosys_*_authored.sigml`) is open. Author identity
  is *not* exported with the sign — only the validated content.

### How to request deletion

Any signer may request deletion of their attributed entries by emailing
**deaf-feedback@kozha.dev**. The board chair acknowledges within 7 days and removes the records
within 30 days. Removal:

- Deletes the session row, draft, and event log for that signer.
- Marks any exported signs as `quarantined` pending re-attribution or re-authoring.
- Removes the signer from the reviewer table if registered.
- Records a deletion row in the audit log (so the chain is preserved without leaking content).

Deletion of an *exported* sign that is in active use by a downstream Kozha translation request is
honored; the sign disappears from the next library refresh. We do not retain backups beyond 30
days specifically so that deletion is final.

## 6. Alignment with Deaf-led organisations

We align our position with:

- **World Federation of the Deaf** — [WFD position statement on accessibility of information and
  communication](https://wfdeaf.org/news/resources/wfd-position-statement-on-accessibility-of-information-and-communication-may-2019/)
  (May 2019).
- **WFD + WASLI joint statement on technology** — [Joint statement on use of signing avatars](https://wfdeaf.org/news/resources/wfd-wasli-joint-statement-use-signing-avatars-13-april-2018/)
  (April 2018), which we cite for its specific guidance on signing-avatar deployment: avatars
  should **not** replace human interpreters in critical settings, and any avatar-based system
  should be co-designed with Deaf community members.
- **European Union of the Deaf** — [EUD position on accessibility](https://www.eud.eu/) and
  associated technology statements.
- **National Association of the Deaf (USA)** for ASL-specific work.
- **British Deaf Association** for BSL-specific work.
- **Deutscher Gehörlosen-Bund** for DGS-specific work.

If any of these organisations updates their position in a way that conflicts with this system's
behavior, the conflict is treated as a P1 incident by the board: pause generation, document the
conflict in the research log, and convene a board meeting to resolve before resuming.

## 7. Funding and conflict-of-interest disclosure

Funding sources, when accepted, must be:

- Disclosed publicly in this document.
- Reviewed by the board for conflicts (e.g. funding from an organisation with a history of
  hostile relations with the Deaf community would be refused).
- Time-bounded — multi-year funding does not buy multi-year governance influence.

As of 2026-04-19, the project is unfunded and developed pro bono by Bogdan Mironov with academic
supervision from the Gómez-Bombarelli group at MIT. There are no commercial sponsors and no
conflicts to disclose.

## 8. Contact

**deaf-feedback@kozha.dev** — Deaf community feedback channel. Monitored daily. Replies within 7
business days. This address is operated by the project, not by an automated system; humans read
every message.

**research@kozha.dev** — Academic / research enquiries.

**security@kozha.dev** — Security-vulnerability reports (see [17-security.md](17-security.md)).

If the project becomes unable to monitor `deaf-feedback@kozha.dev` for any reason, the board chair
must be notified within 24 hours. If the address is unmonitored for more than 7 days, the system
must be paused (set global daily cap to `$0`) until monitoring is restored.

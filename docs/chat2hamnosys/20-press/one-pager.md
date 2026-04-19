# Kozha · chat2hamnosys

**Authoring sign-language vocabulary, with Deaf community oversight built in.**

---

## The problem

Sign-language translation tools work for the most common 3,000–5,000 words. For everything else —
new STEM vocabulary, technical terms, recently-coined community signs, names — they fall back to
fingerspelling, which is slow and not how Deaf signers communicate technical content. The lexicon
gap is the bottleneck.

Closing the gap by hand requires expertise in a phonetic notation system (HamNoSys) that almost
nobody outside a handful of academic groups has studied. The community of people who can extend a
sign-language lexicon is tiny.

## What we built

Kozha is an open-source sign-language translation pipeline. **chat2hamnosys** is its authoring
subsystem: a domain expert types a description of a sign in plain English, and the system
generates the formal phonetic representation, validates it, and renders an animated avatar
preview. The expert can click any part of the avatar that's wrong and describe the correction in
their own words.

What makes this not just another AI sign-language project: **every authored sign passes through a
two-Deaf-reviewer gate before it reaches the live library.** At least one reviewer must be a
native-Deaf signer. The gate is enforced in code in two layers, with a tamper-evident audit
chain. The Deaf reviewer board has unilateral authority to pause the system, retract any sign,
or refuse a funding source.

## Governance

- Reviewer board of native-Deaf signers per language. ≥3 reviewers per language; 24-month
  bounded terms. Hearing engineers cannot vote on linguistic correctness.
- Two-reviewer rule with native-Deaf signoff requirement. No exceptions in production.
- Tamper-evident export audit log. Every sign that reaches the public library has a
  cryptographic chain to its reviewer approvals.
- Public deletion policy. Any signer can request removal of their attributed entries within 30
  days, no questions asked.
- Aligned with WFD, WFD-WASLI, EUD, BDA, NAD, and DGB position statements.

## Status

Open source under the project's existing license. Initial deployment supports BSL, ASL, and DGS.
Three deployment targets supported (Fly.io, Railway, EC2 self-host). 50-fixture evaluation
harness with regression guards in CI. Full security hardening: injection screening, PII
hashing, rate limiting, OpenAI Moderation. Observability: Prometheus metrics, dashboard, alert
rules, runbook.

## What it is not

This is not a replacement for human interpreters. It is not a real-time translator. It does not
generate connected discourse, classifier constructions, or role shift. It is an offline
lexicographer's assistant for growing the sign library, with a Deaf-led governance gate.

## Contact

- **Deaf community feedback:** deaf-feedback@kozha.dev (monitored daily)
- **Research enquiries:** research@kozha.dev
- **Security:** security@kozha.dev
- **Code:** github.com/your-username/Kozha
- **Live demo:** kozha-translate.com

## The team

- **Bogdan Mironov** (maintainer)
- **Zhan** (Kozha core developer)
- **Advisor:** Askhat Zhumabekov
- **Academic supervision:** Gómez-Bombarelli group, MIT
- **Deaf reviewer board:** seating in progress as of Q2 2026

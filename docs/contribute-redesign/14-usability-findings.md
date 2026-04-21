# Usability sessions — protocol and findings ledger (prompt 14)

Authored: 2026-04-21.

## Status: NO SESSIONS HAVE BEEN RUN

This document is the **protocol** for running the usability
sessions that prompt 14 step 2 requires, plus the ledger where the
findings will be written as sessions happen. Today the ledger is
empty. The file exists so that whoever runs the sessions — the
maintainer, an external researcher, or a board member doing their
own diligence — picks up a script that does not need to be
re-invented, and so that the accountability to run these sessions
is visible in the repo rather than hidden in someone's inbox.

**Do not treat this document as evidence of research that has
happened.** If the findings section is empty, no one has sat with
a Deaf signer and watched them use the contribute page.

---

## Why this is a blocker, not a formality

Prompt 14 step 2 says "Do NOT skip this step." The reason is not
process hygiene. The contribute page makes assumptions about what
a contributor will know, expect, and want — and every one of those
assumptions is being made by people who are not the target user.
Five 30-minute sessions will surface friction that the team
cannot see from inside the code. The feasibility study referenced
in prompt 14 sets a minimum Deaf-community compensation floor at
15–20% of project funds; at this scale that is roughly
**$75–$150 per session**, paid within 24 hours of the session.

Skipping this step ships a flow whose core interaction has only
been tested by its implementers. That is not a contribute page for
the Deaf community — it is a contribute page the Deaf community
has not been consulted about.

---

## Recruitment

Target: **five fluent signers**, Deaf native where possible,
across the three languages with the most existing data coverage
in the Kozha corpus — BSL, ASL, DGS. Two BSL, two ASL, one DGS is
a reasonable split given the corpus; adjust if recruitment makes a
different distribution easier.

Channels, in order of preference:

1. **Partner institutions.** Gallaudet University (Washington DC),
   NTID (Rochester NY), UCL DCAL (London), Universität Hamburg
   IDGS. Cold-email the department contact with (a) a link to
   `public/governance.html`, (b) this protocol, and (c) the
   compensation offer. Do not ask the department to unpaid-advise —
   ask them to forward to interested students or staff who will be
   compensated.
2. **Deaf-CS and Deaf-tech communities.** Deaf in AI Slack, r/deaf,
   Deaf Professionals Happy Hour, DeafTEC. Post the same three
   pieces of information. Screen replies for fluency
   self-assertion and language; do not gate on institutional
   affiliation.
3. **Personal network.** If the maintainer or any contributor has
   Deaf acquaintances who would be interested, ask. Pay them at
   the same rate. Do not accept unpaid "favors" from Deaf
   participants.

Do not publish the session link beyond these channels. Do not open
recruitment on Twitter or LinkedIn. The sessions are not
publicity; they are research.

---

## Consent and compensation

A one-page consent form before the session covers:

- What the session is (a usability study of a sign-contribution
  tool).
- What will happen (a 30-minute video call; the participant is
  asked to contribute one sign; the session is not recorded by
  default).
- Data handling (notes are anonymised; quotes are used only with
  written permission; the participant can withdraw at any time
  including after the session, in which case their notes are
  destroyed).
- Compensation ($75–$150 via the participant's choice of wire,
  PayPal, or Wise; paid within 24 hours of the session; not
  contingent on completion of the task).
- Right to refuse any question or task.

The consent form is shared in the participant's preferred written
language. An interpreter is offered if the session is conducted in
English and the participant prefers to sign; pay the interpreter
at their posted professional rate.

**Compensation is paid regardless of whether the participant
completes the contribution task.** If someone gets stuck and
abandons the session after five minutes, they are still paid in
full. The research value is in the friction, not the completion.

---

## Session script

**Preparation.** The facilitator (one person, same person across
all five sessions where possible) opens a browser window in a
shared-screen call, navigates to the contribute URL, and hands
control to the participant. Do not pre-select a language; do not
open the page past the language picker.

**The ask.** One sentence:
> "I'd like you to contribute one sign you know well in
> \<LANGUAGE\>. Take your time. I won't explain the page — I'd
> like to see what makes sense and what doesn't."

Then silence.

**Observation.** The facilitator watches and takes notes. Notes
capture:

1. First three moments of friction — anywhere the participant
   pauses, looks confused, reads something twice, or looks to the
   facilitator for cues.
2. Any element the participant tries to use that does not work as
   they expect (e.g. clicking a region of the avatar that does not
   accept clicks; typing a description in a format the backend
   rejects).
3. Time from "I'd like you to contribute…" to either (a)
   confirmation view or (b) participant declares they are done /
   stuck.

The facilitator does not prompt, explain, or help unless the
participant is fully blocked and explicitly asks. If the
participant gives up, that is a finding — not a failure of the
session.

**Debrief.** After the task or after 20 minutes, whichever comes
first, the facilitator stops observing and asks:

- "What felt wrong, anywhere on the page?"
- "What did you expect that wasn't there?"
- "If you had signed this on your own, without a researcher
  watching, would you have submitted it?"
- "Would you come back to contribute another sign? Why or why
  not?"

Open-ended, no leading questions. Record the participant's
phrasing verbatim where possible.

**Compensation handoff.** Before the call ends, confirm payment
method and send the payment (or schedule it) while the participant
is still on the call. Do not let the participant leave the session
owed money.

---

## Finding severity

Each finding is logged against one of:

- **P0 — task failure.** The participant could not complete the
  contribution at all, or produced a submission that was
  demonstrably broken (wrong gloss attached to wrong sign, wrong
  language, corrupt HamNoSys they could not correct). Any P0 is a
  deploy-blocker. Fix or delay the deploy.
- **P1 — significant confusion.** The participant completed the
  task but visibly struggled for ≥30 seconds at a step, asked a
  question that the page should have pre-empted, or described the
  page as "frustrating" / "unclear" in the debrief. P1s are
  deploy-blockers unless the fix is scoped and tested before
  launch.
- **P2 — minor friction.** A one-liner of polish — wording,
  ordering, microcopy — that the participant named but did not
  block on. P2s become tickets. They do not block deploy.

The P0/P1/P2 taxonomy lives here, not in a separate tracker,
until the volume of findings justifies moving to one.

---

## Findings ledger

### Session 001 — BSL, \<date\>
Facilitator: \<name\>. Participant: \<anonymized ID, e.g. P1\>.
Duration: \<min\>. Language: BSL. Sign attempted: \<gloss\>.
Completion: \<yes | no | partial\>. Time to completion: \<min:sec\>.

*(no data — session not run)*

| # | Severity | Finding | Evidence (quote / step) | Resolution | Ticket |
|---|---|---|---|---|---|
|   |   |   |   |   |   |

### Session 002 — BSL, \<date\>

*(no data)*

### Session 003 — ASL, \<date\>

*(no data)*

### Session 004 — ASL, \<date\>

*(no data)*

### Session 005 — DGS, \<date\>

*(no data)*

---

## Aggregate findings (populated after all sessions)

### P0 — task failures

*(none reported — no sessions run)*

### P1 — significant confusion

*(none reported — no sessions run)*

### P2 — minor friction

*(none reported — no sessions run)*

### Quotes (with permission)

*(none collected)*

### Completion rate

*(0 / 0)*

### Mean time to completion (well-known sign)

*(n/a)*

---

## Decision rule (from prompt 14 step 3)

After all five sessions:

- **Every P0 is fixed** before the next deploy attempt. If a P0
  cannot be fixed within one week, delay the deploy. Do not ship a
  flow that fluent signers cannot complete.
- **Every P1 is either fixed or explicitly accepted** by the Deaf
  advisory board (once seated — see
  [14-deploy-readiness.md § Gate 7](14-deploy-readiness.md)) with
  a written rationale.
- **P2s are logged** and prioritised in the first post-launch
  month.

The recruiter records the decision here after the sessions:

> Decision: \<GO | NO-GO | DELAY-FOR-FIX\>
> Rationale: \<one paragraph\>
> Next step: \<date + owner\>

Until the recruiter writes that paragraph, the decision is
NO-GO by default.

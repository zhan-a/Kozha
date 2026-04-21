# Retrospective — the contribute-redesign series (prompt 14)

Authored: 2026-04-21. The end-of-series debrief. Three columns —
what worked, what didn't, what's next. Written from the actual
work across prompts 1–14, not a summary of what was supposed to
happen.

This file is the handoff. A future contributor picking up this
project should read this before any of the per-prompt docs.

---

## What worked

### The design-first doc cadence.
Every prompt started with a doc — principles, coverage, audit,
gaps — before any code. When the code came, it was narrow because
the scope was already constrained on paper. The prompt 13 wiring
commit (`a29c175`) is a single semantically coherent change
because the `10-backend-gaps.md` and `11-governance-gaps.md`
documents pre-drafted the boundaries.

### Separation between the AI-generation layer and the review gate.
The contribute page accepts AI-generated HamNoSys but does not
publish it. Every accepted sign sits at `draft` or
`pending_review`; only `POST /review/entries/{id}/validate` with
two native-signer approvals moves it forward. This was a
philosophical decision from prompt 1 (see
[01-design-principles.md](01-design-principles.md)) and it held
through every subsequent prompt. The contribute page is not a
publishing pipeline — it is a queue feeder.

### Honest empty states.
`public/governance-data.json` ships empty. The governance page
renders "no reviewers seated" and "no board yet" literally. The
contribute confirmation says "\<LANGUAGE\> does not yet have Deaf
reviewers assigned, so review is on hold" when the backend
returns `draft`. Nothing on the page pretends a human is there
who is not. This is the single most important property the
redesign preserved.

### The a11y pass landed real.
Prompt 12 (`8ca625d`) produced keyboard, screen-reader, and
cognitive-load findings docs that pointed to real friction, not
a checklist run. The skip-link, the `role="log"` chat, the
`aria-live` status strip, the masthead landmark decision (§ on
why it is not a `<banner>`) — all survived the follow-up
accessibility audit in CI.

### The injection defense is layered, not cosmetic.
Regex always on, classifier default on, rejection copy visible to
the contributor instead of silently stripped. The classifier has
caught ambiguous cases that the regex missed in
`test_security_injection.py`. This is not a theoretical guard —
it is running.

### The prompt-driven sequence matched how the repo wants to be
changed.
Each prompt's commit is atomic and reviewable in isolation. The
feature is not a forty-file rewrite; it is fourteen commits, each
with its own gap document.

---

## What didn't

### No deaf humans have touched the contribute page.
The flow has been designed for, documented for, and tested around
Deaf contributors — but no Deaf signer has sat in front of the
page and used it. This is the work that is missing at the point
of this retrospective. Every assumption the page makes about what
is obvious, what is insulting, what is over-explaining, and what
is under-explaining is untested. The usability protocol exists at
[14-usability-findings.md](14-usability-findings.md) and has not
been executed.

### No reviewers. No board.
Six of six go/no-go criteria at
[14-deploy-readiness.md § Go / no-go decision](14-deploy-readiness.md)
fail on this. The system is code-complete for accepting
submissions and routing them to a reviewer — and there is no
reviewer. The contribute page is paused at the end of this series
not because of a code bug but because the project has not yet
done the human work the code was built to support.

### The governance email is a published promise no one has signed
up to keep.
`deaf-feedback@kozha.dev` appears on the governance page with a
24-hour quarantine SLA. No human reads that inbox. This is the
single clearest "check the doc against reality" failure in the
series. Documented in
[11-governance-gaps.md § Gap 5](11-governance-gaps.md) at the
time the page shipped, still not closed.

### `governance-data.json` is hand-shipped.
The plan was a backend job that writes the file from the
reviewers table (`11-governance-gaps.md § Gap 3`). That job has
not landed. A reviewer created through the admin CLI today does
not appear on the page until someone manually edits the JSON.
When the first reviewer is seated, this will be the first thing
someone trips over.

### Footer links 404.
`/privacy.html` and `/contact.html` are referenced in the footer
of `contribute.html` and `governance.html` and do not exist.
Documented as
[11-governance-gaps.md § Gap 8](11-governance-gaps.md). A small
polish item that became visible every single time someone clicked
it. Has not been fixed.

### The `contribute-staging.kozha-translate.com` subdomain does
not exist.
Prompt 14 step 4 specifies it. The subdomain has not been
registered, no DNS record has been added, and no pilot deploy has
been run. The pilot protocol at
[14-pilot-metrics.md](14-pilot-metrics.md) is a schema against a
host that does not exist.

### The DGS/BSL/ASL coverage numbers in the corpus were never
re-checked.
Prompt 14 step 2 says "the languages with the most existing data
coverage." The recruitment document cites BSL/ASL/DGS as the
candidates. The actual corpus breakdown lives in
`docs/contribute-redesign/00-language-coverage.md` and was
written in prompt 0; the recruitment doc did not re-verify it
before writing the language split.

### No load testing against the real chat2hamnosys backend.
The rate limit gate (`Gate 4` in
[13-production-ready.md](13-production-ready.md)) was verified by
unit test. The load-test script at `scripts/loadtest.py` was
never actually run against the integrated stack. This is
acceptable for a paused launch; it is not acceptable once the
page opens.

### AI-coding session ↔ real launch mismatch.
This is the meta-finding. The prompt sequence asked an AI to
close the loop on a project whose remaining work is fundamentally
human-in-the-loop: recruiting Deaf participants, paying them,
seating a board, reading a mailbox, running a fire drill. The
code shipped. The social and operational work that makes the code
honest has not. A future maintainer should not interpret
"prompt 14 commit merged" as "project launched." Those are
different things.

---

## What's next

### In order of blocker severity.

1. **Seat one Deaf advisory board member.** Criterion 3 of the
   go/no-go gate. Without the board, the reviewer-approval
   authority does not exist. Reach out to a Deaf signer the
   project already has a relationship with, ask them to serve,
   pay them a board stipend (this was not included in any
   budget document and needs a line item before the invite goes
   out), and publish their name on `governance-data.json`
   `board[]` with their explicit consent.

2. **Recruit two native-signer reviewers for one language.**
   Narrow scope — pick BSL, ASL, or DGS, not all three.
   Reviewer compensation is the same floor as the usability
   participants ($75–$150 per review is reasonable given a
   10–15 minute review for a well-formed submission; adjust up
   for regions with higher cost of living).

3. **Run the five usability sessions** per
   [14-usability-findings.md](14-usability-findings.md). Pay
   participants. Fix every P0 and P1. Log P2s.

4. **Deploy staging.** Register `contribute-staging.kozha-translate.com`
   (or similar), run the pre-deploy smoke checks at
   [13-production-ready.md § Pre-deploy smoke checks](13-production-ready.md),
   and open the pilot to the usability participants plus
   institution-recruited testers.

5. **Run the quarantine fire drill.** Criterion 6. One
   reviewer publishes a flawed sign, another flags it; the
   quarantine endpoint triggers within 24 hours; write the
   postmortem.

6. **Assign the governance email.** Criterion 5. A specific
   human commits, in writing, to reading
   `deaf-feedback@kozha.dev` daily. Rotation schedule
   documented.

7. **Land the governance-data.json generator.** Closes Gap 3
   of `11-governance-gaps.md`. Script runs on reviewer
   create/update/delete and publishes the JSON. Removes the
   manual-edit footgun.

8. **Fix the footer 404s.** Ship `/privacy.html` and
   `/contact.html`. Five minutes of work; has been a bug for a
   while.

9. **Launch one language, narrowly.** When criteria 1–6 are met
   for that one language, flip `contributions_paused` to false
   in `governance-data.json`, publish the launch blog post at
   [14-launch-blog-draft.md](14-launch-blog-draft.md) (with
   placeholders replaced by real named consenting advisors),
   and share with Deaf-CS communities **before** sharing
   widely.

10. **Run the monthly status update.** Committed to in prompt 14
    step 7. First update is due 30 days after launch; write it
    even if the content is "we shipped, reviewed N signs,
    published M." Skipping the first one sets the tone that the
    rest are optional.

### Not in scope for this retro

- Extending to more languages. First launch one, then decide.
- Adding a contributor-facing form for raising concerns. The
  governance page intentionally refuses this (email-only; see
  [11-governance-gaps.md § Out of scope](11-governance-gaps.md)).
- Building a ticketing portal for reviewers. A text field
  inside the review console is the current interface; leave it
  until there is concrete evidence reviewers want more.
- Any version of "contributor leaderboard" or contribution
  counters. The contribute page is not a game. The
  contributor's dashboard at `/contribute/me` is a status
  aggregator, not a score.

---

## Handoff note

If you are reading this file without having read any of the
earlier prompts: the prompt sequence is
`prompts/prompt-contrib-1.md` through
`prompts/prompt-contrib-14.md`. Each prompt's docs live under
`docs/contribute-redesign/` and follow a numeric prefix. The
commit log has one atomic commit per prompt starting at
`905b218`. The place to start your first day is
[00-current-state.md](00-current-state.md) and
[01-design-principles.md](01-design-principles.md). The place to
start your first week is the "What's next" list above.

# Go / no-go decision — prompt 14 step 5

Authored: 2026-04-21. This is the one-page decision record that
[14-deploy-readiness.md](14-deploy-readiness.md) references. It
exists separately so someone checking the project's launch
posture can read it in under a minute without having to parse the
full readiness matrix.

---

## Decision

**NO-GO.**

The contribute page at `public/contribute.html` is set to paused.
The `governance-data.json` file carries
`"contributions_paused": true` until the criteria below are met.

## Why

Six criteria from
[prompts/prompt-contrib-14.md](../../prompts/prompt-contrib-14.md)
step 5. All six fail as of 2026-04-21.

- **≥80% of invited usability participants completed a full
  contribution without dev intervention.** No participants
  recruited; no sessions run. See
  [14-usability-findings.md](14-usability-findings.md).
- **Mean time-to-completion ≤ 5 minutes for a well-known sign.**
  No timing data.
- **At least one Deaf advisory board member seated and has
  approved the flow.** `public/governance-data.json` `board[]`
  is empty. Documented as
  [11-governance-gaps.md § Gap 2](11-governance-gaps.md).
- **At least two native-signer reviewers exist for the launch
  language(s).** `public/governance-data.json` `reviewers[]` is
  empty across all eight languages.
- **Governance email monitored daily with <24h response SLA.**
  No human has signed up to read `deaf-feedback@kozha.dev`.
  Documented as
  [11-governance-gaps.md § Gap 5](11-governance-gaps.md).
- **Quarantine fire-drilled within 24 hours and postmortem
  documented.** No drill has been performed.

This is not a close call. It is the six criteria plainly not
satisfied.

## What the paused state does

- `governance-data.json` carries `contributions_paused: true`.
- On load, `public/contribute.js` reads the flag and hides the
  language picker, authoring form, and submission panel.
- The page shows a banner with the pause message and a
  `mailto:` to the governance email. The wording is the prompt
  14 step 10 wording: "Contributions are paused while we seat
  reviewers. Email <governance> to be notified."
- The binary is preserved: the page is either accepting
  submissions that get real reviews, or it is not accepting. No
  "coming soon," no beta, no "limited preview."

## What the paused state does *not* do

- It does not prevent `/contribute/me` from rendering. If a
  contributor has session tokens in sessionStorage from a prior
  deployment, the dashboard still shows their status. The
  paused flag affects the submission entry point, not the read
  path for prior work.
- It does not tear down the backend. The API remains up, the
  review console remains functional, any reviewers who are
  onboarded continue to work. The public *entry door* is
  closed; the internal machinery is unchanged.
- It does not change `/governance.html`. The governance page is
  a standing public contract and is correct in either state.

## When NO-GO flips to GO

When — and only when — every criterion in the "why" section
above flips from failing to passing. The path is in
[14-retro.md § What's next](14-retro.md). No shortcut.

When that happens, the steps to re-open:

1. Set `contributions_paused: false` in
   `public/governance-data.json`.
2. Populate `board[]` with the seated advisor(s)' names and
   consented affiliation.
3. Populate `reviewers[]` from the reviewer admin CLI output (or
   the generator job from
   [11-governance-gaps.md § Gap 3](11-governance-gaps.md) if it
   has landed by then).
4. Publish the launch blog post drafted at
   [14-launch-blog-draft.md](14-launch-blog-draft.md),
   replacing placeholders with real names and languages.
5. Share the post in Deaf-CS communities first. Wait at least 48
   hours before sharing it widely.

These five steps are the operational side of the flip. The
decision itself is the six criteria.

## Record

| Date | Decision | Set by | Next review |
|---|---|---|---|
| 2026-04-21 | NO-GO | contribute-redesign series, prompt 14 | when any of the six criteria changes state |

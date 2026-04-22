# Polish 14 — Governance and ethics check

Against the four prompt-14 criteria. Each bullet is a concrete
citation from the current site, not a summary.

## 1. "Nothing about us without us" standard

**Status: observed in policy, not yet in seated-reviewer practice.**

The principle is enforced structurally: no sign is shipped to the
public library without review by Deaf signers of the target
language. From `public/governance.html`:

> Every submission is reviewed by Deaf signers of the language it
> was submitted in. A sign is validated only after two reviewers of
> that language approve it. For each sign language Bridgn supports,
> at least one of the two reviewers must be a Deaf native signer of
> that language.

From the same page, on the advisory board:

> We are in the process of seating a Deaf advisory board. Until
> then, no signs are being exported to the Kozha library. Drafts
> continue to be accepted and queued; they will be reviewed once
> the board is seated and has approved the first reviewers.

The policy is "Deaf-governed at the point of shipping". Currently
zero Deaf reviewers are seated, and the page says so in plain
language rather than dressing up aspiration as active governance.

**Action taken this gate:** the landing-page "Deaf-governed
Contributions" feature card previously read *"Every new sign is
reviewed by two Deaf native signers before it ships."* This was
technically true of the **policy** but misrepresented the **active
state** (zero reviewers seated). Edited to:

> Every new sign is reviewed by two Deaf native signers before it
> ships. The Deaf advisory board is currently being seated; until
> it is, community drafts are queued rather than published.

## 2. The progress dashboard does not shame contributors

**Status: pass.**

The progress section notes and H2s read strictly informational,
not blame-framed:

- *"Aggregate only — no contributor identities."* (on recent
  validations)
- *"If you're a native signer, a single contribution closes the
  gap."* (on help-wanted) — framed as an invitation, not a demand.
- *"If something cannot be computed honestly, it is marked — rather
  than zero."* (on missing data) — acknowledges uncertainty instead
  of under-reporting.

No leaderboards, no named call-outs, no "X is falling behind" copy.
Contributor-level identities never appear on the dashboard.

## 3. Credits does not claim community contributions as project achievements

**Status: pass.**

Every corpus entry on `/credits` names the **upstream maintainer by
name** (Zerrouki, Khan, Esselink, Ablog, KurdishBLARK, Raian Rido,
etc.) and describes their repository as the authoritative source.
The Bridgn project's own claim is the integration work, not the
sign data itself. Sample:

- *"Bangla Sign Language data is drawn from Devr Arif Khan's
  bdsl-3d-animation GitLab repository."* (not "we produced").
- *"Algerian Sign Language data comes from Taha Zerrouki's
  algerianSignLanguage-avatar GitHub repository."*

The reviewer-provenance line per corpus also distinguishes
"native-signer review at corpus level" (upstream research-programme
work, e.g. DictaSign) from Bridgn-level review (pending).

Funding and compensation sections also disclaim:

- *"No external funding to date."*
- *"A compensation policy is not yet in place... Until that is
  done, no reviewers are being asked to work."*

## 4. No unimplemented governance promise is left unmarked

**Status: pass after one edit (criterion 1 above).**

Inventory of governance claims and their implementation state:

| Claim | Source | State | Marked as incomplete? |
| ----- | ------ | ----- | --------------------- |
| Every sign reviewed by two Deaf signers before shipping | `index.html` feature card | **Policy only** — zero reviewers seated | **Yes** (after this gate's edit) |
| Deaf advisory board arbitrates disputes | `governance.html` | **Policy only** — zero candidates confirmed | Yes — "being seated. Current status: zero candidates confirmed." |
| Reviewers listed by name, affiliation, languages | `governance.html` | Seed loader emits empty array → page shows empty state | Yes — loader falls back to "No reviewers seated yet" text |
| Compensation policy for reviewers | `credits.html` §Compensation | **Not yet in place** | Yes — "A compensation policy is not yet in place." |
| Funding transparency | `credits.html` §Funding | No external funding yet | Yes — "No external funding to date." |
| Coverage growth chart | `/progress` | Series has 1 datapoint (first snapshot) | Implicit — chart renders what it has |
| LSE (Spanish) and RSL (Russian) in language picker | `index.html` hero | No database | Yes — option labeled "— coming soon" + status dot reports "No sign database available" on selection |
| `bgHamNoSysUnicode.ttf` font loading | `contribute.css` | @font-face rule exists, binary not committed | **Partial** — documented in `/credits` ("the license file documents the provenance") and in this gate's changelog (`14-changelog.md`, "Honest limitations") |

All governance promises either match the active state or are marked
as forthcoming. The only item below the bar pre-edit was the
landing-page card; that is now fixed.

## Cross-cutting principle: "say nothing stronger than the data"

The project-wide editorial convention observed across `/progress`,
`/credits`, and `/governance`:

- Zero values are shown as **zero** or **—**, not rounded up.
- Upstream corpora without licenses are called "License unclear",
  not "open".
- Reviewer counts are real (zero) rather than aspirational.
- Funding state is "none" rather than "self-funded" (same
  substantive truth, but less glossy).

This editorial discipline is the main defence against silently
over-promising governance. It holds across every page I inspected.

## Conclusion

Governance and ethics check **passes** with one in-gate edit
(criterion 1, landing feature card). Everything else was already
honest about the gap between policy and seated practice.

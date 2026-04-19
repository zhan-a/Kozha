# Demo script

Three timed walkthroughs for distinct audiences, each with a props checklist and a failure-mode
playbook. The 90-second version is for noisy expo floors; the 5-minute version for academic
audiences who want the research framing; the 15-minute version for funders or detailed reviewers
who want the governance model and the roadmap.

All three versions share the same ground truth: **the demo session uses ELECTRON** (a STEM term
that does not exist in any standard sign-language dictionary, illustrating the OOV gap). The fixture
is recorded at `backend/chat2hamnosys/examples/electron.json` and is the reference for what the
audience sees.

---

## A. Props checklist (all versions)

**Hardware:**
- Laptop with Wi-Fi + ethernet adapter (don't trust expo Wi-Fi).
- USB-C / Thunderbolt → HDMI adapter, USB-C → DisplayPort adapter, Lightning → HDMI for iPad.
- Wired mouse (presenter-clicker is too imprecise for the click-to-correct interaction).
- Backup laptop running the same demo (one machine, one OS update, one bricked demo).

**Software pre-flight (run 30 minutes before):**
- `docker compose up` succeeded; `curl http://localhost:8000/api/chat2hamnosys/health` returns 200.
- Authoring UI loads at `http://localhost:8000/chat2hamnosys/` and the avatar panel renders.
- The five example sessions replay end-to-end: `python -m examples.replay --example all`.
- Pre-recorded video of the same demo is open in a second tab, paused at 0:00.
- Browser zoom set to 125% so the avatar is visible from the back of a 6m booth.

**Backups:**
- 30-second screen recording of `electron` end-to-end at `docs/chat2hamnosys/20-press/demo-electron-30s.mp4`.
- Three printed screenshots (initial description, mid-correction, post-validation) on the booth
  table. Audience can see the flow even if the laptop dies.
- One-page handout (the press one-pager from `20-press/`).

---

## B. 90-second version (RESNA, Abilities Expo, conference posters)

**Audience:** mixed engineers, clinicians, accessibility-tech buyers. Attention span: 90 seconds.

**Total runtime:** 1:30. Practice in a mirror until you can deliver it without stumbling.

| Time | What you say | What you do |
|---|---|---|
| **0:00–0:10** | "Kozha is a sign-language pipeline. It works for the 5,000 most common signs, but if you ask it to sign a STEM word like ELECTRON, it has no entry, and just falls back to fingerspelling. That's our gap." | Open the Kozha translation page in a separate tab. Type "electron" → show only fingerspelling. |
| **0:10–0:30** | "We built an LLM-mediated authoring tool. Watch — I describe a sign in plain English." | Open the chat2hamnosys authoring UI. Pick BSL. Paste this exact text: *"ELECTRON is signed with a small round-O handshape, palm down, in neutral signing space. The hand makes a small circle to suggest orbiting motion."* |
| **0:30–0:50** | "It parses what I said into eight phonological slots — handshape, palm direction, location, movement — and notices it doesn't know how big 'small' should be. It asks one question." | Click "Describe". The system asks: *"How small is the circular movement — fingertip-sized, palm-sized, or larger?"* Click "fingertip-sized". |
| **0:50–1:10** | "Now it generates HamNoSys, validates against the grammar, and renders the avatar. If it's wrong, I click the part of the body that's wrong." | Click "Generate". Avatar plays the sign. Click on the hand region in the avatar panel; type *"the handshape should be the round-O, not the F"*. Click "Apply". Avatar re-plays. |
| **1:10–1:30** | "Now it goes into a queue. Two Deaf reviewers — at least one native — have to approve it before it ever reaches the live library. That's the governance gate. Thank you." | Click "Accept". Show the entry appearing in the reviewer console (open in a third tab). Stop. |

**Failure-mode playbook:**

| If… | Do this |
|---|---|
| Wi-Fi fails before 0:10 | Switch to ethernet. If still failing, switch to the pre-recorded video tab and narrate the same script over it. |
| The clarification question is different from "How small…" | Adapt the line to the actual question. Don't argue with the system live. |
| The renderer hangs on "Generate" | Click the cached preview thumbnail (preview cache always has a recent ELECTRON entry). If empty, switch to backup video at `20-press/demo-electron-30s.mp4`, time to 0:50. |
| Avatar fails to load (WebGL error) | Show the SiGML XML output instead — "this is what the sign looks like in our intermediate format" — and switch to backup video. |
| The audience asks a question mid-demo | Finish the script, *then* answer. Don't break the 90-second flow. |

---

## C. 5-minute version (academic audience, MIT lab meeting, computational linguistics seminar)

**Audience:** researchers in NLP, ML, computational linguistics, accessibility. They want the
research contribution framing, the analogy that makes the work memorable, and the architecture
overview.

**Total runtime:** 5:00.

| Time | What you say | What you do |
|---|---|---|
| **0:00–0:30** | "Sign-language synthesis pipelines have a long-tail vocabulary problem. The pre-built lexicon covers the head; everything in the tail — STEM terms, names, recently-coined community signs — falls back to fingerspelling, which is slow and not how Deaf signers communicate technical content." | Title slide: "Authoring out-of-vocabulary signs with LLM mediation and Deaf-reviewer validation." |
| **0:30–1:00** | "Our analogy is SMILES strings in chemistry. SMILES is a structured notation for molecules; you can write 'CCO' and a renderer draws ethanol. HamNoSys is the SMILES of sign language — a phonetic notation system, 231 codepoints, formally specified, renderable. The challenge is that almost no human writes HamNoSys directly." | Slide: SMILES `CCO` → ethanol structure | HamNoSys `îE001îE020...` → sign animation. |
| **1:00–2:00** | "Our pipeline: prose description → 8 phonological slots → at most three clarifying questions → HamNoSys generation with deterministic vocab plus LLM fallback → grammar validator → SiGML rendering → click-targeted correction → Deaf-reviewer queue." | Architecture slide (component diagram from `20-architecture.md`). |
| **2:00–3:30** | Run the live demo from the 90-second script (truncated): describe ELECTRON, answer the clarification, see the avatar render, click-correct one slot, accept. | Live demo. |
| **3:30–4:00** | "Three things that make this novel. One: the parser-then-clarifier-then-generator decomposition keeps the LLM bounded. The LLM never invents a sign; it only fills slots in a phonological grammar that we wrote. Two: every output passes the same Lark validator a human-authored sign would pass, with a repair loop that feeds the validator's error back to the LLM. Three: the export gate is implemented twice — once at the review-policy layer and once at the storage layer — because the cost of a bad sign reaching the public library is high." | Slide: three contributions, one bullet each. |
| **4:00–4:30** | "Evaluation: 50 fixtures across BSL, ASL, DGS. Metrics at four layers — parser per-field accuracy, generator symbol F1, end-to-end exact match, and cost. Three ablations: no-clarification, no-validator-feedback, no-deterministic-vocab. We also bridge to a Deaf-reviewer human-evaluation form using the Huenerfauth grammaticality / naturalness / comprehensibility scales." | Eval slide: numbers from `20-research-log.md` (current as of last run). |
| **4:30–5:00** | "Limitations and what's next. Citation forms only — no connected discourse, no classifiers, no role shift. Reviewer governance is bootstrap mode today; we need a Deaf co-PI to recruit the board. The bigger research question is whether this decomposition generalises beyond lexical authoring to phrasal sign synthesis. Thanks. Code's open source. Email's on the slide." | Limitations + roadmap slide. Contact slide. |

**Failure-mode playbook (additions to 90s playbook):**

| If… | Do this |
|---|---|
| Audience asks "have you compared to <other system>" | Acknowledge: SignWriting tools, Vcom3D's signsmith, the HamNoSys MS Word add-in. Note the difference: ours is *interactive* (clarifies and corrects) and *governed* (review gate); the others are notation editors for trained users. |
| Audience asks about ASL bias / Deaf community concerns | Pivot to the ethics statement. "We do not export anything without two Deaf reviewers, one native. The governance section of the documentation has the full position." |
| Audience asks about classifiers | "Out of scope. Citation forms only. The architecture doc explicitly disclaims that." |

---

## D. 15-minute version (funder, detailed walkthrough, Deaf community board interview)

**Audience:** funder due-diligence call, candidate Deaf co-PI, advisory board candidate. They want
to know whether this is a serious, governable project, not just a demo.

**Total runtime:** 15:00.

### Outline

| Block | Duration | Topic |
|---|---|---|
| 1 | 0:00–1:30 | Problem statement + motivation. Same as the 5-minute opening, expanded. |
| 2 | 1:30–4:30 | Live demo (ELECTRON + one more, e.g. CATALYST or DERIVATIVE). |
| 3 | 4:30–6:30 | Architecture overview using `20-architecture.md` diagrams. |
| 4 | 6:30–9:00 | **Governance model** — the part funders and Deaf candidates care about most. Walk through the two-reviewer rule, the native-Deaf requirement, the audit chain, the refusal mechanism, the deletion policy. Cite the ethics statement directly. |
| 5 | 9:00–11:00 | Evaluation results from the research log. Ablation deltas. Honest limitations: no Deaf-native verification on the BSL/ASL gold set; bootstrap reviewer count below target; OOV definitions vary by community. |
| 6 | 11:00–13:00 | Roadmap. Twelve-month plan. Top three risks and the mitigations. |
| 7 | 13:00–14:30 | Funding ask (if applicable) — what the money is for, what the burn rate looks like, what milestones it buys. |
| 8 | 14:30–15:00 | Q&A start time. Hand off. |

### Block 4 talking points (governance — the high-stakes block)

The funder or Deaf interviewer should leave the meeting believing three things:

1. **The governance is structural, not advisory.** The board has unilateral authority. There is
   no "engineering override". The pause-the-system mechanism is one CLI command and one
   environment variable.
2. **Hearing engineers are accountable, not in charge.** Bogdan is the maintainer; he does not
   vote on linguistic correctness. The board does. This is documented in
   [20-ethics.md § 1](20-ethics.md).
3. **The audit chain is tamper-evident, not just a log.** Every export writes a SHA-256 hash
   chained to the previous row. A forensic reviewer can detect any post-hoc edit. This is what
   makes "we exported X on Y date with reviewer Z's approval" verifiable a year later.

### Block 6 talking points (roadmap)

Three milestones in order:

- **Milestone 1 (next 3 months):** Deaf co-PI recruited, three native-Deaf reviewers per language
  active, bootstrap mode disabled. This is a hiring milestone, not a code milestone.
- **Milestone 2 (next 6 months):** First 100 signs validated and exported in BSL, the most mature
  language. Public release of the validated lexicon as a dataset under a Deaf-community-approved
  license.
- **Milestone 3 (next 12 months):** Honest evaluation against a Deaf-author-produced gold set
  (not the LLM-rephrased one we use today). This is the threshold for any peer-reviewed
  publication.

### Top three risks and mitigations

1. **Risk:** Reviewer fatigue or board attrition. **Mitigation:** time-bounded terms, per-session
   fatigue meter (already implemented; `CHAT2HAMNOSYS_REVIEW_FATIGUE_THRESHOLD=25` defaults to a
   warn at 25 actions per session), recruitment pipeline with a backup reviewer per language.
2. **Risk:** LLM provider cost spike or API discontinuation. **Mitigation:** budget caps at
   per-session, per-IP, and global daily levels; fallback model in `llm/client.py`; the
   architecture is provider-agnostic (one swap point in `llm/client.py`).
3. **Risk:** Cultural misuse — someone authors signs for a community they don't belong to.
   **Mitigation:** the regional-match policy
   (`CHAT2HAMNOSYS_REVIEW_REQUIRE_REGION_MATCH=true`), the deletion policy, the published
   ethics statement, and the public deaf-feedback@kozha.dev channel.

**Failure-mode playbook (additions):**

| If… | Do this |
|---|---|
| The funder says "we want to skip the reviewer gate to ship faster" | Decline. Cite the ethics statement. If the funder insists, walk away from the funding. |
| The Deaf interviewer raises a concern not addressed in the ethics statement | Take notes verbatim. Commit to a written response within 7 days. Do not improvise governance changes live. |
| Someone asks "could this replace interpreters" | The answer is **no** and it's documented as no in three separate places ([README.md](../../README.md), [20-ethics.md](20-ethics.md), this script). Quote those documents. |

---

## E. Demo script style notes

- **Don't apologise for limitations during the demo.** Acknowledge once, move on. "Citation forms
  only — see the architecture doc for the boundaries" is enough.
- **Use the audience's language.** Hearing engineers want to hear about validators and metrics.
  Deaf community members want to hear about governance and refusal mechanisms. Funders want to
  hear about milestones and burn rate. Memorise three different framings of the same demo.
- **Speak slowly during the live demo.** The avatar takes 3–8 seconds to render. Don't fill the
  silence with chatter; let the audience watch.
- **Have a quiet exit ready.** "Thanks. Email's on the slide." Don't trail off into Q&A mid-demo;
  set up Q&A as its own block at the end.

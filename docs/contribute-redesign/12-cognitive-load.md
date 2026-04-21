# Cognitive-load audit — prompt 12

A deliberate read of the contribute flow against a small cognitive-accessibility checklist, meant to surface places where we're asking the contributor to hold too much in working memory, parse dense copy, or choose under ambiguity. Not a WCAG section in itself — but items touched by WCAG 2.2 AA criteria 3.3.7 (Redundant Entry), 3.3.8 (Accessible Authentication), and the aspirational 2.2 additions around predictable UI.

The audit was done by walking the flow as if I were a contributor who had never seen it before, then again as one who had abandoned partway through a draft and come back 48 hours later. Below is what I found — both items addressed in this pass and items left as deliberate trade-offs.

## What's been designed for reduced load

### One primary action per screen
The contribute page at any given moment has one obvious next step:

- Empty state → pick a language (nothing else is visible).
- After language → describe the sign (gloss + description inputs, one submit).
- Mid-session → answer the clarification *or* submit as-is (two choices, both spelled out).
- Preview → play / submit / correct (three choices, with correction surfaced via click-to-target, not an extra form).
- Submission → copy the link (one primary action in the confirmation view).

Measured against earlier drafts of this flow, which had four secondary CTAs competing on the same screen, this is deliberate simplification. The "Help" entry point and "Open examples" shortcuts were removed during prompt 8; the rationale is in `08-removed.md`.

### Persistent state is shown, not hidden
The masthead and context strip make three things visible without hover or click: which language is active, what the current session state is, and the 8-char session ID for sharing. A returning contributor doesn't have to interrogate the page to reconstruct where they left off.

### Error recovery is always one button away
- Chat generation fails → "Submit as-is" appears inline.
- Gloss / description validation fails → input stays focused, error reads below the field.
- Submit fails → inline error with the HTTP detail, submit button re-enables.
- Resume token fails → the token input stays mounted with a specific error.

No dialog that asks the user to "try again later" in the abstract; every error path gives a next action.

### Copy is short, no jargon
The notation panel intentionally does not explain HamNoSys or SiGML in-page. Users who care can follow the footer link to docs; users who just need to submit are never asked to parse the notation themselves (the reviewer does that). The phonological breakdown shows Handshape / Orientation / Location / Movement as four simple labels rather than introducing "phonological" or "parameter" vocabulary.

The confirmation body is two sentences: what's submitted, what happens next. No checklist of things the user should have done; no reassurance template.

## Trade-offs left in place (with reasons)

### The word "gloss"
"Gloss" is linguistics jargon that casual contributors may not recognise. We use it anyway because:
- The label always reads `Gloss — the English word or short phrase for this sign.` — the definition ships with the label, not as a separate tooltip.
- Every reviewer will use the term; keeping contributor- and reviewer-facing copy aligned helps when reviewer comments reference "the gloss you entered".
- Replacing "Gloss" with "English equivalent" or "name" makes sentences on the status page (`Submitted: ELECTRON in BSL`) less precise.

### The submission checklist has six items
Six seems like a lot for a single-action screen. But:
- Only two (`Sign generated and valid` and implicit `Gloss set` / `Language set`) are actual gates on submission.
- The other four are *signals* to the contributor that their draft is progressing — showing "Deaf native signer self-identification" and "At least one correction applied" lets a thorough contributor see their thoroughness reflected back, without blocking hurried ones.
- Collapsing the list into a single "Ready to submit" pill would hide the progress and make it harder for a returning contributor to see what's left.

If user research shows the list intimidates rather than clarifies, we can collapse the optional rows into an `<details>`; noting here that the current shape is a deliberate "visible but not-gating" choice.

### Two notation tabs (HamNoSys + SiGML)
A contributor doesn't need SiGML for anything. We keep the tab because:
- A reviewer (sometimes the contributor's own double-check) may want to paste SiGML into an external tool.
- Having it default-hidden behind a tab keeps it out of the way.
- Removing it would force anyone who does want it to screenshot the HamNoSys and re-derive the XML.

### The resume-token UX
Pasting a token is not a fun task, but we keep it because:
- The alternative (account login) is heavier and out of scope for a no-account contribution flow.
- The stored token in `sessionStorage` means a same-browser same-tab user never sees the prompt; it only appears when someone opens a shared link from another device.
- Error messages are specific: mismatch vs. not-found vs. network failure, so the user isn't left debugging the unknown.

## Items explicitly not solved here

- **Typeface / reading level**: Copy is written at roughly a 10th-grade reading level. Simplifying below that would require plain-language review with d/Deaf contributors who read English as a second language; that is in scope for a later prompt (15: bilingual-signer onboarding materials) but not this one.
- **Dyslexia / ADHD-friendly formatting**: No dyslexic-friendly font toggle or extra line-height option. The base line-height (1.6) is on the generous side of conventional for body text; font stack is system UI. Would add toggles if user research shows demand.
- **Progress over multiple sessions**: A contributor who submits three drafts in a week sees no aggregate "3 signs submitted" summary — that would require account state we don't have. For now, each draft stands alone.

## Where to direct future work

In order of what would most reduce cognitive load if a contributor reports friction:

1. A short ("how this works in 4 bullets") strip above the language picker for first-time contributors. Currently the onboarding is spread across the picker prompt, masthead subtitle, and the context-strip labels — a first-timer sometimes has to read all three to understand what they're signing up for.
2. A "what counts as a good description" inline example once the description field receives focus — currently the placeholder is all they see.
3. A confirmation-screen summary of the clarification turns — useful for the contributor to see what the system picked up before the reviewer sees it. Currently the chat log is not preserved on the status page.

These are not commitments. They are what I'd investigate first if a contributor said "I got confused at step N".

# Copy inventory — every contributor-facing string

All strings that a contributor (not a reviewer, not an admin) can
read on the live site today. Verbatim. Paths are absolute line
references on `main @ 95c9700`. Copy inside `<script>` blocks is
included only when it surfaces in the UI (e.g. status messages
inserted by JS).

---

## 1. `public/index.html` — translator landing (contribute-relevant sections only)

### Nav

- `index.html:750` — `<li><a href="contribute.html">Contribute</a></li>`
- `index.html:753` — `<a href="contribute.html" class="btn-ghost">Volunteer</a>`

### "Open Source" feature card

- `index.html:928` — feature title: "Open Source"
- `index.html:929` — feature body:
  > "The full codebase, sign database, and concept maps are on
  > GitHub. Anyone can add signs, fix translations, or extend the
  > pipeline."

### "Contribute to the project." banner

- `index.html:961` — h2: "Contribute to the project."
- `index.html:962` — body:
  > "Bridgn's sign database has limited vocabulary and needs help.
  > If you know sign language, work in accessibility, or can code,
  > you can add signs, improve translations, or fix bugs."
- `index.html:965` — CTA button: "Get involved"

### Footer

- `index.html:975` — `<li><a href="contribute.html">Contribute</a></li>`

---

## 2. `public/app.html` — translator app (contribute-relevant only)

- `app.html:604` — top-nav link:
  `<a href="contribute.html" class="nav-link-sm">Contribute</a>`

---

## 3. `public/contribute.html` — the contribution landing

### Page `<head>`

- `contribute.html:6` — `<title>`: "Bridgn — Contribute a Sign"
- `contribute.html:7` — meta description:
  > "Help build an open-source sign language dictionary. Register in
  > under a minute and author a sign with the chat2hamnosys
  > pipeline."
- `contribute.html:8` — `og:title`: "Contribute to Bridgn"
- `contribute.html:9` — `og:description`:
  > "Register, describe a sign in plain words, preview the generated
  > animation, and submit it for review. Contribute to an open
  > sign-language dictionary."

### Nav (lines 304-313)

- `contribute.html:306` — logo text: "Bridgn" (with `n` styled as
  accent)
- `contribute.html:308-310` — nav links: "Features", "How it works",
  "Contribute"
- `contribute.html:312` — button: "Open Translator →"

### Hero (lines 315-334)

- `contribute.html:318` — eyebrow: "Open-source · Community-built"
- `contribute.html:320` — h1: "Help grow the / sign dictionary."
  (with "sign dictionary." styled as accented italic)
- `contribute.html:321-326` — paragraph:
  > "Every sign on Bridgn was authored by someone like you. Describe
  > a sign in plain English, preview the generated animation, and
  > our Deaf reviewers validate it before it ships to everyone using
  > the translator."
- `contribute.html:328` — CTA: "Start contributing →"
- `contribute.html:331` — ghost button: "View on GitHub"

### "Why your work matters" (lines 337-367)

- `contribute.html:340` — eyebrow: "Why your work matters"
- `contribute.html:341` — h2: "Small effort, / real reach."
- `contribute.html:348` — card 1 title: "Signs reach real users"
- `contribute.html:349` — card 1 body:
  > "Approved signs land in the live translator at
  > kozha-translate.com. The next person who types your word sees
  > your sign animated."
- `contribute.html:355` — card 2 title: "Deaf-led review"
- `contribute.html:356` — card 2 body:
  > "Every submission goes to two reviewers, with at least one Deaf
  > native signer. Signs only publish after both approve — your work
  > is never rushed out alone."
- `contribute.html:362` — card 3 title: "Under five minutes"
- `contribute.html:363` — card 3 body:
  > "Describe the sign in plain words ("flat hand near the temple,
  > moves to the chest"). Our pipeline asks clarifying questions and
  > shows you a preview to correct before you submit."

### "Four steps to a published sign." (lines 370-400)

- `contribute.html:373` — eyebrow: "What to expect"
- `contribute.html:374` — h2: "Four steps to a / published sign."
- `contribute.html:375` — sub:
  > "The pipeline runs on OpenAI's models, and Bridgn covers the API
  > costs so you can contribute for free."
- `contribute.html:380` — step 1 title: "Register"
- `contribute.html:381` — step 1 body:
  > "Share your name and contact so reviewers can follow up if
  > there's a question about your sign."
- `contribute.html:385` — step 2 title: "Describe"
- `contribute.html:386` — step 2 body:
  > "Type a plain-English description of the sign. The system asks
  > clarifying questions until it has what it needs."
- `contribute.html:390` — step 3 title: "Preview &amp; correct"
- `contribute.html:391` — step 3 body:
  > "Watch the 3D avatar sign it back to you. If something is off,
  > tell the system what to fix — in words, not code."
- `contribute.html:395` — step 4 title: "Submit for review"
- `contribute.html:396` — step 4 body:
  > "Two reviewers inspect and approve. Once published, your sign
  > appears in the translator for everyone."

### Form section (lines 403-469)

- `contribute.html:406` — eyebrow: "Start here"
- `contribute.html:407` — h2: "Create your / contributor profile."
- `contribute.html:410` — form card h2: "Register"
- `contribute.html:411` — form intro:
  > "Name and contact stay with us — reviewers only use them if they
  > need to ask about a submission. We never publish or share them."
- `contribute.html:415` — name label: "Your name"
- `contribute.html:416` — name placeholder: "e.g. Anna Novak"
- `contribute.html:420` — contact label: "Email or phone"
- `contribute.html:421` — contact placeholder:
  `"you@example.com or +1 555 0199"`
- `contribute.html:422` — contact hint:
  > "We use this only to follow up on submissions."
- `contribute.html:427` — honeypot label (off-screen): "Website"
- `contribute.html:432` — captcha label: "Quick check"
- `contribute.html:434` — captcha prompt default text:
  "Loading a math problem…"
- `contribute.html:435` — captcha answer placeholder: "Answer"
- `contribute.html:436-438` — captcha refresh button title /
  aria-label: "Get a new question"
- `contribute.html:447` — BYO key label:
  > "Your OpenAI API key · optional"
- `contribute.html:449` — BYO key placeholder: `"sk-…"`
- `contribute.html:450` — key toggle button: "Show" (toggles to "Hide")
- `contribute.html:452-456` — key hint:
  > "Stored only in your browser (`localStorage`) and sent as a
  > header on each call. Leave blank to use the project key once
  > it's provisioned. Get a key →"
- `contribute.html:461` — submit button: "Create profile &amp; start"
- `contribute.html:465` — fineprint:
  > "By continuing you agree your submitted signs may be published
  > under the project's open licence. Contact details stay private."

### FAQ (lines 472-497)

- `contribute.html:475` — eyebrow: "Good to know"
- `contribute.html:476` — h2: "Answers to / common questions."
- `contribute.html:480` — Q1: "What happens to my contact info?"
- `contribute.html:481` — A1:
  > "It stays in a private database only accessible to reviewers.
  > We never publish it, sell it, or use it for marketing. You can
  > ask us to delete it anytime via the GitHub issues link."
- `contribute.html:484` — Q2: "Do I need to be a Deaf signer?"
- `contribute.html:485` — A2:
  > "No — anyone can contribute. Every sign is reviewed by at least
  > one Deaf native signer before publication, so your draft is
  > always in good hands."
- `contribute.html:488` — Q3: "Which sign languages does Bridgn support?"
- `contribute.html:489` — A3:
  > "BSL (British), DGS (German), ASL (American), LSF (French), LSE
  > (Spanish), PJM (Polish), NGT (Dutch), and GSL (Greek). More
  > coming as the dictionary grows."

  **Flag:** API literal accepts only `bsl|asl|dgs`.

- `contribute.html:492` — Q4: "Who pays for the OpenAI API?"
- `contribute.html:493` — A4:
  > "Bridgn does — contributions are free for you. We cap daily
  > spend to keep things sustainable; if you hit a limit, check back
  > the next day."

### Footer (lines 500-511)

- `contribute.html:502` — logo: "Bridgn"
- `contribute.html:504-507` — link list: "Translator", "Contribute",
  "Features", "GitHub"
- `contribute.html:509` — copy:
  > "© 2025–2026 Bridgn. Open-source research project."

### JS-emitted UI messages (lines 513-688)

Messages inserted into the `#formMsg` status pane, visible to the
user as part of the registration flow:

- `contribute.html:570` — success banner after auto-redirect:
  > "You already have an active profile. Continuing to the
  > authoring tool…"
- `contribute.html:610` — captcha failure:
  > "Could not load — click refresh."
- `contribute.html:625` — name missing: "Please enter your name."
- `contribute.html:626` — contact missing: "Please add an email or phone."
- `contribute.html:628` — captcha missing: "Please answer the math question."
- `contribute.html:629` — captcha stale:
  > "Captcha still loading — try the refresh button."
- `contribute.html:662` — captcha server-reject:
  > "Captcha answer was wrong. Here is a new question."
- `contribute.html:675` — success: "Profile created. Opening the authoring tool…"
- `contribute.html:679` — network failure: "Network error — please try again."

---

## 4. `public/chat2hamnosys/index.html` — authoring UI (contributor-facing only)

### Doc

- `chat2hamnosys/index.html:7` — title: "Kozha — Sign Authoring"
- `chat2hamnosys/index.html:8` — meta description:
  > "Authoring tool for Deaf signers and SL linguists to draft new
  > HamNoSys sign entries via guided conversation."

### Nav / header

- `chat2hamnosys/index.html:20` — skip link: "Skip to chat input"
- `chat2hamnosys/index.html:23` — brand tag: "Kozha authoring"
- `chat2hamnosys/index.html:27` — high-contrast toggle label:
  "High contrast"
- `chat2hamnosys/index.html:31-32` — font size buttons: "A−" / "A+"
- `chat2hamnosys/index.html:34` — translator back-link: "← Translator"
- `chat2hamnosys/index.html:40` — status banner placeholder: "Working…"

### Mobile tabs

- `chat2hamnosys/index.html:44-46` — tabs: "Chat", "Preview",
  "Parameters"

### Chat panel

- `chat2hamnosys/index.html:53` — h1: "Authoring conversation"
- `chat2hamnosys/index.html:54` — initial sub: "No session yet"
- `chat2hamnosys/index.html:62` — gloss label: "Sign name (gloss)"
- `chat2hamnosys/index.html:67` — gloss placeholder: "e.g. SORRY"
- `chat2hamnosys/index.html:72` — gloss hint:
  > "Short uppercase token. Used as the lookup key for the finalized
  > sign entry."
- `chat2hamnosys/index.html:76` — prose label: "Describe the sign"
- `chat2hamnosys/index.html:82` — prose placeholder:
  > "It's signed near the temple, flat hand, moves down to the chest…"
- `chat2hamnosys/index.html:87-89` — prose hint:
  > "Plain English. Mention handshape, location, palm orientation,
  > and movement."
- `chat2hamnosys/index.html:91-93` — buttons: "Reject", "Accept",
  "Send"

### Preview panel

- `chat2hamnosys/index.html:103` — h2: "Avatar preview"
- `chat2hamnosys/index.html:104-105` — sub:
  > "Click on the avatar to flag a body region. Use the timeline to
  > flag a moment."
- `chat2hamnosys/index.html:125` — placeholder:
  > "Send a description to generate a preview."
- `chat2hamnosys/index.html:161-162` — popover label:
  > "What should change here?"
- `chat2hamnosys/index.html:168` — popover placeholder:
  > "e.g. make it a flat-O"
- `chat2hamnosys/index.html:172` — popover submit:
  "Submit correction"
- `chat2hamnosys/index.html:185-188` — "View in sign language"
  button (disabled, tooltip):
  > "Sign-language video of this caption — coming soon"
- `chat2hamnosys/index.html:193-194` — playback controls: "▶ Play",
  "↻ Loop"
- `chat2hamnosys/index.html:211-212` — timeline flag button:
  "Flag this moment"
- `chat2hamnosys/index.html:217` — keyboard region picker label:
  "Select region"
- `chat2hamnosys/index.html:227` — "Specify correction"

### Parameters panel

- `chat2hamnosys/index.html:238` — h2: "Sign parameters"
- `chat2hamnosys/index.html:245` — toggle button: "Collapse"
- `chat2hamnosys/index.html:249` — empty state: "No parameters yet."
- `chat2hamnosys/index.html:253-255` — legend badges: "confirmed",
  "inferred", "gap"

---

## 5. `public/chat2hamnosys/review/index.html` — reviewer console

### Doc

- `review/index.html:7` — title: "Kozha — Reviewer Console"
- `review/index.html:8` — meta description:
  > "Deaf reviewer console: queue of pending signs, action buttons,
  > and the export gate to the Kozha library."

### Header

- `review/index.html:19` — skip link: "Skip to review queue"
- `review/index.html:22` — brand tag: "Kozha reviewer"
- `review/index.html:26` — fatigue meter:
  `"0/25 reviewed this session"`
- `review/index.html:27` — sign-out button: "Sign out"
- `review/index.html:28` — back link: "← Authoring"

### Sign-in gate

- `review/index.html:40` — h1: "Reviewer sign-in"
- `review/index.html:41-45` — sub:
  > "Paste the bearer token your administrator issued. Tokens are
  > stored only in this browser's `localStorage` — closing the tab
  > keeps you signed in until you press *Sign out*."
- `review/index.html:47` — label: "Bearer token"
- `review/index.html:51` — placeholder: "paste token here"
- `review/index.html:55` — submit: "Sign in"

### Queue

- `review/index.html:65` — h1: "Review queue"
- `review/index.html:68-75` — language filter:
  "Sign language" / options: "All", "BSL", "ASL", "DGS"
- `review/index.html:77-78` — region filter:
  "Region" / placeholder: "e.g. BSL-London"
- `review/index.html:80-82` — "Include quarantined" checkbox
- `review/index.html:84` — "Refresh"
- `review/index.html:88` — empty state: "The queue is empty."

### Detail panel

- `review/index.html:94` — h2: "Sign detail"
- `review/index.html:95` — initial sub: "Pick a sign from the queue."
- `review/index.html:100` — empty-state body:
  > "Select a sign on the left to inspect it."
- `review/index.html:111` — meta label: "Sign language"
- `review/index.html:114` — meta label: "Regional variant"
- `review/index.html:117` — meta label: "Domain"
- `review/index.html:120` — meta label: "Approvals"
- `review/index.html:123` — approvals text: "of [N] required"
- `review/index.html:129` — section h4: "Description"
- `review/index.html:133` — section h4: "HamNoSys"
- `review/index.html:138` — section h4: "Reviewer history"
- `review/index.html:146` — action button: "Approve"
- `review/index.html:147` — action button: "Reject"
- `review/index.html:148` — action button: "Request revision"
- `review/index.html:149` — action button: "Flag (quarantine)"
- `review/index.html:150-152` — action button: "Clear quarantine"
- `review/index.html:153-155` — action button: "Export to library"

### Action dialog

- `review/index.html:169` — label: "Comment"
- `review/index.html:174` — placeholder: "Required."
- `review/index.html:179` — fieldset legend: "Category"
- `review/index.html:180-184` — category radios:
  "Inaccurate", "Culturally inappropriate", "Regional mismatch",
  "Poor quality", "Other"
- `review/index.html:188` — fieldset legend: "Send back to"
- `review/index.html:189-191` — clear-target radios:
  "Pending review", "Draft", "Rejected (uphold flag)"
- `review/index.html:194-196` — justify label:
  "Justification (non-native override)"
- `review/index.html:200-201` — non-native checkbox:
  > "Approve as a non-native reviewer (requires justification)"
- `review/index.html:205-206` — buttons: "Cancel", "Submit"

---

## 6. Observations on copy

- **Product name is inconsistent.** The translator site uses
  "Bridgn" (`contribute.html`, `index.html`); the authoring site
  uses "Kozha" (`chat2hamnosys/index.html`,
  `chat2hamnosys/review/index.html`). A contributor completing the
  register form is handed off to a page with a different brand at
  the top.
- **"Open-source" framing.** Hero eyebrow says
  "Open-source · Community-built"; the fineprint says signs
  "may be published under the project's open licence" — licence is
  not named anywhere on the contributor path.
- **"Under five minutes" / "Four steps".** The hero card says
  "Under five minutes"; the next section advertises "Four steps".
  No time estimate per step.
- **The FAQ's language list overstates API coverage** (see
  [00-language-coverage.md](00-language-coverage.md) and the flagged
  issue in [00-current-state.md](00-current-state.md)).
- **Reviewer console copy talks about "Deaf reviewer"** but the UI
  itself never asks "are you Deaf?" — the distinction comes from
  the bearer token's `is_deaf_native` flag which is set when the
  board creates the reviewer.

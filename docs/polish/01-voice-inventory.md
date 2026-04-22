# Voice and copy inventory — main pages

Every piece of user-facing English text on the two main pages, captured verbatim with file:line. This is the voice the contribute page's copy should match.

## Head / meta (landing)

- `public/index.html:6` `<title>Bridgn — Text and Speech to Sign Language</title>`
- `public/index.html:7` `<meta name="description" content="Bridgn translates text and speech into sign language animations. Free, open-source, and supports multiple sign languages.">`
- `public/index.html:8` `<meta property="og:title" content="Bridgn — Text and Speech to Sign Language">`
- `public/index.html:9` `<meta property="og:description" content="Translate text and speech into sign language animations. Free and open-source.">`
- `public/index.html:21` `"description": "Translates text and speech into sign language animations. Supports multiple sign languages."` (schema.org)

## Head / meta (app)

- `public/app.html:6` `<title>Bridgn — Translator</title>`
- `public/app.html:7` `<meta name="description" content="Translate text or speech into sign language animations with Bridgn. Supports German, Polish, and British sign languages.">`
  - Note: this is stale — the app now supports 15 languages, not 3. Fix in a later prompt.
- `public/app.html:9` `<meta property="og:description" content="Type or speak to see sign language animations in real time. Free and open-source.">`

## Nav (landing)

- `public/index.html:757` `Skip to main content`
- `public/index.html:761` `Bridgn` (logo, inline via `Bridg<span>n</span>`)
- `public/index.html:763` `Features`
- `public/index.html:764` `How it works`
- `public/index.html:765` `Contribute`
- `public/index.html:766` `Governance`
- `public/index.html:769` `Volunteer`
- `public/index.html:770` `Open translator →`
- `public/index.html:771` `Menu` (aria-label for hamburger)

## Hero (landing)

- `public/index.html:780` `Open-source research project`
- `public/index.html:781` `Text and speech<br>to <em>sign language.</em>` (heading)
- `public/index.html:782-784` `Bridgn converts typed text, recorded speech, or uploaded video into sign language animations. It runs entirely in your browser — no accounts, no servers, no cost.`
- `public/index.html:786` `Open translator`
- `public/index.html:787` `How it works`
- `public/index.html:791` `15`
- `public/index.html:792` `Sign languages`
- `public/index.html:795` `On-device`
- `public/index.html:796` `Speech recognition`
- `public/index.html:799` `Free`
- `public/index.html:800` `Always for educators`

## Hero demo card (landing)

- `public/index.html:807` `✏️ Text`
- `public/index.html:808` `🎙 Voice`
- `public/index.html:809` `🎬 Video`
- `public/index.html:810` `Translate ▶`
- `public/index.html:815` `Input language`
- `public/index.html:866` `Sign language`
- `public/index.html:890` `Good morning, everyone.` (default textarea value)
- `public/index.html:890` `Type a sentence to translate…` (placeholder)
- `public/index.html:895` `● Start recording`
- `public/index.html:896` `Speech recognition runs locally in your browser.`
- `public/index.html:900` `Video file`
- `public/index.html:902` `Process video`
- `public/index.html:903` `Upload a video to extract and transcribe audio.`
- `public/index.html:906` `Loading avatar…`
- `public/index.html:915` `Connecting…`
- `public/index.html:916` `Open full app →`

Runtime-generated states (index.html script):
- `1041` `Avatar could not load.<br><a …>Try the full app →</a>`
- `1063` `Ready`
- `1240` `Loading…`
- `1244` `No sign database available — fingerspelling only`
- `1247` `Ready (N signs)`
- `1351` `Translation unavailable — using original text`
- `1362` `Translating {SRC} → {TGT}…`
- `1406` `No signs found`
- `1419` `Playing…`
- `1422` `Avatar unavailable — showing text output`
- `1424` `Planned: word1, word2, …`
- `1492` `Mic error: …`
- `1500` `Preparing speech recognition…`
- `1504` `Transcribing audio…`
- `1510` `Transcribed! Hit Translate to see signs.`
- `1516` `Transcription error: …`
- `1551` `Choose a video file first.`
- `1555` `Preparing…`
- `1558` `Extracting audio from video…`

## Features section (landing)

- `public/index.html:921` `Features` (eyebrow)
- `public/index.html:922` `What <em>Bridgn</em> does.` (title)
- `public/index.html:926` `Voice-to-Sign`
- `public/index.html:927` `Record speech in your browser. Transcription uses your device's built-in speech recognition — no audio leaves your machine.`
- `public/index.html:930` `Text-to-Sign`
- `public/index.html:931` `Type a sentence and see it translated into sign language animations. Words not in the dictionary are fingerspelled automatically.`
- `public/index.html:934` `Video Input`
- `public/index.html:935` `Upload a recorded video. Bridgn extracts the audio track, transcribes it, and generates the corresponding sign sequence.`
- `public/index.html:938` `STEM Vocabulary`
- `public/index.html:939` `Pre-loaded with scientific abbreviation packs. "Carbon dioxide" becomes "CO₂" before planning — no manual editing needed.`
- `public/index.html:942` `Automatic Planning`
- `public/index.html:943` `Input text is normalized, lemmatized, and mapped to available signs. The planner reorders words to match sign language grammar.`
- `public/index.html:946` `Deaf-governed Contributions`
- `public/index.html:947` `Every new sign is reviewed by two Deaf native signers before it ships. <a …>Governance →</a>`

## How-it-works section (landing)

- `public/index.html:953` `The process`
- `public/index.html:954` `How it <em>works.</em>`
- `public/index.html:959` `Provide input`
- `public/index.html:960` `Type text, record speech with your microphone, or upload a video file. All three modes work in the same session.`
- `public/index.html:964` `Plan the translation`
- `public/index.html:965` `The planner normalizes your text, applies abbreviations, and looks up each token in the sign database. Unknown words are queued for fingerspelling.`
- `public/index.html:969` `Animate the signs`
- `public/index.html:970` `The matched signs are rendered as a 3D avatar animation using SiGML data. Everything runs client-side in your browser.`

## Contribute banner (landing)

- `public/index.html:979` `Contribute to the project.`
- `public/index.html:980` `Bridgn's sign database has limited vocabulary and needs help. If you know sign language, work in accessibility, or can code, you can add signs, improve translations, or fix bugs.`
- `public/index.html:983` `Get involved`

## Footer (landing)

- `public/index.html:992` `Translator`
- `public/index.html:993` `Contribute`
- `public/index.html:994` `Governance`
- `public/index.html:995` `Features`
- `public/index.html:996` `How it works`
- `public/index.html:998` `© 2025–2026 Bridgn. Open-source research project.`

## App page — chrome and controls (app.html)

- `public/app.html:627` `Translator` (nav badge)
- `public/app.html:630` `Contribute a sign`
- `public/app.html:631` `← Home`
- `public/app.html:639` `Input` (sidebar label)
- `public/app.html:644` `Translate`
- `public/app.html:648` `Microphone`
- `public/app.html:653` `Video Upload`
- `public/app.html:657` `Configuration` (sidebar label)
- `public/app.html:661` `Database`
- `public/app.html:666` `Advanced`
- `public/app.html:670` `Debug` (sidebar label)
- `public/app.html:674` `Planner Log`
- `public/app.html:682` `Text Input`
- `public/app.html:683` `Enter your phrase`
- `public/app.html:684` `Try something like: Hello, how are you?` (placeholder)
- `public/app.html:689` `Input language`
- `public/app.html:740` `Sign language`
- `public/app.html:762` `🤟 Translate to Sign`
- `public/app.html:769` `Microphone Input`
- `public/app.html:771` `● Start recording`
- `public/app.html:772` `Transcribe`
- `public/app.html:773` `Idle`
- `public/app.html:775` `Transcription`
- `public/app.html:776` `Transcription will appear here…`
- `public/app.html:782` `Video Upload → Sign`
- `public/app.html:783` `Select video file`
- `public/app.html:786` `Process Video → Sign`
- `public/app.html:787` `No file selected`
- `public/app.html:794` `Sign Database`
- `public/app.html:796` `Sign database URL`
- `public/app.html:799` `Load`
- `public/app.html:802` `Concept map CSV (URL)`
- `public/app.html:808` `Or upload CSV / TSV file`
- `public/app.html:814` `Loaded Vocabulary`
- `public/app.html:821` `Text Processing`
- `public/app.html:824` `Use advanced text processing`
- `public/app.html:825` `Apply abbreviations`
- `public/app.html:826` `STEM mode (preload STEM abbreviations)`
- `public/app.html:829` `Load abbreviations JSON (URL)`
- `public/app.html:835` `Or upload abbreviations JSON file`
- `public/app.html:838` `Clear`
- `public/app.html:846` `Planner Diagnostics`
- `public/app.html:848` `Planner log`
- `public/app.html:853` `Normalized input`
- `public/app.html:857` `Token sequence`
- `public/app.html:861` `Timeframe guess`
- `public/app.html:865` `Abbreviations applied`
- `public/app.html:885` `Loading…`
- `public/app.html:887` `Avatar`
- `public/app.html:895` `⏹ Stop`
- `public/app.html:903` `Governance`

## Voice characterization

**Tense.** Present, active, declarative. "Bridgn converts …", "Type a sentence and see it translated …". The product is the subject; the user is the implicit actor.

**Person.** Third-person when describing the product ("Bridgn converts", "The planner"). Second-person when addressing the user ("All three modes work in the same session", "Words not in the dictionary are fingerspelled automatically" — note passive here). Never first-person plural ("we", "us") on the landing. The app page uses imperative second person in labels ("Enter your phrase", "Load", "Clear").

**Tone.** Confident but quiet. No exclamation points anywhere in landing prose. Promotional language is absent. Specificity carries the claim: "no audio leaves your machine", "two Deaf native signers before it ships", "runs entirely in your browser — no accounts, no servers, no cost." Numbers are concrete: 15, On-device, Free.

**Vocabulary register.** Mid. Plainspoken but not dumbed down. Technical terms land without apology: "lemmatized", "SiGML", "fingerspelled", "planner". Does not explain what HamNoSys is on the landing — trusts the reader to click through.

**Sentence rhythm.** Short-to-medium. Average landing-prose sentence is ~16 words. Rarely chains more than two clauses. Em-dashes carry the pauses: "— no audio leaves your machine", "— no accounts, no servers, no cost."

**Emoji usage.** Emoji appear *only* as icon substitutes in compact affordances: the three demo tabs (✏️ 🎙 🎬), the translate button's ▶ arrow, the app sidebar items (🤟 🎙️ 🎬 🗄️ ⚙️ 📋), the card titles inside the app, and the Stop button (⏹). They never appear in prose, headings, or meta tags. The contribute page has no emoji — this is correct and consistent.

**Punctuation quirks.**
- `→` for forward navigation ("Open translator →", "Open full app →", "Video Upload → Sign"). Always a full arrow, never `->`.
- Em-dashes (`—`) for appositional clauses and interruption. Never hyphens for this purpose.
- Curly apostrophes in prose when present; straight in code-adjacent strings like "doesn't".
- Dots in placeholder ellipses: `Type a sentence to translate…`, `Loading…`, `Transcription will appear here…`. Always the single `…` character, not `...`.

**What the voice avoids.**
- No "Welcome!" / "Get started!" energy.
- No persona ("Hi, I'm Bridgn — …").
- No promises of quality ("fast", "accurate", "beautiful") — the product just describes what it does.
- No jargon from product-marketing (journey, experience, solution).
- No hedge words ("might", "could", "hopefully").

**Contribute-page drift.** Keys in `public/strings.en.json:contribute.*` — the canonical copy for the contribute surface — are mostly in this voice already (imperative, declarative, short). A handful of exceptions to examine in prompt 4:
- `contribute.chat.generating_msg`: "Enough information to draft the sign. Preparing preview." — the fragment is correct but curt; the period should land after "draft the sign" more clearly.
- `contribute.chat.error_msg`: "The AI had trouble generating a follow-up. Your answer was saved — try rephrasing, or submit the draft as-is and let the reviewer fill any gaps." — fine, matches voice.
- `contribute.authoring.gloss_label`: "Gloss — the English word or short phrase for this sign." — in voice, em-dashed appositional. Keep.
- `contribute.hint_strip.body_change_lang`: "change language" — lower-case start is consistent with the design-aesthetic of the hint strip.

No inconsistencies large enough to flag as drift; the written voice holds up. The gap is copy that exists on the main pages but has no parallel on the contribute page — see `01-contribute-current-state.md`.

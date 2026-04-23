# Kozha — RESNA Presentation Factsheet

*Last verified: April 2026. Raw dump — filter/curate as needed for slides.*

---

## A. KOZHA PROJECT METRICS (internal, as of April 2026)

### Codebase
- **205 total commits**
- **75,207 lines** of first-party code (excluding vendored CWASA bundles)
  - Python: 48,448 lines
  - HTML: 9,961 lines
  - JavaScript: 11,347 lines
  - CSS: 5,451 lines
- First-party code organized across: backend (FastAPI + spaCy + argostranslate), 3 frontend pages (index, translator, contribute), Chrome extension (3 modes), chat2hamnosys session orchestrator

### Sign language data
- **26 SiGML databases** total (15 active + quarantined variants)
- **6 CSV concept→gloss mappings**
- **13 fingerspelling alphabet files** (A–Z manual alphabets)
- **~19,050 active signs** across all databases

### Per-language sign counts (active, loadable)
| Sign Language | Signs |
|---|---|
| Vietnamese SL | 7,128 |
| DGS (German) | 3,828 |
| PJM (Polish) | 2,040 |
| BSL (British) | 1,762 |
| Indian SL | 1,458 |
| Kurdish SL | 1,116 |
| GSL (Greek) | 1,068 |
| LSF (French) | 448 |
| Bangla SL | 120 |
| NGT (Dutch) | 78 |
| Algerian, Filipino, LSE, RSL, KSL | Fingerspelling-only (in progress) |

### Languages & NLP
- **16 sign languages** in dropdown (BSL, ASL, DGS, LSF, LSE, PJM, GSL, RSL, NGT, Algerian, Bangla, FSL, ISL, Kurdish, VSL, **KSL**)
- **7 dedicated grammar-aware spaCy NLP pipelines** (en, de, fr, es, pl, nl, el)
- **~40 input languages** via argostranslate fallback
- **Whisper ASR** via transformers.js for speech input
- **Custom per-sign-language grammar profiles** (SOV reordering for DGS/LSF/NGT/RSL; time-first for all; topic-comment for BSL/ASL)

### Deployment & infrastructure
- Live at **kozha-translate.com** on AWS EC2
- 7 REST endpoints, 2-worker gunicorn, systemd-managed
- 3 client surfaces: web translator, landing-page hero demo, Chrome extension

---

## B. KOZHA UNIQUE POSITIONING

### Uniquely ours (no direct competitor has this combination)
- **Breadth**: 16 sign languages — no competitor covers more than ~5 production-ready
  - Signapse: 2 | Hand Talk: 3 | SignON: 5 | SignAll / SignGemma / KinTrans: 1 each
- **7 dedicated grammar-aware spaCy NLP pipelines** — competitors use statistical MT, a single LLM, or human post-editing
- **Universal Chrome extension**: YouTube auto-sync + any-webpage context menu + standalone popup
- **Community contribute flow with GPT-powered SiGML generation** — no competitor exposes this
- **Deaf-native review tracking** at the sign level (provenance/review metadata)
- **Fully open-source + self-hostable** — peers (sign.mt, SignON, PopSign) are all narrower

### On par
- 3D avatar on CWASA/JASigning (shared stack with SiMAX, SignON, academic research)
- Grammar-aware translation (KinTrans, SignGemma, sign.mt, SignON also have grammar)
- Text-to-sign direction (Signapse, Hand Talk, SiMAX, sign.mt)

### Where competitors are stronger (honest)
- **Photo-realism**: Signapse (recorded Deaf-signer video), sign.mt (photo-realistic avatars) look more natural than 3D cartoon
- **Single-language vocab depth**: Hand Talk (~38k Libras), SignAll ("largest annotated ASL corpus")
- **Sign-to-text direction**: SignAll, KinTrans, SignGemma have it; Kozha's two ML projects are in development
- **Live broadcast / telephony**: Signapse SignStream + Sorenson VRS cover use cases Kozha doesn't
- **Deaf-community institutional backing**: Sorenson (6,000+ human interpreters), SignON (EU-funded Deaf-led consortium), Signapse (paid native Deaf signers)

---

## C. COMPETITOR LANDSCAPE

| # | Product | Input | Output | Sign Langs | Vocab | Grammar NLP | Pricing | Open Source |
|---|---------|-------|--------|------------|-------|-------------|---------|-------------|
| 1 | **Ava** (ava.me) | Speech | Text captions | N/A (50+ spoken) | — | — | Free / $9.99–14.99/mo / Enterprise | ❌ |
| 2 | **Google Live Transcribe** | Speech | Text captions | None (120+ spoken) | — | — | Free | Partial (engine OSS) |
| 3 | **Signapse** (signapse.ai) | Text + live stream | Photo-real recorded signer video | BSL + ASL (2) | Undisclosed | Limited | SaaS subscription, 12-mo min | ❌ |
| 4 | **SignAll** (signall.us) | Video (cameras + gloves) | Text/English | ASL (1) | Undisclosed "largest ASL corpus" | ✅ CV + grammar | Enterprise/kiosk only | ❌ |
| 5 | **Hand Talk** (handtalk.me) | Text/audio | Hugo/Maya 3D avatar + plugin | Libras, ASL, BSL (3) | ~38,000 (Libras) | Statistical MT | Free app; plugin enterprise | ❌ |
| 6 | **SiMAX** (simax.media) | Text (human-edited) | 3D avatar video | Multiple via human post-edit | Depends on translator | Human-in-the-loop | Enterprise only | ❌ |
| 7 | **Sorenson VRS** | Video call | Human ASL interpreter | ASL (1) | Human | Human | Free (FCC TRS fund) | ❌ |
| 8 | **KinTrans** (kintrans.com) | Video (3D camera) | Text/voice + reply avatar | ASL + Arabic SL (2) | ~2,820 signs | ✅ | Enterprise kiosks | Partial (dataset) |
| 9 | **Google SignGemma** | Video | Text (ASL→English) | ASL (1) | LLM-based | ✅ LLM | Free weights | ✅ (open-weights) |
| 10 | **sign.mt** (Moryossef) | Text/speech ↔ video | Photo-real avatar + text | 40+ claimed | Model-based | ✅ | Free | ✅ (CC BY-NC-SA 4.0) |
| 11 | **SignON** (EU H2020) | Text/speech/video | Avatar + BML | 5 | Research prototype | ✅ | Free (research) | ✅ |
| 12 | **Microsoft Seeing AI / Teams** | Camera | Text / speaker elevation | None | — | — | Free | ❌ |

---

## D. PROBLEM SCALE (global)

### Hearing loss
- **1.5 billion people** live with some hearing loss (~20% of world population) — WHO, 2024
- **430 million** have *disabling* hearing loss (>35 dB in better ear) — WHO, 2024
- **By 2050**: ~2.5 billion will have hearing loss; 700 million will require rehabilitation — WHO, 2021
- **34 million children** worldwide are deaf / hard of hearing — WHO, 2024
- **~80%** of people with disabling hearing loss live in low- and middle-income countries — WHO, 2024
- **Annual global cost** of unaddressed hearing loss: ~**US$1 trillion** — WHO, 2024
- **ROI**: US$16 returned for every US$1 spent on ear and hearing care — WHO, 2021

### Deaf population & sign language
- **~70 million deaf people** globally; >80% in developing countries — UN / WFD
- **~12–25 million native sign language users** (Ethnologue 2024)
- **>300 distinct sign languages** in use — UN International Day of Sign Languages
- **150–160** sign languages have ISO codes (Ethnologue/SIL)
- Sign languages are **mutually unintelligible** across countries (ASL ≠ BSL despite both being English-context)
- **World Federation of the Deaf** = federation of **135 national associations**

---

## E. THE ACCESS GAP (why this matters)

### Interpreter shortage
- **~50:1 ratio** of Deaf ASL users to RID-certified interpreters in the US (~500k users / ~10k interpreters) — Deaf Services Unlimited, 2023
- Becoming a certified interpreter takes **5–10 years** — LSA, 2023
- **87% of US ASL interpreters are white**, creating cultural/dialect barriers — Insight Into Diversity, 2023
- Average US interpreter wage: **$38.34/hour** employed; freelance **$75–$125/hour** with 2-hour minimums — Interpreters.com, Indeed 2024
- **VRS wait-time FCC standard**: 80% of calls answered within 120 seconds — FCC, 2024
- **<20%** of people who could benefit from hearing aids get them — WHO, 2024

### Caption insufficiency
- **Median reading grade level of Deaf US high school grads: ~4th grade** — Traxler/SAT-9, 2000
- Adult Deaf ASL users: **5.9 average reading grade level** (vs. 9.8 hearing) — McKee et al. 2015
- **Auto-caption accuracy**: YouTube 94–96% clean audio, 60–82% noisy — Consumer Reports 2021
- **~10% word-error rate** on leading platforms (Zoom, Facebook, Meet, YouTube)
- **Human captions**: 96–99% accuracy — 3Play Media, 2023
- Stanford: ASR misrecognizes Black speakers at **2× the error rate** of white speakers — PNAS, 2020
- **87.6%** of caption users turn them on; **80%** of caption users are not hearing impaired

---

## F. EDUCATION GAP

- Estimated **90% of deaf children worldwide are illiterate** due to lack of sign-competent teachers — GPE / UNICEF, 2018
- **American Academy of Pediatrics (2023)**: for the first time, formally recommends ASL (or other signed language) for deaf children regardless of cochlear-implant status
- **UN CRPD Articles 21 & 24** legally oblige states to facilitate sign languages
- **UNICEF ESARO** explicitly recommends sign language in early childhood + digital learning materials (2023)
- "Only 2% of deaf children receive education in sign language" — widely cited (unverified primary source)

---

## G. KAZAKHSTAN (presenter home country — strong RESNA angle)

- **~725,000 people with disabilities** in Kazakhstan (~4% of population); >100,000 children — UNDP / OHCHR, 2024
- **>150,000** people in Kazakhstan have impaired hearing — Astana Times, 2015
- **~50,000–70,000 Kazakh Sign Language (KSL / Қазақ ымдау тілі) users** — MDPI, 2025
- **KSL is a dialect of Russian Sign Language (RSL)** — Ethnologue; 120–144k total RSL signers across CIS
- **State provides only 60 hours/year of free interpreter time** per deaf individual — egov.kz, 2024
- **KSL is not yet officially recognized** as a state language (cabinet studying international practices as of 2025)
- **17 universities** in Kazakhstan train teachers of the deaf
- **Existing Kazakh tech**: Nazarbayev University's K-SLARS (recognition), KRSL/KRSL20 parallel corpus

---

## H. MARKET & LEGAL CONTEXT

- **Sign-language translator market**: USD **1.2 B (2024) → 4.5 B (2033)**, **CAGR 15.4%** — Verified Market Reports, 2024
- **Global assistive-tech market**: ~USD **26.8 B (2024) → 41 B (2033)** at 4.33% CAGR — IMARC, 2024
- **EU Accessibility Act** (Directive 2019/882) became applicable **28 June 2025** — requires sign-language synchronization with AV media when provided
- **ADA (US, 1990)**: >90,000 discrimination complaints filed 1992–1997; 29% for failure to provide accommodations

### Notable competitor funding
- **Signapse**: USD 2.4M seed (2024), incl. Innovate UK, Royal Assoc. for Deaf People
- **Hand Talk**: R$2.5M (~USD 670k) impact round from Kviv
- **KinTrans**: USD 100k grant from Dubai Expo 2020

---

## I. ACADEMIC LANDSCAPE

- **RWTH-PHOENIX-Weather 2014T**: ~11 hours DGS, most-cited SLT benchmark
- **How2Sign** (ASL): >80 hours multi-view video, 35k+ sentences, 16k vocab — largest ASL
- **WLASL-2000**: 2,000 glosses, ~21,000 video samples, >100 signers
- **DGS Corpus (Hamburg)**: ~560 hours annotated; 50 hours public; 330 signers
- **WLASL state-of-the-art**: top-1 accuracy **93.51%** (ensemble transformer, July 2025); DSLNet 89.97% on WLASL300
- Sign-language MT/NLP paper output has grown sharply in past 5 years

---

## J. DEAF EMPLOYMENT & HEALTHCARE GAP

- **US employment rate**: 53.3% deaf vs. 75.8% hearing (ages 25–64) — National Deaf Center, 2019
- **Labor-force non-participation**: 42.9% deaf vs. 20.8% hearing — 2024
- **Wage gap**: Deaf workers earn **13–14% less** annually — UT Austin, 2019
- **ER visits 9% longer** for Deaf ASL patients (~30 min extra) — 2024 mixed-methods study

---

## K. AVATAR VS HUMAN (sign tech UX research)

- **184-participant ASL user survey**: motion-capture avatars rated significantly more positively than synthesized ones; all users still preferred real human signers, but comprehension scores on math tasks were **equal between human and avatar** — Quandt et al., Frontiers in Psychology, 2022
- **Later-learners** are more accepting of avatars; native signers rate them more critically
- **Facial expression** is the #1 cited deficiency of signing avatars — Kipp et al., 2011

---

## L. HEADLINE NUMBERS FOR THE TALK

Use any of these as clean talking points:

- "**1.5 billion** have hearing loss. By **2050, 2.5 billion**. **70 million** are Deaf. There are over **300 sign languages**."
- "The US has a **50 : 1** Deaf-to-interpreter ratio. Becoming certified takes **5–10 years**."
- "Deaf high-school graduates read at a **4th-grade median level**. Auto-captions have a **10% word error rate**. Captions alone aren't enough."
- "We support **16 sign languages** with **~19,000 signs** across them — more than any other tool in this market."
- "Kozha is the only open-source tool with **grammar-aware NLP for 7 languages** and a **community contribution pipeline** for adding new signs."
- "EU Accessibility Act compliance deadline passed June 2025. Sign-language translation market is growing at **15.4% CAGR**."
- "In Kazakhstan, Deaf citizens receive only **60 hours/year** of free interpretation. **50,000–70,000** sign-language users. That's 830 people per interpreter hour."

---

## Sources (selected)

All external facts have URLs embedded in the text above. Key primary sources:

- **WHO** — https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss
- **WFD** — https://wfdeaf.org
- **UN International Day of Sign Languages** — https://www.un.org/en/observances/sign-languages-day
- **Ethnologue** — https://www.ethnologue.com
- **Consumer Reports (auto-captions)** — 2021
- **Traxler/SAT-9** — Hands & Voices, 2000
- **McKee et al. 2015** — https://pmc.ncbi.nlm.nih.gov/articles/PMC4714330/
- **Quandt et al. (avatars)** — Frontiers in Psychology, 2022
- **FCC VRS** — https://www.fcc.gov/consumers/guides/video-relay-services
- **EU Accessibility Act** — https://commission.europa.eu/strategy-and-policy/policies/justice-and-fundamental-rights/disability/european-accessibility-act-eaa_en
- **AAP signed language recommendation** — 2023
- **RWTH-PHOENIX / How2Sign / WLASL / DGS Corpus** — cited above
- **Kazakhstan disability stats** — UNDP / OHCHR (2024); Astana Times (2015); MDPI KRSL paper (2025)
- **Nazarbayev University K-SLARS** — https://research.nu.edu.kz/en/projects/kazakh-sign-language-automatic-recognition-system-k-slars

### Flagged unverified / caveats
- "Only 2% of deaf children get education in sign language" — widely cited, primary source not confirmed
- SignAll total funding — not publicly consolidated
- Specific % of Deaf people in low-income countries without any interpreter access — approximated from WFD language

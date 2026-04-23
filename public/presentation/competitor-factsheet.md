# Kozha — Competitive Positioning Factsheet

*Last verified: April 2026*

## Competitor Landscape

| # | Product | Input | Output | Sign Languages | Vocab | Grammar NLP | Pricing | Open Source |
|---|---------|-------|--------|----------------|-------|-------------|---------|-------------|
| 1 | **Ava** (ava.me) | Speech | Text captions only | N/A (captions 50+ spoken langs) | N/A | N/A | Free tier; $9.99–14.99/mo; Scribe enterprise | ❌ |
| 2 | **Google Live Transcribe** | Speech | Text captions | N/A (120+ spoken langs, no sign output) | N/A | N/A | Free | Partial (engine on GitHub) |
| 3 | **Signapse** (signapse.ai) | Text + live stream | Photo-realistic recorded-signer video | BSL + ASL (2) | Undisclosed, template-stitched | Limited | SaaS subscription, 12-mo min; Enterprise | ❌ |
| 4 | **SignAll** (signall.us) | Video (cameras + gloves) | Text / English | ASL only (1) | Undisclosed | ✅ CV + grammar | Enterprise/kiosk only (Boeing, etc.) | ❌ |
| 5 | **Hand Talk** (handtalk.me) | Text/audio | 3D avatar (Hugo/Maya) + plugin | Libras, ASL, BSL (3) | ~38,000 (Libras) | Statistical MT | Free app; plugin = enterprise quote | ❌ |
| 6 | **SiMAX** (simax.media) | Text (human-edited) | 3D avatar video | Multiple via human post-edit | Depends on translator | Human-in-the-loop | Enterprise only | ❌ |
| 7 | **Sorenson VRS** | Video call | Human ASL interpreter relay | ASL only | Human (unbounded) | Human | Free (FCC TRS fund) | ❌ |
| 8 | **KinTrans** (kintrans.com) | Video (3D camera) | Text/voice + reply avatar | ASL + Arabic SL (2) | ~2,820 unique signs | ✅ | Enterprise kiosks | Partial (dataset) |
| 9 | **Google SignGemma** | Video | Text (ASL→English) | ASL primarily (1) | LLM-based | ✅ LLM | Free weights | ✅ (open-weights) |
| 10 | **sign.mt** (Moryossef) | Text/speech ↔ video | Photo-realistic avatar + text | 40+ claimed via multilingual model | Model-based | ✅ | Free | ✅ (CC BY-NC-SA 4.0) |
| 11 | **SignON** (EU H2020) | Text/speech/video | Avatar + BML | ISL, BSL, NGT, VGT, LSE (5) | Research prototype | ✅ | Free (research) | ✅ |
| 12 | **Microsoft Seeing AI / Teams** | Camera / Teams video | Text / speaker elevation | No sign generation | N/A | N/A | Free | ❌ |

## Where Kozha is UNIQUELY Positioned

- **Breadth: 16 sign languages** in one tool — no competitor covers more than ~5 production-ready sign languages. Signapse=2, Hand Talk=3, SignON=5. Newly added: **KSL (Kazakh Sign Language)**.
- **7 dedicated grammar-aware spaCy NLP pipelines** (en, de, fr, es, pl, nl, el) + Argos Translate fallback for everything else — unmatched dedicated per-language NLP. Competitors use one statistical MT model (Hand Talk), a single LLM (SignGemma), or human post-editing (SiMAX).
- **Universal Chrome extension** that works on **YouTube and any webpage** via context menu + popup — Hand Talk only has a publisher-installed website plugin; no one else combines universal extension + live avatar translation.
- **Community contribute flow with GPT-powered SiGML generation** — no competitor exposes a pipeline to author HamNoSys/SiGML for new signs collaboratively.
- **Deaf-native review tracking at the sign level** — built-in provenance/review metadata on every sign, distinct from Signapse's internal-only QA process.
- **Fully open-source, free, self-hostable** — peers (sign.mt, SignON, PopSign) are narrower in scope (one model, research prototype, or learning-only).

## Where Kozha is On Par

- 3D avatar pipeline on CWASA / JASigning — shared stack with UHH/UEA research ecosystem; SiMAX and SignON sit on the same HamNoSys→SiGML foundation.
- Grammar-aware translation — KinTrans, SignGemma, sign.mt, SignON also have grammar handling.
- Text-to-sign direction — matches Signapse, Hand Talk, SiMAX, sign.mt.
- Multilingual ambition — sign.mt and SignON explicitly target multilingual coverage too.

## Where Competitors Are Stronger (Honest Assessment)

- **Photo-realism**: Signapse's recorded Deaf-signer video and sign.mt's photo-realistic avatars look more natural than a 3D cartoon.
- **Single-language vocabulary depth**: Hand Talk (~38k Libras signs) and SignAll (claimed "largest annotated ASL corpus") dwarf Kozha's current per-language counts.
- **Sign-to-text (video input)**: SignAll, KinTrans, and SignGemma do bidirectional translation; Kozha is text/speech→sign only (our two ML projects are in development).
- **Live broadcast / telephony**: Signapse SignStream and Sorenson VRS cover live-event and relay-calling use cases Kozha does not yet.
- **Deaf-community institutional backing**: Sorenson (6,000+ human interpreters), SignON (EU-funded Deaf-led consortium), Signapse (paid native Deaf signers) carry stronger institutional endorsement.

## Headline Numbers

- **16** sign languages supported (BSL, ASL, DGS, LSF, LSE, PJM, GSL, RSL, NGT, Algerian, Bangla, FSL, ISL, Kurdish, VSL, **KSL**)
- **7** dedicated spaCy NLP models + **argostranslate** for 40+ input languages
- **15** sign databases with real vocabulary (+ fingerspelling fallback for the rest)
- **3 modes** in the Chrome extension (YouTube auto-sync, context menu, popup)
- **1** text-to-sign live pipeline at [kozha-translate.com](https://kozha-translate.com)
- **2** ML models in development (video→SiGML, video→text)
- **MIT-intended, research/educational-use currently** (non-commercial bindings on some bundled databases)

## Sources

Full source list for competitor facts: see the agent research report in this repo's commit history, or regenerate from the AGENT_QUERIES log. Key pages:
- ava.me, signapse.ai, signall.us, handtalk.me, simax.media, sorenson.com, kintrans.com
- Google blog posts on SignGemma and SignTown
- aclanthology.org for sign.mt (ACL 2024)
- signon-project.eu
- vh.cmp.uea.ac.uk (JASigning/CWASA)

# Press kit

Outreach materials for the chat2hamnosys subsystem. Designed to be useful at conference booths,
in funder pitch decks, and on social media. Plain language; not academic jargon.

## What's in here

| File | Purpose | Audience |
|---|---|---|
| `one-pager.md` | One-page overview: problem, solution, governance, contact. | Funders, journalists, expo visitors. |
| `screenshots/` | Three screenshots of the system at different stages (initial description, click-correction, post-validation). | Slide decks, social posts. |
| `demo-electron-30s.mp4` | 30-second screen recording of an end-to-end demo session for ELECTRON. Captioned in English; sign-language overlay where feasible. | Booth backup video; embed on landing pages. |
| `social/` | Social-media-sized variants of the screenshots and the one-pager (1200×630 OG, 1080×1080 IG, 1500×500 Twitter banner). | Platform-specific posts. |

## Production notes

The PNGs in `screenshots/` and the MP4 in this directory are not committed to git directly — they
are binary assets that bloat history. Instead, the repo contains:

- The Markdown one-pager (text, diff-able).
- A `screenshots/PLACEHOLDER.md` pointing to where the PNGs should be placed.
- A `social/PLACEHOLDER.md` describing the size matrix and the rendering script.

When the production pass is run, the PNGs and MP4 land in the same directory locally and are
served from the project's CDN bucket, not from git. The `.gitignore` excludes `*.png`, `*.mp4`,
`*.jpg` from this directory specifically.

To regenerate the screenshots:

```bash
docker compose up
python -m examples.replay --example electron --no-realtime
# Open browser, take screenshot at:
#   1. After describe (clarification question visible)
#   2. After generate (avatar mid-sign + click-correction overlay)
#   3. After accept (entry in reviewer queue with status pill)
# Save to docs/chat2hamnosys/20-press/screenshots/{1-describe,2-correct,3-review}.png
```

To regenerate the demo video:

```bash
# macOS:
brew install ffmpeg
docker compose up
# In a separate terminal:
python -m examples.replay --example electron
# Use macOS screen recording (Shift+Cmd+5) on the browser window
# Trim to 30 seconds in iMovie or ffmpeg
ffmpeg -i raw.mov -ss 0 -t 30 -vf "scale=1280:720" -c:v libx264 demo-electron-30s.mp4
# Add captions:
ffmpeg -i demo-electron-30s.mp4 -i captions.srt -c:s mov_text demo-electron-30s-captioned.mp4
```

## Style guide for outreach materials

- **Plain language.** The audience does not know what HamNoSys is. Don't make them learn it to
  understand the project.
- **Lead with the gap, not the solution.** "Sign-language pipelines fall back to fingerspelling
  for STEM terms" before "we built an LLM-mediated authoring tool".
- **Always mention the Deaf-reviewer gate.** It is the single most important differentiator from
  hostile-to-Deaf-community AI sign-language projects. Do not leave it implicit.
- **Don't show signers as exhibits.** No video of identified Deaf reviewers without explicit
  consent for the specific use. The avatar is fine to show; humans are not stock footage.
- **Provide the contact email.** Every outreach material must list `deaf-feedback@kozha.dev`.

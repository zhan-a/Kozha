# Social media variants placeholder

This directory holds platform-sized variants of the press kit. Binary assets are not committed to
git. This file is the size matrix and the rendering instructions.

## Size matrix

| Filename | Platform | Size | Aspect | Notes |
|---|---|---|---|---|
| `og-1200x630.png` | Open Graph (Twitter/X, Facebook, Discord, Slack unfurl) | 1200×630 | 1.91:1 | Most important; this is the link-preview image. |
| `twitter-banner-1500x500.png` | Twitter/X profile banner | 1500×500 | 3:1 | Project handle banner. |
| `instagram-1080x1080.png` | Instagram square post | 1080×1080 | 1:1 | One per screenshot stage; carousel. |
| `instagram-1080x1350.png` | Instagram portrait post | 1080×1350 | 4:5 | One-pager render as a single tall image. |
| `linkedin-1200x627.png` | LinkedIn share | 1200×627 | 1.91:1 | Same content as OG; resized. |

## Content guidance per variant

- **OG / LinkedIn:** screenshot 2 (the click-correction one — most visually distinctive). Overlay
  text: "Authoring sign-language vocabulary with Deaf community oversight." Bottom strip:
  `kozha-translate.com`.
- **Twitter banner:** the project name + tagline + `deaf-feedback@kozha.dev` contact. No
  screenshot (banners are too narrow for screenshots to read).
- **Instagram square:** carousel of all three screenshots, one per slide. Slide 4: the one-pager
  governance section as a static text card.
- **Instagram portrait:** a vertical render of the one-pager. Use the same text; reflow to
  portrait.

## Rendering script (placeholder)

Once the screenshots exist (see `../screenshots/PLACEHOLDER.md`), a small ImageMagick / Pillow
script in `scripts/render_social.py` will produce all variants. The script does not yet exist;
it's a closeout-week deliverable.

```bash
# (planned, not yet implemented)
python scripts/render_social.py \
    --screenshots docs/chat2hamnosys/20-press/screenshots/ \
    --out docs/chat2hamnosys/20-press/social/
```

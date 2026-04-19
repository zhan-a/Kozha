# Screenshots placeholder

This directory holds three production screenshots of the chat2hamnosys authoring UI. Binary
assets (PNG/JPG) are not committed to git; they are produced on-demand and served from a CDN
bucket. This file documents what should live here.

## Required screenshots

### 1. `1-describe.png`

**Stage:** initial prose entered, system has parsed it and asked one clarification question.

**What's visible:**
- Authoring UI open at `http://localhost:8000/chat2hamnosys/`.
- Sign-language picker showing "BSL".
- Prose input field with the ELECTRON description filled in.
- A clarification question card on the right: "How small is the circular movement?"
  with the three multiple-choice options.

**Recommended size:** 1920×1080, 16:9.

### 2. `2-correct.png`

**Stage:** avatar has rendered the first generation; user is mid-correction with the click-target
overlay active.

**What's visible:**
- Avatar panel showing the partially-rendered sign frozen on the click-target frame.
- The clicked body region (e.g. the dominant hand) highlighted with a coloured outline.
- Correction text input: "the handshape should be the round-O, not the F".
- Original generated HamNoSys visible in a side panel.

**Recommended size:** 1920×1080, 16:9.

### 3. `3-review.png`

**Stage:** sign has been accepted and now appears in the reviewer console.

**What's visible:**
- Reviewer console at `http://localhost:8000/chat2hamnosys/review/`.
- ELECTRON entry highlighted in the queue, status pill: "pending_review".
- Right-pane detail showing parameters, HamNoSys string, prose description, reviewer history
  (empty so far).
- Action bar with Approve / Reject / Request revision / Flag buttons.
- Fatigue meter in top-right.

**Recommended size:** 1920×1080, 16:9.

## How to produce them

Follow the procedure in [README.md § Production notes](../README.md#production-notes). Save each
PNG with the exact filename above so the one-pager and the social variants can reference them.

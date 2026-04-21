# Design principles — the contribute redesign

Authored: 2026-04-20. This document is the design contract for every
subsequent prompt in the `contribute-redesign` series. Every later
document, schema, and implementation step must be consistent with
what is written here. When a later prompt asks for something this
document rules out, the correct response is to refuse the addition
and cite this file by path and section.

The audit in [00-current-state.md](00-current-state.md) describes
what exists today. This document describes the rules the redesign
will hold itself to. It is intentionally short on _what_ to build
and long on _what not_ to build.

---

## Primary goal

A fluent signer who does not know HamNoSys can sit down at the
contribute page and, within ninety seconds of first opening it,
submit a single sign for review. That is the benchmark. Every
interaction that adds friction to that ninety-second path must
justify itself or be removed. The page does not need to teach
HamNoSys; it needs to get the signer's intent into the database so
that someone else — the generator, the reviewer, the library — can
do the rest.

There are no login walls in that ninety seconds. There is no
tutorial that must be dismissed. There is no preamble explaining
what HamNoSys is or why Bridgn exists. The signer who arrives
already has a sign in mind and a language they speak, and the
page's first duty is to take their word for it. Any copy that
explains sign-notation theory is deferred behind a collapsed
"what is HamNoSys?" affordance the signer can open if they become
curious — it is never the first thing they see.

The first frame visible on the page must answer three questions,
in this order: which sign language am I contributing to, what am
I doing on this screen, and what happens after I click submit. If
a first-time visitor cannot answer all three from a single glance
before scrolling, the page has failed its primary goal.

---

## The one-screen rule

The entire contribution flow fits on one page with progressive
disclosure. There are no multi-step wizards, no "Next" buttons
advancing through numbered steps, no modals that stack, no routes
the user navigates between during authoring. The single page is
both the entry point and the full authoring surface: the same URL
that welcomes a first-time visitor is the URL that holds their
chat, their preview, their notation readout, and their submit
control. This is a deliberate departure from today's
`contribute.html` → `register` → redirect-to-`/chat2hamnosys/`
shape surfaced in the audit ([§1](00-current-state.md)).

"One screen" does not mean fixed viewport. The page is allowed to
grow vertically as the contributor progresses: chat messages
append, the preview pane materialises once the first generation
returns, the HamNoSys and SiGML readouts appear and expand, the
submit region becomes visible when a draft is ready. But the
contributor never feels they have left a step or moved to a new
screen — the motion of the page is additive, not navigational.

Progressive disclosure means the affordances for later stages do
not clutter the early frame. A contributor who has just typed a
gloss should not see Export / Library / Reviewer controls; those
belong later in the same flow. But disclosure is driven by the
contributor's actual progress, not by an imposed wizard sequence:
if they want to edit the description after a preview has rendered,
the description field remains editable without any "Back" ritual.

---

## Explicit language state

The currently selected sign language is visible at all times in a
prominent header element — large text, not a tooltip, not a tab
badge, not a setting buried behind a gear icon. The contributor
can change it with a single click; the selector lives in the same
header as the label itself. Every piece of generated notation the
page displays — the HamNoSys string, the SiGML XML, the preview
avatar caption, the submit-confirmation copy — carries the
language as an explicit label. There is no place in the UI where
"a sign" is shown without "a sign in _\<language\>_" being equally
legible.

If the contributor switches language mid-session, the current
draft is cleared with an explicit confirmation prompt before the
switch completes. Silent data loss is a worse failure than a
minor interruption, so the confirmation makes the
clear-and-restart decision the contributor's, not the system's.
After the switch, the chat, preview, and notation readouts reset
to the empty state, the header reflects the new language, and the
URL is updated so the state is bookmarkable.

This principle exists because the backend's language enum is a
hard `Literal["bsl", "asl", "dgs"]` in three separate models
([§4](00-current-state.md)) — an unknown value 422s at Pydantic
validation. A mismatch between the header's selected language and
the session's declared language is exactly the failure class this
rule is written to prevent.

---

## Minimalist surface

Remove, if present in the current page or in any proposed mock:
hero images, stock illustrations, marketing copy about "helping
the Deaf community," statistics counters, social-share buttons,
testimonials, animated gradients, decorative icons on every
button. The current `contribute.html` carries several of these
today — the `page-hero` section, the three-card "why" block, the
four-step graphic, the FAQ — and the redesign collapses that
rhetorical scaffold into a single working surface.

Keep: one sentence explaining what the page is for, the language
selector, the gloss input, the description input, the chat panel,
the preview panel, the HamNoSys/SiGML readout, the submit button,
and a link to governance. Nothing else is permitted above the
fold. Below the fold, a short collapsible "how review works" and
a governance link are acceptable; neither may auto-expand on load.

The rhetorical weight of the page sits in the notation and the
avatar, not in the chrome. A contributor who arrives without
context should learn what Bridgn does by using it for thirty
seconds, not by reading a hero banner that tells them.

---

## Notation as the hero, not a footnote

HamNoSys and SiGML are the output of the work done on this page.
They are what the contributor and the reviewer will argue over.
The redesign treats them as the primary visible artefact of every
session. The HamNoSys string sits in a dedicated readout panel in
the central column, rendered at a large type size (target: 24px
minimum for the notation glyphs themselves, surrounding chrome at
standard body scale). It is selectable, copyable via a single
button, and — where the glyph set and font stack permit —
syntax-coloured so that handshape, location, movement, and
non-manual features render in distinct hues.

The SiGML readout sits below, collapsible but not collapsed by
default. Its XML is line-broken per `<hamnosys_manual>` and
`<hamnosys_nonmanual>` group, with inner tags indented
conventionally. A contributor reading the two readouts side by
side should be able to see, without clicking anything, which
HamNoSys symbols correspond to which SiGML elements — the visual
layout cross-references them. Legibility here is not cosmetic; it
is load-bearing, because the contributor's ability to spot a
wrong movement symbol depends on being able to find it without
effort.

"Hero" is not a claim about marketing-page prominence. It is a
claim about where the user's eye lands when they arrive at a
completed draft. The avatar is secondary to the notation, not the
other way around, because the notation is what gets committed to
the library and what the reviewer has authority over.

---

## Real, not pretend

Every submit-button click ends with a sign entry written to
storage with `status = "draft"` (if no reviewer is yet configured
for the chosen language) or `status = "pending_review"` (once a
Deaf reviewer has been assigned for that language). The
contributor sees a permanent URL for their submission on the
success screen — a URL that survives browser restarts, can be
pasted to a reviewer out of band, and renders the same
submission's status and notation when returned to days later. Any
copy on the page that says "a real system is coming soon" or "this
is a demo" is forbidden.

Permanence implies the storage path is real. The audit notes that
no authored-sign outputs exist on disk today — `data/chat2hamnosys/`,
`data/hamnosys_*_authored.sigml`, and `data/authored_signs/` are
all absent ([§5](00-current-state.md)). The redesign will produce
them: a draft row in the contributor-facing store and, once
enough approvals exist, a downstream export to the
`hamnosys_<lang>_authored.sigml` target the storage layer already
anticipates. The contributor's URL is the stable handle across
both — it does not change when the draft is promoted to an
exported entry.

If the system cannot write the submission — network failure,
storage outage, reviewer roster not yet provisioned for the
chosen language — the failure mode is visible, named, and offers
the contributor either a retry or a local-draft save. It is
never a silent success that loses the work.

---

## Honesty about review

Before the contributor clicks submit, the page shows them, in
plain language: "Your submission will be reviewed by a Deaf signer
of _\<language\>_. Review typically takes _\<X days\>_. You can
track status at _\<URL\>_." The _\<X days\>_ figure is computed
from actual historical review latency for that language. In the
initial period before such data exists it is a conservative
published SLA, not a marketing estimate, and it updates as real
review data accumulates.

If no reviewer is yet configured for the chosen language, the
page says so plainly: "No Deaf reviewer has been assigned for
_\<language\>_ yet. Your submission will be saved as a draft and
queued; when a reviewer is onboarded, your submission enters their
queue." The contributor is offered the option to save the draft
anyway and is given the status URL. There is no version of this
flow where the absence of a reviewer silently blocks submit, and
no version where the system pretends a review will happen when
the pipeline for it is not yet in place.

The audit surfaces two gaps this principle responds to: the FAQ
advertises eight sign languages but the backend accepts only
three ([§7.1](00-current-state.md)), and `contribute.html`
describes a Deaf-reviewed flow but has no contributor-visible
reviewer-onboarding path ([§7.3](00-current-state.md)). The
redesign cannot close the reviewer-onboarding gap on its own, but
it can refuse to mis-state the status to the contributor.

---

## Accessibility is load-bearing, not decorative

Every interactive element on the page is operable by keyboard
alone, in an order a sighted keyboard user can predict and a
screen-reader user can navigate by landmark. Every interactive
element carries an ARIA label that has been verified against a
real screen reader. VoiceOver on macOS is the project baseline;
NVDA on Windows is the secondary target. "Verified" means the
label has been read aloud during development and the resulting
utterance is intelligible to someone who has never seen the
page — not merely that `aria-label=""` is present in the DOM.

Video and avatar content has captions. The SiGML avatar preview
renders with a text track that describes what the avatar is
doing, in the chosen sign language's local written language where
available and in English as a fallback. The preview does not rely
on audio for any information: a contributor who is themselves
Deaf or Hard-of-Hearing — the project's primary constituency —
must be able to use the page end-to-end without hearing anything.

Text contrast meets WCAG AA at every size. The notation readout,
because of its primacy, is tested at AAA. Interactive focus rings
are visible in both light and dark modes and are not overridden
by `outline: none` anywhere in the stylesheet. These are not
polish items deferred to post-launch; they are prerequisites for
calling the redesign done.

---

## What is explicitly out of scope

This redesign will not add login, social features, contributor
profiles, gamification, points, leaderboards, achievements,
embedded tutorials, live chat with reviewers, or any notification
system beyond the status URL. If a future prompt in this series
asks for any of these — or for adornments that would reintroduce
the marketing-page shape this document is specifically rejecting —
the correct response is to decline the prompt and cite this
section by path (`docs/contribute-redesign/01-design-principles.md`,
§ "What is explicitly out of scope"). Scope creep is how
minimalist designs die; the contract exists so the series can hold
to its shape across fourteen prompts.

---

## Non-goals

Fifteen specific things this redesign will not do. This list is
the brake pedal for later prompts.

1. **No accounts.** No account creation, sign-in, password, or
   password-reset flow of any kind. The contributor identity
   model is a browser-local bearer token issued on first
   submission, consistent with the existing pre-launch
   scaffolding.
2. **No OAuth.** No Google, Apple, GitHub, Microsoft, or other
   third-party sign-in buttons.
3. **No social share buttons.** No Twitter/X, Facebook, LinkedIn,
   Bluesky, Mastodon, Threads, WhatsApp, Reddit, or similar
   share affordances anywhere on the contribute surface.
4. **No contributor profiles.** No public or private profile
   pages, vanity URLs, "@handle" identifiers, or contributor
   directories.
5. **No gamification.** No points, XP, badges, levels, streaks,
   tiers, or progress bars tied to contribution volume.
6. **No leaderboards.** No rankings of contributors by
   submissions, approvals, streaks, or any other metric, public
   or logged-in-only.
7. **No achievement moments.** No "your first sign!" or "100
   signs!" modals, no confetti, no animated celebration states,
   no milestone emails.
8. **No onboarding walkthrough.** No product tour, tutorial
   overlay, coach-marks, or multi-step getting-started wizard.
   Inline context-sensitive help is the only permitted form of
   guidance.
9. **No live chat with reviewers.** No text, voice, or video
   chat between contributor and reviewer inside the product.
   Asynchronous review notes attached to the submission are the
   only channel.
10. **No out-of-band notifications.** No email digests, push
    notifications, SMS, Slack webhooks, or subscription feeds on
    submission status. The status URL is the only channel.
11. **No decorative imagery.** No stock photography, illustrated
    mascots, hand-drawn characters, abstract geometric hero art,
    or other decorative graphics.
12. **No marketing motion.** No animated gradients, parallax
    scrolling, scroll-driven reveal effects, or hero-video
    autoplay.
13. **No testimonials.** No contributor quotes, reviewer
    endorsements, or pull-quote callouts.
14. **No promotional statistics.** No real-time "X signs this
    week" counters, "join N contributors" badges, or similar
    vanity metrics on the contribute page.
15. **No experiments on the critical flow.** No A/B test
    variants, multivariate experiments, or feature flags gating
    the contribute surface. The redesign ships as a single flow.

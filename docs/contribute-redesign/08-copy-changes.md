# Copy audit — decisions against [00-copy-inventory.md](00-copy-inventory.md)

For each entry in the inventory: keep, rewrite, or delete. The rules
from prompt 8 §5:

- No phrases like "join us on our journey," "empower the Deaf
  community," "together we can change the world."
- Concrete over inspirational. "Add a sign. A Deaf reviewer will
  approve it." over "Help us build the largest sign library in
  history."
- Second-person singular. "You describe the sign. We generate a
  draft. A Deaf signer reviews." Not royal "we" about the site itself.

---

## 1. `public/index.html` — landing

Whole file rewritten. The old nav, feature cards, steps, hero stats,
and contribute banner are gone. Only the entries below are decisions
against the inventory; anything not in the inventory was removed as
part of the minimal rebuild (see `08-removed.md`).

### Nav — line 750 / 753

| Before | Decision | After |
|---|---|---|
| `<a href="contribute.html">Contribute</a>` (nav link) | rewrite | `<a href="/contribute.html">Contribute a sign</a>` |
| `<a href="contribute.html" class="btn-ghost">Volunteer</a>` (nav CTA) | delete | — (redundant with the nav link; "Volunteer" is inspirational framing the minimalist design rejects) |

### "Open Source" feature card — line 928-929

| Before | Decision |
|---|---|
| Title: "Open Source" | delete |
| Body: "The full codebase, sign database, and concept maps are on GitHub. Anyone can add signs, fix translations, or extend the pipeline." | delete |

Reason: the whole features grid is gone. The site's open-source
nature is conveyed by the footer's "Open-source research project"
line and by the Credits section, both of which do the same work
without a feature-card gloss.

### "Contribute to the project." banner — lines 961-965

| Before | Decision |
|---|---|
| h2: "Contribute to the project." | delete |
| body: "Bridgn's sign database has limited vocabulary and needs help. If you know sign language, work in accessibility, or can code, you can add signs, improve translations, or fix bugs." | delete |
| CTA: "Get involved" | delete |

Reason: the banner duplicated the nav CTA and used inspirational
framing ("needs help"). The contribute page is linked from the nav
and the hero. That is enough.

### Footer — line 975

| Before | Decision | After |
|---|---|---|
| `<li><a href="contribute.html">Contribute</a></li>` | delete | — (nav already has it; footer is now just the copyright line) |

---

## 2. `public/app.html` — translator app

### Top-nav contribute link — line 604

| Before | Decision | After |
|---|---|---|
| `<a href="contribute.html" class="nav-link-sm">Contribute</a>` | rewrite | `<a href="contribute.html" class="nav-link-sm">Contribute a sign</a>` |

Reason: aligns with the landing page's new phrasing and the prompt's
instruction to make the contribute link read as "Contribute a sign".
No other copy on `app.html` is in-scope per the prompt's "do NOT
redesign this" rule.

---

## 3. `public/contribute.html` — the contribution landing

The pre-redesign file that the inventory describes has been moved to
`public/_deprecated/contribute-pre-redesign.html` (commit 905b218) and
is no longer served. Every inventory entry under §3 is effectively
**deleted** — replaced by the redesigned contribute page authored in
commits 905b218 → 458849d. Those commits have their own, much leaner
copy, and the audit below does not reopen them.

Inventory entries marked **superseded** are left in place on the
deprecated file so the diff between old and new remains legible.

| Inventory entry | Decision |
|---|---|
| `contribute.html:6` `<title>` "Bridgn — Contribute a Sign" | kept (still accurate) |
| `contribute.html:7` meta description "Help build an open-source sign language dictionary. Register in under a minute and author a sign with the chat2hamnosys pipeline." | superseded (new meta description is concrete: "Contribute a single sign to the Bridgn dictionary. Pick a sign language, describe the sign, review the notation, submit for Deaf review.") |
| `contribute.html:8-9` og:title / og:description | superseded |
| `contribute.html:304-313` Nav "Bridgn / Features / How it works / Contribute / Open Translator →" | superseded — new header is just `Bridgn` + `Open translator` back-link |
| `contribute.html:318` eyebrow "Open-source · Community-built" | deleted — inspirational badge, no function |
| `contribute.html:320` h1 "Help grow the / sign dictionary." | superseded — new h1 is "Contribute a sign" (concrete, second-person) |
| `contribute.html:321-326` hero paragraph | deleted — the concrete form is the fields themselves, not prose about them |
| `contribute.html:328` "Start contributing →" | deleted — page now starts with the language picker, no redundant CTA |
| `contribute.html:331` "View on GitHub" | deleted — governance link in footer points to the same story; this button was marketing surface |
| `contribute.html:337-367` "Why your work matters" three-card block | deleted — rhetorical scaffolding, cut by the minimalism rule |
| `contribute.html:370-400` "Four steps to a published sign." | deleted — the page itself is the walkthrough; four-step diagrams duplicate the flow |
| `contribute.html:403-469` Register form + captcha + BYO-key + fineprint | superseded — new page uses a browser-local bearer token model (no registration form) per §4 of the design principles |
| `contribute.html:472-497` FAQ | deleted — the language list overstates coverage; "Deaf signer / anyone can contribute / API payment" all folded into the honest-review copy on the new contribute page |
| `contribute.html:500-509` Footer (Translator / Contribute / Features / GitHub / "© 2025-2026 Bridgn. Open-source research project.") | superseded — new footer is Governance / Privacy / Contact |

### JS-emitted UI messages — lines 513-688

All superseded by the new authoring flow's error surfaces
(`contribute.js`, `contribute-chat.js`, `contribute-context.js`,
`contribute-notation.js`). No registration form exists any more, so
none of these strings are reachable.

---

## 4. `public/chat2hamnosys/index.html` — authoring UI

Out of scope for this prompt. No copy changes. The authoring UI is
still reachable at `/chat2hamnosys/` for contributors who resume a
session started on that surface, but new sessions flow through the
redesigned `/contribute.html` page instead. The prompt's scope is
visual/copy surface on landing + app + extension, not the old
authoring UI.

---

## 5. `public/chat2hamnosys/review/index.html` — reviewer console

Out of scope for this prompt. No copy changes. Reviewer copy is for
a reviewer audience, not the contributor-facing surface this prompt
audits.

---

## 6. Observations on copy — applied decisions

The inventory noted four issues. Each is addressed on this pass:

1. **Brand inconsistency ("Bridgn" vs. "Kozha").** Not resolved on
   this pass. The landing page, app, and new contribute page all
   continue to use "Bridgn"; the extension popup keeps "Kozha" at
   the top but now links to `kozha-translate.com` in its new
   footer link, which at least names the same domain. The Chrome
   extension manifest continues to call the extension "Kozha —
   Sign Language Translator." A full rebrand is a separate
   decision and out of scope here.
2. **"Open-source" framing without named licence.** Addressed by the
   new `Credits` section on the landing page, which links to the
   project README where the licences are enumerated with
   names and URLs.
3. **"Under five minutes" / "Four steps".** Both phrases and both
   sections are gone (see above). The new page does not promise a
   duration or a fixed step count.
4. **FAQ overstating API coverage.** The FAQ is deleted entirely,
   so the overstatement is gone. The active language list on the
   contribute page is driven by `contribute-languages.json` against
   the API's actual `Literal["bsl", "asl", "dgs"]` coverage.

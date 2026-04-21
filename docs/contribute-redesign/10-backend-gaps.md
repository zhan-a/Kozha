# Backend gaps — prompt 10 (submission flow, status page)

Authored: 2026-04-21. Drafted while implementing prompt 10 of the
contribute-redesign series
([prompts/prompt-contrib-10.md](../../prompts/prompt-contrib-10.md)).

The submission flow needs to distinguish two post-accept outcomes so the
contributor sees an honest message:

- **`pending_review`** — the sign language has at least one active Deaf
  reviewer registered. The copy on the confirmation view becomes "Your
  draft is now in the review queue for <LANGUAGE>. The typical review
  time is 3 days."
- **`draft`** — no reviewers yet. Copy becomes "Your draft is saved.
  <LANGUAGE> does not yet have Deaf reviewers assigned, so review is on
  hold."

The frontend already reads `AcceptResponse.sign_entry.status` — the
distinction exists in the type, but the write path does not populate it.

---

## Gap 1 — `SignEntryDraft.to_sign_entry()` hardcodes `"draft"`

File: `backend/chat2hamnosys/session/state.py`, line 297.

```python
return SignEntry(
    gloss=self.gloss,
    sign_language=self.sign_language,
    ...
    status="draft",          # ← hardcoded, never consulted
    author=author,
)
```

The draft knows its language but not whether any reviewer exists for
that language, and the orchestrator's `on_accept`
(`backend/chat2hamnosys/session/orchestrator.py:650`) has no access to
`ReviewerStore`. Every accepted sign is therefore persisted as `"draft"`
regardless of reviewer roster.

Downstream, `review/actions.py::_push_to_pending` promotes to
`"pending_review"` on the first reviewer action, so the backend is
eventually consistent — but the contributor's confirmation view needs
the correct status at the moment of accept, not later.

---

## Gap 2 — `POST /sessions/{id}/accept` does not check the reviewer roster

File: `backend/chat2hamnosys/api/router.py`, `post_accept` at line 740.

Before this prompt the router function was:

```python
session = on_accept(session, store=sign_store)
session_store.save(session)
...
entry = sign_store.get(accepted[-1].sign_entry_id)
return AcceptResponse(sign_entry=_sign_entry_out(entry), session=_envelope(session, request))
```

No reviewer-roster check anywhere, so the `sign_entry.status` field
returned to the frontend is always `"draft"`. The frontend cannot show
the pending_review copy path honestly — hence this document and the
patch that accompanies it.

---

## Gap 3 — no public read path for submission status

The status page at `/contribute/status/<session_id>` needs to render
without the session token for the public view (gloss, language, final
status, validated HamNoSys/SiGML) and with the token for the private
view (description prose, reviewer comments). The existing
`GET /sessions/{id}` returns 403 on a missing or invalid token, which
rules it out for the unauthenticated case.

---

## Smallest change that closes these gaps

Prompt 10 applies the minimal patch inline — a preview of what prompt
15 of the earlier sequence should formalise.

1. **Inject the reviewer store into `post_accept`.** The
   `get_reviewer_store` dependency already exists in
   `backend/chat2hamnosys/review/dependencies.py`; add it as a `Depends`
   to the accept endpoint.

2. **Promote `draft` → `pending_review` when a competent reviewer
   exists.** Immediately after `sign_store.get(...)`, iterate the
   active-reviewer list and call `Reviewer.can_review(...)` against the
   entry's `sign_language` + `regional_variant`. First match wins:
   set `entry.status = "pending_review"` and persist.

   ```python
   if entry.status == "draft":
       try:
           for reviewer in reviewer_store.list(only_active=True):
               if reviewer.can_review(entry.sign_language, entry.regional_variant):
                   entry.status = "pending_review"
                   sign_store.put(entry)
                   break
       except Exception:
           # Reviewer store unavailable — leave as draft.
           # The eventual-consistency promotion in review/actions
           # still runs on the first reviewer action.
           pass
   ```

   Kept defensive because the reviewer DB is optional in dev and the
   accept path must not hard-fail if it's missing.

3. **Add `GET /sessions/{id}/status` — a token-optional read path.**
   Returns the public fields always; folds in the private fields
   (`description_prose`, reviewer comments) only when the
   `X-Session-Token` header verifies against the session. Unknown /
   un-submitted session ids return 404. Response shape is a dedicated
   `StatusResponse` to keep the privacy envelope explicit.

The three changes stay inside
`backend/chat2hamnosys/api/router.py` and `backend/chat2hamnosys/api/models.py`;
the orchestrator, the draft model, and the review action module do not
need to change. If prompt 15 later wants the promotion inside
`on_accept` itself so the `AcceptedEvent.status` history field carries
the real status, the right move is to thread a `reviewer_lookup`
callable into `on_accept` and call it once. This prompt avoids that
widening so the surface area of the minimal change is one endpoint
body.

---

## What this does NOT fix

- **`AcceptedEvent.status` stays `"draft"`** in the session history
  because the event is appended inside `on_accept` before the router
  promotes the entry. The `SignEntry.status` on disk is correct; the
  session history event is a frozen snapshot of the accept moment.
  Acceptable because the history is an audit trail, not the source of
  truth for the current status.

- **No `under_review` state** is tracked. The prompt mentions
  `under_review` "if a reviewer has opened it — prompt 15 could expose
  this." Today the review system has no separate "opened but not yet
  decided" state; `pending_review` covers both. The status page treats
  them identically until the review system grows that field.

- **Public vs. private reviewer comments** — the review system stores
  one `comment` field per `ReviewRecord` and does not yet distinguish
  audiences. Prompt 10 follows the prompt's guidance: treat every
  comment as private (token-gated) except the rejection category,
  which is always public.

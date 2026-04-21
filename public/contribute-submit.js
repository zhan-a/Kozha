/* Contribute page — submission flow and confirmation view (prompt 10).
 *
 * Sits under the notation panel. Subscribes to
 * window.KOZHA_CONTRIB_CONTEXT and:
 *
 *   - Keeps a diagnostic checklist in sync with the session envelope.
 *     Only "sign generated and valid" is an actual gate on submission;
 *     the rest are visible status indicators.
 *   - Enables "Submit for review" when the session is RENDERED and the
 *     generated HamNoSys is non-empty with no pending generation errors.
 *   - On Submit, POSTs /accept and switches to the confirmation view
 *     (permanent /contribute/status/<id> URL + "what next" links).
 *   - "Save draft and leave" is a no-op on the server: the session is
 *     already saved. Clearing the in-memory session lets the user walk
 *     away; sessionStorage keeps the id + token so the existing
 *     deep-link resume flow still works from the #s/<id> fragment.
 *   - "Discard this draft" reuses the existing modal from
 *     window.KOZHA_CONTRIB.showModal.
 *
 * The confirmation view hides #authoring-root and reveals
 * #submissionConfirmation. The language masthead and context strip
 * remain visible so the submission reads as a continuation rather than
 * a separate screen.
 */
(function () {
  'use strict';

  var CTX = window.KOZHA_CONTRIB_CONTEXT;
  if (!CTX) {
    if (window.console) console.error('[contribute-submit] contribute-context.js must load first');
    return;
  }

  // i18n bridge — resolves from the live catalog with the English original
  // as the fallback. Written so the confirmation view is still readable if
  // /strings.en.json fails to load.
  function tr(key, fallback, vars) {
    if (window.KOZHA_I18N && typeof window.KOZHA_I18N.t === 'function') {
      var v = window.KOZHA_I18N.t(key, vars || undefined);
      if (v && v !== key) return v;
    }
    // Tiny interpolator so the fallback supports the same {{placeholder}}
    // syntax as the catalog — keeps call sites uniform.
    if (vars) {
      return String(fallback).replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, function (_m, name) {
        return Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : '{{' + name + '}}';
      });
    }
    return fallback;
  }

  var API_BASE = '/api/chat2hamnosys';
  var STATUS_PATH = '/contribute/status/';
  var COPY_CONFIRM_MS = 1600;

  // The backend can leave an entry as "draft" when no reviewer covers
  // the language; otherwise /accept promotes it to "pending_review".
  var STATUS_PENDING = 'pending_review';
  var STATUS_DRAFT   = 'draft';

  var els = {
    authoringRoot:       document.getElementById('authoring-root'),
    panel:               document.getElementById('submissionPanel'),
    checklist:           document.getElementById('submissionChecklist'),
    glossHint:           document.getElementById('checkGlossHint'),
    languageHint:        document.getElementById('checkLanguageHint'),
    submitBtn:           document.getElementById('submissionSubmitBtn'),
    saveBtn:             document.getElementById('submissionSaveBtn'),
    discardBtn:          document.getElementById('submissionDiscardBtn'),
    error:               document.getElementById('submissionError'),
    confirmation:        document.getElementById('submissionConfirmation'),
    confirmHeading:      document.getElementById('confirmationHeading'),
    confirmBody:         document.getElementById('confirmationBody'),
    confirmUrl:          document.getElementById('confirmationUrl'),
    confirmCopyBtn:      document.getElementById('confirmationCopyBtn'),
    confirmCopyConfirm:  document.getElementById('confirmationCopyConfirm'),
    confirmAnotherBtn:   document.getElementById('confirmationAnotherBtn'),
    confirmChangeLangBtn: document.getElementById('confirmationChangeLangBtn'),
  };

  // A subscription-or-noop sentinel: returns true when every required
  // element resolved. Prompt 10 is additive; on old pages with no
  // submission markup, bail quietly so the rest of the page still works.
  function allPresent() {
    for (var k in els) {
      if (!Object.prototype.hasOwnProperty.call(els, k)) continue;
      if (!els[k]) return false;
    }
    return true;
  }

  if (!allPresent()) {
    if (window.console) {
      console.warn('[contribute-submit] submission markup missing; skipping wire-up');
    }
    return;
  }

  // Most-recent snapshot kept in module scope so click handlers can read
  // it without racing the subscriber callback.
  var lastSnap = CTX.getState();
  // When true, we've switched to the confirmation view and the panel
  // must stay hidden regardless of future subscribe() callbacks until
  // the user opts back in via "Contribute another sign".
  var confirmationShown = false;
  var submitInFlight = false;
  var copyConfirmTimer = null;

  // ---------- checklist ----------

  function itemByKey(key) {
    return els.checklist.querySelector('.submission-item[data-key="' + key + '"]');
  }

  function setItem(key, isComplete, hintText) {
    var li = itemByKey(key);
    if (!li) return;
    li.classList.toggle('is-complete', !!isComplete);
    li.classList.toggle('is-missing',  !isComplete);
    var mark = li.querySelector('.submission-mark');
    if (mark) {
      mark.setAttribute('aria-label', isComplete ? 'complete' : 'pending');
    }
    if (key === 'gloss' && els.glossHint) {
      els.glossHint.textContent = hintText || '';
    } else if (key === 'language' && els.languageHint) {
      els.languageHint.textContent = hintText || '';
    }
  }

  // Description completion is a heuristic — there's no backend signal
  // that says "the description section is done", but if the session
  // entered clarifying, rendered, or a later state, the parser got
  // enough of it to proceed. Matching against the orchestrator's state
  // machine keeps this in sync with what the backend calls "enough".
  var DESCRIPTION_DONE_STATES = {
    clarifying: true,
    generating: true,
    rendered: true,
    awaiting_correction: true,
    applying_correction: true,
    finalized: true,
  };

  function isSignValid(snap) {
    if (snap.sessionState !== 'rendered') return false;
    if (!snap.hamnosys) return false;
    if (snap.generationErrors && snap.generationErrors.length) return false;
    return true;
  }

  function refreshChecklist(snap) {
    var hasSession = !!snap.sessionId;

    setItem('gloss',
      !!snap.gloss,
      hasSession && !snap.gloss ? 'gloss not set yet' : '');

    setItem('language',
      !!snap.language,
      hasSession && !snap.language ? 'language not set' : '');

    setItem('description',
      hasSession && !!DESCRIPTION_DONE_STATES[snap.sessionState]);

    setItem('sign', isSignValid(snap));

    // The self-ID is optional; tick it whether true or false, but only
    // once the author has actually answered (null means "not stated").
    setItem('deafNative', typeof snap.authorIsDeafNative === 'boolean');

    setItem('correction',
      typeof snap.correctionsCount === 'number' && snap.correctionsCount > 0);
  }

  function refreshSubmitEnabled(snap) {
    var ready = isSignValid(snap);
    els.submitBtn.disabled = !ready || submitInFlight;
    els.saveBtn.disabled   = !snap.sessionId || submitInFlight;
    els.discardBtn.disabled = !snap.sessionId || submitInFlight;
  }

  function setPanelVisibility(snap) {
    if (confirmationShown) {
      els.panel.hidden = true;
      return;
    }
    // Once a session exists the panel is visible — the author can see
    // what they still need to do (gate + optional rows) before Submit
    // enables.
    els.panel.hidden = !snap.sessionId;
  }

  function onStateChange(snap) {
    lastSnap = snap;
    setPanelVisibility(snap);
    refreshChecklist(snap);
    refreshSubmitEnabled(snap);
  }

  // ---------- submit ----------

  function showError(text) {
    if (!text) {
      els.error.hidden = true;
      els.error.textContent = '';
      return;
    }
    els.error.textContent = text;
    els.error.hidden = false;
  }

  function onSubmitClick() {
    if (submitInFlight) return;
    var snap = CTX.getState();
    if (!snap.sessionId || !snap.sessionToken) {
      showError(tr('contribute.submission.submit_error_no_session', 'This draft does not have an active session. Refresh and try again.'));
      return;
    }
    if (!isSignValid(snap)) {
      showError(tr('contribute.submission.submit_error_not_ready', 'The generated sign needs to be ready before you can submit.'));
      return;
    }

    submitInFlight = true;
    showError('');
    refreshSubmitEnabled(snap);

    fetch(API_BASE + '/sessions/' + encodeURIComponent(snap.sessionId) + '/accept', {
      method: 'POST',
      headers: {
        'Accept':          'application/json',
        'Content-Type':    'application/json',
        'X-Session-Token': snap.sessionToken,
      },
      body: JSON.stringify({}),
    })
      .then(function (resp) {
        return resp.text().then(function (body) {
          var parsed = null;
          try { parsed = body ? JSON.parse(body) : null; } catch (_e) { /* ignore */ }
          return { ok: resp.ok, status: resp.status, body: parsed, raw: body };
        });
      })
      .then(function (r) {
        if (!r.ok) {
          var msg = tr('contribute.submission.submit_error_http_prefix', 'Could not submit (HTTP {{status}}).', { status: r.status });
          if (r.body && r.body.detail) msg += ' ' + r.body.detail;
          throw new Error(msg);
        }
        showConfirmation(r.body, snap);
      })
      .catch(function (err) {
        showError((err && err.message) || tr('contribute.submission.submit_error_generic', 'Could not submit this draft.'));
      })
      .then(function () {
        submitInFlight = false;
        refreshSubmitEnabled(CTX.getState());
      });
  }

  // ---------- confirmation view ----------

  function permanentUrlFor(sessionId) {
    var loc = window.location;
    return loc.origin + STATUS_PATH + encodeURIComponent(sessionId);
  }

  function languageLabelFor(code) {
    if (!code) return '';
    var upper = String(code).toUpperCase();
    // Prompt 3 / 4 render a full language name via a /contribute-languages
    // manifest. At submit time we can't block on that fetch, so fall back
    // to the upper-cased code (e.g. "BSL") — it matches the language
    // masthead's code column and reads fine in the confirmation copy.
    return upper;
  }

  function showConfirmation(payload, snap) {
    confirmationShown = true;
    // The orchestrator returns { sign_entry, session }. The status
    // promotion from prompt 10's backend patch flips the entry to
    // "pending_review" when a reviewer covers the language; otherwise
    // it stays "draft". Either outcome is a success — the copy just
    // differs.
    var entry = (payload && payload.sign_entry) || {};
    var status = entry.status || STATUS_DRAFT;
    var session = (payload && payload.session) || null;
    if (session && session.state) {
      CTX.setState({ sessionState: session.state });
    }

    var gloss = snap.gloss || entry.gloss || 'this sign';
    var lang  = languageLabelFor(snap.language || entry.sign_language);
    els.confirmHeading.textContent = tr(
      'contribute.confirmation.heading_template',
      'Submitted: {{gloss}} in {{language}}',
      { gloss: gloss, language: lang }
    );

    if (status === STATUS_PENDING) {
      els.confirmBody.textContent = tr(
        'contribute.confirmation.body_pending',
        'Your draft is now in the review queue for {{language}}. The typical review time is 3 days. You can return to this URL any time to check status.',
        { language: lang }
      );
    } else {
      els.confirmBody.textContent = tr(
        'contribute.confirmation.body_draft',
        'Your draft is saved. {{language}} does not yet have Deaf reviewers assigned, so review is on hold. If a reviewer is added, we will update the status here. You can return to this URL any time.',
        { language: lang }
      );
    }

    els.confirmUrl.value = permanentUrlFor(snap.sessionId);
    els.confirmCopyConfirm.hidden = true;
    els.confirmCopyConfirm.classList.remove('is-fading');

    els.authoringRoot.hidden = true;
    els.confirmation.hidden = false;

    // Focus the heading so screen-reader users hear the outcome without
    // manual navigation. h2 is not focusable by default — a tabindex of
    // -1 lets us focus it programmatically.
    try {
      els.confirmHeading.setAttribute('tabindex', '-1');
      els.confirmHeading.focus();
    } catch (_e) { /* ignore */ }
  }

  function onCopyUrlClick() {
    var url = els.confirmUrl.value || '';
    if (!url) return;

    function onOk() {
      if (copyConfirmTimer) clearTimeout(copyConfirmTimer);
      els.confirmCopyConfirm.classList.remove('is-fading');
      els.confirmCopyConfirm.hidden = false;
      copyConfirmTimer = setTimeout(function () {
        els.confirmCopyConfirm.classList.add('is-fading');
        setTimeout(function () {
          els.confirmCopyConfirm.hidden = true;
          els.confirmCopyConfirm.classList.remove('is-fading');
        }, 420);
      }, COPY_CONFIRM_MS);
    }
    function onFail() {
      els.confirmCopyConfirm.textContent = tr('contribute.confirmation.copy_fail', 'Could not copy');
      els.confirmCopyConfirm.hidden = false;
    }

    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      navigator.clipboard.writeText(url).then(onOk).catch(onFail);
      return;
    }
    // Legacy fallback — select the input and execCommand('copy').
    try {
      els.confirmUrl.focus();
      els.confirmUrl.select();
      var ok = document.execCommand && document.execCommand('copy');
      if (ok) { onOk(); } else { onFail(); }
    } catch (_e) { onFail(); }
  }

  function hideConfirmation() {
    confirmationShown = false;
    els.confirmation.hidden = true;
    els.authoringRoot.hidden = false;
    els.confirmCopyConfirm.hidden = true;
  }

  function onAnotherClick() {
    // Keeps the selected language in place, clears the session fields
    // so the form mounts fresh. Re-shows the authoring area for the
    // next draft.
    hideConfirmation();
    CTX.clearSession({ reason: 'submitted_another' });
  }

  function onChangeLanguageClick() {
    hideConfirmation();
    if (window.KOZHA_CONTRIB && typeof window.KOZHA_CONTRIB.requestLanguageChange === 'function') {
      window.KOZHA_CONTRIB.requestLanguageChange();
    } else {
      CTX.setState({ language: null });
      CTX.clearSession({ reason: 'change_language' });
    }
  }

  // ---------- save / discard ----------

  function onSaveClick() {
    // Server side there's nothing extra to persist — /answer, /describe
    // and /correct already saved. The session stays resumable via the
    // #s/<id> fragment (sessionStorage keeps id + token).
    var snap = CTX.getState();
    if (!snap.sessionId) return;
    // Navigate back to Bridgn so the user actually leaves; the /
    // landing page is the obvious place.
    window.location.href = '/';
  }

  function onDiscardClick() {
    var snap = CTX.getState();
    if (!snap.sessionId) return;
    var glossLabel = snap.gloss
      ? '“' + snap.gloss + '”'
      : tr('contribute.submission.discard_modal_body_unnamed', 'this draft');

    var doDiscard = function () {
      // Drop the local session and tell the backend — the existing
      // abandonSession helper clears language too, which matches
      // "discard and start over". Keep the confirmation view off.
      confirmationShown = false;
      els.confirmation.hidden = true;
      els.authoringRoot.hidden = false;
      CTX.abandonSession();
    };

    if (window.KOZHA_CONTRIB && typeof window.KOZHA_CONTRIB.showModal === 'function') {
      window.KOZHA_CONTRIB.showModal({
        title: tr('contribute.submission.discard_modal_title', 'Discard this draft?'),
        body: tr(
          'contribute.submission.discard_modal_body',
          'This will permanently discard {{gloss_label}}. Continue?',
          { gloss_label: glossLabel }
        ),
        cancelLabel: tr('contribute.submission.discard_modal_cancel', 'Cancel'),
        confirmLabel: tr('contribute.submission.discard_modal_confirm', 'Discard'),
        onConfirm: doDiscard,
      });
    } else {
      doDiscard();
    }
  }

  // ---------- wire up ----------

  els.submitBtn.addEventListener('click', onSubmitClick);
  els.saveBtn.addEventListener('click', onSaveClick);
  els.discardBtn.addEventListener('click', onDiscardClick);
  els.confirmCopyBtn.addEventListener('click', onCopyUrlClick);
  els.confirmAnotherBtn.addEventListener('click', onAnotherClick);
  els.confirmChangeLangBtn.addEventListener('click', onChangeLanguageClick);

  CTX.subscribe(onStateChange);
  onStateChange(CTX.getState());
})();

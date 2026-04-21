/* Contribute page — UI controller.
 *
 * All state (language, gloss, session id/token, session state) lives in
 * window.KOZHA_CONTRIB_CONTEXT (see contribute-context.js). This file is
 * the view layer: it subscribes to the store and re-renders the sticky
 * language masthead, the context strip, the language picker, the resume
 * prompt, the confirmation modal, and the one-time keyboard hint.
 *
 * Scope for prompt 5 (cumulative):
 *   - prompt 4: language header + context strip, change-language flow,
 *     deep-link resume, keyboard shortcuts, one-time hint.
 *   - prompt 5: gloss + description inputs inside #authoring-root,
 *     inline validation, character counter, autosave per-language draft
 *     in localStorage, Deaf-native self-ID, submit → POST /sessions,
 *     summary card + edit flow, restored-from-last-visit notice.
 *
 * Out of scope (later prompts):
 *   - chat panel               → prompt 6
 *   - HamNoSys / SiGML readout → prompt 7
 *   - submit + status URL      → later
 */
(function () {
  'use strict';

  var CTX = window.KOZHA_CONTRIB_CONTEXT;
  if (!CTX) {
    if (window.console) console.error('[contribute] contribute-context.js must load first');
    return;
  }

  var HINT_SEEN_KEY = 'kozha.contribute.hintSeen';
  var DRAFT_KEY_PREFIX = 'kozha.contribute.draft.';
  var LANGUAGES_URL = '/contribute-languages.json';
  var TOAST_MS = 1800;
  var RESTORED_NOTICE_MS = 5000;
  var AUTOSAVE_DEBOUNCE_MS = 500;
  var DESCRIPTION_MIN = 20;
  var DESCRIPTION_HINT_THRESHOLD = 40;
  var GENERIC_DESCRIPTION_PLACEHOLDER =
    "e.g. describe the handshape, where the hand starts, how it moves, and which way the palm faces.";

  var view = {
    languages: [],
    // Tracks which language the form inputs have been hydrated for so we
    // don't re-populate on every render. Null when the form isn't mounted.
    formMountedForLanguage: null,
    // Guards autosave from firing while we're programmatically populating
    // inputs during hydration.
    suppressAutosave: false,
    // Tracks whether the user has attempted submission — controls whether
    // inline validation messages appear. Reset on every fresh mount.
    submitAttempted: false,
    // The edit flow clears the session so mountForm re-hydrates with the
    // user's own just-edited draft — the restored-notice doesn't belong
    // on that path, so we suppress it for the next mount and reset.
    suppressNextRestoredNotice: false,
  };

  var els = {
    picker:            document.getElementById('languagePicker'),
    pickerEmpty:       document.getElementById('pickerEmpty'),
    pickerOptions:     document.getElementById('pickerOptions'),
    langMasthead:      document.getElementById('langMasthead'),
    badge:             document.getElementById('languageBadge'),
    badgeCode:         document.getElementById('languageBadgeCode'),
    badgeName:         document.getElementById('languageBadgeName'),
    changeBtn:         document.getElementById('changeLanguageBtn'),
    contextGloss:      document.getElementById('contextGloss'),
    contextState:      document.getElementById('contextState'),
    contextSessionId:  document.getElementById('contextSessionId'),
    contextCopyBtn:    document.getElementById('contextCopyBtn'),
    tokenPrompt:       document.getElementById('tokenPrompt'),
    tokenPromptForm:   document.getElementById('tokenPromptForm'),
    tokenPromptInput:  document.getElementById('tokenPromptInput'),
    tokenPromptError:  document.getElementById('tokenPromptError'),
    notice:            document.getElementById('reviewerNotice'),
    authoringRoot:     document.getElementById('authoring-root'),
    authoringForm:     document.getElementById('authoringForm'),
    authoringSummary:  document.getElementById('authoringSummary'),
    summaryGloss:      document.getElementById('summaryGloss'),
    summaryLang:       document.getElementById('summaryLang'),
    summarySep:        document.getElementById('summarySep'),
    summaryDesc:       document.getElementById('summaryDesc'),
    summaryEditBtn:    document.getElementById('summaryEditBtn'),
    glossInput:        document.getElementById('glossInput'),
    glossError:        document.getElementById('glossError'),
    descriptionInput:  document.getElementById('descriptionInput'),
    descriptionHint:   document.getElementById('descriptionHint'),
    descriptionCount:  document.getElementById('descriptionCount'),
    descriptionError:  document.getElementById('descriptionError'),
    deafNativeInput:   document.getElementById('deafNativeInput'),
    startBtn:          document.getElementById('startAuthoringBtn'),
    submitError:       document.getElementById('submitError'),
    restoredNotice:    document.getElementById('restoredNotice'),
    modalBackdrop:     document.getElementById('modalBackdrop'),
    modalTitle:        document.getElementById('modalTitle'),
    modalBody:         document.getElementById('modalBody'),
    modalCancelBtn:    document.getElementById('modalCancelBtn'),
    modalDiscardBtn:   document.getElementById('modalDiscardBtn'),
    hintStrip:         document.getElementById('hintStrip'),
    hintClose:         document.getElementById('hintClose'),
    toast:             document.getElementById('toast'),
  };

  // ---------- languages ----------

  function findLanguage(code) {
    if (!code) return null;
    for (var i = 0; i < view.languages.length; i++) {
      if (view.languages[i].code === code) return view.languages[i];
    }
    return null;
  }

  function renderOptions() {
    els.pickerOptions.innerHTML = '';
    for (var i = 0; i < view.languages.length; i++) {
      var lang = view.languages[i];
      var li = document.createElement('li');
      li.setAttribute('role', 'presentation');

      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'picker-option';
      btn.setAttribute('role', 'option');
      btn.setAttribute('data-code', lang.code);

      var code = document.createElement('span');
      code.className = 'picker-option-code';
      code.textContent = lang.code.toUpperCase();

      var name = document.createElement('span');
      name.className = 'picker-option-name';
      name.textContent = lang.english_name;

      var coverage = document.createElement('span');
      coverage.className = 'picker-option-coverage';
      coverage.textContent = lang.coverage_count + ' signs in corpus';

      btn.appendChild(code);
      btn.appendChild(name);
      btn.appendChild(coverage);
      btn.addEventListener('click', onOptionClick);

      li.appendChild(btn);
      els.pickerOptions.appendChild(li);
    }
  }

  function onOptionClick(ev) {
    var code = ev.currentTarget.getAttribute('data-code');
    setLanguage(code);
  }

  function setLanguage(code) {
    if (code && !findLanguage(code)) return;
    CTX.setState({ language: code || null });
  }

  function getLanguage() {
    return CTX.getState().language;
  }

  // ---------- per-language draft store ----------

  function draftKeyFor(lang) {
    return lang ? DRAFT_KEY_PREFIX + lang : null;
  }

  function readDraft(lang) {
    var key = draftKeyFor(lang);
    if (!key) return null;
    try {
      var raw = localStorage.getItem(key);
      if (!raw) return null;
      var parsed = JSON.parse(raw);
      return {
        gloss:        typeof parsed.gloss === 'string' ? parsed.gloss : '',
        description:  typeof parsed.description === 'string' ? parsed.description : '',
        isDeafNative: typeof parsed.isDeafNative === 'boolean' ? parsed.isDeafNative : null,
      };
    } catch (_e) { return null; }
  }

  function writeDraft(lang, draft) {
    var key = draftKeyFor(lang);
    if (!key) return;
    try {
      localStorage.setItem(key, JSON.stringify({
        gloss:        draft.gloss || '',
        description:  draft.description || '',
        isDeafNative: typeof draft.isDeafNative === 'boolean' ? draft.isDeafNative : null,
      }));
    } catch (_e) { /* storage blocked — stay in-memory only */ }
  }

  function clearDraft(lang) {
    var key = draftKeyFor(lang);
    if (!key) return;
    try { localStorage.removeItem(key); } catch (_e) { /* ignore */ }
  }

  function draftHasContent(draft) {
    if (!draft) return false;
    return !!(draft.gloss && draft.gloss.length) || !!(draft.description && draft.description.length);
  }

  function currentFormDraft() {
    return {
      gloss:        els.glossInput.value || '',
      description:  els.descriptionInput.value || '',
      isDeafNative: !!els.deafNativeInput.checked,
    };
  }

  // ---------- render ----------

  function render(snapshot) {
    // Deep-link resume flow owns the view — leave the DOM alone.
    if (!els.tokenPrompt.hidden) return;

    var lang = findLanguage(snapshot.language);
    if (!lang) {
      els.picker.hidden = false;
      els.pickerEmpty.hidden = false;
      els.langMasthead.hidden = true;
      els.notice.hidden = true;
      els.authoringRoot.hidden = true;
      els.hintStrip.hidden = true;
      hideFormInputs();
      view.formMountedForLanguage = null;
      return;
    }

    els.picker.hidden = true;
    els.langMasthead.hidden = false;

    els.badgeCode.textContent = lang.code.toUpperCase();
    els.badgeName.textContent = lang.english_name;

    if (lang.has_reviewers) {
      els.notice.hidden = true;
      els.notice.innerHTML = '';
    } else {
      els.notice.innerHTML = '';
      var p = document.createElement('p');
      p.textContent =
        'No Deaf reviewers are currently assigned to ' + lang.english_name +
        '. You can save a draft, but it will not be reviewed until a reviewer is added. ' +
        'We will email you if one becomes available.';
      els.notice.appendChild(p);
      els.notice.hidden = false;
    }

    if (snapshot.gloss && snapshot.gloss.length > 0) {
      els.contextGloss.textContent = snapshot.gloss;
      els.contextGloss.classList.remove('is-empty');
    } else {
      els.contextGloss.textContent = 'No sign selected';
      els.contextGloss.classList.add('is-empty');
    }
    els.contextState.textContent = CTX.stateLabel(snapshot.sessionState);

    if (snapshot.sessionId) {
      els.contextSessionId.textContent = CTX.shortId(snapshot.sessionId);
      els.contextSessionId.setAttribute('title', snapshot.sessionId);
      els.contextCopyBtn.hidden = false;
    } else {
      els.contextSessionId.textContent = '—';
      els.contextSessionId.removeAttribute('title');
      els.contextCopyBtn.hidden = true;
    }

    els.authoringRoot.hidden = false;
    if (snapshot.sessionId) {
      // Post-submit: collapse form into the summary card.
      mountSummary(lang, snapshot);
    } else {
      // Pre-submit: show the form, hydrating from localStorage on first
      // mount for this language.
      mountForm(lang);
    }

    maybeShowHint();
  }

  // ---------- form mount + hydration ----------

  function mountForm(lang) {
    els.authoringSummary.hidden = true;
    els.authoringForm.hidden = false;

    // Language-specific example placeholder on the description field.
    els.descriptionInput.setAttribute(
      'placeholder',
      lang.description_placeholder || GENERIC_DESCRIPTION_PLACEHOLDER
    );

    if (view.formMountedForLanguage === lang.code) {
      // Already mounted for this language — nothing to hydrate.
      updateDescriptionMeta();
      updateSubmitEnabled();
      return;
    }

    view.formMountedForLanguage = lang.code;
    view.submitAttempted = false;
    hideInlineError(els.glossError);
    hideInlineError(els.descriptionError);
    hideInlineError(els.submitError);

    var draft = readDraft(lang.code);
    view.suppressAutosave = true;
    try {
      els.glossInput.value = draft ? draft.gloss : '';
      els.descriptionInput.value = draft ? draft.description : '';
      els.deafNativeInput.checked = !!(draft && draft.isDeafNative);
    } finally {
      view.suppressAutosave = false;
    }

    updateDescriptionMeta();
    updateSubmitEnabled();

    if (draftHasContent(draft) && !view.suppressNextRestoredNotice) {
      showRestoredNotice();
    } else {
      hideRestoredNotice();
    }
    view.suppressNextRestoredNotice = false;
  }

  function mountSummary(lang, snapshot) {
    els.authoringForm.hidden = true;
    els.authoringSummary.hidden = false;

    var gloss = snapshot.gloss || (readDraft(lang.code) || {}).gloss || '';
    var draft = readDraft(lang.code);
    var description = draft ? draft.description : '';

    els.summaryGloss.textContent = gloss;
    els.summaryLang.textContent = lang.code.toUpperCase();
    if (description && description.length > 0) {
      els.summaryDesc.textContent = description;
      els.summarySep.hidden = false;
    } else {
      els.summaryDesc.textContent = '';
      els.summarySep.hidden = true;
    }

    view.formMountedForLanguage = null;
  }

  function hideFormInputs() {
    els.authoringForm.hidden = true;
    els.authoringSummary.hidden = true;
    hideRestoredNotice();
  }

  // ---------- inline validation + description meta ----------

  function showInlineError(node, text) {
    if (!node) return;
    node.textContent = text || '';
    node.hidden = false;
  }
  function hideInlineError(node) {
    if (!node) return;
    node.textContent = '';
    node.hidden = true;
  }

  function updateDescriptionMeta() {
    var len = (els.descriptionInput.value || '').length;
    els.descriptionCount.textContent = String(len);
    els.descriptionHint.hidden = len >= DESCRIPTION_HINT_THRESHOLD;
    // Clear existing errors once the user crosses each threshold so the
    // message doesn't linger while they're actively fixing it.
    if (view.submitAttempted) {
      if ((els.glossInput.value || '').trim().length > 0) {
        hideInlineError(els.glossError);
      }
      if (len >= DESCRIPTION_MIN) {
        hideInlineError(els.descriptionError);
      }
    }
  }

  function updateSubmitEnabled() {
    var glossOk = (els.glossInput.value || '').trim().length > 0;
    var descOk = (els.descriptionInput.value || '').trim().length >= DESCRIPTION_MIN;
    els.startBtn.disabled = !(glossOk && descOk);
  }

  // ---------- autosave ----------

  var autosaveTimer = null;
  function scheduleAutosave() {
    if (view.suppressAutosave) return;
    if (autosaveTimer) clearTimeout(autosaveTimer);
    autosaveTimer = setTimeout(function () {
      autosaveTimer = null;
      var lang = getLanguage();
      if (!lang) return;
      writeDraft(lang, currentFormDraft());
    }, AUTOSAVE_DEBOUNCE_MS);
  }

  function flushAutosave() {
    if (autosaveTimer) {
      clearTimeout(autosaveTimer);
      autosaveTimer = null;
    }
    var lang = getLanguage();
    if (!lang) return;
    writeDraft(lang, currentFormDraft());
  }

  // ---------- input handlers ----------

  function onGlossInput() {
    updateSubmitEnabled();
    if (view.submitAttempted && (els.glossInput.value || '').trim().length > 0) {
      hideInlineError(els.glossError);
    }
    scheduleAutosave();
  }

  function onGlossBlur() {
    var trimmed = (els.glossInput.value || '').trim();
    var upper = trimmed.toUpperCase();
    if (els.glossInput.value !== upper) {
      els.glossInput.value = upper;
      scheduleAutosave();
    }
  }

  function onDescriptionInput() {
    updateDescriptionMeta();
    updateSubmitEnabled();
    scheduleAutosave();
  }

  function onDeafNativeChange() { scheduleAutosave(); }

  // ---------- restored notice ----------

  var restoredTimer = null;
  function showRestoredNotice() {
    els.restoredNotice.hidden = false;
    if (restoredTimer) clearTimeout(restoredTimer);
    restoredTimer = setTimeout(hideRestoredNotice, RESTORED_NOTICE_MS);
  }
  function hideRestoredNotice() {
    els.restoredNotice.hidden = true;
    if (restoredTimer) { clearTimeout(restoredTimer); restoredTimer = null; }
  }

  // ---------- submit ----------

  function onSubmit(ev) {
    ev.preventDefault();
    view.submitAttempted = true;

    var gloss = (els.glossInput.value || '').trim().toUpperCase();
    var description = (els.descriptionInput.value || '').trim();

    var glossOk = gloss.length > 0;
    var descOk = description.length >= DESCRIPTION_MIN;

    if (!glossOk) {
      showInlineError(els.glossError, 'Gloss is required before you describe the sign.');
    } else {
      hideInlineError(els.glossError);
    }
    if (!descOk) {
      showInlineError(els.descriptionError, 'Please add at least 20 characters of description.');
    } else {
      hideInlineError(els.descriptionError);
    }
    if (!glossOk || !descOk) {
      (glossOk ? els.descriptionInput : els.glossInput).focus();
      return;
    }

    // Normalise the gloss in the input so autosave captures it uppercased
    // even if the user submits without blurring the field.
    if (els.glossInput.value !== gloss) {
      els.glossInput.value = gloss;
    }

    // Flush any pending autosave first so the draft matches what we send.
    flushAutosave();

    hideInlineError(els.submitError);
    els.startBtn.disabled = true;
    els.startBtn.textContent = 'Starting…';

    CTX.createSession({
      gloss: gloss,
      prose: description,
      authorIsDeafNative: !!els.deafNativeInput.checked,
    }).then(function () {
      // render() will be invoked via the subscribe hook once state
      // changes; the summary card takes it from there.
      els.startBtn.textContent = 'Start authoring';
    }).catch(function (err) {
      els.startBtn.disabled = false;
      els.startBtn.textContent = 'Start authoring';
      // If a session was created but the chained /describe failed, the
      // form is already hidden behind the summary card. Surface the error
      // inside the chat panel instead of the (now invisible) submit error.
      if (CTX.getState().sessionId &&
          window.KOZHA_CONTRIB_CHAT &&
          typeof window.KOZHA_CONTRIB_CHAT.showError === 'function') {
        window.KOZHA_CONTRIB_CHAT.showError();
        if (window.console) console.error('[contribute] describe failed:', err);
        return;
      }
      var msg = 'Could not start a session. Please try again.';
      if (err && err.status === 422) {
        msg = 'This language is not yet enabled on the server. Please pick another.';
      }
      showInlineError(els.submitError, msg);
      if (window.console) console.error('[contribute] createSession failed:', err);
    });
  }

  // ---------- edit (post-submit) ----------

  function onEditSummary() {
    var snap = CTX.getState();
    var glossLabel = snap.gloss ? '“' + snap.gloss + '”' : 'your current sign';
    showModal({
      title: 'Discard this draft?',
      body: 'This will discard your draft for ' + glossLabel + ' so you can edit the gloss and description. Continue?',
      cancelLabel: 'Cancel',
      confirmLabel: 'Discard',
      onConfirm: function () {
        // Reject server-side (best effort) and clear the session fields
        // while keeping the selected language so the form reopens in
        // place with the values the user just typed — no "restored from
        // your last visit" notice on this path.
        view.suppressNextRestoredNotice = true;
        CTX.clearSession({ reason: 'edit_requested' });
      },
    });
  }

  // ---------- change-language flow ----------

  function onChangeClick() { requestLanguageChange(); }

  function requestLanguageChange() {
    var snap = CTX.getState();
    var lang = snap.language;
    var draft = lang ? readDraft(lang) : null;
    var hasDraft = draftHasContent(draft);

    if (!snap.sessionId && !hasDraft) {
      // Nothing to discard — drop straight back to the picker.
      CTX.setState({ language: null, gloss: '', sessionState: 'awaiting_description' });
      CTX.clearSessionFragment();
      return;
    }

    var glossLabel;
    if (snap.gloss) {
      glossLabel = '“' + snap.gloss + '”';
    } else if (draft && draft.gloss) {
      glossLabel = '“' + draft.gloss + '”';
    } else {
      glossLabel = 'your current sign';
    }

    showModal({
      title: 'Discard this draft?',
      body: 'This will discard your draft for ' + glossLabel + '. Continue?',
      cancelLabel: 'Cancel',
      confirmLabel: 'Discard',
      onConfirm: function () {
        if (lang) clearDraft(lang);
        if (snap.sessionId) {
          CTX.abandonSession();
        } else {
          CTX.setState({ language: null, gloss: '', sessionState: 'awaiting_description' });
          CTX.clearSessionFragment();
        }
      },
    });
  }

  // ---------- modal ----------

  var modalLastFocused = null;
  var modalOnConfirm = null;

  function showModal(opts) {
    els.modalTitle.textContent = opts.title || '';
    els.modalBody.textContent  = opts.body  || '';
    els.modalCancelBtn.textContent  = opts.cancelLabel  || 'Cancel';
    els.modalDiscardBtn.textContent = opts.confirmLabel || 'Confirm';
    modalOnConfirm = typeof opts.onConfirm === 'function' ? opts.onConfirm : null;
    modalLastFocused = document.activeElement;
    els.modalBackdrop.hidden = false;
    // Defer focus so assistive tech sees the dialog mounted first.
    setTimeout(function () { els.modalCancelBtn.focus(); }, 0);
    document.addEventListener('keydown', onModalKey);
  }

  function hideModal() {
    els.modalBackdrop.hidden = true;
    modalOnConfirm = null;
    document.removeEventListener('keydown', onModalKey);
    if (modalLastFocused && typeof modalLastFocused.focus === 'function') {
      modalLastFocused.focus();
    }
    modalLastFocused = null;
  }

  function onModalKey(ev) {
    if (ev.key === 'Escape') {
      ev.preventDefault();
      hideModal();
      return;
    }
    if (ev.key !== 'Tab') return;
    // Trap focus between Cancel and Discard.
    var active = document.activeElement;
    if (ev.shiftKey && active === els.modalCancelBtn) {
      ev.preventDefault(); els.modalDiscardBtn.focus();
    } else if (!ev.shiftKey && active === els.modalDiscardBtn) {
      ev.preventDefault(); els.modalCancelBtn.focus();
    }
  }

  function onModalCancel() { hideModal(); }

  function onModalConfirm() {
    var cb = modalOnConfirm;
    hideModal();
    if (typeof cb === 'function') cb();
  }

  function onModalBackdropClick(ev) {
    if (ev.target === els.modalBackdrop) hideModal();
  }

  // ---------- hint strip ----------

  function hintSeen() {
    try { return localStorage.getItem(HINT_SEEN_KEY) === '1'; } catch (_e) { return true; }
  }

  function maybeShowHint() {
    if (hintSeen()) { els.hintStrip.hidden = true; return; }
    els.hintStrip.hidden = false;
  }

  function dismissHint() {
    try { localStorage.setItem(HINT_SEEN_KEY, '1'); } catch (_e) { /* ignore */ }
    els.hintStrip.hidden = true;
  }

  // ---------- keyboard ----------

  function onGlobalKey(ev) {
    var isMod = ev.metaKey || ev.ctrlKey;
    if (!isMod || ev.altKey) return;
    var key = ev.key ? ev.key.toLowerCase() : '';
    if (key === 'k') {
      ev.preventDefault();
      toggleLanguagePicker();
    } else if (key === 'l') {
      ev.preventDefault();
      copySessionUrl();
    }
  }

  function toggleLanguagePicker() {
    var snap = CTX.getState();
    if (!snap.language) {
      // Already on the picker — give keyboard users the first option.
      var first = els.pickerOptions.querySelector('.picker-option');
      if (first && typeof first.focus === 'function') first.focus();
      return;
    }
    requestLanguageChange();
  }

  // ---------- copy session URL ----------

  function copySessionUrl() {
    var url = CTX.sessionUrl();
    var hasSession = !!CTX.getState().sessionId;
    var success = 'Copied ' + (hasSession ? 'session URL' : 'page URL');
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      navigator.clipboard.writeText(url)
        .then(function () { showToast(success); })
        .catch(function () { showToast('Could not copy'); });
      return;
    }
    // Fallback: transient textarea + execCommand.
    try {
      var ta = document.createElement('textarea');
      ta.value = url;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.opacity = '0';
      document.body.appendChild(ta);
      ta.select();
      var ok = document.execCommand && document.execCommand('copy');
      document.body.removeChild(ta);
      showToast(ok ? success : 'Could not copy');
    } catch (_e) {
      showToast('Could not copy');
    }
  }

  var toastTimer = null;
  function showToast(text) {
    els.toast.textContent = text;
    els.toast.hidden = false;
    if (toastTimer) clearTimeout(toastTimer);
    toastTimer = setTimeout(function () {
      els.toast.hidden = true;
      toastTimer = null;
    }, TOAST_MS);
  }

  // ---------- token resume prompt ----------

  var pendingResumeId = null;

  function showTokenPrompt(sessionId) {
    pendingResumeId = sessionId;
    els.picker.hidden = true;
    els.langMasthead.hidden = true;
    els.notice.hidden = true;
    els.authoringRoot.hidden = true;
    els.tokenPrompt.hidden = false;
    els.tokenPromptError.hidden = true;
    els.tokenPromptError.textContent = '';
    els.tokenPromptInput.value = '';
    setTimeout(function () {
      if (els.tokenPromptInput && typeof els.tokenPromptInput.focus === 'function') {
        els.tokenPromptInput.focus();
      }
    }, 0);
  }

  function hideTokenPrompt() {
    els.tokenPrompt.hidden = true;
    pendingResumeId = null;
    render(CTX.getState());
  }

  function onTokenPromptSubmit(ev) {
    ev.preventDefault();
    var token = (els.tokenPromptInput.value || '').trim();
    if (!token || !pendingResumeId) return;
    var id = pendingResumeId;
    els.tokenPromptError.hidden = true;
    els.tokenPromptError.textContent = '';
    CTX.resumeSession(id, token)
      .then(function () { hideTokenPrompt(); })
      .catch(function (err) {
        els.tokenPromptError.hidden = false;
        if (err && err.status === 403) {
          els.tokenPromptError.textContent = 'That token does not match this session.';
        } else if (err && err.status === 404) {
          els.tokenPromptError.textContent = 'This session was not found.';
        } else {
          els.tokenPromptError.textContent = 'Could not resume the session.';
        }
      });
  }

  // ---------- startup ----------

  function loadLanguages() {
    return fetch(LANGUAGES_URL, { headers: { 'Accept': 'application/json' } })
      .then(function (r) {
        if (!r.ok) throw new Error('languages HTTP ' + r.status);
        return r.json();
      })
      .then(function (data) {
        view.languages = (data && data.languages) || [];
      });
  }

  function hydrateFromFragment() {
    var fragId = CTX.parseFragment();
    if (!fragId) return Promise.resolve(false);
    var snap = CTX.getState();
    if (snap.sessionId === fragId && snap.sessionToken) {
      // Same tab, reload or same-origin navigation — refresh the envelope
      // so the context strip reflects server truth on mount.
      return CTX.resumeSession(fragId, snap.sessionToken)
        .then(function () { return true; })
        .catch(function () { showTokenPrompt(fragId); return true; });
    }
    showTokenPrompt(fragId);
    return Promise.resolve(true);
  }

  function init() {
    els.changeBtn.addEventListener('click', onChangeClick);
    els.modalCancelBtn.addEventListener('click', onModalCancel);
    els.modalDiscardBtn.addEventListener('click', onModalConfirm);
    els.modalBackdrop.addEventListener('click', onModalBackdropClick);
    els.hintClose.addEventListener('click', dismissHint);
    els.contextCopyBtn.addEventListener('click', copySessionUrl);
    els.tokenPromptForm.addEventListener('submit', onTokenPromptSubmit);
    els.authoringForm.addEventListener('submit', onSubmit);
    els.glossInput.addEventListener('input', onGlossInput);
    els.glossInput.addEventListener('blur', onGlossBlur);
    els.descriptionInput.addEventListener('input', onDescriptionInput);
    els.deafNativeInput.addEventListener('change', onDeafNativeChange);
    els.summaryEditBtn.addEventListener('click', onEditSummary);
    document.addEventListener('keydown', onGlobalKey);

    CTX.subscribe(render);

    loadLanguages().then(function () {
      renderOptions();
      return hydrateFromFragment();
    }).then(function () {
      render(CTX.getState());
    }).catch(function (err) {
      els.pickerEmpty.innerHTML = '';
      var p = document.createElement('p');
      p.className = 'picker-prompt';
      p.textContent = 'Could not load the language list. Refresh to retry.';
      els.pickerEmpty.appendChild(p);
      if (window.console) console.error('[contribute] language load failed:', err);
    });
  }

  window.KOZHA_CONTRIB = {
    setLanguage:           setLanguage,
    getLanguage:           getLanguage,
    requestLanguageChange: requestLanguageChange,
    copySessionUrl:        copySessionUrl,
    showModal:             showModal,
    hideModal:             hideModal,
    readDraft:             readDraft,
    writeDraft:            writeDraft,
    clearDraft:            clearDraft,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

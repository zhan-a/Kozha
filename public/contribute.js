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

  // Catalog bridge — return the catalog string when available, else the
  // English fallback. Supports {{placeholder}} via either path.
  function tr(key, fallback, vars) {
    if (window.KOZHA_I18N && typeof window.KOZHA_I18N.t === 'function') {
      var v = window.KOZHA_I18N.t(key, vars || undefined);
      if (v && v !== key) return v;
    }
    if (vars) {
      return String(fallback).replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, function (_m, name) {
        return Object.prototype.hasOwnProperty.call(vars, name) ? String(vars[name]) : '{{' + name + '}}';
      });
    }
    return fallback;
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
    pickerSelect:      document.getElementById('pickerSelect'),
    pickerNote:        document.getElementById('pickerNote'),
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
    tokenPromptDiscard: document.getElementById('tokenPromptDiscardBtn'),
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
    // Prompt 5 reconciled the picker to the main-page <select> widget so
    // the signing-language dropdown is visually identical to the target
    // selector on /app.html. We still populate the legacy <ul> as hidden
    // backing DOM so any external code that reads pickerOptions keeps
    // working during the transition.
    if (els.pickerSelect) {
      els.pickerSelect.innerHTML = '';
      // Leading placeholder so a freshly-rendered select doesn't pre-commit
      // to the first language in the catalog before the user's choice.
      var placeholder = document.createElement('option');
      placeholder.value = '';
      placeholder.textContent = tr(
        'contribute.language_picker.placeholder',
        'Choose a sign language…'
      );
      placeholder.disabled = true;
      placeholder.selected = true;
      els.pickerSelect.appendChild(placeholder);
      for (var j = 0; j < view.languages.length; j++) {
        var langOpt = view.languages[j];
        var opt = document.createElement('option');
        opt.value = langOpt.code;
        opt.textContent = langOpt.code.toUpperCase() + ' — ' + langOpt.english_name +
          ' (' + langOpt.coverage_count + ' signs)';
        els.pickerSelect.appendChild(opt);
      }
      // Sync to current state so reload / back-nav lands on the last choice.
      var currentLang = CTX.getState().language;
      if (currentLang && findLanguage(currentLang)) {
        els.pickerSelect.value = currentLang;
      }
    }

    // Legacy pickerOptions rendering — hidden but preserved so external
    // tests / keyboard-focus fallbacks keep finding .picker-option nodes.
    if (!els.pickerOptions) return;
    els.pickerOptions.innerHTML = '';
    for (var i = 0; i < view.languages.length; i++) {
      var lang = view.languages[i];
      var li = document.createElement('li');

      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'picker-option';
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

  function renderPickerNote() {
    if (!els.pickerNote) return;
    var code = els.pickerSelect ? els.pickerSelect.value : '';
    var lang = findLanguage(code);
    els.pickerNote.classList.remove('limited');
    if (!lang) {
      els.pickerNote.textContent = '';
      return;
    }
    var n = lang.coverage_count || 0;
    if (n === 0) {
      els.pickerNote.textContent = tr(
        'contribute.language_picker.note_empty',
        'No signs in this corpus yet — you would be the first.'
      );
      els.pickerNote.classList.add('limited');
    } else if (n < 200) {
      els.pickerNote.textContent = n + ' signs in corpus — a small dictionary to grow.';
      els.pickerNote.classList.add('limited');
    } else {
      els.pickerNote.textContent = n + ' signs in corpus.';
    }
  }

  function onPickerSelectChange() {
    if (!els.pickerSelect) return;
    var code = els.pickerSelect.value;
    if (!code) return;
    renderPickerNote();
    setLanguage(code);
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
      // Reset the select to its placeholder so "Change language" → pick
      // again starts clean instead of showing the previous selection.
      if (els.pickerSelect) els.pickerSelect.value = '';
      renderPickerNote();
      return;
    }

    els.picker.hidden = true;
    els.langMasthead.hidden = false;
    if (els.pickerSelect && els.pickerSelect.value !== lang.code) {
      els.pickerSelect.value = lang.code;
    }
    renderPickerNote();

    els.badgeCode.textContent = lang.code.toUpperCase();
    els.badgeName.textContent = lang.english_name;

    // Prompt-polish 6: the notation-panel review-status notice uses the
    // picked language name so the contributor sees exactly which native
    // signer community will review this draft.
    var reviewStatusLang = document.getElementById('notationReviewStatusLang');
    if (reviewStatusLang) {
      reviewStatusLang.textContent = lang.english_name;
    }

    if (lang.has_reviewers) {
      els.notice.hidden = true;
      els.notice.innerHTML = '';
    } else {
      els.notice.innerHTML = '';
      var p = document.createElement('p');
      p.textContent = tr(
        'contribute.reviewer_notice.no_reviewers_body',
        'No Deaf reviewers are currently assigned to {{language}}. You can save a draft, but it will not be reviewed until a reviewer is added. We will email you if one becomes available.',
        { language: lang.english_name }
      );
      els.notice.appendChild(p);
      els.notice.hidden = false;
    }

    if (snapshot.gloss && snapshot.gloss.length > 0) {
      els.contextGloss.textContent = snapshot.gloss;
      els.contextGloss.classList.remove('is-empty');
    } else {
      els.contextGloss.textContent = tr('contribute.masthead.context_gloss_empty', 'No sign selected');
      els.contextGloss.classList.add('is-empty');
    }
    els.contextState.textContent = CTX.stateLabel(snapshot.sessionState);

    if (snapshot.sessionId) {
      els.contextSessionId.textContent = CTX.shortId(snapshot.sessionId);
      els.contextSessionId.setAttribute('title', snapshot.sessionId);
      els.contextCopyBtn.hidden = false;
      // aria-label only exists while the button is in the a11y tree —
      // setting it on a hidden element triggers a pa11y HiddenAttr warning.
      els.contextCopyBtn.setAttribute('aria-label', tr('contribute.masthead.context_copy_aria', 'Copy session URL'));
    } else {
      els.contextSessionId.textContent = tr('contribute.masthead.context_session_id_placeholder', '—');
      els.contextSessionId.removeAttribute('title');
      els.contextCopyBtn.hidden = true;
      els.contextCopyBtn.removeAttribute('aria-label');
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
      showInlineError(
        els.glossError,
        tr('contribute.authoring.gloss_error_required', 'Gloss is required before you describe the sign.')
      );
    } else {
      hideInlineError(els.glossError);
    }
    if (!descOk) {
      showInlineError(
        els.descriptionError,
        tr('contribute.authoring.description_error_too_short', 'Please add at least 20 characters of description.')
      );
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
    els.startBtn.textContent = tr('contribute.authoring.submit_button_starting', 'Starting…');

    CTX.createSession({
      gloss: gloss,
      prose: description,
      authorIsDeafNative: !!els.deafNativeInput.checked,
    }).then(function () {
      // render() will be invoked via the subscribe hook once state
      // changes; the summary card takes it from there.
      els.startBtn.textContent = tr('contribute.authoring.submit_button', 'Start authoring');
    }).catch(function (err) {
      els.startBtn.disabled = false;
      els.startBtn.textContent = tr('contribute.authoring.submit_button', 'Start authoring');
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
      var msg = tr('contribute.authoring.submit_error_generic', 'Could not start a session. Please try again.');
      var errCode = null;
      if (err && typeof err.body === 'string') {
        try {
          var parsed = JSON.parse(err.body);
          if (parsed && parsed.error && typeof parsed.error.code === 'string') {
            errCode = parsed.error.code;
          }
        } catch (_e) { /* ignore — fall through to status-based branches */ }
      }
      if (errCode === 'rate_limited' || (err && err.status === 429)) {
        msg = tr('contribute.authoring.submit_error_rate_limited', "You're sending requests faster than the server can process. Wait a moment and try again.");
      } else if (errCode === 'injection_rejected') {
        msg = tr('contribute.authoring.submit_error_injection_rejected', "We didn't interpret this as a sign description. Please describe only the sign itself.");
      } else if (err && err.status === 422) {
        msg = tr('contribute.authoring.submit_error_language_unsupported', 'This language is not yet enabled on the server. Please pick another.');
      }
      showInlineError(els.submitError, msg);
      if (window.console) console.error('[contribute] createSession failed:', err);
    });
  }

  // ---------- edit (post-submit) ----------

  function onEditSummary() {
    var snap = CTX.getState();
    var glossLabel = snap.gloss
      ? '“' + snap.gloss + '”'
      : tr('contribute.submission.edit_modal_body_unnamed', 'your current sign');
    showModal({
      title: tr('contribute.submission.edit_modal_title', 'Discard this draft?'),
      body: tr(
        'contribute.submission.edit_modal_body',
        'This will discard your draft for {{gloss_label}} so you can edit the gloss and description. Continue?',
        { gloss_label: glossLabel }
      ),
      cancelLabel: tr('common.modal_cancel', 'Cancel'),
      confirmLabel: tr('common.modal_discard', 'Discard'),
      onConfirm: function () {
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
      glossLabel = tr('contribute.submission.edit_modal_body_unnamed', 'your current sign');
    }

    showModal({
      title: tr('contribute.submission.change_lang_modal_title', 'Discard this draft?'),
      body: tr(
        'contribute.submission.change_lang_modal_body',
        'This will discard your draft for {{gloss_label}}. Continue?',
        { gloss_label: glossLabel }
      ),
      cancelLabel: tr('common.modal_cancel', 'Cancel'),
      confirmLabel: tr('common.modal_discard', 'Discard'),
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

  // ---------- always-available escape hatches ----------
  //
  // Stale local state (persisted state machine, orphaned fragment, session
  // id resumed without token) must never lock a contributor out of starting
  // a fresh sign. Both handlers below wipe local state unconditionally —
  // the server-side reject is best-effort inside clearSession/abandonSession.

  function onDiscardSessionEvent() {
    var snap = CTX.getState();
    var glossLabel = snap.gloss
      ? '“' + snap.gloss + '”'
      : tr('contribute.submission.edit_modal_body_unnamed', 'your current sign');
    showModal({
      title: tr('contribute.submission.reset_modal_title', 'Discard this draft?'),
      body: tr(
        'contribute.submission.reset_modal_body',
        'This will discard your draft for {{gloss_label}} and return you to the language picker. Continue?',
        { gloss_label: glossLabel }
      ),
      cancelLabel: tr('common.modal_cancel', 'Cancel'),
      confirmLabel: tr('common.modal_discard', 'Discard'),
      onConfirm: function () {
        var lang = snap.language;
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

  function onTokenPromptDiscard() {
    // The fragment points at a session we can't resume (no token on this
    // device). Just clear local state + the fragment and drop back to the
    // picker — leave the server session alone (we lack the token to reject
    // it, and it will be garbage-collected on the server anyway).
    pendingResumeId = null;
    CTX.setState({
      sessionId:          null,
      sessionToken:       null,
      gloss:              '',
      language:           null,
      sessionState:       'awaiting_description',
      pendingQuestions:   [],
      clarifications:     [],
      hamnosys:           null,
      sigml:              null,
      parameters:         null,
      generationErrors:   [],
      correctionsCount:   0,
      authorIsDeafNative: null,
      descriptionProse:   '',
    });
    CTX.clearSessionFragment();
    hideTokenPrompt();
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
      // Already on the picker — give keyboard users the dropdown.
      if (els.pickerSelect && typeof els.pickerSelect.focus === 'function') {
        els.pickerSelect.focus();
        return;
      }
      var first = els.pickerOptions && els.pickerOptions.querySelector('.picker-option');
      if (first && typeof first.focus === 'function') first.focus();
      return;
    }
    requestLanguageChange();
  }

  // ---------- copy session URL ----------

  function copySessionUrl() {
    var url = CTX.sessionUrl();
    var hasSession = !!CTX.getState().sessionId;
    var success = hasSession
      ? tr('contribute.copy_session_success_with_session', 'Copied session URL')
      : tr('contribute.copy_session_success_no_session', 'Copied page URL');
    var failMsg = tr('common.toast_could_not_copy', 'Could not copy');
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      navigator.clipboard.writeText(url)
        .then(function () { showToast(success); })
        .catch(function () { showToast(failMsg); });
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
      showToast(ok ? success : failMsg);
    } catch (_e) {
      showToast(failMsg);
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
          els.tokenPromptError.textContent = tr('contribute.token_prompt.error_mismatch', 'That token does not match this session.');
        } else if (err && err.status === 404) {
          els.tokenPromptError.textContent = tr('contribute.token_prompt.error_not_found', 'This session was not found.');
        } else {
          els.tokenPromptError.textContent = tr('contribute.token_prompt.error_generic', 'Could not resume the session.');
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

  // Prompt 14 step 10 — binary pause. When /governance-data.json carries
  // contributions_paused: true the page hides every interactive section
  // and shows only the pause banner. No new sessions, no authoring, no
  // submit. Prior submissions remain viewable at /contribute/status/<id>
  // and /contribute/me — those pages do not read this flag.
  function checkPaused() {
    return fetch('/governance-data.json', { headers: { 'Accept': 'application/json' } })
      .then(function (resp) { return resp.ok ? resp.json() : null; })
      .then(function (data) {
        if (!data || data.contributions_paused !== true) return false;
        var banner = document.getElementById('contributePaused');
        if (!banner) return false;
        var bodyEl  = document.getElementById('contributePausedBody');
        var emailEl = document.getElementById('contributePausedEmail');
        var reason = typeof data.pause_reason === 'string' && data.pause_reason
          ? data.pause_reason
          : tr('contribute.paused.body', 'Contributions are paused while we seat reviewers. Email the governance address below to be notified when the queue opens.');
        if (bodyEl) bodyEl.textContent = reason;
        if (emailEl && typeof data.governance_email === 'string' && data.governance_email) {
          emailEl.textContent = data.governance_email;
          emailEl.setAttribute('href', 'mailto:' + data.governance_email);
        }
        banner.hidden = false;
        if (els.picker) els.picker.hidden = true;
        if (els.langMasthead) els.langMasthead.hidden = true;
        if (els.authoringRoot) els.authoringRoot.hidden = true;
        if (els.tokenPrompt) els.tokenPrompt.hidden = true;
        if (els.notice) els.notice.hidden = true;
        var confirm = document.getElementById('submissionConfirmation');
        if (confirm) confirm.hidden = true;
        return true;
      })
      .catch(function () {
        // Network / static-file unavailability → proceed without pausing.
        // The flag is a server-driven choice; if it cannot be read we
        // prefer a working page to a blank page with no recourse.
        return false;
      });
  }

  function onSaveDraftClick() {
    flushAutosave();
    showToast(tr('contribute.authoring.save_draft_toast', 'Draft saved locally.'));
  }

  function init() {
    els.changeBtn.addEventListener('click', onChangeClick);
    els.modalCancelBtn.addEventListener('click', onModalCancel);
    els.modalDiscardBtn.addEventListener('click', onModalConfirm);
    els.modalBackdrop.addEventListener('click', onModalBackdropClick);
    els.hintClose.addEventListener('click', dismissHint);
    els.contextCopyBtn.addEventListener('click', copySessionUrl);
    els.tokenPromptForm.addEventListener('submit', onTokenPromptSubmit);
    if (els.tokenPromptDiscard) {
      els.tokenPromptDiscard.addEventListener('click', onTokenPromptDiscard);
    }
    document.addEventListener('kozha:discard-session', onDiscardSessionEvent);
    els.authoringForm.addEventListener('submit', onSubmit);
    els.glossInput.addEventListener('input', onGlossInput);
    els.glossInput.addEventListener('blur', onGlossBlur);
    els.descriptionInput.addEventListener('input', onDescriptionInput);
    els.deafNativeInput.addEventListener('change', onDeafNativeChange);
    els.summaryEditBtn.addEventListener('click', onEditSummary);
    document.addEventListener('keydown', onGlobalKey);
    if (els.pickerSelect) {
      els.pickerSelect.addEventListener('change', onPickerSelectChange);
    }
    var saveDraftBtn = document.getElementById('authoringSaveDraftBtn');
    if (saveDraftBtn) saveDraftBtn.addEventListener('click', onSaveDraftClick);

    CTX.subscribe(render);

    checkPaused().then(function (paused) {
      if (paused) return;
      loadLanguages().then(function () {
        renderOptions();
        return hydrateFromFragment();
      }).then(function () {
        render(CTX.getState());
      }).catch(function (err) {
        els.pickerEmpty.innerHTML = '';
        var p = document.createElement('p');
        p.className = 'picker-prompt';
        p.textContent = tr('contribute.language_picker.load_error', 'Could not load the language list. Refresh to retry.');
        els.pickerEmpty.appendChild(p);
        if (window.console) console.error('[contribute] language load failed:', err);
      });
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

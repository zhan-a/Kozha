/* Contribute page — UI controller.
 *
 * All state (language, gloss, session id/token, session state) lives in
 * window.KOZHA_CONTRIB_CONTEXT (see contribute-context.js). This file is
 * the view layer: it subscribes to the store and re-renders the sticky
 * language masthead, the context strip, the language picker, the resume
 * prompt, the confirmation modal, and the one-time keyboard hint.
 *
 * Scope for prompt 4:
 *   - render language header + context strip from the store
 *   - language selection, "change" with confirmation modal, deep-link
 *     resume flow, Cmd/Ctrl+K and Cmd/Ctrl+L shortcuts, copy-session-URL,
 *     reviewer notice, one-time hint strip
 *
 * Out of scope (later prompts):
 *   - chat panel              → prompt 5
 *   - HamNoSys / SiGML readout → prompt 6
 *   - submit + status URL     → prompt 7
 */
(function () {
  'use strict';

  var CTX = window.KOZHA_CONTRIB_CONTEXT;
  if (!CTX) {
    if (window.console) console.error('[contribute] contribute-context.js must load first');
    return;
  }

  var HINT_SEEN_KEY = 'kozha.contribute.hintSeen';
  // Keys later prompts will write into; clearing them here means changing
  // language invalidates every downstream draft without each later prompt
  // having to wire itself into the change-language logic.
  var DRAFT_KEYS = [
    'kozha.contribute.draftGloss',
    'kozha.contribute.draftDescription',
    'kozha.contribute.draftNotation',
    'kozha.contribute.draftSession',
  ];

  var LANGUAGES_URL = '/contribute-languages.json';
  var TOAST_MS = 1800;

  var view = {
    languages: [],
  };

  var els = {
    picker:           document.getElementById('languagePicker'),
    pickerEmpty:      document.getElementById('pickerEmpty'),
    pickerOptions:    document.getElementById('pickerOptions'),
    langMasthead:     document.getElementById('langMasthead'),
    badge:            document.getElementById('languageBadge'),
    badgeCode:        document.getElementById('languageBadgeCode'),
    badgeName:        document.getElementById('languageBadgeName'),
    changeBtn:        document.getElementById('changeLanguageBtn'),
    contextGloss:     document.getElementById('contextGloss'),
    contextState:     document.getElementById('contextState'),
    contextSessionId: document.getElementById('contextSessionId'),
    contextCopyBtn:   document.getElementById('contextCopyBtn'),
    tokenPrompt:      document.getElementById('tokenPrompt'),
    tokenPromptForm:  document.getElementById('tokenPromptForm'),
    tokenPromptInput: document.getElementById('tokenPromptInput'),
    tokenPromptError: document.getElementById('tokenPromptError'),
    notice:           document.getElementById('reviewerNotice'),
    authoringRoot:    document.getElementById('authoring-root'),
    modalBackdrop:    document.getElementById('modalBackdrop'),
    modalTitle:       document.getElementById('modalTitle'),
    modalBody:        document.getElementById('modalBody'),
    modalCancelBtn:   document.getElementById('modalCancelBtn'),
    modalDiscardBtn:  document.getElementById('modalDiscardBtn'),
    hintStrip:        document.getElementById('hintStrip'),
    hintClose:        document.getElementById('hintClose'),
    toast:            document.getElementById('toast'),
  };

  // ---------- languages ----------

  function findLanguage(code) {
    if (!code) return null;
    for (var i = 0; i < view.languages.length; i++) {
      if (view.languages[i].code === code) return view.languages[i];
    }
    return null;
  }

  function clearDraftKeys() {
    try {
      for (var i = 0; i < DRAFT_KEYS.length; i++) {
        sessionStorage.removeItem(DRAFT_KEYS[i]);
      }
    } catch (_e) { /* ignore */ }
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

    // authoring-root stays hidden until prompts 5-7 mount content.
    els.authoringRoot.hidden = true;

    maybeShowHint();
  }

  // ---------- change-language flow ----------

  function onChangeClick() { requestLanguageChange(); }

  function requestLanguageChange() {
    var snap = CTX.getState();
    if (!snap.sessionId) {
      // Nothing to discard — drop straight back to the picker.
      clearDraftKeys();
      CTX.setState({ language: null, gloss: '', sessionState: 'awaiting_description' });
      CTX.clearSessionFragment();
      return;
    }
    var glossLabel = snap.gloss ? '“' + snap.gloss + '”' : 'your current sign';
    showModal({
      title: 'Discard this draft?',
      body: 'This will discard your draft for ' + glossLabel + '. Continue?',
      cancelLabel: 'Cancel',
      confirmLabel: 'Discard',
      onConfirm: function () {
        clearDraftKeys();
        CTX.abandonSession();
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
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

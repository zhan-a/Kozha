/* Contribute page — language picker state.
 *
 * Scope for prompt 3: load the language list from
 * /contribute-languages.json, render the picker, persist the active
 * language in sessionStorage, show the "no reviewer assigned" notice
 * when the chosen language has has_reviewers === false.
 *
 * Out of scope for this prompt (filled by later prompts):
 *   - chat panel              → prompt 4
 *   - avatar preview          → prompt 5
 *   - HamNoSys / SiGML readout → prompt 6
 *   - submit + status URL     → prompt 7
 *
 * Everything in this file deliberately avoids touching #authoring-root
 * beyond toggling visibility.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'kozha.contribute.activeLanguage';
  var DRAFT_KEYS = [
    // Later prompts will add their own draft keys. Listing them here
    // means switching language from this prompt already clears future
    // state without the switch-logic having to know about each one.
    'kozha.contribute.draftGloss',
    'kozha.contribute.draftDescription',
    'kozha.contribute.draftNotation',
    'kozha.contribute.draftSession',
  ];

  var LANGUAGES_URL = '/contribute-languages.json';

  var state = {
    languages: [],
    activeCode: null,
  };

  var els = {
    picker: document.getElementById('languagePicker'),
    pickerEmpty: document.getElementById('pickerEmpty'),
    pickerOptions: document.getElementById('pickerOptions'),
    pickerActive: document.getElementById('pickerActive'),
    badge: document.getElementById('languageBadge'),
    changeBtn: document.getElementById('changeLanguageBtn'),
    notice: document.getElementById('reviewerNotice'),
    authoringRoot: document.getElementById('authoring-root'),
  };

  function findLanguage(code) {
    for (var i = 0; i < state.languages.length; i++) {
      if (state.languages[i].code === code) return state.languages[i];
    }
    return null;
  }

  function readStoredLanguage() {
    try {
      return sessionStorage.getItem(STORAGE_KEY);
    } catch (_e) {
      return null;
    }
  }

  function writeStoredLanguage(code) {
    try {
      if (code) sessionStorage.setItem(STORAGE_KEY, code);
      else sessionStorage.removeItem(STORAGE_KEY);
    } catch (_e) { /* storage blocked — continue in-memory only */ }
  }

  function clearDraft() {
    try {
      for (var i = 0; i < DRAFT_KEYS.length; i++) {
        sessionStorage.removeItem(DRAFT_KEYS[i]);
      }
    } catch (_e) { /* ignore */ }
  }

  function renderOptions() {
    els.pickerOptions.innerHTML = '';
    for (var i = 0; i < state.languages.length; i++) {
      var lang = state.languages[i];
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

  function renderActive() {
    var lang = findLanguage(state.activeCode);
    if (!lang) {
      els.pickerActive.hidden = true;
      els.pickerEmpty.hidden = false;
      els.notice.hidden = true;
      els.authoringRoot.hidden = true;
      return;
    }
    els.pickerEmpty.hidden = true;
    els.pickerActive.hidden = false;
    els.badge.textContent = lang.code.toUpperCase() + ' — ' + lang.english_name;

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
    // #authoring-root stays hidden until prompts 4-7 mount real content
    // into it; revealing an empty region would be worse than nothing.
    els.authoringRoot.hidden = true;
  }

  function setLanguage(code) {
    if (code && !findLanguage(code)) return;
    var previous = state.activeCode;
    if (previous && code && previous !== code) clearDraft();
    if (!code && previous) clearDraft();
    state.activeCode = code || null;
    writeStoredLanguage(state.activeCode);
    renderActive();
  }

  function getLanguage() {
    return state.activeCode;
  }

  function onChangeClick() {
    setLanguage(null);
  }

  function loadLanguages() {
    return fetch(LANGUAGES_URL, { headers: { 'Accept': 'application/json' } })
      .then(function (r) {
        if (!r.ok) throw new Error('languages HTTP ' + r.status);
        return r.json();
      })
      .then(function (data) {
        state.languages = (data && data.languages) || [];
      });
  }

  function init() {
    els.changeBtn.addEventListener('click', onChangeClick);
    loadLanguages().then(function () {
      renderOptions();
      var stored = readStoredLanguage();
      if (stored && findLanguage(stored)) {
        state.activeCode = stored;
      }
      renderActive();
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
    setLanguage: setLanguage,
    getLanguage: getLanguage,
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

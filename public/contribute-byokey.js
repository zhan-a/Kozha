/* BYO OpenAI API key fallback.

   When the project key on the server is missing, rate-limited, or
   revoked, contributors can paste their own key here. Once saved, every
   subsequent request to /api/chat2hamnosys/* picks up an
   X-OpenAI-Api-Key header; the backend's _ByoOpenAIKeyMiddleware
   parks it in a contextvar and llm/client.py prefers it over the env
   var for the duration of that request.

   The key lives only in this browser (localStorage). It leaves the
   browser solely through that header, over HTTPS, direct to the origin
   that served this page. Nothing else in this file logs or mirrors it.

   This script must load BEFORE the other contribute-*.js scripts so
   the fetch patch is in place when they fire their first call. */
(function () {
  'use strict';

  var STORAGE_KEY = 'bridgn.openai_api_key';
  var API_PREFIX = '/api/chat2hamnosys';
  var HEADER_NAME = 'X-OpenAI-Api-Key';

  function readKey() {
    try {
      return (window.localStorage.getItem(STORAGE_KEY) || '').trim();
    } catch (_e) {
      return '';
    }
  }

  function writeKey(value) {
    try {
      if (!value) window.localStorage.removeItem(STORAGE_KEY);
      else window.localStorage.setItem(STORAGE_KEY, value);
      return true;
    } catch (_e) {
      return false;
    }
  }

  function urlMatchesApi(raw) {
    if (!raw) return false;
    var s = (typeof raw === 'string') ? raw : String(raw);
    if (s.indexOf(API_PREFIX) === 0) return true;
    try {
      var u = new URL(s, window.location.href);
      return u.origin === window.location.origin
        && u.pathname.indexOf(API_PREFIX) === 0;
    } catch (_e) {
      return false;
    }
  }

  // Patch fetch exactly once. Non-pipeline calls pass through untouched;
  // pipeline calls without a saved key also pass through — the server
  // still has its own key for the common case.
  var originalFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    var targetUrl = (typeof input === 'string')
      ? input
      : (input && input.url) || '';
    if (!urlMatchesApi(targetUrl)) {
      return originalFetch(input, init);
    }
    var key = readKey();
    if (!key) {
      return originalFetch(input, init);
    }
    var baseInit = init || {};
    var sourceHeaders = baseInit.headers
      || (typeof input !== 'string' && input && input.headers)
      || {};
    var headers = new Headers(sourceHeaders);
    if (!headers.has(HEADER_NAME)) {
      headers.set(HEADER_NAME, key);
    }
    var merged = {};
    for (var k in baseInit) {
      if (Object.prototype.hasOwnProperty.call(baseInit, k)) {
        merged[k] = baseInit[k];
      }
    }
    merged.headers = headers;
    return originalFetch(input, merged);
  };

  // Small public hook so tests / console can see what's active without
  // touching localStorage keys by name.
  window.BridgnByoKey = {
    has: function () { return !!readKey(); },
    clear: function () { writeKey(''); }
  };

  function wireUI() {
    var input = document.getElementById('byoKeyInput');
    var saveBtn = document.getElementById('byoKeySaveBtn');
    var clearBtn = document.getElementById('byoKeyClearBtn');
    var status = document.getElementById('byoKeyStatus');
    var error = document.getElementById('byoKeyError');
    if (!input || !saveBtn || !clearBtn || !status) return;

    function render() {
      var stored = readKey();
      if (stored) {
        var tail = stored.length >= 4 ? stored.slice(-4) : stored;
        status.textContent = 'Your key is saved in this browser (ends in …' + tail + ').';
        status.classList.add('byo-key-status-saved');
        clearBtn.hidden = false;
      } else {
        status.textContent = 'Using the project key on the server.';
        status.classList.remove('byo-key-status-saved');
        clearBtn.hidden = true;
      }
      if (error) {
        error.hidden = true;
        error.textContent = '';
      }
      input.value = '';
    }

    function showError(msg) {
      if (!error) return;
      error.textContent = msg;
      error.hidden = false;
    }

    saveBtn.addEventListener('click', function () {
      var value = (input.value || '').trim();
      if (!value) {
        showError('Paste a key before saving.');
        return;
      }
      if (value.indexOf('sk-') !== 0) {
        showError('That does not look like an OpenAI key. It should start with "sk-".');
        return;
      }
      if (!writeKey(value)) {
        showError('This browser refused to save the key (private mode?). You can still paste it each visit.');
        return;
      }
      render();
    });

    clearBtn.addEventListener('click', function () {
      writeKey('');
      render();
    });

    input.addEventListener('keydown', function (evt) {
      if (evt.key === 'Enter') {
        evt.preventDefault();
        saveBtn.click();
      }
    });

    render();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wireUI);
  } else {
    wireUI();
  }
})();

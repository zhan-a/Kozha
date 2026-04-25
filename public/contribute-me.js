/* Contribute page — contributor's session dashboard (prompt 14 step 8).
 *
 * A single-page aggregator at /contribute/me. Lists every submission
 * this browser has a token for, with current status pulled from the
 * server. No login — identity is the sessionStorage history array, not
 * a cookie or account.
 *
 * History format (sessionStorage key `kozha.contribute.history`):
 *
 *   [
 *     { sessionId, sessionToken, language, gloss, addedAt },
 *     ...
 *   ]
 *
 * The submit flow (contribute-submit.js) appends to this array on
 * every successful /accept. Order is most-recent first.
 *
 * If the user clears browser data the list is empty. The server-side
 * record for each submission still exists; the permanent status URL
 * (/contribute/status/<id>) continues to resolve when pasted. That is
 * the designed-for behaviour — this dashboard is a convenience, not a
 * source of truth.
 */
(function () {
  'use strict';

  var API_BASE = '/api/chat2hamnosys';
  var STATUS_PATH = '/contribute/status/';
  var HISTORY_KEY = 'kozha.contribute.history';
  var MAX_ITEMS = 50;

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

  var STATUS_LABEL_FALLBACK = {
    draft:           'Draft',
    pending_review:  'Pending review',
    under_review:    'Under review',
    validated:       'Validated',
    rejected:        'Rejected',
    quarantined:     'Quarantined',
  };
  function statusLabel(status) {
    var fb = STATUS_LABEL_FALLBACK[status] || status;
    return tr('status.state.' + status, fb);
  }

  var LANGUAGE_LABEL_FALLBACK = {
    bsl: 'British Sign Language',
    asl: 'American Sign Language',
    dgs: 'German Sign Language',
    lsf: 'French Sign Language',
    lse: 'Spanish Sign Language',
    pjm: 'Polish Sign Language',
    ngt: 'Dutch Sign Language',
    gsl: 'Greek Sign Language',
  };
  function languageLabel(code) {
    if (!code) return '';
    var lc = String(code).toLowerCase();
    var fb = LANGUAGE_LABEL_FALLBACK[lc] || String(code).toUpperCase();
    return tr('status.language_label.' + lc, fb);
  }

  var els = {
    loading:  document.getElementById('meLoading'),
    empty:    document.getElementById('meEmpty'),
    list:     document.getElementById('meList'),
    actions:  document.getElementById('meActions'),
    clearBtn: document.getElementById('meClearBtn'),
    error:    document.getElementById('meError'),
  };

  function readHistory() {
    try {
      var raw = sessionStorage.getItem(HISTORY_KEY);
      if (!raw) return [];
      var parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      // Filter to the subset we can use — drop malformed rows silently.
      var out = [];
      for (var i = 0; i < parsed.length; i++) {
        var r = parsed[i];
        if (r && typeof r === 'object' && typeof r.sessionId === 'string' && r.sessionId) {
          out.push({
            sessionId:    r.sessionId,
            sessionToken: typeof r.sessionToken === 'string' ? r.sessionToken : null,
            language:     typeof r.language === 'string' ? r.language : null,
            gloss:        typeof r.gloss === 'string' ? r.gloss : '',
            addedAt:      typeof r.addedAt === 'number' ? r.addedAt : 0,
          });
        }
        if (out.length >= MAX_ITEMS) break;
      }
      return out;
    } catch (_e) { return []; }
  }

  function writeHistory(items) {
    try {
      if (!items || items.length === 0) {
        sessionStorage.removeItem(HISTORY_KEY);
        return;
      }
      sessionStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, MAX_ITEMS)));
    } catch (_e) { /* storage blocked — accept the loss */ }
  }

  function fetchStatus(sessionId, token) {
    // Prefer the stateless ``/signs/by-token/<token>`` endpoint when a
    // token is available — Option A from
    // ``docs/contrib-fix/01-audit.md`` § 6 makes the token sufficient
    // by itself. The token is in the URL path so we don't add an
    // ``X-Session-Token`` header on this branch. Fall back to the
    // session-id path for entries written before this code shipped
    // (no token recorded in localStorage).
    var url;
    var headers = { 'Accept': 'application/json' };
    if (token) {
      url = API_BASE + '/signs/by-token/' + encodeURIComponent(token);
    } else {
      url = API_BASE + '/sessions/' + encodeURIComponent(sessionId) + '/status';
    }
    return fetch(url, { method: 'GET', headers: headers })
      .then(function (resp) {
        return resp.text().then(function (body) {
          var parsed = null;
          try { parsed = body ? JSON.parse(body) : null; } catch (_e) { /* ignore */ }
          return { ok: resp.ok, status: resp.status, body: parsed };
        });
      })
      .catch(function () { return { ok: false, status: 0, body: null }; });
  }

  function permanentUrlFor(sessionId) {
    var loc = window.location;
    return loc.origin + STATUS_PATH + encodeURIComponent(sessionId);
  }

  function formatDate(ms) {
    if (!ms) return '';
    try {
      var d = new Date(ms);
      if (isNaN(d.getTime())) return '';
      return d.toLocaleString();
    } catch (_e) { return ''; }
  }

  function statusClass(state) {
    if (state === 'validated') return 'me-state is-validated';
    if (state === 'rejected')  return 'me-state is-rejected';
    if (state === 'quarantined') return 'me-state is-quarantine';
    return 'me-state';
  }

  function renderItem(entry, statusBody) {
    var li = document.createElement('li');
    li.className = 'me-item';

    var main = document.createElement('div');
    main.className = 'me-item-main';

    var glossEl = document.createElement('span');
    glossEl.className = 'me-gloss';
    glossEl.textContent = (statusBody && statusBody.gloss) || entry.gloss || tr('me.row_no_gloss', '(no gloss)');
    main.appendChild(glossEl);

    var langCode = (statusBody && statusBody.sign_language) || entry.language || '';
    if (langCode) {
      var langEl = document.createElement('span');
      langEl.className = 'me-language';
      langEl.textContent = languageLabel(langCode);
      main.appendChild(langEl);
    }

    li.appendChild(main);

    var state = (statusBody && statusBody.status) || 'draft';
    var stateEl = document.createElement('span');
    stateEl.className = statusClass(state);
    stateEl.textContent = statusLabel(state);
    li.appendChild(stateEl);

    var meta = document.createElement('div');
    meta.className = 'me-meta';
    var link = document.createElement('a');
    link.href = permanentUrlFor(entry.sessionId);
    link.textContent = permanentUrlFor(entry.sessionId);
    meta.appendChild(link);

    var whenStr = formatDate(entry.addedAt);
    if (whenStr) {
      var sep = document.createElement('span');
      sep.setAttribute('aria-hidden', 'true');
      sep.textContent = ' · ';
      meta.appendChild(sep);
      var whenEl = document.createElement('span');
      whenEl.textContent = tr('me.row_submitted_prefix', 'submitted {{date}}', { date: whenStr });
      meta.appendChild(whenEl);
    }

    li.appendChild(meta);
    return li;
  }

  function showEmpty() {
    els.loading.hidden = true;
    els.list.hidden = true;
    els.actions.hidden = true;
    els.empty.hidden = false;
  }

  function showError(body) {
    els.loading.hidden = true;
    els.error.hidden = false;
    els.error.textContent = body;
  }

  function render(history, results) {
    els.loading.hidden = true;
    if (!history.length) {
      showEmpty();
      return;
    }
    els.list.innerHTML = '';
    for (var i = 0; i < history.length; i++) {
      var entry = history[i];
      var body = results[i] && results[i].ok ? results[i].body : null;
      els.list.appendChild(renderItem(entry, body));
    }
    els.list.hidden = false;
    els.actions.hidden = false;
  }

  function onClear() {
    var confirmed = window.confirm(tr(
      'me.clear_confirm',
      'Clear the list of submissions in this browser? The submissions themselves are not deleted; each permanent status URL continues to work.'
    ));
    if (!confirmed) return;
    writeHistory([]);
    showEmpty();
  }

  function init() {
    els.clearBtn.addEventListener('click', onClear);
    var history = readHistory();
    if (!history.length) {
      showEmpty();
      return;
    }
    // Fan out status lookups in parallel. Each failure collapses to a
    // null status body; the item still renders with the locally-known
    // gloss and language.
    var promises = history.map(function (entry) {
      return fetchStatus(entry.sessionId, entry.sessionToken);
    });
    Promise.all(promises).then(function (results) {
      render(history, results);
    }).catch(function () {
      // Promise.all with caught per-promise shouldn't reject, but defensively:
      render(history, history.map(function () { return { ok: false, body: null }; }));
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

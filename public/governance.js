// Governance page loader. Fetches /governance-data.json and fills in the
// reviewer roster, the advisory board, the per-language coverage list, and
// the contact email. The JSON is the same shape the backend publishes from
// the reviewers table (prompt 15 of the earlier sequence); until it is
// populated, the lists are empty and the page honestly says so.
//
// All DOM reads are guarded so a missing section does not hard-fail.

(function () {
  'use strict';

  var DATA_URL = '/governance-data.json';

  function byId(id) { return document.getElementById(id); }

  function setTextIfPresent(el, text) {
    if (el) el.textContent = text;
  }

  function clear(el) {
    if (!el) return;
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  // Catalog bridge with English fallbacks — the governance page must still
  // read correctly if /strings.en.json failed to load.
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

  function renderReviewers(reviewers) {
    var host = byId('governanceReviewers');
    if (!host) return;
    clear(host);
    if (!reviewers || !reviewers.length) {
      var empty = document.createElement('p');
      empty.className = 'governance-reviewers-empty';
      empty.textContent = tr(
        'governance.reviewers_empty',
        'No reviewers are onboarded yet. Submissions are being accepted and saved as drafts, but the review queue is paused until the Deaf advisory board seats the first reviewers.'
      );
      host.appendChild(empty);
      return;
    }
    var list = document.createElement('ul');
    list.className = 'governance-reviewer-list';
    reviewers.forEach(function (r) {
      var li = document.createElement('li');
      li.className = 'governance-reviewer';
      var name = document.createElement('p');
      name.className = 'governance-reviewer-name';
      name.textContent = r.display_name || tr('governance.reviewer_unnamed', '(unnamed reviewer)');
      if (r.is_deaf_native) {
        var tag = document.createElement('span');
        tag.className = 'governance-reviewer-native';
        tag.textContent = tr('governance.reviewer_native_tag', 'Deaf native signer');
        name.appendChild(tag);
      }
      li.appendChild(name);
      var meta = document.createElement('p');
      meta.className = 'governance-reviewer-meta';
      var parts = [];
      if (r.signs && r.signs.length) {
        parts.push(tr('governance.reviewer_reviews_prefix', 'reviews: ') +
          r.signs.map(function (s) { return s.toUpperCase(); }).join(', '));
      }
      if (r.regional_background) parts.push(tr('governance.reviewer_region_prefix', 'region: ') + r.regional_background);
      if (r.affiliation) parts.push(r.affiliation);
      meta.textContent = parts.join(tr('governance.board_list_separator', ' · '));
      li.appendChild(meta);
      list.appendChild(li);
    });
    host.appendChild(list);
  }

  function renderBoard(board) {
    var host = byId('governanceBoard');
    if (!host) return;
    if (!board || !board.length) {
      // The HTML already ships with the "no board yet" message; leave it.
      return;
    }
    clear(host);
    var intro = document.createElement('p');
    intro.textContent = tr(
      'governance.board_intro_populated',
      'The following Deaf advisors have consented to be listed publicly as members of the board:'
    );
    host.appendChild(intro);
    var list = document.createElement('ul');
    list.className = 'governance-board-list';
    board.forEach(function (m) {
      var li = document.createElement('li');
      li.className = 'governance-board-member';
      var name = document.createElement('p');
      name.className = 'governance-board-name';
      name.textContent = m.name || tr('governance.board_unnamed', '(unnamed advisor)');
      li.appendChild(name);
      var meta = document.createElement('p');
      meta.className = 'governance-board-meta';
      var parts = [];
      if (m.role) parts.push(m.role);
      if (m.affiliation) parts.push(m.affiliation);
      meta.textContent = parts.join(tr('governance.board_list_separator', ' · '));
      li.appendChild(meta);
      list.appendChild(li);
    });
    host.appendChild(list);
  }

  function renderLanguages(languages) {
    var host = byId('governanceLanguageList');
    if (!host) return;
    clear(host);
    if (!languages || !languages.length) {
      var empty = document.createElement('li');
      empty.className = 'governance-language';
      empty.textContent = tr('governance.no_language_data', 'No language data available.');
      host.appendChild(empty);
      return;
    }
    languages.forEach(function (lang) {
      var li = document.createElement('li');
      li.className = 'governance-language';
      var total = lang.reviewers || 0;
      var native = lang.deaf_native_reviewers || 0;
      if (total === 0) li.classList.add('is-uncovered');
      var name = document.createElement('span');
      name.className = 'governance-language-name';
      name.textContent = (lang.name || lang.code || '—') +
        (lang.code ? ' (' + String(lang.code).toUpperCase() + ')' : '');
      var status = document.createElement('span');
      status.className = 'governance-language-status';
      if (total === 0) {
        status.textContent = tr('governance.language_no_reviewers', 'no reviewers seated');
      } else if (native === 0) {
        var key = total === 1
          ? 'governance.language_no_native_singular'
          : 'governance.language_no_native_plural';
        var fb = total === 1
          ? '{{total}} reviewer, none Deaf native — review paused'
          : '{{total}} reviewers, none Deaf native — review paused';
        status.textContent = tr(key, fb, { total: total });
      } else {
        status.textContent = tr(
          'governance.language_coverage',
          '{{native}} Deaf native · {{total}} total',
          { native: native, total: total }
        );
      }
      li.appendChild(name);
      li.appendChild(status);
      host.appendChild(li);
    });
  }

  function renderEmail(email) {
    if (!email) return;
    var link = byId('governanceEmailLink');
    var plain = byId('governanceEmailPlain');
    if (link) {
      link.setAttribute('href', 'mailto:' + email);
      link.textContent = email;
    }
    if (plain) {
      plain.textContent = email;
    }
  }

  function render(data) {
    if (!data || typeof data !== 'object') data = {};
    renderReviewers(data.reviewers);
    renderBoard(data.board);
    renderLanguages(data.languages);
    renderEmail(data.governance_email);
  }

  function showReviewersLoadFailure() {
    var host = byId('governanceReviewers');
    if (!host) return;
    clear(host);
    var msg = document.createElement('p');
    msg.className = 'governance-reviewers-empty';
    msg.textContent = tr(
      'governance.reviewers_load_failure',
      'Reviewer roster unavailable right now. If this persists, email the address at the bottom of this page.'
    );
    host.appendChild(msg);
  }

  function load() {
    fetch(DATA_URL, { credentials: 'same-origin', cache: 'no-cache' })
      .then(function (res) {
        if (!res.ok) throw new Error('HTTP ' + res.status);
        return res.json();
      })
      .then(render)
      .catch(function () { showReviewersLoadFailure(); });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', load);
  } else {
    load();
  }
})();

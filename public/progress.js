/* ---------------------------------------------------------------------
 * /progress page — renders the precomputed snapshot.
 *
 * Loads /progress_snapshot.json (produced by server/progress_snapshot.py
 * in the deploy step) and renders:
 *
 *   1. Top-line stat cards (with reviewed-share meter on the
 *      "Deaf-native-reviewed signs" card)
 *   2. "Help wanted" coverage-gap chips that deep-link into
 *      /contribute.html?lang=<code>&gloss=<word>
 *   3. Sortable + filterable per-language table (desktop) /
 *      card list (mobile), with an inline reviewed-share bar
 *
 * Honest-reporting contract: when a field is null in the snapshot, the
 * UI renders the em-dash ("—"). We never fabricate a zero for a
 * genuinely unknown value.
 *
 * Removed (2026-04-24): "Growth over time" SVG chart and
 * "Recent validations" feed. Both produced low signal-per-pixel and
 * pulled the eye away from the actionable Help-wanted + per-language
 * tables. The snapshot still ships ``progress_series`` and
 * ``recent_activity`` for tooling that wants them.
 * ------------------------------------------------------------------ */
(function () {
  'use strict';

  var SNAPSHOT_URL = '/progress_snapshot.json';
  var DASH = '—';
  var state = {
    snapshot: null,
    sort: { key: 'total', direction: 'desc' },
    filter: '',
  };

  // Keep the DOM targets in one lookup — cleaner than scattering
  // document.getElementById calls throughout the render functions.
  var els = {
    generated:       document.getElementById('progressGenerated'),
    statSigns:       document.querySelector('[data-field="signs"]'),
    statLanguages:   document.querySelector('[data-field="languages"]'),
    statReviewed:    document.querySelector('[data-field="reviewed"]'),
    statAwaiting:    document.querySelector('[data-field="awaiting"]'),
    reviewedMeter:   document.getElementById('reviewedMeter'),
    reviewedMeterBar: document.getElementById('reviewedMeterBar'),
    reviewedMeterPct: document.getElementById('reviewedMeterPct'),
    tableBody:       document.getElementById('progressTableBody'),
    tableHeaders:    document.querySelectorAll('.progress-sort'),
    cards:           document.getElementById('progressCards'),
    helpAslList:     document.getElementById('helpAslList'),
    helpBslList:     document.getElementById('helpBslList'),
    search:          document.getElementById('progressSearch'),
    searchCount:     document.getElementById('progressSearchCount'),
  };

  // ---------- formatting ----------

  function fmtNum(value) {
    if (value === null || value === undefined) return DASH;
    if (typeof value !== 'number' || !isFinite(value)) return DASH;
    // Locale-aware thousands separator, safely falls back without
    // touching server-sent numbers.
    try { return value.toLocaleString('en-US'); }
    catch (_e) { return String(value); }
  }

  function fmtPct(value) {
    if (value === null || value === undefined) return DASH;
    if (typeof value !== 'number' || !isFinite(value)) return DASH;
    return value.toFixed(1) + '%';
  }

  function fmtDate(value) {
    if (!value || typeof value !== 'string') return DASH;
    return value;
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  // ---------- top-line ----------

  function renderTopLine(totals) {
    var fields = [
      [els.statSigns,     totals && totals.signs],
      [els.statLanguages, totals && totals.languages],
      [els.statReviewed,  totals && totals.reviewed],
      [els.statAwaiting,  totals && totals.awaiting_review],
    ];
    fields.forEach(function (pair) {
      var el = pair[0];
      var value = pair[1];
      if (!el) return;
      el.textContent = fmtNum(value);
      el.classList.toggle('is-empty', value === null || value === undefined);
    });

    // Reviewed-share meter on the "Deaf-native-reviewed signs" card.
    // Shows reviewed / total as a horizontal bar so a glance reads
    // "we're at X% of the corpus reviewed" — far more useful than
    // two raw counts the reader has to mentally divide.
    if (els.reviewedMeter && els.reviewedMeterBar && els.reviewedMeterPct) {
      var reviewed = totals && totals.reviewed;
      var total = totals && totals.signs;
      if (typeof reviewed === 'number' && typeof total === 'number' && total > 0) {
        var pct = Math.max(0, Math.min(100, (reviewed / total) * 100));
        els.reviewedMeterBar.style.width = pct.toFixed(1) + '%';
        els.reviewedMeterPct.textContent = pct.toFixed(1) + '% of corpus';
        els.reviewedMeter.hidden = false;
      } else {
        els.reviewedMeter.hidden = true;
      }
    }
  }

  // ---------- source-cell helper ----------

  // Each per-language row links into the /credits page via an anchor.
  // The anchor ids on /credits mirror the language code (e.g. #pjm, #dgs),
  // so the progress dashboard can deep-link the reader straight to the
  // full citation rather than forcing a scroll-hunt through the page.
  // 'alphabet' and any unknown kind route to the top of /credits rather
  // than inventing a missing anchor.
  var CREDITS_ANCHORS = {
    bsl: 'bsl', asl: 'bsl',
    dgs: 'dgs', lsf: 'lsf', gsl: 'gsl', pjm: 'pjm',
    ngt: 'ngt', algerian: 'algerian', bangla: 'bangla',
    isl: 'isl', kurdish: 'kurdish', vsl: 'vsl', fsl: 'fsl',
  };

  function renderSourceCell(lang) {
    var kind = (lang.source_kind || '').toLowerCase();
    var label = kind === 'corpus' ? 'Corpus'
              : kind === 'community' ? 'Community'
              : kind === 'alphabet' ? 'Alphabet' : '';
    var source = lang.source || '';
    var pill = label
      ? '<span class="progress-source-kind' + (kind === 'corpus' ? ' is-corpus' : '') + '">' + escapeHtml(label) + '</span>'
      : '';
    var anchor = CREDITS_ANCHORS[(lang.code || '').toLowerCase()];
    var href = anchor ? '/credits#' + anchor : '/credits';
    if (!source) {
      return pill + '<a class="progress-source-link" href="' + href + '"><span class="progress-dash" aria-label="unknown source">' + DASH + '</span></a>';
    }
    return pill + '<a class="progress-source-link" href="' + href + '" aria-label="Open full citation and license for ' + escapeHtml(lang.name || lang.code || source) + '">' + escapeHtml(source) + '</a>';
  }

  // ---------- reviewed-cell helper ----------

  function renderReviewedCell(lang) {
    if (lang.partial_data) {
      return '<span class="progress-partial" title="'
        + escapeHtml('Review metadata is incomplete for this language; the reviewed count cannot be computed reliably.')
        + '">partial data</span>';
    }
    // Number + inline share bar so a glance answers "how complete is
    // the review pass?" without the reader doing arithmetic. Falls
    // back to bare number when total is unknown or zero.
    var reviewed = lang.reviewed;
    var total = lang.total;
    var num = fmtNum(reviewed);
    if (typeof reviewed !== 'number' || typeof total !== 'number' || total <= 0) {
      return '<div class="progress-reviewed-cell"><span class="progress-reviewed-num">' + num + '</span></div>';
    }
    var pct = Math.max(0, Math.min(100, (reviewed / total) * 100));
    var pctLabel = pct < 0.05 && pct > 0 ? '<0.1%' : pct.toFixed(1) + '%';
    return ''
      + '<div class="progress-reviewed-cell">'
        + '<div class="progress-reviewed-row">'
          + '<span class="progress-reviewed-num">' + num + '</span>'
          + '<span class="progress-reviewed-pct">' + pctLabel + '</span>'
        + '</div>'
        + '<div class="progress-reviewed-bar" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow="' + pct.toFixed(1) + '" aria-label="Reviewed share">'
          + '<div class="progress-reviewed-bar-fill" style="width:' + pct.toFixed(1) + '%"></div>'
        + '</div>'
      + '</div>';
  }

  function renderCommunityCell(lang) {
    if (lang.community_pending === null || lang.community_pending === undefined) return DASH;
    return fmtNum(lang.community_pending);
  }

  function renderAlphabetCell(lang) {
    var status = lang.alphabet || 'unknown';
    var present = lang.alphabet_present;
    var expected = lang.alphabet_expected;
    var glyph = status === 'full' ? '✓'
              : status === 'partial' ? '—'
              : status === 'none' ? '✗'
              : DASH;
    var label = status === 'full' ? 'Complete'
              : status === 'partial' ? 'Partial'
              : status === 'none' ? 'None'
              : 'Unknown';
    var extra = '';
    if (typeof present === 'number' && typeof expected === 'number' && expected > 0) {
      extra = ' <span aria-label="' + present + ' of ' + expected + ' letters">('
        + present + '/' + expected + ')</span>';
    }
    return '<span class="progress-alphabet-badge is-' + escapeHtml(status) + '">'
      + '<span aria-hidden="true">' + glyph + '</span>'
      + escapeHtml(label) + extra
      + '</span>';
  }

  function renderTop500Cell(lang) {
    if (lang.top500_coverage_pct === null || lang.top500_coverage_pct === undefined) {
      return DASH;
    }
    return fmtPct(lang.top500_coverage_pct);
  }

  // ---------- table + cards ----------

  function filteredLanguages() {
    var langs = (state.snapshot && state.snapshot.languages) || [];
    var q = (state.filter || '').trim().toLowerCase();
    if (!q) return langs;
    return langs.filter(function (lang) {
      var hay = (lang.name || '').toLowerCase() + ' ' + (lang.code || '').toLowerCase();
      return hay.indexOf(q) !== -1;
    });
  }

  function sortedLanguages() {
    var langs = filteredLanguages();
    var key = state.sort.key;
    var dir = state.sort.direction === 'asc' ? 1 : -1;
    var copy = langs.slice();
    copy.sort(function (a, b) {
      var av = sortValue(a, key);
      var bv = sortValue(b, key);
      // Push null/undefined to the end regardless of direction so
      // "unknown" never crowds the top of a sorted view.
      if (av === null || av === undefined) {
        if (bv === null || bv === undefined) return 0;
        return 1;
      }
      if (bv === null || bv === undefined) return -1;
      if (typeof av === 'string' && typeof bv === 'string') {
        return av.localeCompare(bv) * dir;
      }
      return (av < bv ? -1 : av > bv ? 1 : 0) * dir;
    });
    return copy;
  }

  function sortValue(lang, key) {
    switch (key) {
      case 'name':      return (lang.name || '').toLowerCase();
      case 'source':    return (lang.source || '').toLowerCase();
      case 'total':     return typeof lang.total === 'number' ? lang.total : null;
      case 'reviewed':  return typeof lang.reviewed === 'number' ? lang.reviewed : null;
      case 'community': return typeof lang.community_pending === 'number' ? lang.community_pending : null;
      case 'alphabet':
        // Sort full > partial > none > unknown, by letters present.
        var order = { full: 3, partial: 2, none: 1, unknown: 0 };
        var primary = order[lang.alphabet || 'unknown'] || 0;
        var secondary = typeof lang.alphabet_present === 'number' ? lang.alphabet_present : 0;
        // Combine into a single comparable number so ties stay stable.
        return primary * 1000 + secondary;
      case 'top500':    return typeof lang.top500_coverage_pct === 'number' ? lang.top500_coverage_pct : null;
      case 'updated':   return lang.last_updated || '';
      default:          return null;
    }
  }

  function renderTable() {
    if (!els.tableBody) return;
    var langs = sortedLanguages();
    if (!langs.length) {
      var empty = state.filter && state.filter.trim()
        ? 'No languages match "' + escapeHtml(state.filter.trim()) + '". Try a different name or code.'
        : 'No languages found in the snapshot.';
      els.tableBody.innerHTML = '<tr class="progress-table-empty"><td colspan="8">' + empty + '</td></tr>';
      renderCards(langs);
      return;
    }
    var rows = langs.map(function (lang) {
      return '<tr>'
        + '<th scope="row"><div class="progress-lang-cell">'
          + '<span class="progress-lang-code">' + escapeHtml(lang.code) + '</span>'
          + '<span class="progress-lang-name">' + escapeHtml(lang.name) + '</span>'
        + '</div></th>'
        + '<td class="progress-source-cell">' + renderSourceCell(lang) + '</td>'
        + '<td class="progress-col-num">' + fmtNum(lang.total) + '</td>'
        + '<td class="progress-col-num">' + renderReviewedCell(lang) + '</td>'
        + '<td class="progress-col-num">' + renderCommunityCell(lang) + '</td>'
        + '<td>' + renderAlphabetCell(lang) + '</td>'
        + '<td class="progress-col-num">' + renderTop500Cell(lang) + '</td>'
        + '<td>' + escapeHtml(fmtDate(lang.last_updated)) + '</td>'
      + '</tr>';
    }).join('');
    els.tableBody.innerHTML = rows;
    renderCards(langs);
  }

  function renderCards(langs) {
    if (!els.cards) return;
    if (!langs.length) {
      els.cards.innerHTML = '<p class="progress-cards-empty">No languages found in the snapshot.</p>';
      return;
    }
    els.cards.innerHTML = langs.map(function (lang) {
      return '<article class="progress-card kz-card" aria-label="' + escapeHtml(lang.name) + ' coverage">'
        + '<div class="progress-card-head">'
          + '<h3 class="progress-card-title">' + escapeHtml(lang.name) + '</h3>'
          + '<span class="progress-card-code">' + escapeHtml(lang.code) + '</span>'
        + '</div>'
        + '<div class="progress-card-grid">'
          + '<div class="progress-card-row"><span class="progress-card-row-label">Total signs</span><span class="progress-card-row-value">' + fmtNum(lang.total) + '</span></div>'
          + '<div class="progress-card-row"><span class="progress-card-row-label">Reviewed</span><span class="progress-card-row-value">' + renderReviewedCell(lang) + '</span></div>'
          + '<div class="progress-card-row"><span class="progress-card-row-label">Community pending</span><span class="progress-card-row-value">' + renderCommunityCell(lang) + '</span></div>'
          + '<div class="progress-card-row"><span class="progress-card-row-label">Alphabet</span><span class="progress-card-row-value">' + renderAlphabetCell(lang) + '</span></div>'
          + '<div class="progress-card-row"><span class="progress-card-row-label">Top-500 English</span><span class="progress-card-row-value">' + renderTop500Cell(lang) + '</span></div>'
          + '<div class="progress-card-row"><span class="progress-card-row-label">Last updated</span><span class="progress-card-row-value">' + escapeHtml(fmtDate(lang.last_updated)) + '</span></div>'
          + '<div class="progress-card-source">' + renderSourceCell(lang) + '</div>'
        + '</div>'
      + '</article>';
    }).join('');
  }

  function updateSortIndicators() {
    els.tableHeaders.forEach(function (btn) {
      var key = btn.getAttribute('data-sort');
      var active = key === state.sort.key;
      btn.classList.toggle('is-active', active);
      // aria-sort is only valid on columnheader/rowheader (the <th>), not
      // on a child <button>, so announce the sort state on the header.
      var th = btn.closest('th');
      if (active) {
        btn.setAttribute('data-direction', state.sort.direction);
        if (th) th.setAttribute('aria-sort', state.sort.direction === 'asc' ? 'ascending' : 'descending');
      } else {
        btn.removeAttribute('data-direction');
        if (th) th.setAttribute('aria-sort', 'none');
      }
    });
  }

  function attachSortHandlers() {
    els.tableHeaders.forEach(function (btn) {
      btn.addEventListener('click', function () {
        var key = btn.getAttribute('data-sort');
        if (!key) return;
        if (state.sort.key === key) {
          state.sort.direction = state.sort.direction === 'asc' ? 'desc' : 'asc';
        } else {
          state.sort.key = key;
          // Numeric columns default descending (show "the most" first);
          // alphabetical columns default ascending (A→Z).
          state.sort.direction = (key === 'name' || key === 'source' || key === 'updated') ? 'asc' : 'desc';
        }
        updateSortIndicators();
        renderTable();
      });
    });
  }

  // ---------- search / filter ----------

  function attachSearchHandler() {
    if (!els.search) return;
    var debounce;
    els.search.addEventListener('input', function () {
      clearTimeout(debounce);
      debounce = setTimeout(function () {
        state.filter = els.search.value || '';
        renderTable();
        renderSearchCount();
      }, 80);
    });
  }

  function renderSearchCount() {
    if (!els.searchCount) return;
    var langs = filteredLanguages();
    var total = (state.snapshot && state.snapshot.languages || []).length;
    if (!state.filter || !state.filter.trim()) {
      els.searchCount.textContent = '';
      return;
    }
    els.searchCount.textContent = langs.length + ' of ' + total + ' shown';
  }

  // ---------- help-wanted ----------

  function renderHelp(list, words, targetLang) {
    if (!list) return;
    if (!Array.isArray(words) || words.length === 0) {
      list.innerHTML = '<li class="progress-help-done">All caught up. Every common word has coverage here.</li>';
      return;
    }
    list.innerHTML = words.map(function (word) {
      var href = '/contribute.html?lang=' + encodeURIComponent(targetLang) + '&gloss=' + encodeURIComponent(word);
      return '<li><a href="' + href + '" data-word="' + escapeHtml(word) + '" data-target-lang="' + escapeHtml(targetLang) + '">'
        + escapeHtml(word)
      + '</a></li>';
    }).join('');
  }

  // ---------- loader ----------

  function renderGeneratedAt(ts) {
    if (!els.generated) return;
    if (!ts) {
      els.generated.textContent = 'Snapshot timestamp unavailable.';
      return;
    }
    els.generated.textContent = 'Snapshot generated ' + ts;
  }

  function renderFailure(message) {
    renderGeneratedAt(null);
    if (els.tableBody) {
      els.tableBody.innerHTML = '<tr class="progress-table-empty"><td colspan="8">' + escapeHtml(message) + '</td></tr>';
    }
    if (els.cards) {
      els.cards.innerHTML = '<p class="progress-cards-empty">' + escapeHtml(message) + '</p>';
    }
    [els.helpAslList, els.helpBslList].forEach(function (list) {
      if (!list) return;
      list.innerHTML = '<li class="progress-help-empty">' + escapeHtml(message) + '</li>';
    });
  }

  function load() {
    fetch(SNAPSHOT_URL, { headers: { 'Accept': 'application/json' }, credentials: 'same-origin' })
      .then(function (resp) {
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        return resp.json();
      })
      .then(function (snap) {
        state.snapshot = snap;
        renderGeneratedAt(snap && snap.generated_at);
        renderTopLine(snap && snap.totals);
        renderTable();
        updateSortIndicators();
        var gaps = (snap && snap.coverage_gaps) || {};
        renderHelp(els.helpAslList, gaps.bsl_missing_from_asl, 'asl');
        renderHelp(els.helpBslList, gaps.asl_missing_from_bsl, 'bsl');
      })
      .catch(function (err) {
        if (window.console) console.error('[progress] snapshot load failed:', err);
        renderFailure('Snapshot is not available right now. The progress dashboard will update once the next scheduled snapshot runs.');
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      attachSortHandlers();
      attachSearchHandler();
      load();
    });
  } else {
    attachSortHandlers();
    attachSearchHandler();
    load();
  }
})();

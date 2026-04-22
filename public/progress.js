/* ---------------------------------------------------------------------
 * /progress page — renders the precomputed snapshot.
 *
 * Loads /progress_snapshot.json (produced by server/progress_snapshot.py
 * in the deploy step) and renders:
 *
 *   1. Top-line stat cards
 *   2. Sortable per-language table (desktop) / card list (mobile)
 *   3. SVG line chart of Deaf-native-reviewed growth
 *   4. Recent validations feed
 *   5. "Help wanted" coverage-gap chips that deep-link into
 *      /contribute.html?lang=<code>&gloss=<word>
 *
 * Honest-reporting contract: when a field is null in the snapshot, the
 * UI renders the em-dash ("—"). We never fabricate a zero for a
 * genuinely unknown value.
 *
 * No external libraries. The chart is hand-rolled SVG so the page stays
 * light (under the 2s-on-3G budget the prompt calls for).
 * ------------------------------------------------------------------ */
(function () {
  'use strict';

  var SNAPSHOT_URL = '/progress_snapshot.json';
  var DASH = '—';
  var state = {
    snapshot: null,
    sort: { key: 'total', direction: 'desc' },
  };

  // Keep the DOM targets in one lookup — cleaner than scattering
  // document.getElementById calls throughout the render functions.
  var els = {
    generated:       document.getElementById('progressGenerated'),
    statSigns:       document.querySelector('[data-field="signs"]'),
    statLanguages:   document.querySelector('[data-field="languages"]'),
    statReviewed:    document.querySelector('[data-field="reviewed"]'),
    statAwaiting:    document.querySelector('[data-field="awaiting"]'),
    tableBody:       document.getElementById('progressTableBody'),
    tableHeaders:    document.querySelectorAll('.progress-sort'),
    cards:           document.getElementById('progressCards'),
    chart:           document.getElementById('progressChart'),
    chartEmpty:      document.getElementById('progressChartEmpty'),
    chartDesc:       document.getElementById('progressChartDescription'),
    recent:          document.getElementById('progressRecent'),
    helpAslList:     document.getElementById('helpAslList'),
    helpBslList:     document.getElementById('helpBslList'),
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
    return fmtNum(lang.reviewed);
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

  function sortedLanguages() {
    var langs = (state.snapshot && state.snapshot.languages) || [];
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
      els.tableBody.innerHTML = '<tr class="progress-table-empty"><td colspan="8">No languages found in the snapshot.</td></tr>';
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
      if (active) {
        btn.setAttribute('data-direction', state.sort.direction);
        btn.setAttribute('aria-sort', state.sort.direction === 'asc' ? 'ascending' : 'descending');
      } else {
        btn.removeAttribute('data-direction');
        btn.setAttribute('aria-sort', 'none');
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

  // ---------- chart ----------

  function renderChart(series) {
    if (!els.chart) return;
    // Series is null → snapshot says "log file missing" → data unavailable.
    // Series is [] → log exists but empty/invalid → also data unavailable.
    if (!Array.isArray(series) || series.length === 0) {
      els.chart.innerHTML = '<p class="progress-chart-empty">Growth data is not yet available. The daily log (<code>data/progress_log.jsonl</code>) will begin populating on the next scheduled snapshot.</p>';
      if (els.chartDesc) els.chartDesc.textContent = '';
      return;
    }

    // Tiny line chart. Intentionally no axis library — keep the dashboard
    // light. Padding is picked so the largest Y-value gets 90% of the
    // drawing height and the oldest/newest dates anchor the X axis.
    var W = 640;
    var H = 240;
    var P = { top: 24, right: 24, bottom: 40, left: 56 };

    var xs = series.map(function (p, i) { return i; });
    var ys = series.map(function (p) { return p.reviewed; });
    var yMin = 0;
    var yMax = Math.max.apply(null, ys);
    if (yMax === yMin) yMax = yMin + 1;
    var xMax = series.length - 1;

    function xCoord(i) {
      if (xMax === 0) return P.left + (W - P.left - P.right) / 2;
      return P.left + (i / xMax) * (W - P.left - P.right);
    }
    function yCoord(v) {
      return P.top + (1 - (v - yMin) / (yMax - yMin)) * (H - P.top - P.bottom);
    }

    var pathCoords = series.map(function (p, i) {
      return (i === 0 ? 'M' : 'L') + xCoord(i).toFixed(1) + ',' + yCoord(p.reviewed).toFixed(1);
    }).join(' ');
    // Close the area under the curve back to the baseline for a soft fill.
    var areaCoords = pathCoords
      + ' L' + xCoord(xMax).toFixed(1) + ',' + yCoord(yMin).toFixed(1)
      + ' L' + xCoord(0).toFixed(1) + ',' + yCoord(yMin).toFixed(1) + ' Z';

    var firstPoint = series[0];
    var lastPoint = series[series.length - 1];
    var firstLabel = escapeHtml(firstPoint.date);
    var lastLabel = escapeHtml(lastPoint.date);

    var ticks = 3;
    var yTickSvg = [];
    for (var t = 0; t <= ticks; t++) {
      var v = yMin + ((yMax - yMin) * t) / ticks;
      var y = yCoord(v);
      yTickSvg.push(
        '<line class="axis-line" x1="' + P.left + '" x2="' + (W - P.right) + '" y1="' + y.toFixed(1) + '" y2="' + y.toFixed(1) + '" />'
        + '<text class="axis-tick-label" x="' + (P.left - 8) + '" y="' + (y + 4).toFixed(1) + '" text-anchor="end">' + Math.round(v).toLocaleString('en-US') + '</text>'
      );
    }

    var pointsSvg = series.map(function (p, i) {
      return '<circle class="series-point" cx="' + xCoord(i).toFixed(1) + '" cy="' + yCoord(p.reviewed).toFixed(1) + '" r="3" />';
    }).join('');

    els.chart.innerHTML =
      '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-labelledby="progressChartTitle progressChartDescription">'
        + '<title id="progressChartTitle">Deaf-native-reviewed signs over time</title>'
        + yTickSvg.join('')
        + '<line class="axis-line" x1="' + P.left + '" x2="' + P.left + '" y1="' + P.top + '" y2="' + (H - P.bottom) + '" />'
        + '<line class="axis-line" x1="' + P.left + '" x2="' + (W - P.right) + '" y1="' + (H - P.bottom) + '" y2="' + (H - P.bottom) + '" />'
        + '<path class="series-area" d="' + areaCoords + '" />'
        + '<path class="series-path" d="' + pathCoords + '" />'
        + pointsSvg
        + '<text class="axis-tick-label" x="' + P.left + '" y="' + (H - 10) + '" text-anchor="start">' + firstLabel + '</text>'
        + '<text class="axis-tick-label" x="' + (W - P.right) + '" y="' + (H - 10) + '" text-anchor="end">' + lastLabel + '</text>'
      + '</svg>';

    // Accessible text summary — required by the spec for a non-visual
    // reading of the trend.
    if (els.chartDesc) {
      var first = firstPoint.reviewed;
      var last = lastPoint.reviewed;
      var delta = last - first;
      var deltaSign = delta > 0 ? '+' : delta < 0 ? '' : '';
      var trend;
      if (series.length === 1) {
        trend = 'A single data point: ' + fmtNum(last) + ' Deaf-native-reviewed signs on ' + lastPoint.date + '.';
      } else if (delta === 0) {
        trend = 'Flat between ' + firstPoint.date + ' and ' + lastPoint.date + ' at ' + fmtNum(last) + ' reviewed signs.';
      } else if (delta > 0) {
        trend = 'Grew from ' + fmtNum(first) + ' on ' + firstPoint.date + ' to ' + fmtNum(last) + ' on ' + lastPoint.date + ' (' + deltaSign + fmtNum(delta) + ' reviewed signs).';
      } else {
        trend = 'Declined from ' + fmtNum(first) + ' on ' + firstPoint.date + ' to ' + fmtNum(last) + ' on ' + lastPoint.date + ' (' + fmtNum(delta) + ' reviewed signs).';
      }
      els.chartDesc.textContent = trend;
    }
  }

  // ---------- recent activity ----------

  function renderRecent(events) {
    if (!els.recent) return;
    if (!Array.isArray(events) || events.length === 0) {
      els.recent.innerHTML = '<li class="progress-recent-empty">No recent validations to show yet. New signs will appear here as they clear review.</li>';
      return;
    }
    els.recent.innerHTML = events.map(function (ev) {
      var reviewerText = (typeof ev.reviewer_count === 'number' && ev.reviewer_count > 0)
        ? ev.reviewer_count + ' reviewer' + (ev.reviewer_count === 1 ? '' : 's')
        : 'reviewed';
      return '<li>'
        + '<span class="progress-recent-gloss">' + escapeHtml(ev.gloss) + '</span>'
        + '<span class="progress-recent-lang">' + escapeHtml(ev.language) + '</span>'
        + '<span class="progress-recent-meta">' + escapeHtml(reviewerText) + ' · ' + escapeHtml(ev.timestamp) + '</span>'
      + '</li>';
    }).join('');
  }

  // ---------- help-wanted ----------

  function renderHelp(list, words, targetLang) {
    if (!list) return;
    if (!Array.isArray(words) || words.length === 0) {
      list.innerHTML = '<li class="progress-help-done">Nothing urgent — every common word has coverage here.</li>';
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
    if (els.chart) {
      els.chart.innerHTML = '<p class="progress-chart-empty">' + escapeHtml(message) + '</p>';
    }
    if (els.recent) {
      els.recent.innerHTML = '<li class="progress-recent-empty">' + escapeHtml(message) + '</li>';
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
        renderChart(snap && snap.progress_series);
        renderRecent(snap && snap.recent_activity);
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
      load();
    });
  } else {
    attachSortHandlers();
    load();
  }
})();

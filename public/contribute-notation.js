/* Contribute page — notation panel controller.
 *
 * Spec: prompts/prompt-contrib-7.md.
 *
 * Scope:
 * - Two-tab panel (HamNoSys default + SiGML). Plain text tabs with an
 *   underline on the active one; no pills, no icons.
 * - HamNoSys tab: large glyph display, per-char legend on hover/focus/tap,
 *   phonological breakdown from the session envelope's parameters, Copy
 *   HamNoSys button.
 * - SiGML tab: hand-tokenised XML highlighter with line numbers, Copy
 *   SiGML + Download .sigml buttons, one-sentence summary.
 * - Crossfade transitions (200ms opacity) on new notation arriving via
 *   the session envelope (covers both POST responses and SSE pushes).
 * - Validation errors render in a muted red box below the glyph. After
 *   two retries the panel collapses the error list into a single-line
 *   fallback so the contributor can still submit.
 * - Accessibility: the glyph display carries an aria-label built from
 *   the phonological breakdown so screen readers read English rather
 *   than trying to pronounce PUA codepoints. The SiGML tab's XML region
 *   uses role="region" + an explicit label (set in the HTML).
 */
(function () {
  'use strict';

  var CTX = window.KOZHA_CONTRIB_CONTEXT;
  if (!CTX) {
    if (window.console) console.error('[contribute-notation] contribute-context.js must load first');
    return;
  }

  // Small shim so each string call gets the live catalog value if present
  // but always has an English fallback — this panel must be readable even
  // if /strings.en.json never arrives.
  function tr(key, fallback) {
    if (window.KOZHA_I18N && typeof window.KOZHA_I18N.t === 'function') {
      var v = window.KOZHA_I18N.t(key);
      if (v && v !== key) return v;
    }
    return fallback;
  }

  var els = {
    panel:           document.getElementById('notationPanel'),
    tabHamnosys:     document.getElementById('notationTabHamnosys'),
    tabSigml:        document.getElementById('notationTabSigml'),
    tabpanelHam:     document.getElementById('notationPanelHamnosys'),
    tabpanelSig:     document.getElementById('notationPanelSigml'),
    display:         document.getElementById('hamnosysDisplay'),
    legendCode:      document.getElementById('legendCode'),
    legendClass:     document.getElementById('legendClass'),
    legendName:      document.getElementById('legendName'),
    errors:          document.getElementById('notationErrors'),
    fallback:        document.getElementById('notationFallback'),
    bHandshape:      document.getElementById('breakdownHandshape'),
    bOrientation:    document.getElementById('breakdownOrientation'),
    bLocation:       document.getElementById('breakdownLocation'),
    bMovement:       document.getElementById('breakdownMovement'),
    copyHam:         document.getElementById('copyHamnosysBtn'),
    copyHamConfirm:  document.getElementById('copyHamnosysConfirm'),
    sigmlDisplay:    document.getElementById('sigmlDisplay'),
    sigmlCode:       document.getElementById('sigmlCode'),
    copySig:         document.getElementById('copySigmlBtn'),
    copySigConfirm:  document.getElementById('copySigmlConfirm'),
    downloadSig:     document.getElementById('downloadSigmlBtn'),
  };

  if (!els.panel || !els.display) {
    return;
  }

  var API_BASE = '/api/chat2hamnosys';
  var SYMBOLS_URL = API_BASE + '/hamnosys/symbols';
  var FADE_MS = 200;
  var COPY_CONFIRM_MS = 1400;
  // After this many consecutive failed validations, swap the error list
  // for the "reviewer will finish it" fallback. Two retries ≡ three total
  // attempts; incrementing *after* the first failure lines up with the
  // common "first try + two retries" reading.
  var MAX_FAILED_ATTEMPTS = 3;

  var state = {
    symbolsByHex:        null,
    fetchingSymbols:     false,
    activeTab:           'hamnosys',
    lastHamnosys:        null,
    lastSigml:           null,
    lastErrorKey:        '',
    failedAttempts:      0,
    rememberedSessionId: null,
    activeGlyphEl:       null,
    // EventSource-equivalent we drive with fetch+ReadableStream so we
    // can attach the session token header.
    sseAbort:            null,
    sseSessionId:        null,
    sseSessionToken:     null,
    sseLastEventId:      null,
    sseReconnectTimer:   null,
    sseRetryMs:          0,
  };

  // Exponential backoff for SSE reconnection. Starts at 1s, caps at 30s.
  // Resets to the base after a successful connection (the first frame
  // delivered) so a transient blip doesn't permanently inflate the delay.
  var SSE_RETRY_BASE_MS = 1000;
  var SSE_RETRY_CAP_MS  = 30000;

  // ---------- symbol table ----------

  function loadSymbolTable() {
    if (state.symbolsByHex || state.fetchingSymbols) return;
    state.fetchingSymbols = true;
    fetch(SYMBOLS_URL, {
      headers: { Accept: 'application/json' },
      cache: 'force-cache',
    })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (payload) {
        state.fetchingSymbols = false;
        var map = {};
        if (payload && payload.symbols) {
          for (var i = 0; i < payload.symbols.length; i++) {
            var s = payload.symbols[i];
            map[s.hex] = s;
          }
        }
        state.symbolsByHex = map;
        if (state.activeGlyphEl) {
          updateLegendForHex(state.activeGlyphEl.getAttribute('data-hex'));
        }
      })
      .catch(function () {
        state.fetchingSymbols = false;
        state.symbolsByHex = {};
      });
  }

  function hexFor(ch) {
    var cp = ch.codePointAt(0);
    var hex = cp.toString(16).toUpperCase();
    while (hex.length < 4) hex = '0' + hex;
    return 'U+' + hex;
  }

  function lookupSymbol(hex) {
    if (!hex || !state.symbolsByHex) return null;
    return state.symbolsByHex[hex] || null;
  }

  function updateLegendForHex(hex) {
    if (!hex) {
      resetLegend();
      return;
    }
    els.legendCode.textContent = hex;
    var sym = lookupSymbol(hex);
    if (!sym) {
      els.legendClass.textContent = state.symbolsByHex
        ? tr('contribute.notation.legend_unknown', 'Unknown symbol')
        : tr('contribute.notation.legend_loading', 'Loading…');
      els.legendName.textContent = '';
      return;
    }
    els.legendClass.textContent = sym.class_label || sym['class'] || '';
    els.legendName.textContent = sym.short_name || '';
  }

  function resetLegend() {
    els.legendCode.textContent = tr('contribute.notation.legend_default_code', '—');
    els.legendClass.textContent = tr('contribute.notation.legend_default_class', 'Hover or tap a glyph');
    els.legendName.textContent = '';
  }

  // ---------- HamNoSys glyph rendering ----------

  function clearChildren(el) {
    while (el.firstChild) el.removeChild(el.firstChild);
  }

  function renderHamnosysString(s) {
    clearChildren(els.display);
    state.activeGlyphEl = null;
    if (!s) {
      els.display.classList.add('is-empty');
      els.display.textContent = tr('contribute.notation.display_empty', 'No HamNoSys generated yet.');
      return;
    }
    els.display.classList.remove('is-empty');
    // Array.from iterates by code point, which matters for any
    // non-BMP codepoint (HamNoSys is PUA in the BMP today but the
    // helper keeps the code future-proof).
    var chars = Array.from(s);
    for (var i = 0; i < chars.length; i++) {
      var ch = chars[i];
      if (ch === '\n' || ch === ' ' || ch === '\t') {
        els.display.appendChild(document.createTextNode(ch));
        continue;
      }
      var span = document.createElement('span');
      span.className = 'notation-glyph';
      span.textContent = ch;
      span.setAttribute('data-hex', hexFor(ch));
      span.setAttribute('tabindex', '0');
      span.addEventListener('mouseenter', onGlyphActivate);
      span.addEventListener('focus', onGlyphActivate);
      span.addEventListener('click', onGlyphActivate);
      els.display.appendChild(span);
    }
  }

  function onGlyphActivate(ev) {
    var el = ev.currentTarget;
    if (state.activeGlyphEl && state.activeGlyphEl !== el) {
      state.activeGlyphEl.classList.remove('is-active');
    }
    el.classList.add('is-active');
    state.activeGlyphEl = el;
    updateLegendForHex(el.getAttribute('data-hex'));
  }

  // ---------- breakdown ----------

  function setBreakdownSlot(dd, text) {
    if (text) {
      dd.textContent = text;
      dd.classList.remove('is-blank');
    } else {
      dd.textContent = '—';
      dd.classList.add('is-blank');
    }
  }

  function buildHandshapeText(p) {
    if (!p) return null;
    var dom = p.handshape_dominant;
    var non = p.handshape_nondominant;
    if (!dom && !non) return null;
    if (dom && non) return dom + ' (dominant); ' + non + ' (non-dominant)';
    if (dom) return dom;
    return non + ' (non-dominant)';
  }

  function buildOrientationText(p) {
    if (!p) return null;
    var parts = [];
    if (p.orientation_palm) parts.push('palm ' + p.orientation_palm);
    if (p.orientation_extended_finger) parts.push('fingers ' + p.orientation_extended_finger);
    return parts.length ? parts.join(', ') : null;
  }

  function buildMovementText(p) {
    if (!p || !p.movement || !p.movement.length) return null;
    var out = [];
    for (var i = 0; i < p.movement.length; i++) {
      var m = p.movement[i];
      if (!m) continue;
      var pieces = [];
      if (m.path) pieces.push(m.path);
      if (m.size_mod) pieces.push(m.size_mod);
      if (m.speed_mod) pieces.push(m.speed_mod);
      if (m.repeat) pieces.push(m.repeat);
      if (pieces.length) out.push(pieces.join(' '));
    }
    return out.length ? out.join('; ') : null;
  }

  function renderBreakdown(p) {
    setBreakdownSlot(els.bHandshape,   buildHandshapeText(p));
    setBreakdownSlot(els.bOrientation, buildOrientationText(p));
    setBreakdownSlot(els.bLocation,    p && p.location ? p.location : null);
    setBreakdownSlot(els.bMovement,    buildMovementText(p));
  }

  function buildAriaLabel(p, hamnosys) {
    var lines = [];
    var hs = buildHandshapeText(p);
    if (hs) lines.push(tr('contribute.notation.breakdown_handshape', 'Handshape') + ': ' + hs);
    var or = buildOrientationText(p);
    if (or) lines.push(tr('contribute.notation.breakdown_orientation', 'Orientation') + ': ' + or);
    if (p && p.location) lines.push(tr('contribute.notation.breakdown_location', 'Location') + ': ' + p.location);
    var mv = buildMovementText(p);
    if (mv) lines.push(tr('contribute.notation.breakdown_movement', 'Movement') + ': ' + mv);
    if (!lines.length) {
      return hamnosys
        ? tr('contribute.notation.display_aria_no_breakdown', 'HamNoSys notation (phonological breakdown not available)')
        : tr('contribute.notation.display_aria_empty', 'HamNoSys notation (not yet generated)');
    }
    return lines.join('. ');
  }

  // ---------- SiGML highlighter ----------

  function renderSigml(xml) {
    clearChildren(els.sigmlCode);
    if (!xml) return;
    var lines = xml.split(/\r?\n/);
    for (var i = 0; i < lines.length; i++) {
      var lineEl = document.createElement('span');
      lineEl.className = 'sigml-line';
      tokenizeSigmlLine(lineEl, lines[i]);
      els.sigmlCode.appendChild(lineEl);
    }
  }

  function tokenizeSigmlLine(container, line) {
    if (!line) {
      container.appendChild(document.createTextNode('\n'));
      return;
    }
    // Phase 1: split top-level into comments / tags / plain text.
    var topRe = /(<!--[\s\S]*?-->)|(<\/?[A-Za-z][^>]*>)/g;
    var idx = 0;
    var m;
    while ((m = topRe.exec(line)) !== null) {
      if (m.index > idx) {
        container.appendChild(document.createTextNode(line.slice(idx, m.index)));
      }
      if (m[1]) {
        appendSpan(container, 'sigml-comment', m[1]);
      } else {
        tokenizeTag(container, m[2]);
      }
      idx = m.index + m[0].length;
    }
    if (idx < line.length) {
      container.appendChild(document.createTextNode(line.slice(idx)));
    }
    container.appendChild(document.createTextNode('\n'));
  }

  function tokenizeTag(container, tag) {
    // Walks a tag token, classifying:
    //   - tag brackets + name   → sigml-tag
    //   - attribute names       → sigml-attr
    //   - quoted attribute vals → sigml-str
    //   - whitespace / punct    → plain text
    var re = /(<\/?)([A-Za-z][\w:.-]*)|(\/?>)|(\s+)|([A-Za-z_][\w:.-]*)|(=)|("[^"]*"|'[^']*')|([^\s<>\/="']+)/g;
    var m;
    while ((m = re.exec(tag)) !== null) {
      if (m[1] !== undefined) {
        appendSpan(container, 'sigml-tag', m[1] + (m[2] || ''));
      } else if (m[3]) {
        appendSpan(container, 'sigml-tag', m[3]);
      } else if (m[4]) {
        container.appendChild(document.createTextNode(m[4]));
      } else if (m[5]) {
        appendSpan(container, 'sigml-attr', m[5]);
      } else if (m[6]) {
        container.appendChild(document.createTextNode(m[6]));
      } else if (m[7]) {
        appendSpan(container, 'sigml-str', m[7]);
      } else if (m[8]) {
        container.appendChild(document.createTextNode(m[8]));
      }
    }
  }

  function appendSpan(container, cls, text) {
    var s = document.createElement('span');
    s.className = cls;
    s.textContent = text;
    container.appendChild(s);
  }

  // ---------- tabs ----------

  function setActiveTab(name) {
    state.activeTab = name === 'sigml' ? 'sigml' : 'hamnosys';
    var isHam = state.activeTab === 'hamnosys';
    els.tabHamnosys.classList.toggle('is-active', isHam);
    els.tabSigml.classList.toggle('is-active', !isHam);
    els.tabHamnosys.setAttribute('aria-selected', isHam ? 'true' : 'false');
    els.tabSigml.setAttribute('aria-selected', isHam ? 'false' : 'true');
    els.tabHamnosys.setAttribute('tabindex', isHam ? '0' : '-1');
    els.tabSigml.setAttribute('tabindex', isHam ? '-1' : '0');
    els.tabpanelHam.hidden = !isHam;
    els.tabpanelSig.hidden = isHam;
  }

  function onTabClick(ev) {
    setActiveTab(ev.currentTarget.getAttribute('data-tab'));
  }

  function onTabKey(ev) {
    var key = ev.key;
    if (key !== 'ArrowLeft' && key !== 'ArrowRight' &&
        key !== 'Home'      && key !== 'End') return;
    ev.preventDefault();
    if (key === 'Home') {
      setActiveTab('hamnosys'); els.tabHamnosys.focus(); return;
    }
    if (key === 'End') {
      setActiveTab('sigml'); els.tabSigml.focus(); return;
    }
    var next = state.activeTab === 'hamnosys' ? 'sigml' : 'hamnosys';
    setActiveTab(next);
    (next === 'hamnosys' ? els.tabHamnosys : els.tabSigml).focus();
  }

  // ---------- errors + fallback ----------

  function errorKey(errors) {
    return (errors || []).join('');
  }

  function renderValidationErrors(errors, hamnosys) {
    var hasErrors = errors && errors.length > 0;
    if (!hasErrors) {
      els.errors.hidden = true;
      els.errors.textContent = '';
      els.fallback.hidden = true;
      return;
    }
    if (state.failedAttempts >= MAX_FAILED_ATTEMPTS) {
      els.errors.hidden = true;
      els.fallback.hidden = false;
      return;
    }
    els.fallback.hidden = true;
    els.errors.hidden = false;
    els.errors.textContent = errors.map(function (line) {
      return '• ' + line;
    }).join('\n');
  }

  // ---------- copy + download ----------

  function copyToClipboard(text) {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      try {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.setAttribute('readonly', '');
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        var ok = document.execCommand && document.execCommand('copy');
        document.body.removeChild(ta);
        if (ok) resolve(); else reject(new Error('execCommand copy failed'));
      } catch (e) { reject(e); }
    });
  }

  function showCopyConfirm(el) {
    if (!el) return;
    el.hidden = false;
    el.classList.remove('is-fading');
    if (el._hideT) clearTimeout(el._hideT);
    el._hideT = setTimeout(function () {
      el.classList.add('is-fading');
      el._hideT = setTimeout(function () {
        el.hidden = true;
        el.classList.remove('is-fading');
      }, 400);
    }, COPY_CONFIRM_MS);
  }

  function onCopyHam() {
    if (!state.lastHamnosys) return;
    copyToClipboard(state.lastHamnosys).then(function () {
      showCopyConfirm(els.copyHamConfirm);
    }).catch(function () { /* user will see no confirmation; no toast here */ });
  }

  function onCopySig() {
    if (!state.lastSigml) return;
    copyToClipboard(state.lastSigml).then(function () {
      showCopyConfirm(els.copySigConfirm);
    }).catch(function () { /* silent */ });
  }

  function downloadFilename() {
    var snap = CTX.getState();
    var gloss = (snap.gloss || 'sign').toLowerCase().replace(/[^a-z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') || 'sign';
    var lang = (snap.language || 'xx').toLowerCase();
    return gloss + '_' + lang + '.sigml';
  }

  function onDownloadSig() {
    if (!state.lastSigml) return;
    var blob = new Blob([state.lastSigml], { type: 'application/xml' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = downloadFilename();
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    // Defer revoke so Safari actually honours the click before GC.
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  // ---------- crossfade ----------

  function crossfade(applyUpdate) {
    els.display.classList.add('is-fading');
    els.sigmlDisplay.classList.add('is-fading');
    setTimeout(function () {
      applyUpdate();
      // Next frame so the browser sees the updated content before
      // we clear is-fading (otherwise the opacity would snap).
      requestAnimationFrame(function () {
        els.display.classList.remove('is-fading');
        els.sigmlDisplay.classList.remove('is-fading');
      });
    }, FADE_MS);
  }

  // ---------- snapshot + render ----------

  function sessionHasNotation(snap) {
    return !!(snap.hamnosys || snap.sigml);
  }

  function shouldShowPanel(snap) {
    if (!snap.sessionId) return false;
    if (snap.sessionState === 'finalized' || snap.sessionState === 'abandoned') {
      // Once finalized, the envelope still carries the notation, so
      // surfacing it is harmless and sometimes desirable; but for the
      // "session gone" case (abandoned), hide entirely.
      return snap.sessionState === 'finalized' && sessionHasNotation(snap);
    }
    // Otherwise show as soon as any notation has landed (GENERATING
    // has the preview shimmer in the chat panel, so the notation panel
    // doesn't need its own spinner).
    return sessionHasNotation(snap);
  }

  function resetForNewSession() {
    state.lastHamnosys = null;
    state.lastSigml = null;
    state.lastErrorKey = '';
    state.failedAttempts = 0;
    state.activeGlyphEl = null;
    renderHamnosysString(null);
    renderSigml('');
    renderBreakdown(null);
    resetLegend();
    els.errors.hidden = true;
    els.fallback.hidden = true;
    els.copyHam.disabled = true;
    els.copySig.disabled = true;
    els.downloadSig.disabled = true;
    els.display.setAttribute('aria-label', 'HamNoSys notation (not yet generated)');
    setActiveTab('hamnosys');
  }

  function applySnapshot(snap) {
    var newHam = snap.hamnosys || null;
    var newSig = snap.sigml || null;
    var newErrorKey = errorKey(snap.generationErrors);

    var hamnosysChanged = newHam !== state.lastHamnosys;
    var sigmlChanged = newSig !== state.lastSigml;

    // Increment the failed-attempts counter when a *new* error set lands.
    // Clears when a successful notation comes through.
    if (newErrorKey && newErrorKey !== state.lastErrorKey) {
      state.failedAttempts += 1;
      state.lastErrorKey = newErrorKey;
    } else if (!newErrorKey) {
      state.lastErrorKey = '';
      // Don't reset failedAttempts here: a success after a retry is
      // expected; keeping the counter lets the fallback stay if the
      // contributor does a second failing correction after that.
      // However, if a fresh, valid hamnosys arrives, zero it — the
      // validator passed.
      if (newHam && !snap.generationErrors.length) {
        state.failedAttempts = 0;
      }
    }

    var applyUpdate = function () {
      renderHamnosysString(newHam);
      renderSigml(newSig || '');
      renderBreakdown(snap.parameters);
      els.display.setAttribute('aria-label', buildAriaLabel(snap.parameters, newHam));
      els.copyHam.disabled = !newHam;
      els.copySig.disabled = !newSig;
      els.downloadSig.disabled = !newSig;
      renderValidationErrors(snap.generationErrors, newHam);
      state.lastHamnosys = newHam;
      state.lastSigml = newSig;
    };

    if (hamnosysChanged || sigmlChanged) {
      crossfade(applyUpdate);
    } else {
      // Breakdown / errors can change without the glyph changing; still
      // push the update but skip the fade.
      applyUpdate();
    }
  }

  function onSnapshot(snap) {
    if (!shouldShowPanel(snap)) {
      els.panel.hidden = true;
      stopSse();
      return;
    }
    els.panel.hidden = false;
    if (state.rememberedSessionId !== snap.sessionId) {
      resetForNewSession();
      state.rememberedSessionId = snap.sessionId;
      startSse(snap.sessionId, snap.sessionToken);
    }
    applySnapshot(snap);
  }

  // ---------- SSE push ----------
  //
  // Subscribe to the per-session event stream so a background render
  // pipeline (e.g. an async generator that finishes after the POST has
  // already returned) still triggers the crossfade. Uses fetch +
  // ReadableStream instead of EventSource because we need to send
  // ``X-Session-Token`` as a header, which EventSource cannot do.

  function stopSse() {
    if (state.sseAbort) {
      try { state.sseAbort.abort(); } catch (_e) {}
    }
    if (state.sseReconnectTimer) {
      clearTimeout(state.sseReconnectTimer);
      state.sseReconnectTimer = null;
    }
    state.sseAbort = null;
    state.sseSessionId = null;
    state.sseSessionToken = null;
    state.sseLastEventId = null;
    state.sseRetryMs = 0;
  }

  function scheduleSseReconnect() {
    // Exponential backoff: double the previous wait, capped at the
    // configured ceiling. A missed frame is not a data-loss event — the
    // server honours Last-Event-ID so resuming picks up where we left
    // off.
    var sessionId = state.sseSessionId;
    var token = state.sseSessionToken;
    if (!sessionId || !token) return;
    if (state.sseReconnectTimer) return;
    var next = state.sseRetryMs
      ? Math.min(state.sseRetryMs * 2, SSE_RETRY_CAP_MS)
      : SSE_RETRY_BASE_MS;
    state.sseRetryMs = next;
    state.sseReconnectTimer = setTimeout(function () {
      state.sseReconnectTimer = null;
      // Guard against a session swap landing during the backoff delay.
      if (state.sseSessionId === sessionId && state.rememberedSessionId === sessionId) {
        openSseConnection();
      }
    }, next);
  }

  function startSse(sessionId, token) {
    stopSse();
    if (!sessionId || !token) return;
    if (typeof AbortController === 'undefined') return;
    if (typeof TextDecoder === 'undefined') return;
    state.sseSessionId = sessionId;
    state.sseSessionToken = token;
    state.sseLastEventId = null;
    state.sseRetryMs = 0;
    openSseConnection();
  }

  function openSseConnection() {
    var sessionId = state.sseSessionId;
    var token = state.sseSessionToken;
    if (!sessionId || !token) return;
    var ctrl = new AbortController();
    state.sseAbort = ctrl;
    var headers = {
      'Accept':          'text/event-stream',
      'X-Session-Token': token,
    };
    // Only send Last-Event-ID on a reconnect — the first open should
    // replay the whole history so the chat panel can populate from a
    // cold start too.
    if (state.sseLastEventId !== null && state.sseLastEventId !== undefined) {
      headers['Last-Event-ID'] = String(state.sseLastEventId);
    }
    fetch(API_BASE + '/sessions/' + encodeURIComponent(sessionId) + '/events', {
      method: 'GET',
      headers: headers,
      signal: ctrl.signal,
    })
      .then(function (r) {
        if (!r.ok || !r.body) {
          // 4xx with a body usually means the session was rejected or
          // forbidden — reconnecting won't fix either, so stop.
          if (r.status >= 400 && r.status < 500) return;
          scheduleSseReconnect();
          return;
        }
        return consumeEventStream(r.body, sessionId).then(function () {
          // Clean stream end (server said timeout / closed). Reconnect so
          // events emitted after the next poll round still reach us.
          if (state.sseSessionId === sessionId) scheduleSseReconnect();
        });
      })
      .catch(function (err) {
        // AbortController-triggered abort surfaces as AbortError; that's
        // intentional stoppage, not a network drop.
        if (err && err.name === 'AbortError') return;
        scheduleSseReconnect();
      });
  }

  function consumeEventStream(body, sessionId) {
    var reader = body.getReader();
    var decoder = new TextDecoder('utf-8');
    var buf = '';
    function pump() {
      return reader.read().then(function (res) {
        if (res.done) return;
        buf += decoder.decode(res.value, { stream: true });
        // SSE frames are separated by a blank line.
        var parts = buf.split(/\n\n/);
        buf = parts.pop() || '';
        for (var i = 0; i < parts.length; i++) {
          handleSseFrame(parts[i], sessionId);
        }
        return pump();
      });
    }
    return pump();
  }

  function handleSseFrame(frame, sessionId) {
    // Only GeneratedEvent (type=generated) frames drive the crossfade;
    // other events are handled by the chat panel via envelope replay.
    var lines = frame.split('\n');
    var evName = '';
    var data = '';
    var eventId = null;
    for (var i = 0; i < lines.length; i++) {
      var line = lines[i];
      if (line.indexOf('event:') === 0) {
        evName = line.slice(6).trim();
      } else if (line.indexOf('data:') === 0) {
        data += (data ? '\n' : '') + line.slice(5).trim();
      } else if (line.indexOf('id:') === 0) {
        eventId = line.slice(3).trim();
      }
    }
    // A successfully-parsed frame confirms the server is reachable —
    // reset the backoff so any later drop starts fresh.
    if (state.sseRetryMs) state.sseRetryMs = 0;
    if (eventId !== null && eventId !== '') {
      state.sseLastEventId = eventId;
    }
    if (evName !== 'generated' || !data) return;
    if (state.rememberedSessionId !== sessionId) return;
    var payload;
    try { payload = JSON.parse(data); }
    catch (_e) { return; }
    if (!payload || payload.type !== 'generated') return;
    // Short-circuit: the POST /answer (etc.) responses already call
    // applyEnvelope via the context store. Only fire the crossfade when
    // this frame delivers notation the store hasn't observed yet.
    if (payload.hamnosys && payload.hamnosys !== state.lastHamnosys) {
      CTX.setState({
        hamnosys:         payload.hamnosys,
        sigml:            payload.sigml || null,
        generationErrors: payload.errors || [],
      });
    }
  }

  // ---------- public surface ----------

  window.KOZHA_CONTRIB_NOTATION = {
    setActiveTab:  setActiveTab,
    getActiveTab:  function () { return state.activeTab; },
    renderForTest: function (opts) {
      // Tiny hook so tests can drive the panel end-to-end through the
      // store — the subscriber handles show/hide, the crossfade, and
      // the breakdown. Production code never calls this.
      opts = opts || {};
      CTX.setState({
        sessionId:        opts.sessionId || 'test',
        sessionToken:     opts.sessionToken || 'test',
        sessionState:     opts.sessionState || 'rendered',
        language:         opts.language || 'bsl',
        gloss:            opts.gloss || 'TEMPLE',
        hamnosys:         opts.hamnosys || null,
        sigml:            opts.sigml || null,
        parameters:       opts.parameters || null,
        generationErrors: opts.generationErrors || [],
      });
    },
  };

  // ---------- init ----------

  function init() {
    els.tabHamnosys.addEventListener('click', onTabClick);
    els.tabSigml.addEventListener('click', onTabClick);
    els.tabHamnosys.addEventListener('keydown', onTabKey);
    els.tabSigml.addEventListener('keydown', onTabKey);
    els.copyHam.addEventListener('click', onCopyHam);
    els.copySig.addEventListener('click', onCopySig);
    els.downloadSig.addEventListener('click', onDownloadSig);
    els.display.addEventListener('mouseleave', function () {
      // Leaving the display entirely clears the legend selection —
      // clicking a glyph re-sticks it.
      if (state.activeGlyphEl) {
        state.activeGlyphEl.classList.remove('is-active');
        state.activeGlyphEl = null;
      }
      resetLegend();
    });
    resetLegend();
    loadSymbolTable();
    CTX.subscribe(onSnapshot);
    onSnapshot(CTX.getState());
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

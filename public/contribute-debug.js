/* Contribute page — shared debug log + drawer.
 *
 * What it does
 * ------------
 * - Exposes ``window.KOZHA_CONTRIB_DEBUG`` with a tiny ``log(msg, data?)``
 *   API every contribute-page module calls when something interesting
 *   happens (an LLM call started/retried/failed, the CWASA bundle
 *   loaded, a snapshot arrived, etc.).
 * - Holds the last 200 entries in memory.
 * - Renders a fixed-position toggle button at the bottom-right of the
 *   screen (only when debug mode is on) that opens a slide-up drawer
 *   showing the live log.
 * - Debug mode is enabled by ``?debug=1`` in the URL or by setting
 *   ``localStorage.kozha_debug = '1'``. Once enabled it persists across
 *   reloads of the same browser; ``?debug=0`` clears it.
 *
 * Why it exists
 * -------------
 * The contribute pipeline is multi-step and crosses the LLM boundary in
 * several places (clarification, slot resolution, SiGML-direct, repair,
 * preview). When something goes wrong, the contributor gets a single
 * vague "AI had trouble" message; the developer needs to see the chain
 * of API calls + envelopes + CWASA boot states to diagnose. This panel
 * is that view, opt-in so it never clutters the default UX.
 */
(function () {
  'use strict';

  var MAX_ENTRIES = 200;
  var STORAGE_KEY = 'kozha_debug';

  var enabled = computeEnabled();
  var entries = [];
  var listeners = [];
  var drawer = null;

  function computeEnabled() {
    try {
      var url = new URL(window.location.href);
      var p = url.searchParams.get('debug');
      if (p === '1' || p === 'true') {
        try { localStorage.setItem(STORAGE_KEY, '1'); } catch (_e) {}
        return true;
      }
      if (p === '0' || p === 'false') {
        try { localStorage.removeItem(STORAGE_KEY); } catch (_e) {}
        return false;
      }
    } catch (_e) { /* ignore — bad URL parse */ }
    try {
      return localStorage.getItem(STORAGE_KEY) === '1';
    } catch (_e) {
      return false;
    }
  }

  function timestamp() {
    var now = new Date();
    var pad = function (n, w) { var s = String(n); while (s.length < w) s = '0' + s; return s; };
    return (
      pad(now.getHours(), 2) + ':' +
      pad(now.getMinutes(), 2) + ':' +
      pad(now.getSeconds(), 2) + '.' +
      pad(now.getMilliseconds(), 3)
    );
  }

  function safeStringify(value) {
    if (value === undefined) return '';
    if (value === null) return 'null';
    if (typeof value === 'string') return value;
    try {
      // Truncate very long strings inside the payload so an envelope
      // dump doesn't blow up the drawer.
      var seen = [];
      return JSON.stringify(value, function (_k, v) {
        if (typeof v === 'string' && v.length > 800) {
          return v.slice(0, 800) + '… (' + v.length + ' chars total)';
        }
        if (v && typeof v === 'object') {
          if (seen.indexOf(v) >= 0) return '[circular]';
          seen.push(v);
        }
        return v;
      }, 2);
    } catch (_e) {
      try { return String(value); } catch (_e2) { return '<unserialisable>'; }
    }
  }

  function log(message, data) {
    var entry = {
      at:   timestamp(),
      msg:  String(message || ''),
      data: data === undefined ? null : data,
    };
    entries.push(entry);
    if (entries.length > MAX_ENTRIES) entries.splice(0, entries.length - MAX_ENTRIES);
    notify();
    if (window.console && typeof window.console.debug === 'function') {
      window.console.debug('[contrib-debug]', entry.msg, data === undefined ? '' : data);
    }
  }

  function notify() {
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](entries); } catch (_e) { /* swallow */ }
    }
  }

  function snapshot() { return entries.slice(); }

  function clear() { entries = []; notify(); }

  function isEnabled() { return enabled; }

  // ---------- drawer UI ----------

  function ensureDrawer() {
    if (drawer || !enabled) return drawer;
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'kz-debug-toggle';
    btn.setAttribute('aria-label', 'Open debug log');
    btn.setAttribute('aria-expanded', 'false');
    btn.textContent = 'Debug';
    document.body.appendChild(btn);

    var panel = document.createElement('aside');
    panel.className = 'kz-debug-panel';
    panel.setAttribute('aria-label', 'Debug log');
    panel.setAttribute('role', 'log');
    panel.setAttribute('aria-live', 'polite');
    panel.hidden = true;
    document.body.appendChild(panel);

    var head = document.createElement('header');
    head.className = 'kz-debug-head';
    var title = document.createElement('span');
    title.className = 'kz-debug-title';
    title.textContent = 'Debug log';
    var clearBtn = document.createElement('button');
    clearBtn.type = 'button';
    clearBtn.className = 'kz-debug-action';
    clearBtn.textContent = 'Clear';
    clearBtn.addEventListener('click', clear);
    var copyBtn = document.createElement('button');
    copyBtn.type = 'button';
    copyBtn.className = 'kz-debug-action';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', function () {
      var text = entries.map(function (e) {
        return '[' + e.at + '] ' + e.msg + (e.data === null ? '' : '\n  ' + safeStringify(e.data));
      }).join('\n');
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).catch(function () {});
      }
    });
    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'kz-debug-action';
    closeBtn.setAttribute('aria-label', 'Close debug panel');
    closeBtn.textContent = '×';
    closeBtn.addEventListener('click', function () { setOpen(false); });
    head.appendChild(title);
    head.appendChild(clearBtn);
    head.appendChild(copyBtn);
    head.appendChild(closeBtn);
    panel.appendChild(head);

    var body = document.createElement('div');
    body.className = 'kz-debug-body';
    panel.appendChild(body);

    function setOpen(open) {
      panel.hidden = !open;
      btn.setAttribute('aria-expanded', open ? 'true' : 'false');
      if (open) render();
    }

    btn.addEventListener('click', function () {
      setOpen(panel.hidden);
    });

    function render() {
      // Render in newest-first order; cap to MAX_ENTRIES (already
      // bounded but defensive against future tweaks).
      while (body.firstChild) body.removeChild(body.firstChild);
      var rows = entries.slice().reverse();
      for (var i = 0; i < rows.length; i++) {
        var e = rows[i];
        var row = document.createElement('div');
        row.className = 'kz-debug-row';
        var ts = document.createElement('span');
        ts.className = 'kz-debug-ts';
        ts.textContent = e.at;
        var msg = document.createElement('span');
        msg.className = 'kz-debug-msg';
        msg.textContent = e.msg;
        row.appendChild(ts);
        row.appendChild(msg);
        if (e.data !== null) {
          var pre = document.createElement('pre');
          pre.className = 'kz-debug-data';
          pre.textContent = safeStringify(e.data);
          row.appendChild(pre);
        }
        body.appendChild(row);
      }
    }

    listeners.push(function () { if (!panel.hidden) render(); });

    drawer = { btn: btn, panel: panel, render: render };
    return drawer;
  }

  // ---------- public surface ----------

  window.KOZHA_CONTRIB_DEBUG = {
    log:        enabled ? log : function () { /* no-op when disabled */ },
    isEnabled:  isEnabled,
    snapshot:   snapshot,
    clear:      clear,
    // Always-on emergency log: lets a module record an event regardless
    // of whether the toggle is enabled. Useful for hard errors we want
    // to capture even if the user later switches debug on.
    forceLog:   log,
    enable: function () {
      enabled = true;
      try { localStorage.setItem(STORAGE_KEY, '1'); } catch (_e) {}
      window.KOZHA_CONTRIB_DEBUG.log = log;
      ensureDrawer();
    },
    disable: function () {
      enabled = false;
      try { localStorage.removeItem(STORAGE_KEY); } catch (_e) {}
      window.KOZHA_CONTRIB_DEBUG.log = function () {};
      if (drawer) {
        drawer.btn.remove();
        drawer.panel.remove();
        drawer = null;
      }
    },
  };

  function init() { if (enabled) ensureDrawer(); }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

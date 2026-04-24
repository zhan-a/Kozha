/* Contribute page — shared debug log + downloadable drawer.
 *
 * What it does
 * ------------
 * - Exposes ``window.KOZHA_CONTRIB_DEBUG.log(msg, data?)`` which every
 *   contribute-page module calls at every interesting moment (LLM call
 *   start/retry/fail, envelope arrived, CWASA boot stage, snapshot,
 *   correction submitted, font detection result, …). Logging is
 *   ALWAYS-ON regardless of whether the drawer is visible — the
 *   in-memory ring buffer captures the last 500 entries and is
 *   downloadable as a text file even if the user never opened debug
 *   mode beforehand. This is the user's "send me a log file" path.
 * - Renders a small fixed pill at the bottom-right that opens the
 *   drawer. The pill is hidden until either ``?debug=1`` is in the URL,
 *   ``localStorage.kozha_debug='1'`` is set, OR a hard error fires
 *   (auto-reveal so the contributor can grab the log without knowing
 *   the URL trick).
 * - Drawer header has Clear, Copy, and Download buttons. Download
 *   produces ``kozha-contrib-log-YYYY-MM-DD-HHMMSS.txt`` with full
 *   user-agent, URL, and timestamped entries.
 *
 * Why it exists
 * -------------
 * The contribute pipeline crosses the LLM boundary in several places
 * (clarification, slot resolution, SiGML-direct, repair, preview).
 * When something goes wrong the contributor sees a vague "AI had
 * trouble" message; the developer needs to see the chain of API calls
 * + envelopes + CWASA boot states to diagnose. Always-on capture +
 * downloadable export makes that chain shippable in a bug report.
 */
(function () {
  'use strict';

  var MAX_ENTRIES = 500;
  var STORAGE_KEY = 'kozha_debug';

  var entries = [];
  var listeners = [];
  var drawer = null;
  // Pill is always visible — the user asked for "log accessible all
  // the time" after hitting several opaque stalls. Opt-out via
  // ``?debug=0`` or ``localStorage.kozha_debug='0'`` stays available
  // for the rare operator who wants a clean screenshot.
  var drawerEnabled = computeDrawerEnabled();
  // Auto-reveal hook kept for external callers (e.g. the
  // window-level error listener), but now functionally equivalent
  // to drawerEnabled being true: the drawer always mounts.
  var autoRevealed = false;
  // Capture the page session start so the downloaded log carries
  // wall-clock context.
  var sessionStartedAt = new Date();

  function computeDrawerEnabled() {
    // Default-on: every contributor needs the Download button in
    // reach the moment something stalls. Only ``?debug=0`` or
    // ``localStorage.kozha_debug='0'`` opts out.
    try {
      var url = new URL(window.location.href);
      var p = url.searchParams.get('debug');
      if (p === '0' || p === 'false') {
        try { localStorage.setItem(STORAGE_KEY, '0'); } catch (_e) {}
        return false;
      }
      if (p === '1' || p === 'true') {
        try { localStorage.removeItem(STORAGE_KEY); } catch (_e) {}
        return true;
      }
    } catch (_e) { /* ignore — bad URL parse */ }
    try {
      if (localStorage.getItem(STORAGE_KEY) === '0') return false;
    } catch (_e) { /* ignore */ }
    return true;
  }

  function timestamp(d) {
    d = d || new Date();
    var pad = function (n, w) { var s = String(n); while (s.length < w) s = '0' + s; return s; };
    return (
      pad(d.getHours(), 2) + ':' +
      pad(d.getMinutes(), 2) + ':' +
      pad(d.getSeconds(), 2) + '.' +
      pad(d.getMilliseconds(), 3)
    );
  }

  function downloadStamp(d) {
    d = d || new Date();
    var pad = function (n) { var s = String(n); while (s.length < 2) s = '0' + s; return s; };
    return (
      d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
      '-' + pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds())
    );
  }

  function safeStringify(value) {
    if (value === undefined) return '';
    if (value === null) return 'null';
    if (typeof value === 'string') return value;
    try {
      var seen = [];
      return JSON.stringify(value, function (_k, v) {
        if (typeof v === 'string' && v.length > 1500) {
          return v.slice(0, 1500) + '… (' + v.length + ' chars total)';
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

  function log(message, data, opts) {
    var entry = {
      at:    timestamp(),
      atMs:  Date.now(),
      level: (opts && opts.level) || 'info',
      msg:   String(message || ''),
      data:  data === undefined ? null : data,
    };
    entries.push(entry);
    if (entries.length > MAX_ENTRIES) entries.splice(0, entries.length - MAX_ENTRIES);
    notify();
    // Mirror to the browser console so the developer who already has
    // devtools open sees the live stream too.
    if (window.console && typeof window.console.debug === 'function') {
      var cons = entry.level === 'error' ? window.console.error
              : entry.level === 'warn'  ? window.console.warn
              : window.console.debug;
      try { cons.call(window.console, '[contrib-debug]', entry.msg, data === undefined ? '' : data); } catch (_e) {}
    }
    // Auto-reveal the pill on errors so a contributor who hit a hard
    // failure can find + download the log without the URL trick.
    if (entry.level === 'error' && !autoRevealed) {
      autoRevealed = true;
      ensureDrawer();
    }
  }

  function logError(message, data) { log(message, data, { level: 'error' }); }
  function logWarn(message, data)  { log(message, data, { level: 'warn'  }); }

  function notify() {
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](entries); } catch (_e) { /* swallow */ }
    }
  }

  function snapshot() { return entries.slice(); }
  function clear() { entries = []; notify(); }

  function exportText() {
    var head = [
      '# Kozha contribute debug log',
      '# session started ' + sessionStartedAt.toISOString(),
      '# exported       ' + new Date().toISOString(),
      '# url            ' + window.location.href,
      '# user agent     ' + (navigator.userAgent || '<unknown>'),
      '# entries        ' + entries.length + ' (cap ' + MAX_ENTRIES + ')',
      '',
    ].join('\n');
    var body = entries.map(function (e) {
      var line = '[' + e.at + '] ' + (e.level !== 'info' ? e.level.toUpperCase() + ' ' : '') + e.msg;
      if (e.data !== null) {
        line += '\n  ' + safeStringify(e.data).split('\n').join('\n  ');
      }
      return line;
    }).join('\n');
    return head + body + '\n';
  }

  function downloadLog() {
    var text = exportText();
    var blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    var url = URL.createObjectURL(blob);
    var a = document.createElement('a');
    a.href = url;
    a.download = 'kozha-contrib-log-' + downloadStamp() + '.txt';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  // ---------- drawer UI ----------

  function ensureDrawer() {
    if (drawer) return drawer;
    if (!drawerEnabled && !autoRevealed) return null;
    if (!document.body) {
      // Body not ready yet — defer to DOMContentLoaded.
      document.addEventListener('DOMContentLoaded', ensureDrawer, { once: true });
      return null;
    }

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
    title.textContent = 'Debug log (' + entries.length + ')';
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
      var text = exportText();
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).catch(function () {});
      }
    });
    var dlBtn = document.createElement('button');
    dlBtn.type = 'button';
    dlBtn.className = 'kz-debug-action';
    dlBtn.textContent = 'Download';
    dlBtn.addEventListener('click', downloadLog);
    var closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.className = 'kz-debug-action';
    closeBtn.setAttribute('aria-label', 'Close debug panel');
    closeBtn.textContent = '×';
    closeBtn.addEventListener('click', function () { setOpen(false); });
    head.appendChild(title);
    head.appendChild(clearBtn);
    head.appendChild(copyBtn);
    head.appendChild(dlBtn);
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
      title.textContent = 'Debug log (' + entries.length + ')';
      while (body.firstChild) body.removeChild(body.firstChild);
      var rows = entries.slice().reverse();
      for (var i = 0; i < rows.length; i++) {
        var e = rows[i];
        var row = document.createElement('div');
        row.className = 'kz-debug-row is-level-' + e.level;
        var ts = document.createElement('span');
        ts.className = 'kz-debug-ts';
        ts.textContent = e.at;
        var msg = document.createElement('span');
        msg.className = 'kz-debug-msg';
        msg.textContent = (e.level !== 'info' ? '[' + e.level + '] ' : '') + e.msg;
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

    listeners.push(function () {
      title.textContent = 'Debug log (' + entries.length + ')';
      if (!panel.hidden) render();
    });

    drawer = { btn: btn, panel: panel, render: render, setOpen: setOpen };
    return drawer;
  }

  function reveal() {
    autoRevealed = true;
    ensureDrawer();
  }

  // ---------- error capture (window-level) ----------

  window.addEventListener('error', function (ev) {
    if (!ev) return;
    logError('window: error', {
      message: ev.message,
      filename: ev.filename,
      lineno: ev.lineno,
      colno: ev.colno,
      stack: ev.error && ev.error.stack ? String(ev.error.stack).slice(0, 2000) : undefined,
    });
  });
  window.addEventListener('unhandledrejection', function (ev) {
    var r = ev && ev.reason;
    logError('window: unhandled promise rejection', {
      reason: r && r.message ? r.message : String(r),
      stack: r && r.stack ? String(r.stack).slice(0, 2000) : undefined,
    });
  });

  // ---------- public surface ----------

  window.KOZHA_CONTRIB_DEBUG = {
    // Logging is always-on. Pill / drawer visibility is the only thing
    // gated by the opt-in flag.
    log:       log,
    warn:      logWarn,
    error:     logError,
    forceLog:  log,
    snapshot:  snapshot,
    clear:     clear,
    download:  downloadLog,
    exportText: exportText,
    reveal:    reveal,
    isEnabled: function () { return drawerEnabled || autoRevealed; },
    enable: function () {
      drawerEnabled = true;
      try { localStorage.setItem(STORAGE_KEY, '1'); } catch (_e) {}
      ensureDrawer();
    },
    disable: function () {
      drawerEnabled = false;
      autoRevealed = false;
      try { localStorage.removeItem(STORAGE_KEY); } catch (_e) {}
      if (drawer) {
        drawer.btn.remove();
        drawer.panel.remove();
        drawer = null;
      }
    },
  };

  // First baseline entry — useful as the anchor in any downloaded log.
  log('debug: page session started', {
    url: window.location.href,
    userAgent: navigator.userAgent,
    drawerEnabled: drawerEnabled,
  });

  function init() { if (drawerEnabled) ensureDrawer(); }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

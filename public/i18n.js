/* Minimalist i18n layer — prep work for localisation (prompt 12 step 7).
 *
 * Shape: load /strings.en.json once at startup, expose t(key, vars?) for
 * dynamic strings, and hydrate [data-i18n] / [data-i18n-attr-*] attributes
 * on the current document so HTML files don't have to duplicate English
 * copy in two places.
 *
 * Philosophy: English is the source of truth right now. We do not fall
 * back to a locale file — if a key is missing, we return the key itself
 * (makes the gap visible in the UI rather than silently showing '').
 *
 * Public API on window.KOZHA_I18N:
 *   t(key, vars?)          → resolved string; leaves {{placeholders}}
 *                            replaced when vars are provided.
 *   ready                  → Promise that resolves once the catalog is
 *                            loaded and any data-i18n attributes have
 *                            been hydrated.
 *   reinject(root?)        → re-runs attribute hydration over a DOM
 *                            subtree; useful after dynamic insertions.
 *
 * Attribute conventions:
 *   data-i18n="some.key"                → textContent = t(some.key)
 *   data-i18n-attr-aria-label="k"       → el.setAttribute('aria-label', t(k))
 *   data-i18n-attr-placeholder="k"      → el.setAttribute('placeholder', t(k))
 *   data-i18n-attr-title="k"            → el.setAttribute('title', t(k))
 *
 * Only set-once attributes are supported via markup — anything that
 * needs to change at runtime routes through t() from JS.
 */
(function () {
  'use strict';

  var CATALOG_URL = '/strings.en.json';
  var catalog = null;

  function resolve(key) {
    if (!catalog || typeof key !== 'string') return undefined;
    var parts = key.split('.');
    var node = catalog;
    for (var i = 0; i < parts.length; i++) {
      if (node == null || typeof node !== 'object') return undefined;
      node = node[parts[i]];
    }
    return typeof node === 'string' ? node : undefined;
  }

  function interpolate(template, vars) {
    if (!vars) return template;
    return template.replace(/\{\{\s*([a-zA-Z0-9_]+)\s*\}\}/g, function (_m, name) {
      return Object.prototype.hasOwnProperty.call(vars, name)
        ? String(vars[name])
        : '{{' + name + '}}';
    });
  }

  function t(key, vars) {
    var raw = resolve(key);
    if (raw === undefined) return key; // Visible gap — easier to spot than ''.
    return interpolate(raw, vars);
  }

  function hydrateRoot(root) {
    if (!root || !root.querySelectorAll) return;
    var textNodes = root.querySelectorAll('[data-i18n]');
    for (var i = 0; i < textNodes.length; i++) {
      var el = textNodes[i];
      var key = el.getAttribute('data-i18n');
      if (!key) continue;
      var val = resolve(key);
      if (val !== undefined) el.textContent = val;
    }
    // Generic attr hydration — data-i18n-attr-<attrname>="<key>".
    var all = root.querySelectorAll('*');
    for (var j = 0; j < all.length; j++) {
      var node = all[j];
      if (!node.attributes) continue;
      for (var k = 0; k < node.attributes.length; k++) {
        var a = node.attributes[k];
        if (a.name.indexOf('data-i18n-attr-') !== 0) continue;
        var attrName = a.name.substring('data-i18n-attr-'.length);
        var v = resolve(a.value);
        if (v !== undefined) node.setAttribute(attrName, v);
      }
    }
  }

  var ready = fetch(CATALOG_URL, {
    credentials: 'same-origin',
    cache: 'no-cache',
    headers: { Accept: 'application/json' },
  })
    .then(function (r) {
      if (!r.ok) throw new Error('strings.en.json HTTP ' + r.status);
      return r.json();
    })
    .then(function (data) {
      catalog = data || {};
      if (document.readyState === 'loading') {
        return new Promise(function (resolveDom) {
          document.addEventListener('DOMContentLoaded', function () {
            hydrateRoot(document);
            resolveDom();
          });
        });
      }
      hydrateRoot(document);
    })
    .catch(function (err) {
      if (window.console) console.warn('[i18n] failed to load catalog:', err);
      catalog = {};
    });

  window.KOZHA_I18N = {
    t: t,
    ready: ready,
    reinject: hydrateRoot,
  };
})();

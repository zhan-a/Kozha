/* Interactive SiGML annotated editor.
 *
 * Renders the current SiGML in the contribute panel as a row of
 * click-swappable chips. Each chip is one <ham...> tag from the
 * <hamnosys_manual> block. Clicking a chip opens a picker scoped to
 * the same category (handshape, palm direction, location, etc.) and
 * lists every alternative with its plain-English role pulled from the
 * SiGML reference catalog at /api/chat2hamnosys/reference/sigml.
 *
 * Picking an alternative swaps the tag in-place, re-emits canonical
 * SiGML, and pushes the change through CTX.setState({ sigml }) — the
 * preview module re-renders automatically.
 *
 * Goals
 * -----
 * - Make SiGML the primary, browseable surface (not raw HamNoSys).
 * - Give the contributor freedom to fix any slot the LLM got wrong
 *   without typing XML by hand.
 * - Surface the semantic role of every tag so reviewers and authors
 *   share a vocabulary.
 *
 * Public API on window.KOZHA_SIGML_EDIT (small): currently nothing —
 * the module wires itself to CTX.subscribe + DOM events at load time
 * and needs no external entry points.
 */
(function () {
  'use strict';

  if (!window.KOZHA_CONTRIB_CONTEXT) {
    if (window.console) console.warn('[contribute-sigml-edit] CTX not ready');
    return;
  }
  var CTX = window.KOZHA_CONTRIB_CONTEXT;
  var DEBUG = window.KOZHA_CONTRIB_DEBUG || { log: function () {}, error: function () {} };

  var REFERENCE_URL = '/api/chat2hamnosys/reference/sigml';

  var els = {
    panel:    document.getElementById('notationPanelSigml'),
    annotated: document.getElementById('sigmlAnnotated'),
    picker:   document.getElementById('sigmlPicker'),
    pickerTitle: document.getElementById('sigmlPickerTitle'),
    pickerCurrent: document.getElementById('sigmlPickerCurrent'),
    pickerList: document.getElementById('sigmlPickerList'),
    pickerSearch: document.getElementById('sigmlPickerSearch'),
    pickerClose: document.getElementById('sigmlPickerCloseBtn'),
  };

  // Bail quietly if the markup isn't on the page (the module is
  // loaded by every contribute view; some sub-pages may not have the
  // notation panel mounted).
  if (!els.annotated) return;

  // -----------------------------------------------------------------
  // Reference catalog — fetched once, cached for the page lifetime.
  // -----------------------------------------------------------------
  var catalog = null;
  var catalogPromise = null;

  function fetchCatalog() {
    if (catalog) return Promise.resolve(catalog);
    if (catalogPromise) return catalogPromise;
    catalogPromise = fetch(REFERENCE_URL, { headers: { Accept: 'application/json' } })
      .then(function (r) {
        if (!r.ok) throw new Error('reference HTTP ' + r.status);
        return r.json();
      })
      .then(function (data) {
        catalog = data;
        return catalog;
      })
      .catch(function (err) {
        if (window.console) console.warn('[contribute-sigml-edit] catalog fetch failed', err);
        catalogPromise = null;
        throw err;
      });
    return catalogPromise;
  }

  // -----------------------------------------------------------------
  // SiGML parsing / serialisation
  // -----------------------------------------------------------------

  // Pull the <ham*/> tags out of <hamnosys_manual>. Returns an array
  // of { tagName, raw } in document order. Falls back to an empty
  // array on any parse error so the UI degrades to "show source".
  function extractManualTags(sigmlSource) {
    if (!sigmlSource || typeof sigmlSource !== 'string') return [];
    var manualMatch = sigmlSource.match(
      /<\s*hamnosys_manual\s*>([\s\S]*?)<\s*\/\s*hamnosys_manual\s*>/i
    );
    if (!manualMatch) return [];
    var body = manualMatch[1];
    var out = [];
    var re = /<\s*(ham[a-z0-9_]+)\s*\/?\s*>/gi;
    var m;
    while ((m = re.exec(body)) !== null) {
      out.push({ tagName: m[1].toLowerCase(), raw: m[0] });
    }
    return out;
  }

  // Replace the Nth <ham*/> tag inside <hamnosys_manual> with a new
  // tag name. Returns the new SiGML string (or original on miss).
  function replaceTagAtIndex(sigmlSource, index, newTagName) {
    var manualMatch = sigmlSource.match(
      /(<\s*hamnosys_manual\s*>)([\s\S]*?)(<\s*\/\s*hamnosys_manual\s*>)/i
    );
    if (!manualMatch) return sigmlSource;
    var before = sigmlSource.slice(0, manualMatch.index + manualMatch[1].length);
    var body = manualMatch[2];
    var after = sigmlSource.slice(
      manualMatch.index + manualMatch[1].length + manualMatch[2].length
    );
    var re = /<\s*(ham[a-z0-9_]+)\s*\/?\s*>/gi;
    var i = 0;
    var newBody = body.replace(re, function (full) {
      if (i === index) {
        i++;
        return '<' + newTagName + '/>';
      }
      i++;
      return full;
    });
    return before + newBody + after;
  }

  // -----------------------------------------------------------------
  // Render — annotated chips
  // -----------------------------------------------------------------

  function clearChildren(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  function renderAnnotated() {
    var snap = CTX.getState();
    var sigml = snap.sigml || '';
    if (!sigml) {
      clearChildren(els.annotated);
      var hint = document.createElement('p');
      hint.className = 'sigml-annotated-empty field-hint';
      hint.textContent = 'No SiGML yet — describe the sign and the model will draft one.';
      els.annotated.appendChild(hint);
      return;
    }
    fetchCatalog().then(function (cat) {
      var tags = extractManualTags(sigml);
      clearChildren(els.annotated);
      if (!tags.length) {
        var p = document.createElement('p');
        p.className = 'sigml-annotated-empty field-hint';
        p.textContent = 'No <hamnosys_manual> block found in the SiGML.';
        els.annotated.appendChild(p);
        return;
      }
      var byName = (cat && cat.by_name) || {};
      tags.forEach(function (t, idx) {
        var entry = byName[t.tagName];
        var chip = document.createElement('button');
        chip.type = 'button';
        chip.className = 'sigml-chip';
        chip.dataset.index = String(idx);
        chip.dataset.tagName = t.tagName;
        if (entry) {
          chip.dataset.category = entry.category;
          chip.title = entry.role;
        } else {
          chip.dataset.category = 'unknown';
          chip.title = 'Unknown tag — not in catalog';
          chip.classList.add('is-unknown');
        }
        var name = document.createElement('span');
        name.className = 'sigml-chip-name';
        name.textContent = '<' + t.tagName + '/>';
        var role = document.createElement('span');
        role.className = 'sigml-chip-role';
        role.textContent = entry ? entry.label : '?';
        chip.appendChild(name);
        chip.appendChild(role);
        chip.addEventListener('click', function () { openPicker(idx, t.tagName); });
        els.annotated.appendChild(chip);
      });
    }).catch(function () {
      // Catalog fetch failed — show the raw source as a fallback so
      // the contributor can still see what got generated.
      clearChildren(els.annotated);
      var pre = document.createElement('pre');
      pre.className = 'sigml-annotated-fallback';
      pre.textContent = sigml;
      els.annotated.appendChild(pre);
    });
  }

  // -----------------------------------------------------------------
  // Picker
  // -----------------------------------------------------------------

  var pickerCtx = { index: -1, tagName: null, category: null, options: [] };

  function openPicker(index, currentTagName) {
    fetchCatalog().then(function (cat) {
      var byName = cat.by_name || {};
      var byCategory = cat.by_category || {};
      var entry = byName[currentTagName];
      if (!entry) {
        DEBUG.log('sigml-edit: unknown tag, no picker', { tagName: currentTagName });
        return;
      }
      var category = entry.category;
      var options = byCategory[category] || [];
      pickerCtx = {
        index: index,
        tagName: currentTagName,
        category: category,
        options: options,
      };
      var label = (cat.category_labels && cat.category_labels[category]) || category;
      els.pickerTitle.textContent = label + ' — pick an option';
      els.pickerCurrent.textContent = 'Current: <' + currentTagName + '/> — ' + entry.role;
      els.pickerSearch.value = '';
      renderPickerList('');
      els.picker.hidden = false;
      // Anchor the picker near the chip if possible.
      var chip = els.annotated.querySelector('.sigml-chip[data-index="' + index + '"]');
      if (chip) {
        var rect = chip.getBoundingClientRect();
        var panelRect = els.panel.getBoundingClientRect();
        // Position relative to the notation panel so the picker scrolls
        // with it. Top-anchored just below the chip.
        els.picker.style.top = (rect.bottom - panelRect.top + 8) + 'px';
        els.picker.style.left = Math.max(0, rect.left - panelRect.left) + 'px';
      }
      setTimeout(function () {
        if (els.pickerSearch && typeof els.pickerSearch.focus === 'function') {
          els.pickerSearch.focus();
        }
      }, 0);
    }).catch(function () { /* fetch already warned */ });
  }

  function renderPickerList(filter) {
    clearChildren(els.pickerList);
    var query = (filter || '').trim().toLowerCase();
    var shown = 0;
    pickerCtx.options.forEach(function (opt) {
      var hay = (opt.name + ' ' + opt.role + ' ' + opt.label).toLowerCase();
      if (query && hay.indexOf(query) === -1) return;
      var li = document.createElement('li');
      li.className = 'sigml-picker-item';
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'sigml-picker-btn';
      if (opt.name === pickerCtx.tagName) btn.classList.add('is-current');
      var n = document.createElement('span');
      n.className = 'sigml-picker-btn-name';
      n.textContent = '<' + opt.name + '/>';
      var r = document.createElement('span');
      r.className = 'sigml-picker-btn-role';
      r.textContent = opt.role;
      btn.appendChild(n);
      btn.appendChild(r);
      btn.addEventListener('click', function () { applyPick(opt.name); });
      li.appendChild(btn);
      els.pickerList.appendChild(li);
      shown++;
    });
    if (!shown) {
      var empty = document.createElement('li');
      empty.className = 'sigml-picker-empty field-hint';
      empty.textContent = 'No matches.';
      els.pickerList.appendChild(empty);
    }
  }

  function closePicker() {
    els.picker.hidden = true;
    pickerCtx = { index: -1, tagName: null, category: null, options: [] };
  }

  function applyPick(newTagName) {
    var snap = CTX.getState();
    var current = snap.sigml || '';
    if (!current || pickerCtx.index < 0) {
      closePicker();
      return;
    }
    if (newTagName === pickerCtx.tagName) {
      closePicker();
      return;
    }
    var next = replaceTagAtIndex(current, pickerCtx.index, newTagName);
    if (next === current) {
      closePicker();
      return;
    }
    var fromTag = pickerCtx.tagName;
    var toTag = newTagName;
    var swapIndex = pickerCtx.index;
    DEBUG.log('sigml-edit: tag swap', {
      index: swapIndex,
      from: fromTag,
      to: toTag,
    });
    // Optimistic local update: paint the new SiGML into context
    // immediately so the avatar re-renders without waiting for the
    // round-trip. The server response (or SSE GeneratedEvent) will
    // overwrite this with the canonical SiGML when it lands —
    // typically within a few hundred milliseconds because the
    // backend handles structured chip swaps without an LLM call.
    CTX.setState({ sigml: next });
    closePicker();
    // Persist the swap to the backend so a reload doesn't lose it,
    // a CorrectionAppliedEvent + GeneratedEvent enter the session
    // history (audit trail), and the SSE channel notifies any other
    // tabs watching this session. The structured ``swap`` payload
    // bypasses the LLM-backed correction interpreter on the server
    // side — see backend/chat2hamnosys/correct/sigml_rewrite.py.
    if (CTX && typeof CTX.correct === 'function') {
      CTX.correct('chip swap: <' + fromTag + '/> → <' + toTag + '/>', {
        swap: { from_tag: fromTag, to_tag: toTag, index: swapIndex },
      }).catch(function (err) {
        if (window.console) {
          console.warn('[contribute-sigml-edit] chip swap POST failed', err);
        }
        DEBUG.error('sigml-edit: chip swap POST failed', {
          status: err && err.status,
          body:   err && err.body,
        });
      });
    }
  }

  // -----------------------------------------------------------------
  // Wire-up
  // -----------------------------------------------------------------

  els.pickerClose.addEventListener('click', closePicker);
  els.pickerSearch.addEventListener('input', function () {
    renderPickerList(els.pickerSearch.value);
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && !els.picker.hidden) closePicker();
  });
  // Click outside the picker closes it (only if it's open).
  document.addEventListener('click', function (e) {
    if (els.picker.hidden) return;
    if (els.picker.contains(e.target)) return;
    if (e.target.closest && e.target.closest('.sigml-chip')) return;
    closePicker();
  });

  // Re-render on every state change. SiGML changes after generation,
  // corrections, or our own apply — all should refresh the chips.
  var lastSigml = '';
  CTX.subscribe(function (snap) {
    if ((snap.sigml || '') !== lastSigml) {
      lastSigml = snap.sigml || '';
      renderAnnotated();
    }
  });
  // Initial paint.
  renderAnnotated();

  window.KOZHA_SIGML_EDIT = {
    refresh: renderAnnotated,
  };
})();

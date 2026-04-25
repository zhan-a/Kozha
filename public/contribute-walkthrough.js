// contribute-walkthrough.js
//
// Drives the playable HAMBURG demo on /contribute walkthrough step 4
// ("Watch the avatar perform it"). Owns:
//   - The single canonical HAMBURG SiGML constant for the page (also
//     exposed as window.KOZHA_DEMO_HAMBURG so other surfaces don't
//     bundle a second copy).
//   - Reparenting CWASA's rendered <canvas> between the walkthrough
//     mount and the live-preview mount so a single CWASA.init covers
//     every visible stage on the page.
//   - Autoplay-on-step-visible (IntersectionObserver + step-active
//     MutationObserver), Replay button, and pause when the panel
//     leaves view.
//   - Building a keyboard-navigable chip strip — one chip per
//     <ham*-/> tag in the SiGML, each carrying its semantic role from
//     hamnosys-sigml-reference.md § 1, with a read-only inspector on
//     Enter/click. Chip click-to-swap is prompt 05's job; this file
//     stays read-only.
(function () {
  'use strict';

  // Canonical HAMBURG demo. Lifted verbatim from
  // hamnosys-sigml-reference.md § 5.1 (the document cited as the
  // ground-truth HamNoSys ↔ SiGML mapping). Keep this string the only
  // copy on the page — the chat2hamnosys generator's few-shot example
  // module reads its own copy server-side; we don't bundle a second
  // browser-side duplicate.
  var HAMBURG_SIGML = [
    '<?xml version="1.0" encoding="utf-8"?>',
    '<sigml>',
    '  <hns_sign gloss="HAMBURG">',
    '    <hamnosys_manual>',
    '      <hamceeall/><hamthumbopenmod/><hamfingerstraightmod/><hamextfingerul/>',
    '      <hampalmdl/><hamforehead/><hamlrat/><hamclose/>',
    '      <hamparbegin/><hammover/><hamreplace/><hampinchall/>',
    '      <hamfingerstraightmod/><hamparend/>',
    '    </hamnosys_manual>',
    '  </hns_sign>',
    '</sigml>'
  ].join('\n');
  window.KOZHA_DEMO_HAMBURG = HAMBURG_SIGML;

  // Tag → { role, category }. Role text is taken verbatim from the
  // reference doc's `role:` field. The reference's category names are
  // condensed to one display word so chips stay scannable.
  var TAG_INFO = {
    hamceeall:            { role: 'C-shape using all fingers',                        cat: 'handshape'      },
    hamthumbopenmod:      { role: 'Thumb away from fingers (open)',                   cat: 'modifier'       },
    hamfingerstraightmod: { role: 'Fingers fully straight',                           cat: 'modifier'       },
    hamextfingerul:       { role: 'Fingers point up-left',                            cat: 'finger dir.'    },
    hampalmdl:            { role: 'Palm faces down-left',                             cat: 'palm'           },
    hamforehead:          { role: 'Location: forehead',                               cat: 'location'       },
    hamlrat:              { role: 'Locative qualifier (proximity: at the location)',  cat: 'location mod.'  },
    hamclose:             { role: 'Contact: close to the locative surface',           cat: 'movement mod.'  },
    hamparbegin:          { role: 'Parallel-action group: begin',                     cat: 'movement'       },
    hammover:             { role: 'Movement: out (away from signer)',                 cat: 'movement'       },
    hamreplace:           { role: 'Replace handshape mid-sign',                       cat: 'movement'       },
    hampinchall:          { role: 'All fingers pinched together with thumb',          cat: 'handshape'      },
    hamparend:            { role: 'Parallel-action group: end',                       cat: 'movement'       }
  };

  function extractHamTags(sigml) {
    var tags = [];
    var re = /<(ham[a-z0-9]+)\/>/gi;
    var m;
    while ((m = re.exec(sigml)) !== null) tags.push(m[1].toLowerCase());
    return tags;
  }

  // ---------- DOM resolution (every selector is optional; bail if the
  // walkthrough markup isn't on this page) ----------

  var walkMount      = document.getElementById('walkAvatarMount');
  var liveMount      = document.getElementById('avatarCanvas');
  var stepPanel      = document.getElementById('c2-panel-4');
  var replayBtn      = document.getElementById('walkReplayBtn');
  var chipHost       = document.getElementById('walkChipStrip');
  var inspector      = document.getElementById('walkInspector');
  var inspectorTag   = inspector && inspector.querySelector('[data-walk-inspector-tag]');
  var inspectorCat   = inspector && inspector.querySelector('[data-walk-inspector-cat]');
  var inspectorRole  = inspector && inspector.querySelector('[data-walk-inspector-role]');
  var inspectorClose = inspector && inspector.querySelector('[data-walk-inspector-close]');
  var previewSection = document.getElementById('avatarPreview');

  if (!walkMount || !stepPanel || !chipHost) return;

  // ---------- CWASA canvas ownership ----------
  //
  // CWASA scans for `.CWASAAvatar.av0` at init time and writes its
  // canvas into the FIRST match (see cwa/allcsa.js: `avaDiv[0].innerHTML
  // = htmlgen.htmlForAv()`). The page now has multiple .CWASAAvatar.av0
  // hosts (hero, walkthrough, live preview); we pick whichever one
  // currently matches what the user is looking at and physically move
  // the rendered <canvas> into it. WebGL contexts survive reparenting,
  // so this doesn't lose the avatar's animation state.

  function findCanvas() {
    return document.querySelector('.CWASAAvatar.av0 canvas');
  }
  function currentHost() {
    var c = findCanvas();
    if (!c) return null;
    var host = c.parentNode;
    while (host && !(host.classList && host.classList.contains('CWASAAvatar'))) {
      host = host.parentNode;
    }
    return host;
  }
  function claimCanvasFor(targetHost) {
    var canvas = findCanvas();
    if (!canvas || !targetHost) return false;
    var host = currentHost();
    if (!host || host === targetHost) return Boolean(canvas);
    while (host.firstChild) targetHost.appendChild(host.firstChild);
    return true;
  }

  // ---------- CWASA boot + ready ----------

  function ensureCwasaInit() {
    var hook = window.KOZHA_CONTRIB_PREVIEW;
    if (hook && typeof hook.ensureCWASA === 'function') {
      try { hook.ensureCWASA(); } catch (_e) { /* logged by preview */ }
    }
  }

  var cwasaReadyPromise = new Promise(function (resolve) {
    var attempts = 0;
    function tick() {
      attempts++;
      if (window.CWASA && findCanvas()) { resolve(true); return; }
      // 60s ceiling — CWASA's bundle is ~4.6 MB and the lazy loader
      // waits 2.5s after `load`. After ~60 polls @ 200ms = 12s without
      // a canvas we give up; the chip strip and Replay button still
      // work, just no playback.
      if (attempts > 300) { resolve(false); return; }
      setTimeout(tick, 200);
    }
    tick();
  });

  var isPlaying = false;
  function playHamburg() {
    if (!window.CWASA || typeof window.CWASA.playSiGMLText !== 'function') return;
    try { window.CWASA.stop(0); } catch (_e) {}
    try {
      window.CWASA.playSiGMLText(HAMBURG_SIGML, 0);
      isPlaying = true;
    } catch (e) {
      if (window.console) console.warn('[contribute-walkthrough] play failed:', e);
    }
  }
  function pause() {
    if (!window.CWASA) return;
    try { window.CWASA.stop(0); } catch (_e) {}
    isPlaying = false;
  }

  // ---------- Chip strip + inspector ----------

  function buildChips() {
    var tags = extractHamTags(HAMBURG_SIGML);
    chipHost.innerHTML = '';
    tags.forEach(function (tag) {
      var info = TAG_INFO[tag] || { role: tag, cat: 'tag' };
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'c2-viz-4__chip';
      btn.setAttribute('tabindex', '0');
      btn.dataset.tag = tag;
      btn.dataset.role = info.role;
      btn.dataset.category = info.cat;
      btn.setAttribute(
        'aria-label',
        '<' + tag + '/>: ' + info.role + ' (' + info.cat + '). Press Enter to inspect.'
      );

      var name = document.createElement('span');
      name.className = 'c2-viz-4__chip-name';
      name.textContent = '<' + tag + '/>';
      var cat = document.createElement('span');
      cat.className = 'c2-viz-4__chip-cat';
      cat.textContent = info.cat;
      btn.appendChild(name);
      btn.appendChild(cat);

      btn.addEventListener('click', function () { openInspector(btn); });
      btn.addEventListener('keydown', function (ev) {
        if (ev.key === 'Enter' || ev.key === ' ') {
          ev.preventDefault();
          openInspector(btn);
        }
      });
      chipHost.appendChild(btn);
    });
  }

  function openInspector(btn) {
    if (!inspector) return;
    if (inspectorTag)  inspectorTag.textContent  = '<' + btn.dataset.tag + '/>';
    if (inspectorCat)  inspectorCat.textContent  = btn.dataset.category || '';
    if (inspectorRole) inspectorRole.textContent = btn.dataset.role || '';
    inspector.hidden = false;
  }
  function closeInspector() { if (inspector) inspector.hidden = true; }
  if (inspectorClose) inspectorClose.addEventListener('click', closeInspector);

  // ---------- Visibility orchestration ----------
  //
  // Step 4 must satisfy two conditions to play: the walkthrough has
  // selected step 4 (panel is-active + not [hidden]) AND the panel sits
  // in the viewport. Either condition flipping false pauses CWASA.

  var inViewport = false;
  function isStepActive() {
    return stepPanel.classList.contains('is-active') && !stepPanel.hidden;
  }
  function shouldPlay() { return isStepActive() && inViewport; }

  function update() {
    if (shouldPlay()) {
      ensureCwasaInit();
      cwasaReadyPromise.then(function (ok) {
        if (!ok || !shouldPlay()) return;
        if (claimCanvasFor(walkMount)) playHamburg();
      });
    } else if (isPlaying) {
      pause();
    }
  }

  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        // Use a generous threshold — the panel is full-width on desktop
        // and effectively in-view long before 50% intersects. 0.25 is
        // the "user can clearly see the avatar" cutoff in practice.
        inViewport = entry.isIntersecting && entry.intersectionRatio > 0.25;
      });
      update();
    }, { threshold: [0, 0.25, 0.5, 1] });
    io.observe(stepPanel);
  } else {
    inViewport = true;
  }

  // The walkthrough stepper flips `is-active` and the `hidden` attr on
  // the panel; observe both so a step change inside the page (no
  // scroll) still flips playback.
  var stepObserver = new MutationObserver(update);
  stepObserver.observe(stepPanel, { attributes: true, attributeFilter: ['class', 'hidden'] });

  // Cede the canvas back to the live preview the moment the user
  // actually has a session and avatarPreview becomes visible — that's
  // where contribute-preview.js expects to render playback.
  if (previewSection) {
    var previewObserver = new MutationObserver(function () {
      if (!previewSection.hidden) {
        if (isPlaying) pause();
        claimCanvasFor(liveMount);
      }
    });
    previewObserver.observe(previewSection, { attributes: true, attributeFilter: ['hidden'] });
  }

  if (replayBtn) {
    replayBtn.addEventListener('click', function () {
      ensureCwasaInit();
      cwasaReadyPromise.then(function (ok) {
        if (!ok) return;
        claimCanvasFor(walkMount);
        playHamburg();
      });
    });
  }

  // Initial render: chips first (works without CWASA), then evaluate
  // whether step 4 is the current step. The IntersectionObserver above
  // fires its first callback synchronously after observe(), so the
  // viewport flag is initialised before the first user-driven update.
  buildChips();
  update();
})();

// contribute-walkthrough.js
//
// Drives both demo cards on /contribute:
//   1. The hero conversation snapshot card (BSL · ELECTRON, prompt 04) —
//      bubbles + HamNoSys + SiGML rendered statically. The "Replay in
//      avatar" button opens #heroReplayModal, lazy-loads CWASA, and
//      plays the inline data-demo-payload SiGML at viewport scale.
//   2. The walkthrough Step 4 ("Watch the avatar perform it") demo
//      (DGS · HAMBURG2^, prompt 05) — also a snapshot card; the inline
//      auto-play stage was retired because the cramped 220 px slot
//      never reached the spec's 280×280 floor and competed with the
//      chip strip / inspector for vertical space. The Play button now
//      opens #walkReplayModal at viewport scale.
//
// SiGML payloads live inside the cards as inline
// <script type="application/xml" data-demo-payload="..."> elements so
// the gloss/payload alignment is enforced by markup (the test at
// tests/contrib_demo_signs.spec.ts asserts each payload matches the
// named corpus entry or chat2hamnosys fixture).
//
// Reparents CWASA's rendered <canvas> between the hero modal mount,
// the walk modal mount, and the live-preview mount so a single
// CWASA.init covers every visible stage on the page. Honors
// prefers-reduced-motion: nothing here auto-plays — every playback is
// gated entirely on an explicit click.
(function () {
  'use strict';

  // Read SiGML payload from the inline <script type="application/xml">
  // anchored to a demo card. Returns null if the card or payload
  // element isn't on the page. Single source of truth — the visible
  // <pre> excerpt and the played SiGML are sourced from the same
  // markup so they cannot drift.
  function readPayload(slot) {
    var el = document.querySelector('script[type="application/xml"][data-demo-payload="' + slot + '"]');
    return el ? (el.textContent || '').trim() : null;
  }

  var HAMBURG_SIGML = readPayload('walk');
  var HERO_SIGML    = readPayload('hero');

  // Backward-compatible export: the chat2hamnosys generator and other
  // surfaces previously read window.KOZHA_DEMO_HAMBURG to avoid bundling
  // a second copy. Preserved here.
  if (HAMBURG_SIGML) window.KOZHA_DEMO_HAMBURG = HAMBURG_SIGML;

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
    if (!sigml) return tags;
    var re = /<(ham[a-z0-9]+)\s*\/>/gi;
    var m;
    while ((m = re.exec(sigml)) !== null) tags.push(m[1].toLowerCase());
    return tags;
  }

  // ---------- DOM resolution (every selector is optional; bail if the
  // walkthrough markup isn't on this page) ----------

  var heroPlayBtn    = document.getElementById('heroPlayBtn');
  var heroModal      = document.getElementById('heroReplayModal');
  var heroModalClose = document.getElementById('heroReplayModalClose');
  var heroMount      = document.getElementById('heroAvatarMount');

  var walkPlayBtn    = document.getElementById('walkPlayBtn');
  var walkModal      = document.getElementById('walkReplayModal');
  var walkModalClose = document.getElementById('walkReplayModalClose');
  var walkMount      = document.getElementById('walkReplayMount');

  var liveMount      = document.getElementById('avatarCanvas');
  var chipHost       = document.getElementById('walkChipStrip');
  var inspector      = document.getElementById('walkInspector');
  var inspectorTag   = inspector && inspector.querySelector('[data-walk-inspector-tag]');
  var inspectorCat   = inspector && inspector.querySelector('[data-walk-inspector-cat]');
  var inspectorRole  = inspector && inspector.querySelector('[data-walk-inspector-role]');
  var inspectorClose = inspector && inspector.querySelector('[data-walk-inspector-close]');
  var previewSection = document.getElementById('avatarPreview');

  if (!chipHost && !heroPlayBtn && !walkPlayBtn) return;

  // ---------- CWASA canvas ownership ----------
  //
  // CWASA scans for `.CWASAAvatar.av0` at init time and writes its
  // canvas into the FIRST match (see cwa/allcsa.js: `avaDiv[0].innerHTML
  // = htmlgen.htmlForAv()`). The page now has multiple .CWASAAvatar.av0
  // hosts (hero modal, walk modal, live preview); we pick whichever one
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
      // waits 2.5s after `load`. After ~300 polls @ 200ms = 60s without
      // a canvas we give up; the chip strip and Play buttons still
      // work (chip strip never needed CWASA), just no playback.
      if (attempts > 300) { resolve(false); return; }
      setTimeout(tick, 200);
    }
    tick();
  });

  var isPlaying = false;
  function playSigml(sigml) {
    if (!sigml) return;
    if (!window.CWASA || typeof window.CWASA.playSiGMLText !== 'function') return;
    try { window.CWASA.stop(0); } catch (_e) {}
    try {
      window.CWASA.playSiGMLText(sigml, 0);
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
    if (!chipHost) return;
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

  // ---------- Replay modal helper ----------
  //
  // Used by both the hero and walkthrough cards. Identical contract:
  // play button opens a viewport-sized dialog, claims the CWASA canvas
  // into the dialog's mount, plays the inline SiGML, and on close
  // pauses + returns focus. No auto-anything, so prefers-reduced-motion
  // is satisfied without an extra branch.

  // Track which modal currently holds the canvas so a second open
  // (e.g. walk while hero is open) closes the first one cleanly.
  var openModals = [];

  function setupReplayModal(opts) {
    var playBtn   = opts.playBtn;
    var modal     = opts.modal;
    var modalClose = opts.modalClose;
    var mount     = opts.mount;
    var sigml     = opts.sigml;
    if (!playBtn || !modal || !modalClose || !mount || !sigml) return null;

    var lastFocused = null;

    function open() {
      if (!modal.hidden) return;
      // Dismiss any other open modal first — only one canvas exists,
      // and reparenting it mid-playback into a hidden mount looks like
      // a frozen frame.
      openModals.slice().forEach(function (other) {
        if (other !== api) other.close();
      });

      lastFocused = document.activeElement;
      modal.hidden = false;
      playBtn.setAttribute('aria-expanded', 'true');
      document.addEventListener('keydown', onKey);
      // Defer focus by a tick so assistive tech announces the dialog
      // mount before the focus move.
      setTimeout(function () { modalClose.focus(); }, 0);
      openModals.push(api);

      ensureCwasaInit();
      cwasaReadyPromise.then(function (ok) {
        // Modal may have been dismissed while the bundle was still
        // loading — bail rather than yank the canvas in after the user
        // has already moved on.
        if (!ok || modal.hidden) return;
        if (claimCanvasFor(mount)) playSigml(sigml);
      });
    }

    function close() {
      if (modal.hidden) return;
      modal.hidden = true;
      playBtn.setAttribute('aria-expanded', 'false');
      document.removeEventListener('keydown', onKey);
      if (isPlaying) pause();
      var idx = openModals.indexOf(api);
      if (idx >= 0) openModals.splice(idx, 1);
      if (lastFocused && typeof lastFocused.focus === 'function') {
        lastFocused.focus();
      }
      lastFocused = null;
    }

    function onKey(ev) {
      if (ev.key === 'Escape') {
        ev.preventDefault();
        close();
        return;
      }
      // The modal has exactly one focusable control (the close button);
      // trap Tab/Shift+Tab on it so focus cannot escape into the page
      // beneath.
      if (ev.key === 'Tab') {
        ev.preventDefault();
        modalClose.focus();
      }
    }

    playBtn.setAttribute('aria-expanded', 'false');
    playBtn.addEventListener('click', open);
    modalClose.addEventListener('click', close);
    // Backdrop click — only when the user clicks the dimmed area
    // outside .c2-replay-modal__panel. Inside-panel clicks bubble up
    // with currentTarget !== target if they originated on a child.
    modal.addEventListener('click', function (ev) {
      if (ev.target === modal) close();
    });

    var api = { open: open, close: close, isOpen: function () { return !modal.hidden; } };
    return api;
  }

  setupReplayModal({
    playBtn:    heroPlayBtn,
    modal:      heroModal,
    modalClose: heroModalClose,
    mount:      heroMount,
    sigml:      HERO_SIGML
  });

  setupReplayModal({
    playBtn:    walkPlayBtn,
    modal:      walkModal,
    modalClose: walkModalClose,
    mount:      walkMount,
    sigml:      HAMBURG_SIGML
  });

  // ---------- Live preview hand-off ----------
  //
  // Cede the canvas back to the live preview the moment the user
  // actually has a session and avatarPreview becomes visible — that's
  // where contribute-preview.js expects to render playback. Any open
  // demo modal is dismissed so the canvas isn't yanked mid-playback.
  if (previewSection) {
    var previewObserver = new MutationObserver(function () {
      if (!previewSection.hidden) {
        if (isPlaying) pause();
        openModals.slice().forEach(function (m) { m.close(); });
        claimCanvasFor(liveMount);
      }
    });
    previewObserver.observe(previewSection, { attributes: true, attributeFilter: ['hidden'] });
  }

  // Initial render: chips first so the inspector affordance works
  // independently of CWASA. Modal playback is gated on explicit clicks
  // (no IntersectionObserver, no auto-play) — prefers-reduced-motion is
  // respected by construction.
  buildChips();
})();

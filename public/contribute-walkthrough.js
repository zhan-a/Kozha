// contribute-walkthrough.js
//
// Drives both demo cards on /contribute:
//   1. The hero snapshot card (BSL · name(v)#1) — Pattern B by default,
//      swaps to a real CWASA mount on play-button click.
//   2. The walkthrough Step 4 ("Watch the avatar perform it") demo
//      (DGS · HAMBURG2^) — autoplays in-viewport on step-active,
//      Replay button re-triggers, pauses when the panel leaves view.
//
// SiGML payloads live inside the cards as inline
// <script type="application/xml" data-demo-payload="..."> elements so
// the gloss/payload alignment is enforced by markup (the test at
// tests/contrib_demo_signs.spec.ts asserts each payload matches the
// named corpus entry). The fake silhouette posters that used to live
// here were removed — see docs/contrib-fix/prompts/03-fake-avatar-audit.md.
//
// Reparents CWASA's rendered <canvas> between the hero mount, the
// walkthrough mount, and the live-preview mount so a single CWASA.init
// covers every visible stage on the page. Honors
// prefers-reduced-motion (no auto-play; explicit click only) and small
// viewports (auto-play suppressed below 600 px wide).
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

  var walkMount      = document.getElementById('walkAvatarMount');
  var heroMount      = document.getElementById('heroAvatarMount');
  var heroPlayBtn    = document.getElementById('heroPlayBtn');
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

  // ---------- Adaptive defaults (Pattern B fallback) ----------
  //
  // The fake-avatar audit (docs/contrib-fix/prompts/03-fake-avatar-audit.md)
  // requires that prefers-reduced-motion AND small viewports default to
  // the static snapshot card and only swap to the live avatar on
  // explicit click. Both checks are evaluated lazily so a user changing
  // their OS motion preference or rotating their device picks up the
  // new state without reload.
  function reduceMotion() {
    return window.matchMedia &&
           window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }
  function tinyViewport() {
    if (typeof window.innerWidth !== 'number') return false;
    return window.innerWidth < 600;
  }
  function autoPlayAllowed() {
    // Reduced motion always wins. Small viewports skip auto-play
    // because the canvas would render under 200 px tall on phones —
    // the snapshot card is more legible than a postage-stamp avatar.
    return !reduceMotion() && !tinyViewport();
  }

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
  function playHamburg() { playSigml(HAMBURG_SIGML); }
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
  // Auto-play requires the step to be active AND visible AND for the
  // user not to have asked us to back off (reduced motion or tiny
  // viewport — see autoPlayAllowed). The Replay button bypasses this
  // gate so the user can always trigger playback explicitly.
  function shouldPlay() {
    return isStepActive() && inViewport && autoPlayAllowed();
  }

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

  // ---------- Hero snapshot card play button ----------
  //
  // The hero card stays in Pattern B (HamNoSys + SiGML snapshot) until
  // the user clicks "Play with avatar". On click we lazy-init CWASA
  // (the head loader has likely already injected the bundle on the
  // user's first interaction; ensureCwasaInit is idempotent), claim
  // the canvas to the hero mount, and play the embedded payload.
  // aria-pressed flips true/false to mirror visible play state.
  if (heroPlayBtn && heroMount && HERO_SIGML) {
    heroPlayBtn.addEventListener('click', function () {
      var pressed = heroPlayBtn.getAttribute('aria-pressed') === 'true';
      if (pressed) {
        // Toggle off: stop playback and release the canvas back to the
        // walkthrough so the chip strip / Replay button stay live.
        pause();
        heroPlayBtn.setAttribute('aria-pressed', 'false');
        if (isStepActive() && inViewport && autoPlayAllowed()) {
          claimCanvasFor(walkMount);
        }
        return;
      }
      heroPlayBtn.setAttribute('aria-pressed', 'true');
      ensureCwasaInit();
      cwasaReadyPromise.then(function (ok) {
        if (!ok) {
          heroPlayBtn.setAttribute('aria-pressed', 'false');
          return;
        }
        claimCanvasFor(heroMount);
        playSigml(HERO_SIGML);
      });
    });

    // Reflect canvas hand-off back to the button: if the walkthrough
    // step grabs the canvas, the hero is no longer playing — drop
    // aria-pressed to false so the button label re-reads "▶ Play".
    var heroOwnershipObserver = new MutationObserver(function () {
      if (heroPlayBtn.getAttribute('aria-pressed') !== 'true') return;
      if (!heroMount.querySelector('canvas')) {
        heroPlayBtn.setAttribute('aria-pressed', 'false');
      }
    });
    heroOwnershipObserver.observe(heroMount, { childList: true, subtree: true });
  }

  // Initial render: chips first (works without CWASA), then evaluate
  // whether step 4 is the current step. The IntersectionObserver above
  // fires its first callback synchronously after observe(), so the
  // viewport flag is initialised before the first user-driven update.
  buildChips();
  update();
})();

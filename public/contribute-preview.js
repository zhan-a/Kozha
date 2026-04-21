/* Contribute page — avatar preview pane controller.
 *
 * Spec: prompts/prompt-contrib-9.md.
 *
 * Scope:
 * - Renders the CWASA signing-avatar canvas inside a 16:9 stage with a
 *   light-gray backdrop (Kipp et al. 2011 avatar UX: dark backdrops
 *   reduce legibility for many Deaf users).
 * - Control bar: Play/Pause button, Loop checkbox, timeline scrubber
 *   with current/total time labels, three playback-speed buttons.
 * - Body-region SVG overlay — hovering a region softens its outline;
 *   clicking captures region + current time, focuses the chat input
 *   in correction mode, and posts a dismissible pill showing the
 *   captured target.
 * - Generation progress is handled *inside* the pane — a muted pulse
 *   on the backdrop during GENERATING/RENDERING, never a spinner or
 *   percentage. Status strip underneath narrates in plain text.
 * - Failure handling: if CWASA / WebGL / the script bundle is missing
 *   the pane collapses to a plain-text fallback and the notation
 *   panel (prompt 7) becomes the primary surface. Submit stays
 *   enabled.
 * - Cache signal: if we're asked to render a SiGML string we've
 *   already seen this session, show "From cache" for 1.5s.
 * - Accessibility: keyboard controls (space play/pause, arrows
 *   scrub ±100ms, shift+arrows ±1s), aria-live status strip, the
 *   scrubber carries role="slider" + min/max/valuenow/valuetext,
 *   captions show the gloss and description so Deaf reviewers can
 *   cross-check contributor intent.
 * - Layout stability: the pane never moves once mounted — an empty
 *   state shows a neutral avatar so the notation panel below keeps
 *   its y-position across generation.
 */
(function () {
  'use strict';

  var CTX = window.KOZHA_CONTRIB_CONTEXT;
  if (!CTX) {
    if (window.console) console.error('[contribute-preview] contribute-context.js must load first');
    return;
  }

  // ---------- constants ----------

  var SCRUB_SMALL_STEP_MS = 100;
  var SCRUB_LARGE_STEP_MS = 1000;
  var DEFAULT_DURATION_MS = 1400;
  var CACHE_BADGE_MS = 1500;
  var CWASA_WAIT_MS = 6000; // if CWASA never arrives, assume the bundle failed
  var CWASA_POLL_MS  = 100;
  // SVG viewBox is 160x90 (16:9). Convert clicked SVG coords back into
  // percentages of the stage for the caption-friendly "at X.XXs"
  // timestamp; nothing else reads these numbers.
  var REGION_LABELS = {
    head:                  'head',
    face:                  'face',
    neck:                  'neck',
    shoulder_dominant:     'shoulder (dominant)',
    shoulder_nondominant:  'shoulder (non-dominant)',
    chest:                 'chest',
    torso:                 'torso',
    arm_dominant:          'arm (dominant)',
    arm_nondominant:       'arm (non-dominant)',
    hand_dominant:         'hand (dominant)',
    hand_nondominant:      'hand (non-dominant)',
  };

  var STATUS_GENERATING = 'Generating HamNoSys…';
  var STATUS_RENDERING  = 'Rendering avatar…';
  var STATUS_READY      = 'Ready.';
  var STATUS_CACHED     = 'From cache';

  // ---------- DOM ----------

  var els = {
    panel:           document.getElementById('avatarPreview'),
    stage:           document.getElementById('avatarStage'),
    backdrop:        document.getElementById('avatarBackdrop'),
    canvas:          document.getElementById('avatarCanvas'),
    regions:         document.getElementById('avatarRegions'),
    pulse:           document.getElementById('avatarPulse'),
    fallback:        document.getElementById('avatarFallback'),
    status:          document.getElementById('avatarStatus'),
    controls:        null, // resolved in init()
    playBtn:         document.getElementById('avatarPlayBtn'),
    loopInput:       document.getElementById('avatarLoopInput'),
    scrubber:        document.getElementById('avatarScrubber'),
    scrubberTicks:   document.getElementById('avatarScrubberTicks'),
    time:            document.getElementById('avatarTime'),
    speedBtns:       null, // resolved in init()
    captions:        document.getElementById('avatarCaptions'),
    captionGloss:    document.getElementById('avatarCaptionGloss'),
    captionDesc:     document.getElementById('avatarCaptionDesc'),
  };

  if (!els.panel || !els.canvas || !els.playBtn) {
    // No preview markup on this page — bail silently.
    return;
  }

  // ---------- internal state ----------

  var state = {
    // Lifecycle flags.
    cwasaReady:          false,
    cwasaFailed:         false,
    webglOk:             null,  // null = not yet probed
    initStarted:         false,
    renderFailed:        false,
    // The sigml currently loaded into CWASA (or most recently played).
    currentSigml:        null,
    // Per-browser-session cache of SiGML strings we've played at least
    // once. If a second render request lands for a string we've seen
    // already, we flash "From cache" next to the status.
    playedSigml:         Object.create(null),
    // Virtual playback cursor. CWASA does not expose a seek API in its
    // public surface, so the scrubber is driven by `avatarframe` hook
    // updates during playback and by arrow-key / pointer drags when
    // paused. The captured value is what we forward into /correct's
    // target_time_ms.
    currentTimeMs:       0,
    durationMs:          DEFAULT_DURATION_MS,
    playing:             false,
    scrubbing:           false,
    scrubberWasPlaying:  false,
    // Observed frame/sign data from CWASA hooks — we use the animidle
    // edge to flip playing → false (and to schedule loop restarts).
    hookAttached:        false,
    loopPending:         false,
    // Cache of the last observed session id so we can reset duration,
    // played-set, etc. on a session swap.
    rememberedSessionId: null,
    // Current session state from the store. Drives the status strip
    // and the empty/pulse/ready visual mode.
    sessionState:        null,
    // Cache-badge timer.
    cacheBadgeTimer:     null,
    // Deferred auto-play request set when SiGML arrives before CWASA
    // has finished booting; consumed by maybePlayPending().
    pendingPlay:         false,
  };

  // ---------- WebGL + CWASA availability probe ----------

  function probeWebGL() {
    if (state.webglOk !== null) return state.webglOk;
    try {
      var c = document.createElement('canvas');
      var gl = c.getContext('webgl') || c.getContext('experimental-webgl');
      state.webglOk = !!gl;
    } catch (_e) {
      state.webglOk = false;
    }
    return state.webglOk;
  }

  function waitForCWASA() {
    // The CWASA bundle loads with `defer`, so on DOMContentLoaded it's
    // often still parsing. Poll briefly for the global to arrive; if
    // it doesn't in CWASA_WAIT_MS, assume the load failed (network,
    // CSP, bundle corrupted) and mark the pane for the fallback path.
    return new Promise(function (resolve) {
      if (window.CWASA) { resolve(true); return; }
      var waited = 0;
      var tick = setInterval(function () {
        if (window.CWASA) {
          clearInterval(tick);
          resolve(true);
          return;
        }
        waited += CWASA_POLL_MS;
        if (waited >= CWASA_WAIT_MS) {
          clearInterval(tick);
          resolve(false);
        }
      }, CWASA_POLL_MS);
    });
  }

  function initCWASA() {
    if (state.initStarted) return;
    state.initStarted = true;

    if (!probeWebGL()) {
      markRenderFailed('webgl_unavailable');
      return;
    }

    waitForCWASA().then(function (ok) {
      if (!ok || !window.CWASA) {
        markRenderFailed('cwasa_missing');
        return;
      }
      try {
        window.CWASA.init({
          useClientConfig: false,
          useCwaConfig:    true,
          avSettings: [{
            width:           384,
            height:          320,
            avList:          'avs',
            initAv:          'luna',
            ambIdle:         true,
            allowFrameSteps: false,
            allowSiGMLText:  false,
            // Let the control bar's speed buttons drive playback rate.
            // initSpeed is a log2 offset on CWASA's internal rate; 0 is
            // neutral (≡ 1× in the UI).
            initSpeed:       0,
          }],
        });
      } catch (e) {
        if (window.console) console.error('[contribute-preview] CWASA.init threw', e);
        markRenderFailed('cwasa_init_threw');
        return;
      }

      attachCwasaHooks();

      if (window.CWASA.ready && typeof window.CWASA.ready.then === 'function') {
        window.CWASA.ready.then(function () {
          state.cwasaReady = true;
          applyReadyState();
          maybePlayPending();
        }).catch(function () {
          markRenderFailed('cwasa_ready_rejected');
        });
      }
    });
  }

  function attachCwasaHooks() {
    if (state.hookAttached) return;
    if (!window.CWASA || typeof window.CWASA.addHook !== 'function') return;
    state.hookAttached = true;

    // Avatar model finished loading — the pane can show a resting pose.
    window.CWASA.addHook('avatarready', function () {
      state.cwasaReady = true;
      applyReadyState();
      maybePlayPending();
    }, 0);

    // `avatarframe` fires at render tick with { s: sequenceIdx, f: frame }.
    // We convert the frame into a ms timestamp using CWASA's declared
    // FPS (30 by default) and refine durationMs upward as playback
    // reveals frames past our initial estimate.
    window.CWASA.addHook('avatarframe', function (info) {
      if (!info || typeof info.f !== 'number') return;
      var fps = 30;
      var ms = Math.round((info.f / fps) * 1000);
      if (ms > state.durationMs - 40) {
        // We're approaching the tail of the estimated window — stretch
        // it so the scrubber has room to reach the end cleanly.
        state.durationMs = ms + 200;
        applyDurationChange();
      }
      if (!state.scrubbing) {
        setCurrentTime(ms, { fromHook: true });
      }
    }, 0);

    window.CWASA.addHook('animactive', function () {
      state.playing = true;
      state.renderFailed = false;
      applyReadyState();
      applyPlayButtonState();
      applyPulse();
    }, 0);

    window.CWASA.addHook('animidle', function () {
      state.playing = false;
      applyPlayButtonState();
      applyPulse();
      // Loop restart: fire from the idle edge rather than a timer so we
      // follow the avatar's actual end-of-sign, not a predicted clock.
      if (els.loopInput.checked && state.currentSigml && state.cwasaReady) {
        state.loopPending = true;
        setTimeout(function () {
          if (!state.loopPending) return;
          state.loopPending = false;
          if (els.loopInput.checked && !state.playing) doPlay();
        }, 80);
      }
    }, 0);

    window.CWASA.addHook('sigmlloading', function () {
      applyStatus();
    }, 0);
  }

  // ---------- render-state transitions ----------

  function markRenderFailed(reason) {
    state.cwasaFailed = true;
    state.renderFailed = true;
    if (window.console) console.warn('[contribute-preview] falling back to text:', reason);
    applyReadyState();
    applyStatus();
  }

  function applyReadyState() {
    var failed = state.cwasaFailed || state.renderFailed;
    els.fallback.hidden = !failed;
    els.stage.classList.toggle('is-failed', !!failed);
    els.canvas.setAttribute('aria-hidden', failed ? 'true' : 'false');
    els.regions.setAttribute('aria-hidden', failed ? 'true' : 'false');

    var interactive = state.cwasaReady && !failed;
    els.playBtn.disabled = !interactive;
    els.scrubber.disabled = !interactive;
  }

  function applyPlayButtonState() {
    if (state.playing) {
      els.playBtn.textContent = 'Pause';
      els.playBtn.setAttribute('aria-pressed', 'true');
      els.playBtn.setAttribute('aria-label', 'Pause preview');
    } else {
      els.playBtn.textContent = 'Play';
      els.playBtn.setAttribute('aria-pressed', 'false');
      els.playBtn.setAttribute('aria-label', 'Play preview');
    }
  }

  function applyPulse() {
    // Subtle backdrop pulse during generation or rendering — no
    // spinners, no percentage, never over the canvas itself.
    var s = state.sessionState;
    var showPulse = (s === 'generating' || s === 'applying_correction') && !state.renderFailed;
    els.pulse.hidden = !showPulse;
  }

  function applyStatus() {
    if (state.renderFailed) { els.status.textContent = ''; return; }
    var s = state.sessionState;
    if (s === 'generating') {
      els.status.textContent = STATUS_GENERATING;
    } else if (s === 'applying_correction') {
      els.status.textContent = STATUS_RENDERING;
    } else if (s === 'rendered' || s === 'awaiting_correction') {
      els.status.textContent = state.currentSigml ? STATUS_READY : '';
    } else {
      els.status.textContent = '';
    }
  }

  function flashCacheBadge() {
    if (state.renderFailed) return;
    els.status.textContent = STATUS_CACHED;
    if (state.cacheBadgeTimer) clearTimeout(state.cacheBadgeTimer);
    state.cacheBadgeTimer = setTimeout(function () {
      state.cacheBadgeTimer = null;
      applyStatus();
    }, CACHE_BADGE_MS);
  }

  // ---------- scrubber + duration ----------

  function formatSeconds(ms) {
    var s = Math.max(0, ms) / 1000;
    return s.toFixed(2);
  }

  function setCurrentTime(ms, opts) {
    opts = opts || {};
    var clamped = Math.max(0, Math.min(ms, state.durationMs));
    state.currentTimeMs = clamped;
    els.scrubber.value = String(clamped);
    els.scrubber.setAttribute('aria-valuenow', String(clamped));
    els.scrubber.setAttribute('aria-valuetext',
      formatSeconds(clamped) + ' seconds of ' +
      formatSeconds(state.durationMs) + ' seconds');
    els.time.textContent = formatSeconds(clamped) + ' / ' +
                           formatSeconds(state.durationMs) + ' s';
  }

  function applyDurationChange() {
    els.scrubber.max = String(state.durationMs);
    els.scrubber.setAttribute('aria-valuemax', String(state.durationMs));
    renderScrubberTicks();
    setCurrentTime(state.currentTimeMs); // re-render time label
  }

  function renderScrubberTicks() {
    // Native range tick marks via <datalist>. 100ms granularity, kept
    // light — we only emit integer deciseconds.
    while (els.scrubberTicks.firstChild) {
      els.scrubberTicks.removeChild(els.scrubberTicks.firstChild);
    }
    var step = SCRUB_SMALL_STEP_MS;
    for (var t = 0; t <= state.durationMs; t += step) {
      var opt = document.createElement('option');
      opt.value = String(t);
      els.scrubberTicks.appendChild(opt);
    }
  }

  function onScrubberInput(ev) {
    var ms = parseInt(ev.target.value, 10);
    if (!isFinite(ms)) return;
    setCurrentTime(ms);
  }

  function onScrubberPointerDown() {
    state.scrubbing = true;
    state.scrubberWasPlaying = state.playing;
    if (state.playing) doPause();
  }

  function onScrubberPointerUp() {
    state.scrubbing = false;
    // We don't attempt to seek CWASA — the captured time is forwarded
    // into the next correction. If the user was playing before they
    // grabbed the handle, we don't auto-resume (the typical flow is:
    // scrub → click a region → describe a correction).
    state.scrubberWasPlaying = false;
  }

  function onScrubberKey(ev) {
    if (els.scrubber.disabled) return;
    var delta = 0;
    if (ev.key === 'ArrowLeft' || ev.key === 'ArrowDown') {
      delta = ev.shiftKey ? -SCRUB_LARGE_STEP_MS : -SCRUB_SMALL_STEP_MS;
    } else if (ev.key === 'ArrowRight' || ev.key === 'ArrowUp') {
      delta = ev.shiftKey ? SCRUB_LARGE_STEP_MS : SCRUB_SMALL_STEP_MS;
    } else {
      return;
    }
    ev.preventDefault();
    setCurrentTime(state.currentTimeMs + delta);
  }

  // ---------- play / pause / loop / speed ----------

  function doPlay() {
    if (state.renderFailed) return;
    if (!state.cwasaReady) {
      state.pendingPlay = true;
      return;
    }
    if (!state.currentSigml) return;
    try {
      window.CWASA.playSiGMLText(state.currentSigml, 0);
      state.playing = true;
      state.loopPending = false;
    } catch (e) {
      if (window.console) console.error('[contribute-preview] playSiGMLText threw', e);
      markRenderFailed('play_threw');
      return;
    }
    applyPlayButtonState();
  }

  function doPause() {
    if (!state.cwasaReady) return;
    try {
      if (typeof window.CWASA.stopSiGML === 'function') {
        window.CWASA.stopSiGML(0);
      } else if (typeof window.CWASA.stop === 'function') {
        window.CWASA.stop(0);
      }
    } catch (_e) { /* swallow — animidle fires the state update */ }
    state.playing = false;
    state.loopPending = false;
    applyPlayButtonState();
  }

  function togglePlay() {
    if (state.playing) doPause();
    else doPlay();
  }

  function onSpeedClick(ev) {
    var btn = ev.currentTarget;
    var rate = parseFloat(btn.getAttribute('data-speed')) || 1;
    for (var i = 0; i < els.speedBtns.length; i++) {
      var b = els.speedBtns[i];
      var on = b === btn;
      b.classList.toggle('is-active', on);
      b.setAttribute('aria-pressed', on ? 'true' : 'false');
    }
    applySpeedToCWASA(rate);
  }

  function applySpeedToCWASA(rate) {
    // CWASA's built-in hidden GUI carries a speed slider we can drive
    // via DOM. 0 is neutral; each ±1 step is a doubling/halving. We map
    // 0.5→−1, 1→0, 2→+1. If the slider isn't present (older CWASA,
    // GUI removed), we silently skip — the scrubber still reflects
    // wall-clock time from avatarframe either way.
    var logStep = Math.round(Math.log(rate) / Math.log(2));
    var slider = document.querySelector('.CWASAGUI.av0 .spdBase, .CWASAGUI.av0 input[type="range"]');
    if (slider) {
      try {
        slider.value = String(logStep);
        slider.dispatchEvent(new Event('input', { bubbles: true }));
        slider.dispatchEvent(new Event('change', { bubbles: true }));
      } catch (_e) { /* ignore */ }
    }
  }

  // ---------- SiGML supply: driven by the store ----------

  function maybePlayPending() {
    // Covers the case where the session envelope already carried a
    // SiGML string before CWASA finished booting. Called once when
    // the ready flag flips.
    if (!state.cwasaReady) return;
    if (state.pendingPlay && state.currentSigml) {
      state.pendingPlay = false;
      doPlay();
    }
  }

  function applyNewSigml(sigml) {
    if (!sigml) {
      state.currentSigml = null;
      return;
    }
    var firstSeen = !state.playedSigml[sigml];
    var changed = sigml !== state.currentSigml;
    state.currentSigml = sigml;

    if (!changed) return;

    // Always reset the scrubber to 0 on new SiGML so the user isn't
    // silently forwarded into the tail of the previous clip.
    setCurrentTime(0);

    // Reset estimated duration — we'll refine upward from avatarframe.
    state.durationMs = DEFAULT_DURATION_MS;
    applyDurationChange();

    if (!firstSeen) {
      flashCacheBadge();
    }
    state.playedSigml[sigml] = true;

    // Auto-play as soon as notation arrives. If CWASA isn't ready yet,
    // stash the request so we play once `ready` resolves.
    if (state.cwasaReady && !state.renderFailed) {
      doPlay();
    } else if (!state.renderFailed) {
      state.pendingPlay = true;
    }
  }

  // ---------- captions ----------

  function updateCaptions(snap) {
    els.captionGloss.textContent = snap.gloss || '';
    // Description isn't persisted on the envelope — pull it from the
    // language-scoped draft store written by contribute.js (prompt 5).
    var desc = '';
    try {
      if (window.KOZHA_CONTRIB && typeof window.KOZHA_CONTRIB.readDraft === 'function') {
        var d = window.KOZHA_CONTRIB.readDraft(snap.language);
        if (d && typeof d.description === 'string') desc = d.description;
      }
    } catch (_e) { /* non-fatal */ }
    els.captionDesc.textContent = desc;
    els.captionDesc.hidden = !desc;
  }

  // ---------- body-region targeting ----------

  function onRegionClick(ev) {
    var tgt = ev.target;
    if (!tgt || !tgt.classList || !tgt.classList.contains('avatar-region')) return;
    var region = tgt.getAttribute('data-region');
    if (!region) return;
    // Focus the chat input labeled "Describe what should change" and
    // hand over the (region, time) pair via the chat module's public
    // surface. The chat panel shows the pill; we don't own that DOM.
    var chat = window.KOZHA_CONTRIB_CHAT;
    if (!chat || typeof chat.setCorrectionTarget !== 'function') return;
    chat.setCorrectionTarget({
      region:   region,
      label:    REGION_LABELS[region] || region,
      timeMs:   state.currentTimeMs,
      timeText: formatSeconds(state.currentTimeMs),
    });
  }

  function onRegionHover(ev) {
    var tgt = ev.target;
    if (!tgt || !tgt.classList) return;
    if (tgt.classList.contains('avatar-region')) {
      tgt.classList.add('is-hover');
    }
  }
  function onRegionLeave(ev) {
    var tgt = ev.target;
    if (!tgt || !tgt.classList) return;
    if (tgt.classList.contains('avatar-region')) {
      tgt.classList.remove('is-hover');
    }
  }

  // ---------- global keyboard (within the pane) ----------

  function onPaneKey(ev) {
    // Only steal keys when the pane contains focus — so typing in the
    // chat textarea still spaces-in-text as expected.
    if (!els.panel.contains(document.activeElement)) return;
    if (ev.target && ev.target.tagName === 'INPUT' &&
        ev.target.type !== 'range' && ev.target.type !== 'checkbox') return;
    if (ev.target && ev.target.tagName === 'TEXTAREA') return;

    if (ev.key === ' ' || ev.key === 'Spacebar') {
      if (els.playBtn.disabled) return;
      ev.preventDefault();
      togglePlay();
    }
    // ArrowLeft/Right on the scrubber are handled by onScrubberKey.
  }

  // ---------- subscriber ----------

  function shouldShowPane(snap) {
    if (!snap.sessionId) return false;
    if (snap.sessionState === 'abandoned') return false;
    // Visible from the moment a session exists — empty state shows the
    // neutral resting pose so the notation panel below never jumps.
    return true;
  }

  function onSnapshot(snap) {
    if (!shouldShowPane(snap)) {
      els.panel.hidden = true;
      return;
    }
    if (els.panel.hidden) {
      els.panel.hidden = false;
      initCWASA();
    }

    if (state.rememberedSessionId !== snap.sessionId) {
      state.rememberedSessionId = snap.sessionId;
      state.playedSigml = Object.create(null);
      state.currentSigml = null;
      state.durationMs = DEFAULT_DURATION_MS;
      applyDurationChange();
      setCurrentTime(0);
    }

    state.sessionState = snap.sessionState;

    if (snap.sigml && snap.sigml !== state.currentSigml) {
      applyNewSigml(snap.sigml);
    } else if (!snap.sigml && state.currentSigml) {
      // Correction wiped the draft — stop playback and reset.
      doPause();
      state.currentSigml = null;
      setCurrentTime(0);
    }

    updateCaptions(snap);
    applyPulse();
    applyStatus();
  }

  // ---------- test hook ----------

  window.KOZHA_CONTRIB_PREVIEW = {
    setDurationForTest: function (ms) {
      state.durationMs = Math.max(100, ms | 0);
      applyDurationChange();
    },
    forceFallbackForTest: function () { markRenderFailed('forced_for_test'); },
    getState: function () {
      return {
        durationMs:    state.durationMs,
        currentTimeMs: state.currentTimeMs,
        playing:       state.playing,
        renderFailed:  state.renderFailed,
      };
    },
  };

  // ---------- init ----------

  function init() {
    els.speedBtns = els.panel.querySelectorAll('.avatar-speed-btn');

    els.playBtn.addEventListener('click', togglePlay);
    els.loopInput.addEventListener('change', function () {
      if (!els.loopInput.checked) state.loopPending = false;
    });
    els.scrubber.addEventListener('input', onScrubberInput);
    els.scrubber.addEventListener('pointerdown', onScrubberPointerDown);
    els.scrubber.addEventListener('pointerup', onScrubberPointerUp);
    els.scrubber.addEventListener('keydown', onScrubberKey);
    for (var i = 0; i < els.speedBtns.length; i++) {
      els.speedBtns[i].addEventListener('click', onSpeedClick);
    }
    els.regions.addEventListener('click', onRegionClick);
    els.regions.addEventListener('mouseover', onRegionHover);
    els.regions.addEventListener('mouseout', onRegionLeave);
    document.addEventListener('keydown', onPaneKey);

    applyDurationChange();
    setCurrentTime(0);
    applyPlayButtonState();
    applyPulse();

    CTX.subscribe(onSnapshot);
    onSnapshot(CTX.getState());
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

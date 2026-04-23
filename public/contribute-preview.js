/* Contribute page — avatar preview pane controller.
 *
 * Spec: prompts/prompt-contrib-9.md.
 *
 * Scope:
 * - Renders the CWASA signing-avatar canvas inside a 16:9 stage with a
 *   light-gray backdrop (Kipp et al. 2011 avatar UX: dark backdrops
 *   reduce legibility for many Deaf users).
 * - Control bar: Play/Pause button, Loop checkbox, three playback-speed
 *   buttons. No timeline scrubber — CWASA exposes no seek API, so a
 *   draggable cursor would advertise control we can't honor.
 * - Body-region SVG overlay — hovering a region softens its outline;
 *   clicking captures the region (plus the best-effort current time)
 *   and hands it to the chat module as a pill. Targeting is optional:
 *   corrections may also be submitted as free text with no region.
 * - Generation progress is handled *inside* the pane — a muted pulse
 *   on the backdrop during GENERATING/RENDERING, never a spinner or
 *   percentage. Status strip underneath narrates in plain text.
 * - Failure handling: if CWASA / WebGL / the script bundle is missing
 *   the pane collapses to a plain-text fallback and the notation
 *   panel (prompt 7) becomes the primary surface. Submit stays
 *   enabled.
 * - Cache signal: if we're asked to render a SiGML string we've
 *   already seen this session, show "From cache" for 1.5s.
 * - Accessibility: keyboard play/pause (space), aria-live status
 *   strip, captions show the gloss and description so Deaf reviewers
 *   can cross-check contributor intent.
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

  // Optional debug logger — see contribute-debug.js. No-op when the
  // user hasn't enabled debug mode, so this stays free of overhead in
  // the default flow.
  var DEBUG = (window.KOZHA_CONTRIB_DEBUG && window.KOZHA_CONTRIB_DEBUG.log)
    ? window.KOZHA_CONTRIB_DEBUG
    : { log: function () {}, forceLog: function () {} };

  // ---------- constants ----------

  var DEFAULT_DURATION_MS = 1400;
  var CACHE_BADGE_MS = 1500;
  var CWASA_WAIT_MS = 6000; // if CWASA never arrives, assume the bundle failed
  var CWASA_POLL_MS  = 100;
  // SVG viewBox is 160x90 (16:9). Convert clicked SVG coords back into
  // percentages of the stage for the caption-friendly "at X.XXs"
  // timestamp; nothing else reads these numbers.
  // Body-region IDs carry a catalog key; the English label is the fallback
  // used when the i18n catalog has not loaded yet or is absent.
  function tr(key, fallback) {
    if (window.KOZHA_I18N && typeof window.KOZHA_I18N.t === 'function') {
      var v = window.KOZHA_I18N.t(key);
      if (v && v !== key) return v;
    }
    return fallback;
  }
  var REGION_LABEL_FALLBACK = {
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
  function regionLabel(region) {
    var fb = REGION_LABEL_FALLBACK[region] || region;
    return tr('contribute.preview.region_' + region, fb);
  }

  function statusGenerating() { return tr('contribute.preview.status_generating', 'Generating HamNoSys…'); }
  function statusRendering()  { return tr('contribute.preview.status_rendering', 'Rendering avatar…'); }
  function statusReady()      { return tr('contribute.preview.status_ready', 'Ready.'); }
  function statusCached()     { return tr('contribute.preview.status_cached', 'From cache'); }

  // ---------- DOM ----------

  var els = {
    panel:           document.getElementById('avatarPreview'),
    stage:           document.getElementById('avatarStage'),
    backdrop:        document.getElementById('avatarBackdrop'),
    canvas:          document.getElementById('avatarCanvas'),
    regions:         document.getElementById('avatarRegions'),
    pulse:           document.getElementById('avatarPulse'),
    fallback:        document.getElementById('avatarFallback'),
    fallbackMsg:     document.getElementById('avatarFallbackMsg'),
    fallbackDetail:  document.getElementById('avatarFallbackDetail'),
    fallbackRetry:   document.getElementById('avatarFallbackRetry'),
    fallbackDiag:    document.getElementById('avatarFallbackDiag'),
    fallbackRegen:   document.getElementById('avatarFallbackRegen'),
    status:          document.getElementById('avatarStatus'),
    controls:        null, // resolved in init()
    playBtn:         document.getElementById('avatarPlayBtn'),
    loopInput:       document.getElementById('avatarLoopInput'),
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
    // The reason ``markRenderFailed`` was last invoked with — kept on
    // state so ``applyReadyState`` can render an actionable detail
    // line and the debug log can replay it on retry.
    lastFailureReason:   null,
    webglOk:             null,  // null = not yet probed
    initStarted:         false,
    renderFailed:        false,
    // The sigml currently loaded into CWASA (or most recently played).
    currentSigml:        null,
    // Per-browser-session cache of SiGML strings we've played at least
    // once. If a second render request lands for a string we've seen
    // already, we flash "From cache" next to the status.
    playedSigml:         Object.create(null),
    // Virtual playback cursor, driven by `avatarframe` hook updates
    // during playback. Without a seek API there is no user-facing
    // scrubber; we still track a best-effort current time so a region
    // click can forward an approximate target_time_ms to /correct.
    currentTimeMs:       0,
    durationMs:          DEFAULT_DURATION_MS,
    playing:             false,
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

    DEBUG.log('preview: initCWASA start', {
      cwasaPresent: !!window.CWASA,
      webglProbe: probeWebGL(),
    });

    if (!probeWebGL()) {
      markRenderFailed('webgl_unavailable');
      return;
    }

    waitForCWASA().then(function (ok) {
      DEBUG.log('preview: waitForCWASA result', { ok: ok, cwasaPresent: !!window.CWASA });
      if (!ok || !window.CWASA) {
        markRenderFailed('cwasa_missing');
        return;
      }
      try {
        DEBUG.log('preview: calling CWASA.init');
        window.CWASA.init({
          useClientConfig: false,
          useCwaConfig:    true,
          avSettings: [{
            // 720x540 native render — large enough that the avatar
            // fills the contribute preview stage on desktop without
            // upscaling. CSS (.CWASAAvatar.av0 rule below in
            // contribute.css) scales the canvas to fill the stage
            // with object-fit: contain to preserve proportions.
            width:           720,
            height:          540,
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
        DEBUG.error('preview: CWASA.init threw', { message: String(e), stack: e && e.stack ? String(e.stack).slice(0, 800) : undefined });
        markRenderFailed('cwasa_init_threw');
        return;
      }

      attachCwasaHooks();

      if (window.CWASA.ready && typeof window.CWASA.ready.then === 'function') {
        window.CWASA.ready.then(function () {
          DEBUG.log('preview: CWASA.ready resolved');
          state.cwasaReady = true;
          applyReadyState();
          maybePlayPending();
        }).catch(function (err) {
          DEBUG.error('preview: CWASA.ready rejected', { error: String(err) });
          markRenderFailed('cwasa_ready_rejected');
        });
      } else {
        DEBUG.warn('preview: CWASA exposes no .ready promise — relying on avatarready hook');
      }
    });
  }

  function attachCwasaHooks() {
    if (state.hookAttached) return;
    if (!window.CWASA || typeof window.CWASA.addHook !== 'function') return;
    state.hookAttached = true;

    // Avatar model finished loading — the pane can show a resting pose.
    window.CWASA.addHook('avatarready', function () {
      DEBUG.log('preview: hook avatarready');
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
        state.durationMs = ms + 200;
        applyDurationChange();
      }
      setCurrentTime(ms, { fromHook: true });
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
    state.lastFailureReason = reason;
    if (window.console) console.warn('[contribute-preview] falling back to text:', reason);
    DEBUG.log('preview: render failed', { reason: reason });
    applyReadyState();
    applyStatus();
  }

  // Human-readable, actionable explanation for each failure reason. The
  // generic "Preview unavailable in this environment" is a polite
  // shrug; this gives the contributor (and any debug-mode reader) the
  // specific cause and what they can do about it.
  function failureDetailFor(reason) {
    switch (reason) {
      case 'webgl_unavailable':
        return tr('contribute.preview.fail_webgl',
          'WebGL is disabled or unavailable in this browser. Try Chrome or Firefox with hardware acceleration enabled.');
      case 'cwasa_missing':
        return tr('contribute.preview.fail_cwasa_missing',
          'The signing-avatar bundle did not load (network blocked, ad-blocker, or strict CSP). Reload to retry, or check the browser console for /cwa/allcsa.js.');
      case 'cwasa_init_threw':
      case 'cwasa_ready_rejected':
        return tr('contribute.preview.fail_cwasa_init',
          'The signing-avatar engine could not start. Reload to retry — if this persists, your browser may not support the player.');
      case 'play_threw':
        return tr('contribute.preview.fail_play',
          'The player rejected this SiGML. Reload and try again, or submit as-is so a reviewer can play it server-side.');
      case 'avatar_load_timeout':
        return tr('contribute.preview.fail_avatar_load',
          'The avatar took too long to download. Reload to retry on a stable connection.');
      case 'sigml_invalid_object-literal-in-xml':
        return tr('contribute.preview.fail_sigml_obj',
          'The generated SiGML contains a [object Object] string — the upstream pipeline emitted a JS object where text was expected. The pipeline will retry on the next correction, or click Retry preview after submitting another correction.');
      case 'sigml_invalid_empty-or-non-string':
        return tr('contribute.preview.fail_sigml_empty',
          'The generated SiGML is empty. Send a correction or reload to ask the model again.');
      case 'sigml_invalid_missing-sigml-root':
        return tr('contribute.preview.fail_sigml_root',
          'The generated SiGML is missing its <sigml> wrapper. The pipeline will retry on the next correction.');
      default:
        return '';
    }
  }

  function retryPreview() {
    DEBUG.log('preview: manual retry requested', { previousReason: state.lastFailureReason });
    state.cwasaFailed = false;
    state.renderFailed = false;
    state.cwasaReady = false;
    state.initStarted = false;
    state.lastFailureReason = null;
    applyReadyState();
    applyStatus();
    initCWASA();
  }

  function applyReadyState() {
    var failed = state.cwasaFailed || state.renderFailed;
    els.fallback.hidden = !failed;
    els.stage.classList.toggle('is-failed', !!failed);
    els.canvas.setAttribute('aria-hidden', failed ? 'true' : 'false');
    els.regions.setAttribute('aria-hidden', failed ? 'true' : 'false');

    if (failed && els.fallbackDetail) {
      var detail = failureDetailFor(state.lastFailureReason);
      if (detail) {
        els.fallbackDetail.textContent = detail;
        els.fallbackDetail.hidden = false;
      } else {
        els.fallbackDetail.hidden = true;
      }
    } else if (els.fallbackDetail) {
      els.fallbackDetail.hidden = true;
    }
    // The retry button is shown for failure modes a reload won't fix
    // automatically (notably cwasa_missing, init/play threw). For
    // webgl_unavailable retry won't help, so hide it there.
    if (els.fallbackRetry) {
      var canRetry = failed && state.lastFailureReason !== 'webgl_unavailable';
      els.fallbackRetry.hidden = !canRetry;
    }
    // The regenerate button is shown when CWASA crashed on a SiGML
    // we shipped — that's a server-side generation problem and
    // re-rendering won't help; we have to ask the server for a new
    // SiGML. Modes that warrant regen: play_threw (the avatar engine
    // refused the markup) and any sigml_invalid_* prefix (the
    // pre-flight pre-emptively rejected it).
    if (els.fallbackRegen) {
      var reason = state.lastFailureReason || '';
      var canRegen = failed && (
        reason === 'play_threw' ||
        reason.indexOf('sigml_invalid_') === 0
      );
      els.fallbackRegen.hidden = !canRegen;
    }

    var interactive = state.cwasaReady && !failed;
    els.playBtn.disabled = !interactive;
  }

  function applyPlayButtonState() {
    if (state.playing) {
      els.playBtn.textContent = tr('contribute.preview.pause', 'Pause');
      els.playBtn.setAttribute('aria-pressed', 'true');
      els.playBtn.setAttribute('aria-label', tr('contribute.preview.pause_aria', 'Pause preview'));
    } else {
      els.playBtn.textContent = tr('contribute.preview.play', 'Play');
      els.playBtn.setAttribute('aria-pressed', 'false');
      els.playBtn.setAttribute('aria-label', tr('contribute.preview.play_aria', 'Play preview'));
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
      els.status.textContent = statusGenerating();
    } else if (s === 'applying_correction') {
      els.status.textContent = statusRendering();
    } else if (s === 'rendered' || s === 'awaiting_correction') {
      els.status.textContent = state.currentSigml ? statusReady() : '';
    } else {
      els.status.textContent = '';
    }
  }

  function flashCacheBadge() {
    if (state.renderFailed) return;
    els.status.textContent = statusCached();
    if (state.cacheBadgeTimer) clearTimeout(state.cacheBadgeTimer);
    state.cacheBadgeTimer = setTimeout(function () {
      state.cacheBadgeTimer = null;
      applyStatus();
    }, CACHE_BADGE_MS);
  }

  // ---------- time tracking ----------

  function formatSeconds(ms) {
    var s = Math.max(0, ms) / 1000;
    return s.toFixed(2);
  }

  function setCurrentTime(ms, _opts) {
    var clamped = Math.max(0, Math.min(ms, state.durationMs));
    state.currentTimeMs = clamped;
  }

  function applyDurationChange() { /* scrubber removed — duration drifts silently */ }

  // ---------- play / pause / loop / speed ----------

  // Pre-flight SiGML check: catches the `[object Object]` regression
  // (a JS-side variable getting stringified into the SiGML stream)
  // before CWASA's grammar parser barfs with "Ham4HMLGen.g: node from
  // line 0:0 mismatched input '[object Object]' expecting MICFG2".
  // When detected we suppress the play, log loudly so the issue is
  // visible in the downloaded debug log, and ask the store for a
  // re-generation rather than leaving the user staring at a frozen
  // preview.
  function sigmlLooksValid(s) {
    if (typeof s !== 'string' || !s.trim()) {
      return { ok: false, reason: 'empty-or-non-string' };
    }
    if (s.indexOf('[object Object]') !== -1) {
      return { ok: false, reason: 'object-literal-in-xml' };
    }
    if (s.indexOf('<sigml') === -1 && s.indexOf('<hns_sign') === -1) {
      return { ok: false, reason: 'missing-sigml-root' };
    }
    return { ok: true };
  }

  function doPlay() {
    if (state.renderFailed) {
      DEBUG.log('preview: doPlay skipped (renderFailed)');
      return;
    }
    if (!state.cwasaReady) {
      DEBUG.log('preview: doPlay deferred (cwasaReady=false) — pendingPlay=true');
      state.pendingPlay = true;
      return;
    }
    if (!state.currentSigml) {
      DEBUG.log('preview: doPlay skipped (no currentSigml)');
      return;
    }
    var v = sigmlLooksValid(state.currentSigml);
    if (!v.ok) {
      DEBUG.error('preview: refusing to play malformed SiGML', {
        reason: v.reason,
        sample: String(state.currentSigml).slice(0, 200),
      });
      // Surface the failure visually so the contributor knows the
      // sign needs another generation pass; the chat panel shows a
      // recoverable error and the preview stays at the resting pose
      // rather than the (broken) attempt.
      markRenderFailed('sigml_invalid_' + v.reason);
      return;
    }
    try {
      DEBUG.log('preview: CWASA.playSiGMLText', { sigmlLen: state.currentSigml.length });
      window.CWASA.playSiGMLText(state.currentSigml, 0);
      state.playing = true;
      state.loopPending = false;
    } catch (e) {
      if (window.console) console.error('[contribute-preview] playSiGMLText threw', e);
      DEBUG.error('preview: playSiGMLText threw', {
        message: e && e.message ? String(e.message) : String(e),
        name:    e && e.name,
        stack:   e && e.stack ? String(e.stack).slice(0, 1500) : undefined,
        sigmlHead: String(state.currentSigml).slice(0, 400),
        sigmlTail: String(state.currentSigml).slice(-200),
      });
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
    // GUI removed), we silently skip.
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
    DEBUG.log('preview: applyNewSigml', {
      sigmlLen: sigml.length,
      changed: changed,
      firstSeen: firstSeen,
      cwasaReady: state.cwasaReady,
      renderFailed: state.renderFailed,
    });
    state.currentSigml = sigml;

    if (!changed) return;

    // Reset the internal cursor on new SiGML so a subsequent region
    // click doesn't forward a target_time_ms from the previous clip.
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
      label:    regionLabel(region),
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
    if (ev.target && ev.target.tagName === 'INPUT' && ev.target.type !== 'checkbox') return;
    if (ev.target && ev.target.tagName === 'TEXTAREA') return;

    if (ev.key === ' ' || ev.key === 'Spacebar') {
      if (els.playBtn.disabled) return;
      ev.preventDefault();
      togglePlay();
    }
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
    if (els.fallbackRetry) {
      els.fallbackRetry.addEventListener('click', retryPreview);
    }
    if (els.fallbackDiag) {
      els.fallbackDiag.addEventListener('click', function () {
        if (window.KOZHA_CONTRIB_DEBUG && typeof window.KOZHA_CONTRIB_DEBUG.download === 'function') {
          DEBUG.log('preview: diagnostics download requested');
          window.KOZHA_CONTRIB_DEBUG.download();
        }
      });
    }
    if (els.fallbackRegen) {
      els.fallbackRegen.addEventListener('click', function () {
        // Send a "fix the sign" correction to the backend. The new
        // server-side regenerate fast-path (correction_interpreter.py
        // _looks_like_regenerate) routes this straight to
        // run_generation, which now self-corrects via the SiGML-direct
        // retry loop with the slot-completeness check. From the user's
        // perspective: one click and the broken sign is replaced with
        // a fresh attempt without re-typing a description.
        if (!CTX || typeof CTX.correct !== 'function') return;
        DEBUG.log('preview: regen requested via fallback button', { reason: state.lastFailureReason });
        // Clear the failed flag optimistically so the pulse shows
        // and the fallback card disappears the moment we POST.
        state.cwasaFailed = false;
        state.renderFailed = false;
        state.lastFailureReason = null;
        applyReadyState();
        applyStatus();
        CTX.correct(
          'fix the sign — the previous SiGML failed to play in CWASA, please regenerate'
        ).catch(function (err) {
          DEBUG.error('preview: regen correction failed', { status: err && err.status, body: err && err.body });
          // Surface the failure back into the preview pane so the
          // user sees something happen.
          markRenderFailed('regen_request_failed');
        });
      });
    }
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

/**
 * chat2hamnosys — minimal authoring frontend.
 *
 * State is a single `state` object mutated via API responses. The DOM is
 * re-rendered from scratch on every envelope update; this keeps the model
 * the only source of truth and avoids drift.
 */
(() => {
  'use strict';

  const API_BASE = '/api/chat2hamnosys';

  // Slot label + display order for the inspector.
  const SLOTS = [
    ['handshape_dominant',         'Handshape (dominant)'],
    ['handshape_nondominant',      'Handshape (non-dominant)'],
    ['orientation_extended_finger','Finger orientation'],
    ['orientation_palm',           'Palm orientation'],
    ['location',                   'Location'],
    ['contact',                    'Contact'],
    ['movement',                   'Movement'],
    ['non_manual.mouth_picture',   'Mouth picture'],
    ['non_manual.eye_gaze',        'Eye gaze'],
    ['non_manual.head_movement',   'Head movement'],
    ['non_manual.eyebrows',        'Eyebrows'],
    ['non_manual.facial_expression','Facial expression'],
  ];

  // Body-region click → suggested correction-target field path.
  const REGION_TO_FIELD = {
    'head':         'non_manual.head_movement',
    'face':         'non_manual.facial_expression',
    'torso':        'location',
    'hand-dom':     'handshape_dominant',
    'hand-nondom':  'handshape_nondominant',
    'arm-dom':      'movement[0].path',
    'arm-nondom':   'orientation_palm',
  };

  // Friendly progress text per SSE event type.
  const PROGRESS_TEXT = {
    described:              'Parsing your description…',
    clarification_asked:    'Preparing clarification questions…',
    clarification_answered: 'Applying your answer…',
    generated:              'Generated HamNoSys, rendering avatar…',
    correction_requested:   'Interpreting your correction…',
    correction_applied:     'Applying correction…',
    accepted:               'Sign accepted.',
    rejected:               'Session rejected.',
  };

  // -----------------------------------------------------------------------
  // State
  // -----------------------------------------------------------------------
  const state = {
    sessionId: null,
    sessionToken: null,
    envelope: null,
    eventSource: null,
    selectedRegion: null,        // {regionId, fieldPath}
    selectedTimeMs: null,        // captured timeline ms
    isLooping: false,
    cwasaInited: false,
    activeMobileTab: 'chat',
  };

  // -----------------------------------------------------------------------
  // DOM refs
  // -----------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const dom = {
    chatLog:      $('chatLog'),
    chatForm:     $('chatForm'),
    chatInput:    $('chatInput'),
    chatLabel:    $('chatInputLabel'),
    chatHint:     $('chatHint'),
    glossInput:   $('glossInput'),
    glossWrap:    $('glossWrap'),
    sendBtn:      $('sendBtn'),
    acceptBtn:    $('acceptBtn'),
    rejectBtn:    $('rejectBtn'),
    correctionCtx:$('correctionContext'),
    sessionMeta:  $('sessionMeta'),
    statusBanner: $('statusBanner'),
    statusText:   () => dom.statusBanner.querySelector('.status-text'),

    previewMount:    $('previewMount'),
    previewVideo:    $('previewVideo'),
    cwasaMount:      $('cwasaMount'),
    previewPh:       $('previewPlaceholder'),
    regionOverlay:   $('regionOverlay'),
    captionGloss:    $('captionGloss'),
    captionProse:    $('captionProse'),
    playBtn:         $('playBtn'),
    loopBtn:         $('loopBtn'),
    timeline:        $('timeline'),
    timelineLabel:   $('timelineLabel'),
    flagBtn:         $('flagMomentBtn'),
    selectionHint:   $('selectionHint'),

    inspectorEmpty:  $('inspectorEmpty'),
    inspectorList:   $('inspectorList'),
    inspectorBody:   $('inspectorBody'),
    inspectorToggle: $('inspectToggle'),

    srAnnounce:      $('srAnnounce'),
    tplMessage:      $('tplMessage'),
    tplOption:       $('tplOption'),
    tplInspectorRow: $('tplInspectorRow'),
  };

  // -----------------------------------------------------------------------
  // API
  // -----------------------------------------------------------------------
  async function api(method, path, body) {
    const headers = { 'Content-Type': 'application/json' };
    if (state.sessionToken) headers['X-Session-Token'] = state.sessionToken;
    const opts = { method, headers };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(API_BASE + path, opts);
    let payload;
    try { payload = await res.json(); }
    catch (_e) { payload = null; }
    if (!res.ok) {
      const code = (payload && payload.error && payload.error.code) || `http_${res.status}`;
      const msg  = (payload && payload.error && payload.error.message) || res.statusText;
      const err = new Error(msg);
      err.code = code;
      err.status = res.status;
      err.payload = payload;
      throw err;
    }
    return payload;
  }

  // -----------------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------------
  function announce(text) {
    dom.srAnnounce.textContent = '';
    setTimeout(() => { dom.srAnnounce.textContent = text; }, 50);
  }

  function setStatus(text) {
    dom.statusText().textContent = text;
    dom.statusBanner.hidden = false;
  }
  function clearStatus() { dom.statusBanner.hidden = true; }

  function fmtTime(iso) {
    if (!iso) return '';
    try { return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
    catch (_e) { return ''; }
  }

  function getSlotValue(params, path) {
    if (!params) return undefined;
    if (path === 'movement') return params.movement || [];
    if (path.startsWith('non_manual.')) {
      const leaf = path.split('.', 2)[1];
      return params.non_manual ? params.non_manual[leaf] : undefined;
    }
    return params[path];
  }

  function gapPaths() {
    if (!state.envelope) return new Set();
    return new Set((state.envelope.gaps || []).map((g) => g.field));
  }

  function clarifiedPaths() {
    if (!state.envelope) return new Set();
    return new Set((state.envelope.clarifications || []).map((c) => c.question_field || c.field));
  }

  function pendingFor(field) {
    if (!state.envelope) return null;
    return (state.envelope.pending_questions || []).find((q) => q.field === field) || null;
  }

  // -----------------------------------------------------------------------
  // Boot
  // -----------------------------------------------------------------------
  async function boot() {
    try {
      setStatus('Starting session…');
      const created = await api('POST', '/sessions', { sign_language: 'bsl' });
      state.sessionId = created.session_id;
      state.sessionToken = created.session_token;
      applyEnvelope(created.session);
      subscribeEvents();
      clearStatus();
      announce('Session ready. Describe a sign to begin.');
    } catch (err) {
      setStatus(`Failed to start session: ${err.message}`);
      console.error(err);
    }
  }

  // -----------------------------------------------------------------------
  // Envelope → re-render
  // -----------------------------------------------------------------------
  function applyEnvelope(env) {
    state.envelope = env;
    dom.sessionMeta.textContent =
      `Session ${env.session_id.slice(0, 8)} · ${env.sign_language.toUpperCase()} · state: ${env.state}`;
    renderChat();
    renderInspector();
    renderPreview();
    updateActions();
  }

  function updateActions() {
    const env = state.envelope;
    if (!env) return;
    const isTerminal = env.state === 'finalized' || env.state === 'abandoned';
    const isRendered = env.state === 'rendered';
    const hasDescription = (env.description_prose || '').length > 0;

    dom.acceptBtn.hidden = !isRendered;
    dom.rejectBtn.hidden = isTerminal || !hasDescription;
    dom.sendBtn.disabled = isTerminal;
    dom.chatInput.disabled = isTerminal;

    // Gloss input is only useful before the first describe call.
    dom.glossWrap.hidden = !!hasDescription || isTerminal;
    if (hasDescription) dom.glossInput.required = false;

    if (env.state === 'awaiting_description') {
      dom.chatLabel.textContent = 'Describe the sign';
      dom.chatHint.textContent =
        'Plain English. Mention handshape, location, palm orientation, and movement.';
      dom.chatInput.placeholder = "It's signed near the temple, flat hand, moves down to the chest…";
    } else if (env.state === 'clarifying') {
      dom.chatLabel.textContent = 'Answer a clarification or refine the description';
      dom.chatHint.textContent = 'Click an option above, or type your own.';
      dom.chatInput.placeholder = 'Type a free-form answer…';
    } else if (isRendered || env.state === 'awaiting_correction') {
      dom.chatLabel.textContent = state.selectedRegion || state.selectedTimeMs !== null
        ? 'What should change at this moment?'
        : 'Submit a correction (or accept the sign)';
      dom.chatHint.textContent =
        'Click on the avatar to flag a body region, or use the timeline to flag a moment.';
      dom.chatInput.placeholder = 'e.g. handshape should be flat, not a fist';
    } else if (isTerminal) {
      dom.chatLabel.textContent = 'Session ended';
      dom.chatHint.textContent =
        env.state === 'finalized'
          ? 'Sign saved as a draft entry. Awaiting Deaf review.'
          : 'Session abandoned.';
      dom.chatInput.placeholder = '';
    }

    renderCorrectionContext();
  }

  // -----------------------------------------------------------------------
  // Chat panel
  // -----------------------------------------------------------------------
  function renderChat() {
    dom.chatLog.innerHTML = '';
    const env = state.envelope;
    if (!env) return;

    for (const event of env.history) {
      appendHistoryEvent(event);
    }
    // Pending questions = assistant turn waiting on the user.
    for (const q of env.pending_questions || []) {
      appendQuestion(q);
    }

    if (env.state === 'finalized') {
      appendSystem('Sign accepted as draft entry — awaiting Deaf review.');
    }

    dom.chatLog.scrollTop = dom.chatLog.scrollHeight;
  }

  function appendMessage({ role, author, time, body, actions }) {
    const node = dom.tplMessage.content.firstElementChild.cloneNode(true);
    node.dataset.role = role;
    node.querySelector('.msg-author').textContent = author;
    node.querySelector('.msg-time').textContent = time || '';
    node.querySelector('.msg-body').textContent = body || '';
    const actionsBox = node.querySelector('.msg-actions');
    if (actions && actions.length) {
      actionsBox.hidden = false;
      for (const a of actions) actionsBox.appendChild(a);
    }
    dom.chatLog.appendChild(node);
    return node;
  }

  function appendSystem(text) {
    appendMessage({ role: 'system', author: '', time: '', body: text });
  }

  function appendHistoryEvent(event) {
    const t = fmtTime(event.timestamp);
    switch (event.type) {
      case 'described':
        appendMessage({ role: 'user', author: 'You', time: t, body: event.prose });
        if (event.gaps_found > 0) {
          appendSystem(`Parsed — ${event.gaps_found} clarification${event.gaps_found === 1 ? '' : 's'} needed.`);
        }
        break;
      case 'clarification_asked':
        // Pending questions are rendered separately; only show a note here.
        if (!event.questions || !event.questions.length) break;
        appendMessage({
          role: 'assistant',
          author: 'Kozha',
          time: t,
          body: `Asked ${event.questions.length} clarification${event.questions.length === 1 ? '' : 's'}.`,
        });
        break;
      case 'clarification_answered':
        appendMessage({
          role: 'user',
          author: 'You',
          time: t,
          body: `${event.question_field}: ${event.resolved_value}`,
        });
        break;
      case 'generated':
        if (event.success) {
          const conf = typeof event.confidence === 'number' ? ` (confidence ${event.confidence.toFixed(2)})` : '';
          appendMessage({ role: 'generated', author: 'Generator', time: t, body: `Generated HamNoSys${conf}.` });
        } else {
          appendMessage({
            role: 'error',
            author: 'Generator',
            time: t,
            body: `Generation failed: ${(event.errors || []).join('; ') || 'unknown error'}`,
          });
        }
        break;
      case 'correction_requested':
        appendMessage({
          role: 'user',
          author: 'You',
          time: t,
          body: event.raw_text + (event.target_region ? ` — region: ${event.target_region}` : '')
                              + (event.target_time_ms != null ? ` — ${event.target_time_ms} ms` : ''),
        });
        break;
      case 'correction_applied':
        appendMessage({
          role: 'assistant',
          author: 'Kozha',
          time: t,
          body: event.summary || 'Correction applied.',
        });
        break;
      case 'accepted':
        appendSystem('Sign accepted as draft.');
        break;
      case 'rejected':
        appendSystem(`Session rejected${event.reason ? `: ${event.reason}` : ''}.`);
        break;
      case 'abandoned':
        appendSystem(`Session abandoned: ${event.reason || 'inactivity'}.`);
        break;
    }
  }

  function appendQuestion(q) {
    const actions = [];
    for (const opt of (q.options || [])) {
      const btn = dom.tplOption.content.firstElementChild.cloneNode(true);
      btn.textContent = opt.label;
      btn.setAttribute('aria-label', `Answer ${q.field}: ${opt.label}`);
      btn.addEventListener('click', () => onAnswer(q.field, opt.value));
      actions.push(btn);
    }
    if (q.allow_freeform) {
      const note = document.createElement('span');
      note.className = 'msg-opt freeform';
      note.textContent = 'or type your own answer below';
      note.setAttribute('aria-hidden', 'true');
      actions.push(note);
    }
    appendMessage({
      role: 'assistant',
      author: 'Kozha',
      time: '',
      body: q.text,
      actions,
    });
  }

  // -----------------------------------------------------------------------
  // Inspector
  // -----------------------------------------------------------------------
  function renderInspector() {
    const env = state.envelope;
    if (!env || !env.parameters) {
      dom.inspectorEmpty.hidden = false;
      dom.inspectorList.hidden = true;
      dom.inspectorList.innerHTML = '';
      return;
    }
    dom.inspectorEmpty.hidden = true;
    dom.inspectorList.hidden = false;
    dom.inspectorList.innerHTML = '';

    const gaps = gapPaths();
    const clarified = clarifiedPaths();

    for (const [field, label] of SLOTS) {
      const value = getSlotValue(env.parameters, field);
      const isMovement = field === 'movement';
      const filled = isMovement ? (Array.isArray(value) && value.length > 0) : (value != null && value !== '');
      const isGap = gaps.has(field);
      const isClarified = clarified.has(field);
      const isPending = pendingFor(field) != null;

      let badgeClass, badgeText;
      if (isGap || isPending) { badgeClass = 'gap'; badgeText = 'gap'; }
      else if (filled && isClarified) { badgeClass = 'confirmed'; badgeText = 'confirmed'; }
      else if (filled) { badgeClass = 'inferred'; badgeText = 'inferred'; }
      else { badgeClass = 'gap'; badgeText = 'empty'; }

      const row = dom.tplInspectorRow.content.firstElementChild.cloneNode(true);
      const btn = row.querySelector('.ip-row-btn');
      btn.querySelector('.ip-name').textContent = label;
      const valueEl = btn.querySelector('.ip-value');
      const displayValue = formatSlotValue(value, isMovement);
      valueEl.textContent = displayValue || '—';
      if (!displayValue) valueEl.classList.add('empty');
      const badge = btn.querySelector('.ip-badge');
      badge.classList.add(badgeClass);
      badge.textContent = badgeText;
      btn.setAttribute('aria-label',
        `${label}: ${displayValue || 'empty'} (${badgeText}). Click to clarify or correct.`);
      btn.addEventListener('click', () => onInspectorRowClick(field));

      if (isGap) {
        const reasonEl = row.querySelector('.ip-reason');
        const gap = (env.gaps || []).find((g) => g.field === field);
        if (gap) { reasonEl.textContent = gap.reason; reasonEl.hidden = false; }
      }

      dom.inspectorList.appendChild(row);
    }
  }

  function formatSlotValue(value, isMovement) {
    if (isMovement) {
      if (!Array.isArray(value) || !value.length) return '';
      return value.map((seg) => {
        const parts = [seg.path, seg.size_mod, seg.speed_mod, seg.repeat].filter(Boolean);
        return parts.join(' · ');
      }).join(' → ');
    }
    return value || '';
  }

  function onInspectorRowClick(field) {
    const pending = pendingFor(field);
    if (pending) {
      // Scroll the chat to the question for this field; focus the input.
      dom.chatInput.focus();
      announce(`Answer the ${field} question.`);
      switchMobileTab('chat');
      return;
    }
    // Otherwise prime a correction targeting this field.
    state.selectedRegion = { regionId: field, fieldPath: field };
    renderCorrectionContext();
    dom.chatInput.focus();
    switchMobileTab('chat');
    announce(`Targeting ${field} for correction.`);
  }

  // -----------------------------------------------------------------------
  // Preview
  // -----------------------------------------------------------------------
  function renderPreview() {
    const env = state.envelope;
    if (!env) return;
    const preview = env.preview || {};

    // Captions
    dom.captionGloss.textContent = env.gloss ? `Gloss: ${env.gloss}` : '';
    dom.captionProse.textContent = env.description_prose || '';

    if (preview.video_url) {
      dom.previewPh.hidden = true;
      dom.cwasaMount.hidden = true;
      dom.previewVideo.hidden = false;
      if (dom.previewVideo.dataset.url !== preview.video_url) {
        dom.previewVideo.src = preview.video_url;
        dom.previewVideo.dataset.url = preview.video_url;
        dom.previewVideo.load();
        dom.previewVideo.addEventListener('loadedmetadata', onVideoLoaded, { once: true });
      }
      dom.regionOverlay.hidden = false;
      enablePlaybackControls(true);
      autoPlayOnce();
    } else if (preview.sigml) {
      dom.previewVideo.hidden = true;
      dom.previewPh.hidden = true;
      dom.cwasaMount.hidden = false;
      dom.regionOverlay.hidden = false;
      enablePlaybackControls(true, /* timelineDisabled */ true);
      maybeInitCWASA().then(() => {
        if (window.CWASA && CWASA.playSiGMLText) {
          try { CWASA.playSiGMLText(preview.sigml, 0); } catch (_e) {}
        }
      });
      dom.timelineLabel.textContent = preview.message || 'no timeline (server-side render unavailable)';
    } else {
      dom.previewVideo.hidden = true;
      dom.cwasaMount.hidden = true;
      dom.previewPh.hidden = false;
      dom.regionOverlay.hidden = true;
      enablePlaybackControls(false);
    }
  }

  function enablePlaybackControls(enabled, timelineDisabled) {
    dom.playBtn.disabled = !enabled;
    dom.loopBtn.disabled = !enabled;
    dom.flagBtn.disabled = !enabled;
    dom.timeline.disabled = !enabled || !!timelineDisabled;
  }

  function onVideoLoaded() {
    const ms = Math.round(dom.previewVideo.duration * 1000);
    dom.timeline.max = ms || 0;
    dom.timeline.value = 0;
    dom.timelineLabel.textContent = `0 / ${ms} ms`;
    dom.timeline.setAttribute('aria-valuetext', `0 of ${ms} milliseconds`);
  }

  let _autoPlayed = '';
  function autoPlayOnce() {
    const url = dom.previewVideo.src;
    if (!url || _autoPlayed === url) return;
    _autoPlayed = url;
    dom.previewVideo.loop = state.isLooping;
    const p = dom.previewVideo.play();
    if (p && typeof p.then === 'function') p.catch(() => { /* user gesture required */ });
  }

  async function maybeInitCWASA() {
    if (state.cwasaInited) return;
    if (!window.CWASA || !CWASA.init) return;
    try {
      CWASA.init({
        useClientConfig: false,
        useCwaConfig: true,
        avSettings: [{
          width: 360, height: 300,
          avList: 'avs', initAv: 'luna',
          ambIdle: false, allowFrameSteps: false, allowSiGMLText: false,
        }],
      });
      state.cwasaInited = true;
    } catch (e) {
      console.warn('CWASA init failed', e);
    }
  }

  // Playback control handlers
  function onPlayClick() {
    if (dom.previewVideo.hidden) {
      // CWASA replay
      const sigml = state.envelope && state.envelope.preview && state.envelope.preview.sigml;
      if (sigml && window.CWASA) { try { CWASA.playSiGMLText(sigml, 0); } catch (_e) {} }
      return;
    }
    if (dom.previewVideo.paused) {
      dom.previewVideo.play();
      dom.playBtn.textContent = '⏸ Pause';
    } else {
      dom.previewVideo.pause();
      dom.playBtn.textContent = '▶ Play';
    }
  }

  function onLoopClick() {
    state.isLooping = !state.isLooping;
    dom.previewVideo.loop = state.isLooping;
    dom.loopBtn.setAttribute('aria-pressed', String(state.isLooping));
    dom.loopBtn.textContent = state.isLooping ? '↻ Looping' : '↻ Loop';
  }

  function onTimelineInput() {
    if (dom.previewVideo.hidden) return;
    const ms = parseInt(dom.timeline.value, 10) || 0;
    dom.previewVideo.currentTime = ms / 1000;
    dom.timelineLabel.textContent = `${ms} / ${dom.timeline.max} ms`;
    dom.timeline.setAttribute('aria-valuetext', `${ms} milliseconds`);
  }

  function onVideoTimeUpdate() {
    const ms = Math.round(dom.previewVideo.currentTime * 1000);
    dom.timeline.value = ms;
    dom.timelineLabel.textContent = `${ms} / ${dom.timeline.max} ms`;
  }

  function onFlagMomentClick() {
    let ms = null;
    if (!dom.previewVideo.hidden) ms = Math.round(dom.previewVideo.currentTime * 1000);
    state.selectedTimeMs = ms;
    renderCorrectionContext();
    dom.chatInput.focus();
    switchMobileTab('chat');
    announce(ms != null ? `Flagged moment at ${ms} milliseconds.` : 'Flagged moment.');
  }

  function onRegionClick(ev) {
    const target = ev.target.closest('button.region');
    if (!target) return;
    const id = target.dataset.region;
    state.selectedRegion = { regionId: id, fieldPath: REGION_TO_FIELD[id] || null };
    renderCorrectionContext();
    dom.chatInput.focus();
    switchMobileTab('chat');
    announce(`Selected region: ${target.getAttribute('aria-label')}.`);
  }

  function renderCorrectionContext() {
    const env = state.envelope;
    if (!env || (env.state !== 'rendered' && env.state !== 'awaiting_correction')
        || (state.selectedRegion == null && state.selectedTimeMs == null)) {
      dom.correctionCtx.hidden = true;
      dom.correctionCtx.textContent = '';
      return;
    }
    const parts = [];
    if (state.selectedRegion) parts.push(`region: ${state.selectedRegion.fieldPath || state.selectedRegion.regionId}`);
    if (state.selectedTimeMs != null) parts.push(`@ ${state.selectedTimeMs} ms`);
    dom.correctionCtx.hidden = false;
    dom.correctionCtx.innerHTML = '';
    const span = document.createElement('span');
    span.textContent = `Correction target — ${parts.join(', ')}`;
    const clear = document.createElement('button');
    clear.type = 'button';
    clear.className = 'clear';
    clear.textContent = 'clear';
    clear.setAttribute('aria-label', 'Clear correction target');
    clear.addEventListener('click', () => {
      state.selectedRegion = null;
      state.selectedTimeMs = null;
      renderCorrectionContext();
    });
    dom.correctionCtx.append(span, clear);
  }

  // -----------------------------------------------------------------------
  // Submit handlers
  // -----------------------------------------------------------------------
  async function onChatSubmit(ev) {
    ev.preventDefault();
    const text = dom.chatInput.value.trim();
    if (!text) return;
    const env = state.envelope;
    if (!env) return;

    if (env.state === 'awaiting_description') {
      const gloss = (dom.glossInput.value || '').trim().toUpperCase();
      if (!gloss) {
        dom.glossInput.focus();
        setStatus('Sign name (gloss) is required before describing.');
        return;
      }
      await sendDescribe(text, gloss);
    } else if (env.state === 'clarifying') {
      // Treat as freeform answer to the first pending question.
      const q = (env.pending_questions || [])[0];
      if (q) await onAnswer(q.field, text);
      else await sendDescribe(text); // fallback: re-describe
    } else if (env.state === 'rendered' || env.state === 'awaiting_correction') {
      await sendCorrection(text);
    }
  }

  async function sendDescribe(prose, gloss) {
    try {
      setStatus('Parsing your description…');
      dom.chatInput.value = '';
      const body = { prose };
      if (gloss) body.gloss = gloss;
      const env = await api('POST', `/sessions/${state.sessionId}/describe`, body);
      applyEnvelope(env);
      clearStatus();
    } catch (err) {
      setStatus(`Describe failed: ${err.message}`);
    }
  }

  async function onAnswer(questionId, answer) {
    try {
      setStatus(`Applying answer to ${questionId}…`);
      dom.chatInput.value = '';
      const env = await api('POST', `/sessions/${state.sessionId}/answer`,
        { question_id: questionId, answer });
      applyEnvelope(env);
      clearStatus();
    } catch (err) {
      setStatus(`Answer failed: ${err.message}`);
    }
  }

  async function sendCorrection(text) {
    try {
      setStatus('Submitting correction…');
      dom.chatInput.value = '';
      const body = { raw_text: text };
      if (state.selectedTimeMs != null) body.target_time_ms = state.selectedTimeMs;
      if (state.selectedRegion && state.selectedRegion.fieldPath) {
        body.target_region = state.selectedRegion.fieldPath;
      }
      const env = await api('POST', `/sessions/${state.sessionId}/correct`, body);
      // Clear the targeting context once submitted.
      state.selectedRegion = null;
      state.selectedTimeMs = null;
      applyEnvelope(env);
      clearStatus();
    } catch (err) {
      setStatus(`Correction failed: ${err.message}`);
    }
  }

  async function onAccept() {
    try {
      setStatus('Finalizing sign…');
      const result = await api('POST', `/sessions/${state.sessionId}/accept`);
      applyEnvelope(result.session);
      announce(`Sign saved as draft. Status: ${result.sign_entry.status}. Awaiting Deaf review.`);
      clearStatus();
    } catch (err) {
      setStatus(`Accept failed: ${err.message}`);
    }
  }

  async function onReject() {
    if (!confirm('Reject this session and discard the draft?')) return;
    try {
      setStatus('Rejecting session…');
      const env = await api('POST', `/sessions/${state.sessionId}/reject`, { reason: 'user rejected' });
      applyEnvelope(env);
      clearStatus();
    } catch (err) {
      setStatus(`Reject failed: ${err.message}`);
    }
  }

  // -----------------------------------------------------------------------
  // SSE
  // -----------------------------------------------------------------------
  function subscribeEvents() {
    if (!state.sessionId) return;
    if (state.eventSource) state.eventSource.close();
    // EventSource cannot send custom headers; the backend accepts the token
    // as the X-Session-Token header. Standard browsers support neither —
    // we pass it as a query string. The router additionally supports the
    // header via the load helper; if SSE auth fails, the stream simply
    // closes and the REST flow remains usable.
    const url = `${API_BASE}/sessions/${state.sessionId}/events`;
    let es;
    try { es = new EventSource(url, { withCredentials: false }); }
    catch (_e) { return; }
    state.eventSource = es;

    const types = Object.keys(PROGRESS_TEXT);
    for (const t of types) {
      es.addEventListener(t, (ev) => onSSEEvent(t, ev));
    }
    es.addEventListener('closed',  () => es.close());
    es.addEventListener('timeout', () => es.close());
    es.addEventListener('error',   () => { /* let browser auto-retry */ });
  }

  function onSSEEvent(type, _ev) {
    const text = PROGRESS_TEXT[type];
    if (text) setStatus(text);
    if (type === 'generated' || type === 'correction_applied' || type === 'accepted') {
      // Refresh state from authoritative GET to pick up the latest envelope.
      refreshSession();
    }
    if (type === 'accepted' || type === 'rejected') {
      setTimeout(clearStatus, 1500);
    }
  }

  async function refreshSession() {
    if (!state.sessionId) return;
    try {
      const env = await api('GET', `/sessions/${state.sessionId}`);
      applyEnvelope(env);
    } catch (_e) { /* non-fatal */ }
  }

  // -----------------------------------------------------------------------
  // Misc UI handlers
  // -----------------------------------------------------------------------
  function setupTopbar() {
    $('hcToggle').addEventListener('click', (e) => {
      const on = !document.body.classList.contains('hc');
      document.body.classList.toggle('hc', on);
      e.currentTarget.setAttribute('aria-pressed', String(on));
      try { localStorage.setItem('c2h.hc', on ? '1' : '0'); } catch (_e) {}
    });
    $('fontDown').addEventListener('click', () => bumpFont(-1));
    $('fontUp').addEventListener('click', () => bumpFont(+1));

    try {
      if (localStorage.getItem('c2h.hc') === '1') {
        document.body.classList.add('hc');
        $('hcToggle').setAttribute('aria-pressed', 'true');
      }
      const fs = localStorage.getItem('c2h.fs');
      if (fs) document.documentElement.classList.add(fs);
    } catch (_e) {}
  }

  function bumpFont(delta) {
    const root = document.documentElement;
    const order = ['', 'fs-large', 'fs-xlarge'];
    const current = order.find((c) => c === '' ? !root.classList.contains('fs-large') && !root.classList.contains('fs-xlarge') : root.classList.contains(c));
    let idx = Math.max(0, order.indexOf(current));
    idx = Math.min(order.length - 1, Math.max(0, idx + delta));
    root.classList.remove('fs-large', 'fs-xlarge');
    if (order[idx]) root.classList.add(order[idx]);
    try { localStorage.setItem('c2h.fs', order[idx] || ''); } catch (_e) {}
  }

  function setupTabs() {
    document.querySelectorAll('.mobile-tabs .tab').forEach((tab) => {
      tab.addEventListener('click', () => switchMobileTab(tab.dataset.tab));
    });
    switchMobileTab('chat');
  }

  function switchMobileTab(name) {
    state.activeMobileTab = name;
    document.querySelectorAll('.mobile-tabs .tab').forEach((t) => {
      const on = t.dataset.tab === name;
      t.classList.toggle('is-active', on);
      t.setAttribute('aria-selected', String(on));
    });
    document.querySelectorAll('.panel').forEach((p) => {
      p.classList.toggle('is-active', p.dataset.panel === name);
    });
  }

  function setupInspectorToggle() {
    dom.inspectorToggle.addEventListener('click', () => {
      const expanded = dom.inspectorToggle.getAttribute('aria-expanded') === 'true';
      const next = !expanded;
      dom.inspectorToggle.setAttribute('aria-expanded', String(next));
      dom.inspectorToggle.textContent = next ? 'Collapse' : 'Expand';
      dom.inspectorBody.hidden = !next;
    });
  }

  // -----------------------------------------------------------------------
  // Wire up
  // -----------------------------------------------------------------------
  function wire() {
    dom.chatForm.addEventListener('submit', onChatSubmit);
    dom.acceptBtn.addEventListener('click', onAccept);
    dom.rejectBtn.addEventListener('click', onReject);
    dom.playBtn.addEventListener('click', onPlayClick);
    dom.loopBtn.addEventListener('click', onLoopClick);
    dom.timeline.addEventListener('input', onTimelineInput);
    dom.previewVideo.addEventListener('timeupdate', onVideoTimeUpdate);
    dom.flagBtn.addEventListener('click', onFlagMomentClick);
    dom.regionOverlay.addEventListener('click', onRegionClick);
    setupTopbar();
    setupTabs();
    setupInspectorToggle();

    // Submit chat with Cmd/Ctrl+Enter as a power-user shortcut.
    dom.chatInput.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
        e.preventDefault();
        dom.chatForm.requestSubmit();
      }
    });
  }

  // -----------------------------------------------------------------------
  // Go
  // -----------------------------------------------------------------------
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => { wire(); boot(); });
  } else {
    wire();
    boot();
  }

  // Expose minimal hooks for the Playwright smoke test.
  window.__c2h = {
    state,
    api,
    applyEnvelope,
    refreshSession,
  };
})();

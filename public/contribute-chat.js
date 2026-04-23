/* Contribute page — clarification chat panel.
 *
 * Spec: prompts/prompt-contrib-6.md.
 * Design contract: docs/contribute-redesign/01-design-principles.md.
 *
 * Behaviour summary:
 * - Mounts inside #authoring-root once a session exists; hides again
 *   when the session is finalized or abandoned.
 * - Two message kinds — 'system' (left, "Clarification:" label) and
 *   'you' (right, no label). Plain text only, no markdown / emoji.
 * - During CLARIFYING, shows the first pending question. When that
 *   question carries multiple-choice options, renders option buttons
 *   above the textarea; clicking sends the option's `value`. Free-form
 *   typed answers also POST to /answer.
 * - Detects contributor questions (input ends with "?") and replies
 *   with a fixed message rather than forwarding to the LLM. Keeps
 *   that reply in a constant — no scope creep.
 * - Three fixed transition messages, each shown once per session:
 *     CLARIFYING → GENERATING : "Enough information to draft the sign…"
 *     ANY → RENDERED          : "Draft is ready…"
 * - Once RENDERED, the input relabels to corrections; sends call
 *   /correct.
 * - On API failure, appends the fixed "Something went wrong…" message
 *   and reveals a "Submit as-is" button that dispatches a custom
 *   `kozha:submit-as-is` event for the submission flow (later prompt)
 *   to handle.
 */
(function () {
  'use strict';

  var CTX = window.KOZHA_CONTRIB_CONTEXT;
  if (!CTX) {
    if (window.console) console.error('[contribute-chat] contribute-context.js must load first');
    return;
  }

  // Optional debug-log shim: contribute-debug.js loads first and exposes
  // .log() when ?debug=1 is on; we fall back to a no-op so this module
  // works in any load order. Contributors only see the live drawer when
  // they explicitly opt in.
  var DEBUG = (window.KOZHA_CONTRIB_DEBUG && window.KOZHA_CONTRIB_DEBUG.log)
    ? window.KOZHA_CONTRIB_DEBUG
    : { log: function () {}, forceLog: function () {} };

  // ---------- exact copy (no variation, per spec) ----------
  //
  // Source of truth: strings.en.json under contribute.chat.*. The fallbacks
  // below are the English originals; we read from i18n each time a string
  // is used so the catalog can override once loaded (and so we stay sane
  // if the catalog fetch fails — nothing on this panel goes blank).

  function tr(key, fallback) {
    if (window.KOZHA_I18N && typeof window.KOZHA_I18N.t === 'function') {
      var v = window.KOZHA_I18N.t(key);
      if (v && v !== key) return v;
    }
    return fallback;
  }

  var COPY = {
    get CLARIFICATION_LABEL()          { return tr('contribute.chat.clarification_label', 'Sign wizard'); },
    get ERROR_LABEL()                  { return tr('contribute.chat.error_label', 'Error:'); },
    get NOTICE_LABEL()                 { return tr('contribute.chat.notice_label', 'Notice:'); },
    get INPUT_LABEL_ANSWER()           { return tr('contribute.chat.input_label_answer', 'Your answer'); },
    get INPUT_LABEL_CORRECTION()       { return tr('contribute.chat.input_label_correction', 'Describe what should change'); },
    get INPUT_PLACEHOLDER_ANSWER()     { return tr('contribute.chat.input_placeholder_answer', 'Type your answer…'); },
    get INPUT_PLACEHOLDER_CORRECTION() { return tr('contribute.chat.input_placeholder_correction', 'Type a correction…'); },
    get INPUT_PLACEHOLDER_GENERATING() { return tr('contribute.chat.input_placeholder_generating', 'Generating sign…'); },
    get CORRECTION_HINT()              { return tr('contribute.chat.correction_hint', 'Just describe what should change. Clicking a body region is optional.'); },
    get REPLY_USER_QUESTION()          { return tr('contribute.chat.reply_user_question', "I can only ask about this sign. If you'd like to learn HamNoSys, see the docs link below."); },
    get GENERATING_MSG()               { return tr('contribute.chat.generating_msg', 'Enough information to draft the sign. Preparing preview.'); },
    get READY_MSG()                    { return tr('contribute.chat.ready_msg', 'Draft is ready. Review the preview below and either submit or describe a correction.'); },
    get ERROR_MSG()                    { return tr('contribute.chat.error_msg', 'The AI had trouble generating a follow-up. Your answer was saved — try rephrasing, or submit the draft as-is and let the reviewer fill any gaps.'); },
    get GENERATION_FAILED_DEBUG()      { return tr('contribute.chat.generation_failed_debug', 'Generator detail: '); },
    get RETRY_BUTTON()                 { return tr('contribute.chat.retry', 'Try again'); },
    get CORRECTION_WORKING()           { return tr('contribute.chat.correction_working', 'Working on your correction — the avatar will update once the new sign is ready.'); },
    get GENERATION_CANDIDATE_LABEL()   { return tr('contribute.chat.generation_candidate_label', 'Tried candidate: '); },
    get GENERATION_PATH_LABEL()        { return tr('contribute.chat.generation_path_label', 'Generation path: '); },
    get ERROR_RATE_LIMITED()           { return tr('contribute.chat.error_rate_limited', "You're sending requests faster than the server can process. Wait a moment and try again."); },
    get ERROR_INJECTION_REJECTED()     { return tr('contribute.chat.error_injection_rejected', "We didn't interpret this as a sign description. Please describe only the sign itself."); },
    get ERROR_BUDGET()                 { return tr('contribute.chat.error_budget', "Today's shared AI quota is used up. Your draft is saved — you can submit it as-is, or come back tomorrow to continue."); },
    get ERROR_MODEL_UNAVAILABLE()      { return tr('contribute.chat.error_model_unavailable', "The AI model is temporarily unavailable. Your draft is saved — please retry in a minute."); },
    get PROGRESS_THINKING()            { return tr('contribute.chat.progress_thinking', 'Thinking…'); },
    get PROGRESS_CONTACTING()          { return tr('contribute.chat.progress_contacting', 'Contacting the AI…'); },
  };

  var TERMINAL_STATES = { finalized: true, abandoned: true };
  var INPUT_MIN_HEIGHT_PX = 44;
  var INPUT_MAX_HEIGHT_PX = 132;

  // ---------- DOM ----------

  var els = {
    panel:         document.getElementById('chatPanel'),
    log:           document.getElementById('chatLog'),
    options:       document.getElementById('chatOptions'),
    hint:          document.getElementById('chatInputHint'),
    progress:      document.getElementById('chatProgress'),
    inputLabel:    document.getElementById('chatInputLabel'),
    input:         document.getElementById('chatInput'),
    sendBtn:       document.getElementById('chatSendBtn'),
    errorActions:  document.getElementById('chatErrorActions'),
    retryBtn:      document.getElementById('chatRetryBtn'),
    submitAsIs:    document.getElementById('chatSubmitAsIsBtn'),
    discardBtn:    document.getElementById('chatDiscardBtn'),
    targetPill:    document.getElementById('chatTargetPill'),
    targetText:    document.getElementById('chatTargetPillText'),
    targetClear:   document.getElementById('chatTargetPillClear'),
  };

  if (!els.panel || !els.log || !els.input || !els.sendBtn) {
    // No chat HTML on this page — bail without errors.
    return;
  }

  // ---------- internal state ----------

  var chat = {
    // The pending question being shown; null when no question is active.
    currentQuestion: null,
    // Session id we last rendered for; resets the log on a session swap.
    rememberedSessionId: null,
    // Last server state we observed; drives transition-message detection.
    lastObservedState: null,
    // Once-per-session guards for the two transition messages.
    shownGeneratingMsg: false,
    shownReadyMsg: false,
    // True while a fetch is outstanding.
    inFlight: false,
    // Timer id for the "Thinking…" → "Contacting the AI…" upgrade;
    // null when no upgrade is scheduled.
    progressUpgradeTimer: null,
    // True after we surface the error message and the Submit-as-is path.
    inError: false,
    // Signature of the last generation_errors list we surfaced in chat —
    // prevents the "generation failed" message from duplicating on
    // repeated envelopes that still carry the same errors list.
    lastSurfacedGenerationError: null,
    // Correction target set by the preview pane when the contributor
    // clicks a body region. Carries { region, label, timeMs, timeText }.
    // Cleared once a correction is submitted or the user dismisses the
    // pill. Null means "no target — send correction un-targeted."
    correctionTarget: null,
    // Last successfully-dispatched answer or correction; the "Try
    // again" button replays this when the previous attempt errored.
    lastAction: null,
    // True after maybeSurfaceGenerationError fired its silent
    // /generate auto-retry. Reset on resetChat() (session swap or
    // discard) so a fresh authoring round still gets one auto-retry.
    autoRetriedGen: false,
  };

  // ---------- log + options rendering ----------

  function fmtTime(d) {
    var iso = (d || new Date()).toISOString();
    // "2026-04-20T13:45:23.123Z" → "2026-04-20 13:45:23 UTC"
    return iso.replace('T', ' ').slice(0, 19) + ' UTC';
  }

  function appendMessage(opts) {
    var msg = document.createElement('div');
    msg.className = 'chat-msg chat-msg-' + opts.kind;
    msg.setAttribute('title', fmtTime(opts.ts));

    var labelText = null;
    if (opts.kind === 'error') {
      labelText = COPY.ERROR_LABEL;
    } else if (opts.kind === 'notice') {
      labelText = COPY.NOTICE_LABEL;
    } else if (opts.kind === 'system') {
      labelText = COPY.CLARIFICATION_LABEL;
    }
    if (labelText) {
      var label = document.createElement('span');
      label.className = 'chat-msg-label';
      label.textContent = labelText;
      msg.appendChild(label);
    }

    var text = document.createElement('p');
    text.className = 'chat-msg-text';
    // textContent (not innerHTML) — plain text only, line breaks via
    // CSS white-space: pre-wrap. No markdown, no emoji, no rich.
    text.textContent = opts.text || '';
    msg.appendChild(text);

    els.log.appendChild(msg);
    els.log.scrollTop = els.log.scrollHeight;
  }

  function clearLog() {
    while (els.log.firstChild) els.log.removeChild(els.log.firstChild);
  }

  // Walk back from the end of the chat log and return the text of the
  // most recent ``chat-msg-system`` entry. Used to dedupe a pending
  // clarification question whose text was already replayed via the
  // envelope's clarifications history.
  function lastSystemMessageText() {
    if (!els.log) return '';
    for (var n = els.log.lastChild; n; n = n.previousSibling) {
      if (n.nodeType !== 1) continue;
      if (n.classList && n.classList.contains('chat-msg-system')) {
        var t = n.querySelector ? n.querySelector('.chat-msg-text') : null;
        return ((t && t.textContent) || n.textContent || '').trim();
      }
    }
    return '';
  }

  function renderOptions(question) {
    while (els.options.firstChild) els.options.removeChild(els.options.firstChild);
    if (!question || !question.options || !question.options.length) {
      els.options.hidden = true;
      return;
    }
    question.options.forEach(function (opt) {
      var btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'chat-option-btn';
      btn.textContent = opt.label;
      btn.setAttribute('data-value', opt.value);
      btn.setAttribute('aria-label', (window.KOZHA_I18N && window.KOZHA_I18N.t)
        ? window.KOZHA_I18N.t('contribute.chat.option_aria_prefix', { label: opt.label })
        : 'Answer: ' + opt.label);
      btn.addEventListener('click', function () { onOptionClick(opt); });
      els.options.appendChild(btn);
    });
    els.options.hidden = false;
  }

  // ---------- input mode + progress ----------

  function inCorrectionMode(s) {
    return s === 'rendered' ||
           s === 'awaiting_correction' ||
           s === 'applying_correction';
  }

  function isLoadingState(s) {
    return s === 'generating' || s === 'applying_correction';
  }

  function updateInputMode(snapshot) {
    var s = snapshot.sessionState;
    var loading = chat.inFlight || isLoadingState(s);
    var noActiveTurn =
      s === 'awaiting_description' && !chat.currentQuestion;

    // Hint: only meaningful in correction mode.
    if (inCorrectionMode(s) && !loading) {
      els.hint.textContent = COPY.CORRECTION_HINT;
      els.hint.hidden = false;
    } else {
      els.hint.hidden = true;
    }

    if (loading) {
      els.input.setAttribute('placeholder', COPY.INPUT_PLACEHOLDER_GENERATING);
      els.input.disabled = true;
      els.sendBtn.disabled = true;
      els.options.hidden = true;
      return;
    }

    if (inCorrectionMode(s)) {
      els.inputLabel.textContent = COPY.INPUT_LABEL_CORRECTION;
      els.input.setAttribute('placeholder', COPY.INPUT_PLACEHOLDER_CORRECTION);
    } else {
      els.inputLabel.textContent = COPY.INPUT_LABEL_ANSWER;
      els.input.setAttribute('placeholder', COPY.INPUT_PLACEHOLDER_ANSWER);
    }

    // When an in-flight request completes, this function runs again
    // with loading=false. It must restore the options-row visibility
    // that the loading branch above turns off, otherwise a second
    // clarification's buttons render into a container that stays
    // display:none.
    var q = chat.currentQuestion;
    els.options.hidden = !(q && q.options && q.options.length);

    els.input.disabled = noActiveTurn;
    els.sendBtn.disabled = noActiveTurn || !(els.input.value || '').trim();
  }

  function setInFlight(yes) {
    chat.inFlight = !!yes;
    els.progress.hidden = !yes;
    if (chat.progressUpgradeTimer) {
      clearTimeout(chat.progressUpgradeTimer);
      chat.progressUpgradeTimer = null;
    }
    if (yes) {
      // The progress element is visual-only by default; announce and
      // label the stage so contributors can see the request is active.
      // Start as "Thinking…" and upgrade to "Contacting the AI…" after
      // a short beat so long-running calls read as deliberate rather
      // than stuck.
      els.progress.removeAttribute('aria-hidden');
      els.progress.setAttribute('role', 'status');
      els.progress.setAttribute('aria-live', 'polite');
      setProgressText(COPY.PROGRESS_THINKING);
      chat.progressUpgradeTimer = setTimeout(function () {
        chat.progressUpgradeTimer = null;
        if (chat.inFlight) setProgressText(COPY.PROGRESS_CONTACTING);
      }, 1200);
      els.input.disabled = true;
      els.sendBtn.disabled = true;
    } else {
      els.progress.setAttribute('aria-hidden', 'true');
      setProgressText('');
      updateInputMode(CTX.getState());
    }
  }

  function setProgressText(text) {
    // The visual bar is a separate span; clear it, then replace any
    // previous label so multiple calls overwrite rather than stack.
    var bar = els.progress.querySelector('.chat-progress-bar');
    els.progress.textContent = '';
    if (bar) els.progress.appendChild(bar);
    if (text) {
      var label = document.createElement('span');
      label.className = 'chat-progress-label';
      label.textContent = text;
      els.progress.appendChild(label);
    }
  }

  // ---------- send paths ----------

  function onOptionClick(opt) {
    if (chat.inFlight) return;
    if (!chat.currentQuestion) return;
    var q = chat.currentQuestion;
    appendMessage({ kind: 'you', text: opt.label });
    sendAnswer(q.field || q.question_id, opt.value);
  }

  function onSendClick() {
    if (chat.inFlight) return;
    var raw = (els.input.value || '').trim();
    if (!raw) return;
    var snap = CTX.getState();

    // Guard: contributor asking the system a question. Reply locally
    // with the fixed message — never forward to the LLM. The /? trailing
    // heuristic catches "what is X?", "how do I Y?" etc.
    if (looksLikeUserQuestion(raw)) {
      appendMessage({ kind: 'you', text: raw });
      els.input.value = '';
      autoGrowInput();
      appendMessage({ kind: 'system', text: COPY.REPLY_USER_QUESTION });
      updateInputMode(snap);
      return;
    }

    appendMessage({ kind: 'you', text: raw });
    els.input.value = '';
    autoGrowInput();

    if (inCorrectionMode(snap.sessionState)) {
      sendCorrection(raw);
      return;
    }
    if (chat.currentQuestion) {
      var q = chat.currentQuestion;
      sendAnswer(q.field || q.question_id, raw);
    }
  }

  function looksLikeUserQuestion(text) {
    return /\?\s*$/.test(text);
  }

  // Auto-retry policy for /answer and /correct: a single second attempt
  // after a 600ms backoff, but only for errors we believe are transient
  // (network failure, 5xx, generic LLM hiccup). Hard errors — rate
  // limited, injection rejected, budget exhausted, model unavailable —
  // surface immediately because retrying won't change anything and the
  // contributor needs to see what to do next. The reload-and-it-worked
  // pattern is the symptom of skipping this retry, so this is the most
  // impactful single change to smooth out the experience.
  function isTransientError(err) {
    if (!err) return true;
    var code = parseErrorCode(err);
    if (code === 'rate_limited' ||
        code === 'injection_rejected' ||
        code === 'budget_exceeded' ||
        code === 'daily_budget_exceeded' ||
        code === 'cost_cap_exceeded' ||
        code === 'llm_config_error' ||
        code === 'model_unavailable') {
      return false;
    }
    var status = err.status || 0;
    if (status === 429 || status === 401 || status === 403 || status === 422) {
      return false;
    }
    return true;
  }

  function withAutoRetry(thunk) {
    return thunk().catch(function (err) {
      if (!isTransientError(err)) {
        return Promise.reject(err);
      }
      DEBUG.log('chat: transient call failed, retrying once', { status: err && err.status });
      return new Promise(function (resolve) { setTimeout(resolve, 600); })
        .then(function () { return thunk(); });
    });
  }

  function sendAnswer(field, answerText) {
    setInFlight(true);
    clearError();
    chat.lastAction = { kind: 'answer', field: field, text: answerText };
    withAutoRetry(function () { return CTX.answer(field, answerText); })
      .catch(handleError)
      .then(function () { setInFlight(false); });
  }

  function sendCorrection(text) {
    setInFlight(true);
    clearError();
    var opts = {};
    if (chat.correctionTarget) {
      if (typeof chat.correctionTarget.timeMs === 'number') {
        opts.targetTimeMs = chat.correctionTarget.timeMs;
      }
      if (chat.correctionTarget.region) {
        opts.targetRegion = chat.correctionTarget.region;
      }
    }
    chat.lastAction = { kind: 'correction', text: text, opts: opts };
    // The target applies to this one correction only — clear the pill
    // so the next correction starts fresh unless the user re-targets.
    clearCorrectionTarget();
    // Visible "working on it" chat notice so the contributor knows
    // the correction was received and the system is regenerating —
    // not silently doing nothing. The notice carries a marker class
    // so the next snapshot can swap it for the real follow-up.
    appendMessage({ kind: 'system', text: COPY.CORRECTION_WORKING });
    withAutoRetry(function () { return CTX.correct(text, opts); })
      .catch(handleError)
      .then(function () { setInFlight(false); });
  }

  function retryLastAction() {
    var last = chat.lastAction;
    if (!last) return;
    if (last.kind === 'answer') {
      sendAnswer(last.field, last.text);
    } else if (last.kind === 'correction') {
      sendCorrection(last.text);
    }
  }

  function parseErrorCode(err) {
    if (!err || typeof err.body !== 'string') return null;
    try {
      var parsed = JSON.parse(err.body);
      if (parsed && parsed.error && typeof parsed.error.code === 'string') {
        return parsed.error.code;
      }
    } catch (_e) { /* non-JSON body — caller falls back to status */ }
    return null;
  }

  function pickErrorMessage(err) {
    // Soft-fail classes (rate limit, injection) don't need the
    // "Submit as-is" escape hatch — the contributor just needs to wait
    // or rephrase. Hard errors fall through to the generic message.
    var code = parseErrorCode(err);
    var status = err ? err.status : null;
    if (code === 'rate_limited') {
      return { text: COPY.ERROR_RATE_LIMITED, offerSubmitAsIs: false };
    }
    if (code === 'injection_rejected') {
      return { text: COPY.ERROR_INJECTION_REJECTED, offerSubmitAsIs: false };
    }
    if (code === 'budget_exceeded' ||
        code === 'daily_budget_exceeded' ||
        code === 'cost_cap_exceeded') {
      return { text: COPY.ERROR_BUDGET, offerSubmitAsIs: true };
    }
    if (code === 'llm_config_error' ||
        code === 'model_unavailable') {
      return { text: COPY.ERROR_MODEL_UNAVAILABLE, offerSubmitAsIs: true };
    }
    if (status === 429) {
      return { text: COPY.ERROR_RATE_LIMITED, offerSubmitAsIs: false };
    }
    return { text: COPY.ERROR_MSG, offerSubmitAsIs: true };
  }

  function handleError(err) {
    if (window.console) console.error('[contribute-chat] action failed:', err);
    var code = parseErrorCode(err);
    DEBUG.log('chat: action failed (after retry)', {
      status: err && err.status,
      code:   code,
      body:   err && (err.body || err.message),
    });

    // No-key situation: the panel-opening UX from contribute.js's
    // describe-failure path is just as relevant here. Pop the BYO-key
    // panel so the contributor can recover with one paste, instead of
    // staring at "AI model is temporarily unavailable" with no clue
    // what to do.
    if (code === 'llm_config_error' || code === 'llm_no_key') {
      var panel = document.querySelector('.byo-key');
      if (panel && typeof panel.open !== 'undefined') {
        panel.open = true;
        panel.classList.add('is-required');
        if (typeof panel.scrollIntoView === 'function') {
          panel.scrollIntoView({ block: 'center', behavior: 'smooth' });
        }
      }
      var keyInput = document.getElementById('byoKeyInput');
      if (keyInput && typeof keyInput.focus === 'function') {
        setTimeout(function () { try { keyInput.focus(); } catch (_e) {} }, 250);
      }
    }

    chat.inError = true;
    var picked = pickErrorMessage(err);
    appendMessage({ kind: 'error', text: picked.text });
    // The error actions strip is shown when *anything* useful sits in
    // it. Retry is offered whenever we have a lastAction to replay
    // (i.e. the failure came from a /answer or /correct call); the
    // submit-as-is escape hatch follows pickErrorMessage's verdict.
    var hasRetryTarget = !!chat.lastAction;
    if (els.retryBtn) els.retryBtn.hidden = !hasRetryTarget;
    if (els.submitAsIs) els.submitAsIs.hidden = !picked.offerSubmitAsIs;
    els.errorActions.hidden = !(hasRetryTarget || picked.offerSubmitAsIs);
  }

  function clearError() {
    chat.inError = false;
    els.errorActions.hidden = true;
  }

  function onSubmitAsIs() {
    var ev;
    try { ev = new CustomEvent('kozha:submit-as-is'); }
    catch (_e) {
      ev = document.createEvent('Event');
      ev.initEvent('kozha:submit-as-is', true, true);
    }
    document.dispatchEvent(ev);
  }

  function onDiscardClick() {
    // Dispatch a page-wide event so contribute.js owns the confirmation
    // modal and the actual clearSession call — keeps the chat panel
    // ignorant of the modal machinery and lets the same escape hatch
    // fire from other panels later if needed.
    var ev;
    try { ev = new CustomEvent('kozha:discard-session'); }
    catch (_e) {
      ev = document.createEvent('Event');
      ev.initEvent('kozha:discard-session', true, true);
    }
    document.dispatchEvent(ev);
  }

  // ---------- envelope / state syncing ----------

  function applyState(newState, pending) {
    var prevState = chat.lastObservedState;

    // Render the next pending question if its field differs from the
    // one we last rendered. Backend may queue multiple questions but
    // the chat surfaces them one at a time, in order.
    var nextQ = pending && pending.length ? pending[0] : null;
    var nextField = nextQ ? (nextQ.field || nextQ.question_id) : null;
    var prevField = chat.currentQuestion
      ? (chat.currentQuestion.field || chat.currentQuestion.question_id)
      : null;

    if (nextField && nextField !== prevField) {
      // Text-level dedupe: when an envelope replays the clarification
      // history *and* carries the same question still pending, the
      // history loop renders the question once and applyState would
      // render it a second time. Compare against the last system
      // message text and skip if identical.
      var lastSystemText = lastSystemMessageText();
      var nextText = (nextQ.text || '').trim();
      if (nextText && nextText === lastSystemText) {
        DEBUG.log('chat: dedupe — pending question already in log', { field: nextField });
      } else {
        appendMessage({ kind: 'system', text: nextQ.text });
      }
      chat.currentQuestion = nextQ;
      renderOptions(nextQ);
    } else if (!nextField && chat.currentQuestion) {
      chat.currentQuestion = null;
      renderOptions(null);
    }

    // Transition messages — exactly once per session.
    if (newState === 'generating' && !chat.shownGeneratingMsg) {
      appendMessage({ kind: 'system', text: COPY.GENERATING_MSG });
      chat.shownGeneratingMsg = true;
    }
    if (newState === 'rendered' && !chat.shownReadyMsg) {
      // Synchronous /describe + /answer paths can leap straight from
      // CLARIFYING (or AWAITING_DESCRIPTION) to RENDERED without ever
      // being observed in GENERATING. Surface the "preparing preview"
      // message anyway so the transition reads in order.
      var leapt =
        !chat.shownGeneratingMsg &&
        (prevState === 'clarifying' || prevState === 'awaiting_description');
      if (leapt) {
        appendMessage({ kind: 'system', text: COPY.GENERATING_MSG });
        chat.shownGeneratingMsg = true;
      }
      appendMessage({ kind: 'system', text: COPY.READY_MSG });
      chat.shownReadyMsg = true;
    }

    chat.lastObservedState = newState;
  }

  // ---------- subscriber ----------

  function onSnapshot(snapshot) {
    if (!snapshot.sessionId || TERMINAL_STATES[snapshot.sessionState]) {
      els.panel.hidden = true;
      return;
    }
    els.panel.hidden = false;

    // New session id (eg. user edited and resubmitted) → wipe local
    // chat state so the new turn starts from a clean log.
    if (chat.rememberedSessionId &&
        chat.rememberedSessionId !== snapshot.sessionId) {
      resetChat();
    }
    chat.rememberedSessionId = snapshot.sessionId;

    // Replay clarification history when the log is empty (typically a
    // page reload that resumed an existing session). The envelope's
    // clarifications carry both author and assistant turns.
    var clarifications = snapshot.clarifications || [];
    if (clarifications.length && !els.log.firstChild) {
      clarifications.forEach(function (turn) {
        appendMessage({
          kind: turn.role === 'author' ? 'you' : 'system',
          text: turn.text,
        });
      });
    }

    applyState(snapshot.sessionState, snapshot.pendingQuestions || []);
    maybeSurfaceGenerationError(snapshot);
    updateInputMode(snapshot);
  }

  function resetChat() {
    clearLog();
    chat.currentQuestion = null;
    chat.lastObservedState = null;
    chat.shownGeneratingMsg = false;
    chat.shownReadyMsg = false;
    chat.inError = false;
    chat.inFlight = false;
    chat.lastSurfacedGenerationError = null;
    chat.autoRetriedGen = false;
    els.errorActions.hidden = true;
    els.options.hidden = true;
    els.progress.hidden = true;
    els.input.value = '';
    clearCorrectionTarget();
    autoGrowInput();
  }

  // Surface a generator failure inline: when the backend bounced out of
  // the generating/applying_correction loading state *without* producing
  // a HamNoSys string, we want the contributor to see (a) something went
  // wrong, (b) what the generator actually complained about, and (c) the
  // Submit-as-is escape hatch. Runs idempotently — repeated envelopes
  // carrying the same errors list don't re-append.
  function maybeSurfaceGenerationError(snap) {
    var errs = snap.generationErrors || [];
    if (!errs.length) return;
    var s = snap.sessionState;
    if (s === 'generating' || s === 'applying_correction') return;
    if (snap.hamnosys) return;

    // Include the generation path and candidate in the signature so a
    // contributor who retries and gets a different failure sees the
    // fresh debug trail.
    var path = (snap.generationPath || []).join(',');
    var cand = snap.candidateHamnosys || '';
    var sig = errs.join('|') + '::' + path + '::' + cand;
    if (sig === chat.lastSurfacedGenerationError) return;
    chat.lastSurfacedGenerationError = sig;

    // Auto-recover: the most common cause of a clean error envelope on
    // /answer's response is a transient LLM stumble (rate-limit blip,
    // BadRequestError on a freshly-warmed model, etc.). The exact
    // pattern the user sees today is "first attempt fails, refresh
    // produces the sign" — refresh works because resume + a saved
    // GENERATING state replays the generator with the answers
    // intact. Calling /generate here does the same thing without the
    // user noticing: keep a "Generating…" notice in the chat, fire
    // the request, and only surface the hard error if the retry also
    // fails. The auto-retry runs once per snapshot signature so a
    // truly broken pipeline doesn't loop forever.
    if (CTX && typeof CTX.forceGenerate === 'function' && !chat.autoRetriedGen) {
      chat.autoRetriedGen = true;
      DEBUG.log('chat: auto-retrying generation after error envelope', {
        firstErr: errs[0],
        path: path,
      });
      appendMessage({ kind: 'system', text: COPY.GENERATING_MSG });
      setInFlight(true);
      CTX.forceGenerate()
        .catch(function (retryErr) {
          DEBUG.error('chat: auto-retry /generate also failed', {
            status: retryErr && retryErr.status,
          });
          chat.inError = true;
          appendMessage({ kind: 'error', text: COPY.ERROR_MSG });
          appendMessage({ kind: 'notice', text: COPY.GENERATION_FAILED_DEBUG + errs[0] });
          if (path) appendMessage({ kind: 'notice', text: COPY.GENERATION_PATH_LABEL + path });
          if (cand) appendMessage({ kind: 'notice', text: COPY.GENERATION_CANDIDATE_LABEL + cand });
          els.errorActions.hidden = false;
        })
        .then(function () { setInFlight(false); });
      return;
    }

    chat.inError = true;
    appendMessage({ kind: 'error', text: COPY.ERROR_MSG });
    appendMessage({ kind: 'notice', text: COPY.GENERATION_FAILED_DEBUG + errs[0] });
    if (path) {
      appendMessage({ kind: 'notice', text: COPY.GENERATION_PATH_LABEL + path });
    }
    if (cand) {
      appendMessage({ kind: 'notice', text: COPY.GENERATION_CANDIDATE_LABEL + cand });
    }
    els.errorActions.hidden = false;
  }

  // ---------- input behaviour ----------

  function autoGrowInput() {
    els.input.style.height = INPUT_MIN_HEIGHT_PX + 'px';
    var h = Math.min(els.input.scrollHeight, INPUT_MAX_HEIGHT_PX);
    els.input.style.height = h + 'px';
  }

  function onInputChange() {
    autoGrowInput();
    updateInputMode(CTX.getState());
  }

  function onKeyDown(ev) {
    if (ev.key === 'Enter' && !ev.shiftKey) {
      ev.preventDefault();
      onSendClick();
      return;
    }
    if (ev.key === 'Escape') {
      els.input.blur();
    }
  }

  // ---------- correction targeting pill ----------

  function setCorrectionTarget(tgt) {
    if (!els.targetPill) return;
    if (!tgt || (!tgt.region && typeof tgt.timeMs !== 'number')) {
      clearCorrectionTarget();
      return;
    }
    chat.correctionTarget = {
      region:   tgt.region || null,
      label:    tgt.label  || tgt.region || '',
      timeMs:   typeof tgt.timeMs === 'number' ? tgt.timeMs : null,
      timeText: tgt.timeText || (typeof tgt.timeMs === 'number'
        ? (tgt.timeMs / 1000).toFixed(2) : ''),
    };
    var parts = [];
    if (chat.correctionTarget.timeText) {
      var timePrefix = (window.KOZHA_I18N && window.KOZHA_I18N.t)
        ? window.KOZHA_I18N.t('contribute.chat.target_pill_time_prefix', { time: chat.correctionTarget.timeText })
        : 'At ' + chat.correctionTarget.timeText + 's';
      parts.push(timePrefix);
    }
    if (chat.correctionTarget.label) {
      parts.push(chat.correctionTarget.label);
    }
    els.targetText.textContent = parts.join(tr('contribute.chat.target_pill_separator', ' • '));
    els.targetPill.hidden = false;
    // Focus the input so the contributor can start typing the
    // correction right away. The pill stays visible above the
    // textarea until they send or dismiss it.
    if (els.input && !els.input.disabled && typeof els.input.focus === 'function') {
      els.input.focus();
    }
  }

  function clearCorrectionTarget() {
    chat.correctionTarget = null;
    if (els.targetPill) {
      els.targetPill.hidden = true;
      if (els.targetText) els.targetText.textContent = '';
    }
  }

  // ---------- public surface ----------

  window.KOZHA_CONTRIB_CHAT = {
    reset: resetChat,
    showError: function () {
      chat.inError = true;
      appendMessage({ kind: 'error', text: COPY.ERROR_MSG });
      els.errorActions.hidden = false;
    },
    setCorrectionTarget:   setCorrectionTarget,
    clearCorrectionTarget: clearCorrectionTarget,
  };

  // ---------- mount ----------

  function init() {
    els.input.addEventListener('input', onInputChange);
    els.input.addEventListener('keydown', onKeyDown);
    els.sendBtn.addEventListener('click', onSendClick);
    els.submitAsIs.addEventListener('click', onSubmitAsIs);
    if (els.retryBtn) {
      els.retryBtn.addEventListener('click', function () {
        clearError();
        retryLastAction();
      });
    }
    if (els.discardBtn) {
      els.discardBtn.addEventListener('click', onDiscardClick);
    }
    if (els.targetClear) {
      els.targetClear.addEventListener('click', clearCorrectionTarget);
    }
    autoGrowInput();
    CTX.subscribe(onSnapshot);
    onSnapshot(CTX.getState());
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

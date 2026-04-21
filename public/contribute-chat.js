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

  // ---------- exact copy (no variation, per spec) ----------

  var COPY = {
    CLARIFICATION_LABEL:           'Clarification:',
    INPUT_LABEL_ANSWER:            'Your answer',
    INPUT_LABEL_CORRECTION:        'Describe what should change',
    INPUT_PLACEHOLDER_ANSWER:      'Type your answer…',
    INPUT_PLACEHOLDER_CORRECTION:  'Type a correction…',
    INPUT_PLACEHOLDER_GENERATING:  'Generating sign…',
    CORRECTION_HINT:               'Click on the avatar to target a specific moment or body part (optional).',
    REPLY_USER_QUESTION:           "I can only ask about this sign. If you'd like to learn HamNoSys, see the docs link below.",
    GENERATING_MSG:                'Enough information to draft the sign. Preparing preview.',
    READY_MSG:                     'Draft is ready. Review the preview below and either submit or describe a correction.',
    ERROR_MSG:                     'Something went wrong generating a clarification. Try rephrasing your description, or submit the draft as-is and let the reviewer fill gaps.',
  };

  var TERMINAL_STATES = { finalized: true, abandoned: true };
  var INPUT_MIN_HEIGHT_PX = 44;
  var INPUT_MAX_HEIGHT_PX = 132;

  // ---------- DOM ----------

  var els = {
    panel:        document.getElementById('chatPanel'),
    log:          document.getElementById('chatLog'),
    options:      document.getElementById('chatOptions'),
    hint:         document.getElementById('chatInputHint'),
    progress:     document.getElementById('chatProgress'),
    inputLabel:   document.getElementById('chatInputLabel'),
    input:        document.getElementById('chatInput'),
    sendBtn:      document.getElementById('chatSendBtn'),
    errorActions: document.getElementById('chatErrorActions'),
    submitAsIs:   document.getElementById('chatSubmitAsIsBtn'),
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
    // True after we surface the error message and the Submit-as-is path.
    inError: false,
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

    if (opts.kind === 'system') {
      var label = document.createElement('span');
      label.className = 'chat-msg-label';
      label.textContent = COPY.CLARIFICATION_LABEL;
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
      btn.setAttribute('aria-label', 'Answer: ' + opt.label);
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

    els.input.disabled = noActiveTurn;
    els.sendBtn.disabled = noActiveTurn || !(els.input.value || '').trim();
  }

  function setInFlight(yes) {
    chat.inFlight = !!yes;
    els.progress.hidden = !yes;
    if (yes) {
      els.input.disabled = true;
      els.sendBtn.disabled = true;
    } else {
      updateInputMode(CTX.getState());
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

  function sendAnswer(field, answerText) {
    setInFlight(true);
    clearError();
    CTX.answer(field, answerText)
      .catch(handleError)
      .then(function () { setInFlight(false); });
  }

  function sendCorrection(text) {
    setInFlight(true);
    clearError();
    CTX.correct(text, {})
      .catch(handleError)
      .then(function () { setInFlight(false); });
  }

  function handleError(err) {
    if (window.console) console.error('[contribute-chat] action failed:', err);
    chat.inError = true;
    appendMessage({ kind: 'system', text: COPY.ERROR_MSG });
    els.errorActions.hidden = false;
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
      appendMessage({ kind: 'system', text: nextQ.text });
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
    els.errorActions.hidden = true;
    els.options.hidden = true;
    els.progress.hidden = true;
    els.input.value = '';
    autoGrowInput();
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

  // ---------- public surface ----------

  window.KOZHA_CONTRIB_CHAT = {
    reset: resetChat,
    showError: function () {
      chat.inError = true;
      appendMessage({ kind: 'system', text: COPY.ERROR_MSG });
      els.errorActions.hidden = false;
    },
  };

  // ---------- mount ----------

  function init() {
    els.input.addEventListener('input', onInputChange);
    els.input.addEventListener('keydown', onKeyDown);
    els.sendBtn.addEventListener('click', onSendClick);
    els.submitAsIs.addEventListener('click', onSubmitAsIs);
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

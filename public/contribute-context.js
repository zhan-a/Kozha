/* Contribute-page session context store.
 *
 * A tiny observable store for the state the contribute page needs to keep
 * in sync between the language header, the context strip, and whichever
 * authoring component mounts next (chat / preview / notation readout).
 *
 * Public API (window.KOZHA_CONTRIB_CONTEXT):
 *
 *   getState() → { language, gloss, sessionId, sessionToken,
 *                  sessionState, lastUpdated }
 *   setState(patch)     — shallow-merges; persists the session subset;
 *                         notifies subscribers.
 *   subscribe(fn) → unsubscribe fn.
 *   reset()             — back to default, clears persistence.
 *   stateLabel(raw?)    — backend state → plain English label.
 *   shortId(id?)        — first 8 chars of a UUID.
 *   sessionUrl()        — absolute URL carrying the #s/<id> fragment.
 *   parseFragment(hash?) → <id> if the URL carries #s/<id>, else null.
 *   pushSessionFragment() / clearSessionFragment()
 *
 * Session actions (all return Promises that resolve with the envelope):
 *   createSession({prose?, gloss?, authorIsDeafNative?})
 *                                    POST /sessions (+ /describe if prose)
 *   describe(prose, gloss?)          POST /sessions/:id/describe
 *   answer(field, answerText)        POST /sessions/:id/answer
 *   correct(rawText, {targetTimeMs?, targetRegion?})
 *                                    POST /sessions/:id/correct
 *   resumeSession(id, token)         GET  /sessions/:id (verifies token)
 *   abandonSession()                 POST /sessions/:id/reject + full reset
 *   clearSession({reason?})          POST /sessions/:id/reject but keeps
 *                                    the selected language
 *
 * Persistence rule: only {language, sessionId, sessionToken} land in
 * sessionStorage. gloss / sessionState / pendingQuestions / clarifications
 * are derived from the server envelope after a resume, so persisting them
 * would fight the source of truth.
 */
(function () {
  'use strict';

  var STORAGE_KEY = 'kozha.contribute.context';
  // Prompt 3 wrote the active language to this key directly. Read it as a
  // fallback during hydration so the selector doesn't blank out for users
  // who had it set before this file shipped.
  var LEGACY_LANGUAGE_KEY = 'kozha.contribute.activeLanguage';

  var API_BASE = '/api/chat2hamnosys';

  // Backend session state → plain English label shown in the context strip.
  // The prompt names five: Draft / Awaiting clarification / Rendering /
  // Ready to submit / Submitted. Correction states collapse into the most
  // natural neighbouring label so the strip never flashes unfamiliar text.
  var STATE_LABELS = {
    'awaiting_description': 'Draft',
    'clarifying':           'Awaiting clarification',
    'generating':           'Rendering',
    'applying_correction':  'Rendering',
    'rendered':             'Ready to submit',
    'awaiting_correction':  'Ready to submit',
    'finalized':            'Submitted',
    'abandoned':            'Draft',
  };

  function defaultState() {
    return {
      language:         null,
      gloss:            '',
      sessionId:        null,
      sessionToken:     null,
      sessionState:     'awaiting_description',
      // Latest pending clarification questions from the server envelope.
      // Transient — never persisted. The chat panel reads the first one
      // as the question to show next.
      pendingQuestions: [],
      // Author/assistant clarification turns from the envelope, used by
      // the chat panel to replay history on a resume.
      clarifications:   [],
      // Generator outputs — the notation panel renders these once the
      // session reaches the RENDERED state. Null until generation has
      // produced a string.
      hamnosys:         null,
      sigml:            null,
      parameters:       null,
      generationErrors: [],
      // Tracked for the submission checklist — the server envelope
      // carries the running count; the "at least one correction" row is
      // optional but a useful signal of author engagement.
      correctionsCount:   0,
      authorIsDeafNative: null,
      descriptionProse:   '',
      lastUpdated:      0,
    };
  }

  var state = defaultState();
  var subscribers = [];

  // ----- persistence -----

  function readPersisted() {
    try {
      var raw = sessionStorage.getItem(STORAGE_KEY);
      if (raw) {
        var parsed = JSON.parse(raw);
        return {
          language:     parsed.language     || null,
          sessionId:    parsed.sessionId    || null,
          sessionToken: parsed.sessionToken || null,
        };
      }
      var legacy = sessionStorage.getItem(LEGACY_LANGUAGE_KEY);
      if (legacy) return { language: legacy, sessionId: null, sessionToken: null };
      return null;
    } catch (_e) {
      return null;
    }
  }

  function writePersisted() {
    try {
      if (!state.language && !state.sessionId) {
        sessionStorage.removeItem(STORAGE_KEY);
        sessionStorage.removeItem(LEGACY_LANGUAGE_KEY);
        return;
      }
      var subset = {
        language:     state.language,
        sessionId:    state.sessionId,
        sessionToken: state.sessionToken,
      };
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(subset));
      // Keep the legacy single-key mirror in sync so the prompt-3 smoke
      // test and any older code path still sees the active language.
      if (state.language) sessionStorage.setItem(LEGACY_LANGUAGE_KEY, state.language);
      else sessionStorage.removeItem(LEGACY_LANGUAGE_KEY);
    } catch (_e) { /* storage blocked — stay in-memory only */ }
  }

  // ----- subscribers -----

  function notify() {
    var snap = getState();
    for (var i = 0; i < subscribers.length; i++) {
      try { subscribers[i](snap); } catch (e) {
        if (window.console) console.error('[contribute-context] subscriber error', e);
      }
    }
  }

  function getState() {
    return {
      language:           state.language,
      gloss:              state.gloss,
      sessionId:          state.sessionId,
      sessionToken:       state.sessionToken,
      sessionState:       state.sessionState,
      pendingQuestions:   state.pendingQuestions.slice(),
      clarifications:     state.clarifications.slice(),
      hamnosys:           state.hamnosys,
      sigml:              state.sigml,
      parameters:         state.parameters,
      generationErrors:   state.generationErrors.slice(),
      correctionsCount:   state.correctionsCount,
      authorIsDeafNative: state.authorIsDeafNative,
      descriptionProse:   state.descriptionProse,
      lastUpdated:        state.lastUpdated,
    };
  }

  function setState(patch) {
    if (!patch || typeof patch !== 'object') return;
    var changed = false;
    for (var k in patch) {
      if (!Object.prototype.hasOwnProperty.call(patch, k)) continue;
      if (!(k in state)) continue;
      var prev = state[k];
      var next = patch[k];
      // Arrays come from server envelopes; reference-compare would
      // miss in-place changes we never make, but we always assign so
      // subscribers re-render with the latest envelope payload.
      if (Array.isArray(prev) || Array.isArray(next)) {
        state[k] = Array.isArray(next) ? next : [];
        changed = true;
      } else if (prev !== next) {
        state[k] = next;
        changed = true;
      }
    }
    if (!changed) return;
    state.lastUpdated = Date.now();
    writePersisted();
    notify();
  }

  function subscribe(fn) {
    if (typeof fn !== 'function') return function () {};
    subscribers.push(fn);
    return function unsubscribe() {
      for (var i = 0; i < subscribers.length; i++) {
        if (subscribers[i] === fn) { subscribers.splice(i, 1); return; }
      }
    };
  }

  function reset() {
    state = defaultState();
    writePersisted();
    notify();
  }

  // ----- helpers -----

  function stateLabel(raw) {
    var v = raw || state.sessionState;
    return STATE_LABELS[v] || 'Draft';
  }

  function shortId(id) {
    var v = id || state.sessionId;
    if (!v || typeof v !== 'string') return '';
    return v.slice(0, 8);
  }

  function sessionUrl() {
    var loc = window.location;
    var base = loc.origin + loc.pathname;
    if (!state.sessionId) return base;
    return base + '#s/' + state.sessionId;
  }

  function parseFragment(hash) {
    if (typeof hash !== 'string') hash = (window.location.hash || '');
    if (hash.indexOf('#s/') !== 0) return null;
    var id = hash.substring(3).trim();
    return id || null;
  }

  function pushSessionFragment() {
    var loc = window.location;
    var target = state.sessionId
      ? loc.pathname + loc.search + '#s/' + state.sessionId
      : loc.pathname + loc.search;
    try {
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, '', target);
      } else {
        loc.hash = state.sessionId ? 's/' + state.sessionId : '';
      }
    } catch (_e) { /* ignore — fragment is a UX nicety, not load-bearing */ }
  }

  function clearSessionFragment() {
    var loc = window.location;
    try {
      if (window.history && window.history.replaceState) {
        window.history.replaceState(null, '', loc.pathname + loc.search);
      } else {
        loc.hash = '';
      }
    } catch (_e) { /* ignore */ }
  }

  // ----- network -----

  function parseError(resp) {
    return resp.text().then(function (body) {
      var err = new Error('HTTP ' + resp.status);
      err.status = resp.status;
      err.body = body;
      return err;
    });
  }

  function jsonOr(resp) {
    if (resp.ok) return resp.json();
    return parseError(resp).then(function (err) { throw err; });
  }

  // Pulls the chat- and notation-relevant fields off a server envelope
  // and mirrors them into the store. Centralised so /describe, /answer,
  // /correct and the resume GET stay in sync without each remembering
  // which fields each panel reads.
  function applyEnvelope(env) {
    if (!env) return;
    var patch = {
      sessionState:     env.state || state.sessionState,
      gloss:            env.gloss || state.gloss,
      pendingQuestions: env.pending_questions || [],
      clarifications:   env.clarifications || [],
      // Nullable so a correction that wipes the draft blanks the panel
      // (instead of leaving the previous notation on screen).
      hamnosys:         typeof env.hamnosys === 'string' ? env.hamnosys : null,
      sigml:            typeof env.sigml === 'string' ? env.sigml : null,
      parameters:       env.parameters || null,
      generationErrors: env.generation_errors || [],
    };
    if (typeof env.corrections_count === 'number') {
      patch.correctionsCount = env.corrections_count;
    }
    if (typeof env.description_prose === 'string') {
      patch.descriptionProse = env.description_prose;
    }
    setState(patch);
  }

  function createSession(opts) {
    opts = opts || {};
    if (!state.language) return Promise.reject(new Error('no language selected'));
    var body = { sign_language: state.language };
    if (opts.gloss) body.gloss = opts.gloss;
    // The checkbox is optional on the form — only forward the flag when
    // the caller actually set it, so unchecked stays null (not false) and
    // downstream reviewer routing can distinguish "not stated" from "no".
    if (typeof opts.authorIsDeafNative === 'boolean') {
      body.author_is_deaf_native = opts.authorIsDeafNative;
    }
    return fetch(API_BASE + '/sessions', {
      method: 'POST',
      headers: {
        'Accept':       'application/json',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })
      .then(jsonOr)
      .then(function (created) {
        var patch = {
          sessionId:    created.session_id,
          sessionToken: created.session_token,
          sessionState: created.state,
          gloss:        (created.session && created.session.gloss) || opts.gloss || '',
        };
        if (typeof opts.authorIsDeafNative === 'boolean') {
          patch.authorIsDeafNative = opts.authorIsDeafNative;
        }
        setState(patch);
        // Mirror nested envelope fields too so subscribers see a fully
        // populated snapshot before the chained /describe (if any) fires.
        if (created.session) applyEnvelope(created.session);
        pushSessionFragment();
        if (opts.prose) {
          return describe(opts.prose, opts.gloss);
        }
        return created;
      });
  }

  function describe(prose, gloss) {
    if (!state.sessionId || !state.sessionToken) {
      return Promise.reject(new Error('no active session'));
    }
    var body = { prose: prose };
    if (gloss) body.gloss = gloss;
    return fetch(API_BASE + '/sessions/' + encodeURIComponent(state.sessionId) + '/describe', {
      method: 'POST',
      headers: {
        'Accept':          'application/json',
        'Content-Type':    'application/json',
        'X-Session-Token': state.sessionToken,
      },
      body: JSON.stringify(body),
    })
      .then(jsonOr)
      .then(function (env) {
        applyEnvelope(env);
        return env;
      });
  }

  function answer(questionField, answerText) {
    if (!state.sessionId || !state.sessionToken) {
      return Promise.reject(new Error('no active session'));
    }
    if (!questionField) {
      return Promise.reject(new Error('question field required'));
    }
    var body = { question_id: questionField, answer: answerText };
    return fetch(API_BASE + '/sessions/' + encodeURIComponent(state.sessionId) + '/answer', {
      method: 'POST',
      headers: {
        'Accept':          'application/json',
        'Content-Type':    'application/json',
        'X-Session-Token': state.sessionToken,
      },
      body: JSON.stringify(body),
    })
      .then(jsonOr)
      .then(function (env) {
        applyEnvelope(env);
        return env;
      });
  }

  function correct(rawText, opts) {
    opts = opts || {};
    if (!state.sessionId || !state.sessionToken) {
      return Promise.reject(new Error('no active session'));
    }
    var body = { raw_text: rawText };
    if (typeof opts.targetTimeMs === 'number') body.target_time_ms = opts.targetTimeMs;
    if (opts.targetRegion) body.target_region = opts.targetRegion;
    return fetch(API_BASE + '/sessions/' + encodeURIComponent(state.sessionId) + '/correct', {
      method: 'POST',
      headers: {
        'Accept':          'application/json',
        'Content-Type':    'application/json',
        'X-Session-Token': state.sessionToken,
      },
      body: JSON.stringify(body),
    })
      .then(jsonOr)
      .then(function (env) {
        applyEnvelope(env);
        return env;
      });
  }

  function resumeSession(sessionId, token) {
    if (!sessionId || !token) return Promise.reject(new Error('id and token required'));
    return fetch(API_BASE + '/sessions/' + encodeURIComponent(sessionId), {
      method: 'GET',
      headers: {
        'Accept':          'application/json',
        'X-Session-Token': token,
      },
    })
      .then(jsonOr)
      .then(function (env) {
        setState({
          language:     env.sign_language || state.language,
          sessionId:    env.session_id,
          sessionToken: token,
        });
        applyEnvelope(env);
        pushSessionFragment();
        return env;
      });
  }

  function abandonSession() {
    var id = state.sessionId;
    var token = state.sessionToken;
    if (!id || !token) {
      reset();
      clearSessionFragment();
      return Promise.resolve();
    }
    return fetch(API_BASE + '/sessions/' + encodeURIComponent(id) + '/reject', {
      method: 'POST',
      headers: {
        'Accept':          'application/json',
        'Content-Type':    'application/json',
        'X-Session-Token': token,
      },
      body: JSON.stringify({ reason: 'language_changed' }),
    })
      // Best-effort: a failed reject still clears local state so the UI
      // isn't stuck on a dead session. The orphan is harmless on the
      // backend; the user sees a clean slate.
      .catch(function () { /* swallow */ })
      .then(function () {
        reset();
        clearSessionFragment();
      });
  }

  // Rejects the session server-side (best-effort) and clears the local
  // session fields, but leaves the selected language in place. The
  // contribute form's "edit" flow uses this so the user lands back on
  // the input with the same language still selected.
  function clearSession(opts) {
    opts = opts || {};
    var reason = opts.reason || 'cleared';
    var id = state.sessionId;
    var token = state.sessionToken;
    setState({
      sessionId:          null,
      sessionToken:       null,
      gloss:              '',
      sessionState:       'awaiting_description',
      pendingQuestions:   [],
      clarifications:     [],
      hamnosys:           null,
      sigml:              null,
      parameters:         null,
      generationErrors:   [],
      correctionsCount:   0,
      authorIsDeafNative: null,
      descriptionProse:   '',
    });
    clearSessionFragment();
    if (!id || !token) return Promise.resolve();
    return fetch(API_BASE + '/sessions/' + encodeURIComponent(id) + '/reject', {
      method: 'POST',
      headers: {
        'Accept':          'application/json',
        'Content-Type':    'application/json',
        'X-Session-Token': token,
      },
      body: JSON.stringify({ reason: reason }),
    }).catch(function () { /* best-effort */ });
  }

  // ----- hydrate + export -----

  var persisted = readPersisted();
  if (persisted) {
    state.language     = persisted.language;
    state.sessionId    = persisted.sessionId;
    state.sessionToken = persisted.sessionToken;
  }

  window.KOZHA_CONTRIB_CONTEXT = {
    getState:             getState,
    setState:             setState,
    subscribe:            subscribe,
    reset:                reset,
    stateLabel:           stateLabel,
    shortId:              shortId,
    sessionUrl:           sessionUrl,
    parseFragment:        parseFragment,
    pushSessionFragment:  pushSessionFragment,
    clearSessionFragment: clearSessionFragment,
    createSession:        createSession,
    describe:             describe,
    answer:               answer,
    correct:              correct,
    resumeSession:        resumeSession,
    abandonSession:       abandonSession,
    clearSession:         clearSession,
  };
})();

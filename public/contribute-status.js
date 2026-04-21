/* Contribute page — submission status viewer (prompt 10).
 *
 * Renders GET /api/chat2hamnosys/sessions/{id}/status for the session
 * id in the URL path (/contribute/status/<id>). The status endpoint is
 * token-optional: without a token the public envelope is returned
 * (gloss, language, final status, validated HamNoSys/SiGML); with a
 * valid X-Session-Token, the backend folds in the private description
 * and reviewer comments.
 *
 * Flow:
 *   1. Parse <id> from the path.
 *   2. Try sessionStorage for a token (same browser — the submission
 *      flow leaves id + token there). If present, send it on the first
 *      GET so private fields come back immediately.
 *   3. Render the envelope. If the caller has no token, show a resume-
 *      token form that re-fetches with the X-Session-Token header on
 *      submit.
 *   4. On 404, show a simple "not found" screen with a link back to
 *      the contribute page.
 */
(function () {
  'use strict';

  var API_BASE = '/api/chat2hamnosys';
  var STORAGE_KEY = 'kozha.contribute.context';
  var PATH_PREFIX = '/contribute/status/';

  var LANGUAGE_LABELS = {
    bsl: 'British Sign Language',
    asl: 'American Sign Language',
    dgs: 'German Sign Language',
  };

  var STATUS_LABELS = {
    draft:           'Draft',
    pending_review:  'Pending review',
    under_review:    'Under review',
    validated:       'Validated',
    rejected:        'Rejected',
    quarantined:     'Quarantined',
  };

  var STATUS_NOTES = {
    draft:           'No Deaf reviewer is currently assigned to this language. Review is on hold.',
    pending_review:  'Typical review time is 3 days.',
    under_review:    'A reviewer has this draft open.',
    validated:       'A Deaf reviewer has approved this sign.',
    rejected:        'A Deaf reviewer declined to publish this sign.',
    quarantined:     'This sign is on hold pending further review.',
  };

  var els = {
    loading:           document.getElementById('statusLoading'),
    error:             document.getElementById('statusError'),
    errorHeading:      document.getElementById('statusErrorHeading'),
    errorBody:         document.getElementById('statusErrorBody'),
    body:              document.getElementById('statusBody'),
    gloss:             document.getElementById('statusGloss'),
    language:          document.getElementById('statusLanguage'),
    state:             document.getElementById('statusState'),
    stateNote:         document.getElementById('statusStateNote'),
    dates:             document.getElementById('statusDates'),
    privateSec:        document.getElementById('statusPrivate'),
    description:       document.getElementById('statusDescription'),
    notationSec:       document.getElementById('statusNotation'),
    hamnosys:          document.getElementById('statusHamnosys'),
    sigml:             document.getElementById('statusSigml'),
    commentsSec:       document.getElementById('statusComments'),
    commentsList:      document.getElementById('statusCommentsList'),
    tokenGate:         document.getElementById('statusTokenGate'),
    tokenForm:         document.getElementById('statusTokenForm'),
    tokenInput:        document.getElementById('statusTokenInput'),
    tokenError:        document.getElementById('statusTokenError'),
  };

  function parseSessionId() {
    var path = window.location.pathname || '';
    if (path.indexOf(PATH_PREFIX) !== 0) return null;
    var rest = path.substring(PATH_PREFIX.length);
    // Strip any trailing slash or query ambiguity.
    var slash = rest.indexOf('/');
    if (slash >= 0) rest = rest.substring(0, slash);
    rest = decodeURIComponent(rest).trim();
    return rest || null;
  }

  function readStoredToken(sessionId) {
    try {
      var raw = sessionStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      var parsed = JSON.parse(raw);
      if (!parsed) return null;
      if (parsed.sessionId && parsed.sessionId !== sessionId) return null;
      return parsed.sessionToken || null;
    } catch (_e) { return null; }
  }

  function writeStoredToken(sessionId, token, language) {
    try {
      var raw = sessionStorage.getItem(STORAGE_KEY);
      var parsed = raw ? JSON.parse(raw) : {};
      parsed.sessionId = sessionId;
      parsed.sessionToken = token;
      if (language) parsed.language = language;
      sessionStorage.setItem(STORAGE_KEY, JSON.stringify(parsed));
    } catch (_e) { /* ignore */ }
  }

  function languageLabelFor(code) {
    if (!code) return '';
    return LANGUAGE_LABELS[code] || String(code).toUpperCase();
  }

  function formatDate(iso) {
    if (!iso) return '';
    try {
      var d = new Date(iso);
      if (isNaN(d.getTime())) return '';
      return d.toLocaleString();
    } catch (_e) { return ''; }
  }

  function fetchStatus(sessionId, token) {
    var url = API_BASE + '/sessions/' + encodeURIComponent(sessionId) + '/status';
    var headers = { 'Accept': 'application/json' };
    if (token) headers['X-Session-Token'] = token;
    return fetch(url, { method: 'GET', headers: headers })
      .then(function (resp) {
        return resp.text().then(function (body) {
          var parsed = null;
          try { parsed = body ? JSON.parse(body) : null; } catch (_e) { /* ignore */ }
          return { ok: resp.ok, status: resp.status, body: parsed, raw: body };
        });
      });
  }

  function renderComments(comments) {
    els.commentsList.innerHTML = '';
    for (var i = 0; i < comments.length; i++) {
      var c = comments[i] || {};
      var li = document.createElement('li');
      li.className = 'status-comment';

      var verdict = document.createElement('p');
      verdict.className = 'status-comment-verdict';
      verdict.textContent = c.verdict || '';
      li.appendChild(verdict);

      if (c.category) {
        var cat = document.createElement('p');
        cat.className = 'status-comment-category';
        cat.textContent = c.category;
        li.appendChild(cat);
      }

      if (c.comment) {
        var body = document.createElement('p');
        body.className = 'status-comment-body';
        body.textContent = c.comment;
        li.appendChild(body);
      }

      var when = formatDate(c.reviewed_at);
      if (when) {
        var stamp = document.createElement('p');
        stamp.className = 'status-comment-stamp';
        stamp.textContent = when;
        li.appendChild(stamp);
      }

      els.commentsList.appendChild(li);
    }
  }

  function render(envelope) {
    els.loading.hidden = true;
    els.error.hidden = true;
    els.body.hidden = false;

    var status = envelope.status || 'draft';
    var hasToken = !!envelope.has_token;

    els.gloss.textContent = envelope.gloss || '';
    var langCode = envelope.sign_language || '';
    var langName = languageLabelFor(langCode);
    var regional = envelope.regional_variant ? ' (' + envelope.regional_variant + ')' : '';
    els.language.textContent = langName + regional;

    els.state.textContent = STATUS_LABELS[status] || status;
    var note = STATUS_NOTES[status] || '';
    if (status === 'rejected' && envelope.rejection_category) {
      note = 'Category: ' + envelope.rejection_category + '. ' + note;
    }
    els.stateNote.textContent = note;

    var created = formatDate(envelope.created_at);
    var updated = formatDate(envelope.updated_at);
    var datePieces = [];
    if (created) datePieces.push('Submitted ' + created);
    if (updated && updated !== created) datePieces.push('Last update ' + updated);
    els.dates.textContent = datePieces.join(' · ');

    // Validated HamNoSys/SiGML is public once status is validated; for
    // any other state it's author-only. The backend enforces this and
    // only populates these fields when the caller has access.
    if (envelope.hamnosys) {
      els.hamnosys.textContent = envelope.hamnosys;
      els.sigml.textContent = envelope.sigml || '';
      els.notationSec.hidden = false;
    } else {
      els.notationSec.hidden = true;
    }

    if (hasToken && typeof envelope.description_prose === 'string' && envelope.description_prose) {
      els.description.textContent = envelope.description_prose;
      els.privateSec.hidden = false;
    } else {
      els.privateSec.hidden = true;
    }

    var comments = Array.isArray(envelope.reviewer_comments) ? envelope.reviewer_comments : [];
    if (hasToken && comments.length > 0) {
      renderComments(comments);
      els.commentsSec.hidden = false;
    } else {
      els.commentsSec.hidden = true;
    }

    // Show the token gate only when the caller is unauthenticated. With
    // a token, we've already pulled the private fields — the gate would
    // just add noise.
    els.tokenGate.hidden = hasToken;
  }

  function showError(heading, body) {
    els.loading.hidden = true;
    els.body.hidden = true;
    els.error.hidden = false;
    if (heading) els.errorHeading.textContent = heading;
    els.errorBody.textContent = body || '';
  }

  function handleFailure(sessionId, result) {
    if (result.status === 404) {
      showError('Not found',
        'No submission matches this URL. It may have been discarded, or the link may be mistyped.');
      return;
    }
    if (result.status === 403) {
      showError('Access denied',
        'The resume token did not match this submission.');
      return;
    }
    var detail = (result.body && result.body.detail) || '';
    showError('Something went wrong',
      'Could not load this submission (HTTP ' + result.status + '). ' + detail);
  }

  function onTokenSubmit(sessionId) {
    return function (ev) {
      ev.preventDefault();
      var token = (els.tokenInput.value || '').trim();
      if (!token) return;
      els.tokenError.hidden = true;
      els.tokenError.textContent = '';
      fetchStatus(sessionId, token).then(function (r) {
        if (!r.ok || !r.body) {
          els.tokenError.hidden = false;
          els.tokenError.textContent = r.status === 403
            ? 'That token does not match this submission.'
            : 'Could not unlock this submission.';
          return;
        }
        writeStoredToken(sessionId, token, r.body.sign_language);
        render(r.body);
      }).catch(function () {
        els.tokenError.hidden = false;
        els.tokenError.textContent = 'Could not reach the server. Check your connection and try again.';
      });
    };
  }

  function init() {
    var sessionId = parseSessionId();
    if (!sessionId) {
      showError('Invalid link',
        'This URL does not contain a submission id.');
      return;
    }

    els.tokenForm.addEventListener('submit', onTokenSubmit(sessionId));

    var token = readStoredToken(sessionId);
    fetchStatus(sessionId, token).then(function (r) {
      if (!r.ok || !r.body) {
        // If our stored token turned out to be stale (403), retry anonymously
        // so the public envelope still shows.
        if (r.status === 403 && token) {
          fetchStatus(sessionId, null).then(function (r2) {
            if (r2.ok && r2.body) {
              render(r2.body);
            } else {
              handleFailure(sessionId, r2);
            }
          }).catch(function () {
            handleFailure(sessionId, { status: 0 });
          });
          return;
        }
        handleFailure(sessionId, r);
        return;
      }
      render(r.body);
    }).catch(function () {
      showError('Something went wrong',
        'Could not reach the server. Check your connection and try again.');
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

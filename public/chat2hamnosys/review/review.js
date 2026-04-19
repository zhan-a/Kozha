/**
 * Reviewer console — vanilla JS app.
 *
 * Talks to /api/chat2hamnosys/review. The bearer token lives in
 * localStorage and is sent on every request as X-Reviewer-Token.
 *
 * Fatigue counter: a per-tab integer incremented every time a review
 * action lands. At 25 we surface a non-blocking nudge — the prompt
 * spec calls for 25/session as the rough informed-consent default.
 */
(() => {
  'use strict';

  const API_BASE = '/api/chat2hamnosys/review';
  const TOKEN_KEY = 'kozha:reviewer:token';
  const FATIGUE_LIMIT = 25;

  // ---------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------
  const state = {
    token: null,
    me: null,           // ReviewerPublic dict
    queue: [],          // queue summary list
    activeId: null,     // currently-selected sign id
    detail: null,       // SignEntryDetail dict
    fatigueCount: 0,
  };

  // ---------------------------------------------------------------------
  // DOM refs
  // ---------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const dom = {
    statusBanner: $('statusBanner'),
    statusText:   () => $('statusBanner').querySelector('.status-text'),
    reviewerBadge:$('reviewerBadge'),
    fatigueMeter: $('fatigueMeter'),
    signOutBtn:   $('signOutBtn'),

    signinGate:   $('signinGate'),
    signinForm:   $('signinForm'),
    tokenInput:   $('tokenInput'),
    signinError:  $('signinError'),

    layout:       $('reviewLayout'),
    refreshBtn:   $('refreshBtn'),

    queueList:    $('queueList'),
    queueEmpty:   $('queueEmpty'),
    filterLang:   $('filterLang'),
    filterRegion: $('filterRegion'),
    filterQuarantine: $('filterQuarantine'),

    detailEmpty:  $('detailEmpty'),
    detailBody:   $('detailBody'),
    detailGloss:  $('detailGloss'),
    detailStatus: $('detailStatus'),
    detailLang:   $('detailLang'),
    detailRegion: $('detailRegion'),
    detailDomain: $('detailDomain'),
    detailProse:  $('detailProse'),
    detailHns:    $('detailHns'),
    detailHistory:$('detailHistory'),
    approvalsPresent:  $('approvalsPresent'),
    approvalsRequired: $('approvalsRequired'),
    detailSub:    $('detailSub'),

    actApprove:   $('actApprove'),
    actReject:    $('actReject'),
    actRevise:    $('actRevise'),
    actFlag:      $('actFlag'),
    actClear:     $('actClear'),
    actExport:    $('actExport'),
    actionError:  $('actionError'),

    dialog:       $('actionDialog'),
    dialogForm:   $('actionDialogForm'),
    dialogTitle:  $('actionDialogTitle'),
    dialogSub:    $('actionDialogSub'),
    dialogComment:$('actionDialogComment'),
    dialogCancel: $('actionDialogCancel'),
    dialogSubmit: $('actionDialogSubmit'),
    dialogCommentLbl: $('actionDialogCommentLbl'),
    dialogCategoryWrap: $('actionDialogCategoryWrap'),
    dialogClearWrap:    $('actionDialogClearWrap'),
    dialogJustifyLbl:   $('actionDialogJustifyLbl'),
    dialogJustify:      $('actionDialogJustify'),
    dialogAllowNonNativeLbl: $('actionDialogAllowNonNativeLbl'),
    dialogAllowNonNative:    $('actionDialogAllowNonNative'),

    srAnnounce:   $('srAnnounce'),
    tplQueueItem: $('tplQueueItem'),
    tplHistoryItem: $('tplHistoryItem'),
  };

  // ---------------------------------------------------------------------
  // API
  // ---------------------------------------------------------------------
  async function api(method, path, body) {
    const headers = { 'Content-Type': 'application/json' };
    if (state.token) headers['X-Reviewer-Token'] = state.token;
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

  // ---------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------
  function setStatus(text) {
    dom.statusText().textContent = text;
    dom.statusBanner.hidden = false;
  }
  function clearStatus() { dom.statusBanner.hidden = true; }

  function announce(msg) {
    dom.srAnnounce.textContent = '';
    setTimeout(() => { dom.srAnnounce.textContent = msg; }, 50);
  }

  function fmtTime(iso) {
    if (!iso) return '';
    try { return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }); }
    catch (_e) { return iso; }
  }
  function fmtAge(iso) {
    if (!iso) return '';
    const ms = Date.now() - new Date(iso).getTime();
    const mins = Math.floor(ms / 60000);
    if (mins < 60) return `${mins}m old`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h old`;
    return `${Math.floor(hrs / 24)}d old`;
  }

  function setStatusPill(el, status) {
    el.className = 'status-pill s-' + status;
    el.textContent = status.replace('_', ' ');
  }

  function showActionError(msg) {
    dom.actionError.textContent = msg;
    dom.actionError.hidden = false;
  }
  function clearActionError() { dom.actionError.hidden = true; }

  // ---------------------------------------------------------------------
  // Sign-in
  // ---------------------------------------------------------------------
  async function trySignIn(token) {
    state.token = token;
    try {
      const me = await api('GET', '/me');
      state.me = me;
      window.localStorage.setItem(TOKEN_KEY, token);
      renderSignedIn();
      await loadQueue();
    } catch (e) {
      state.token = null;
      window.localStorage.removeItem(TOKEN_KEY);
      throw e;
    }
  }

  function renderSignedIn() {
    dom.signinGate.hidden = true;
    dom.layout.hidden = false;
    dom.signOutBtn.hidden = false;
    dom.fatigueMeter.hidden = false;
    dom.reviewerBadge.hidden = false;
    const flags = [];
    if (state.me.is_deaf_native) flags.push('Deaf-native');
    if (state.me.is_board) flags.push('board');
    const flagStr = flags.length ? ' · ' + flags.join(', ') : '';
    dom.reviewerBadge.textContent = state.me.display_name + flagStr;
    dom.reviewerBadge.classList.toggle('is-board', state.me.is_board);
    if (!state.me.is_board) {
      dom.actClear.hidden = true;
      dom.actExport.hidden = true;
    }
    updateFatigueMeter();
  }

  function signOut() {
    state.token = null;
    state.me = null;
    state.queue = [];
    state.detail = null;
    state.activeId = null;
    state.fatigueCount = 0;
    window.localStorage.removeItem(TOKEN_KEY);
    dom.layout.hidden = true;
    dom.signinGate.hidden = false;
    dom.signOutBtn.hidden = true;
    dom.fatigueMeter.hidden = true;
    dom.reviewerBadge.hidden = true;
    dom.tokenInput.value = '';
  }

  function updateFatigueMeter() {
    dom.fatigueMeter.textContent = `${state.fatigueCount}/${FATIGUE_LIMIT} reviewed this session`;
    dom.fatigueMeter.classList.toggle(
      'is-warning',
      state.fatigueCount >= FATIGUE_LIMIT
    );
    if (state.fatigueCount === FATIGUE_LIMIT) {
      announce(
        `You have reviewed ${FATIGUE_LIMIT} signs in this session — please consider taking a break before continuing.`
      );
    }
  }

  // ---------------------------------------------------------------------
  // Queue
  // ---------------------------------------------------------------------
  async function loadQueue() {
    setStatus('Loading queue…');
    try {
      const params = new URLSearchParams();
      if (dom.filterLang.value) params.set('sign_language', dom.filterLang.value);
      if (dom.filterRegion.value.trim()) params.set('regional_variant', dom.filterRegion.value.trim());
      if (dom.filterQuarantine.checked) params.set('include_quarantined', 'true');
      const qs = params.toString();
      const data = await api('GET', '/queue' + (qs ? '?' + qs : ''));
      state.queue = data.items || [];
      renderQueue();
    } catch (e) {
      announce('Failed to load queue: ' + e.message);
      console.error(e);
    } finally {
      clearStatus();
    }
  }

  function renderQueue() {
    dom.queueList.innerHTML = '';
    if (state.queue.length === 0) {
      dom.queueEmpty.hidden = false;
      return;
    }
    dom.queueEmpty.hidden = true;
    for (const item of state.queue) {
      const node = dom.tplQueueItem.content.firstElementChild.cloneNode(true);
      const btn = node.querySelector('.queue-item-btn');
      btn.querySelector('.qi-gloss').textContent = item.gloss;
      setStatusPill(btn.querySelector('.status-pill'), item.status);
      btn.querySelector('.qi-lang').textContent = item.sign_language;
      btn.querySelector('.qi-region').textContent = item.regional_variant || '—';
      btn.querySelector('.qi-time').textContent = fmtAge(item.created_at);
      btn.querySelector('.qi-approvals').textContent =
        `Approvals: ${item.qualifying_approvals}/${item.min_approvals_required}`;
      btn.querySelector('.qi-reviews').textContent =
        item.review_count + ' review' + (item.review_count === 1 ? '' : 's');
      if (item.id === state.activeId) btn.classList.add('is-active');
      btn.addEventListener('click', () => loadDetail(item.id));
      dom.queueList.appendChild(node);
    }
  }

  // ---------------------------------------------------------------------
  // Detail
  // ---------------------------------------------------------------------
  async function loadDetail(signId) {
    state.activeId = signId;
    clearActionError();
    setStatus('Loading sign…');
    try {
      const detail = await api('GET', '/entries/' + signId);
      state.detail = detail;
      renderDetail();
      // Re-mark active row in the queue.
      renderQueue();
    } catch (e) {
      announce('Failed to load sign: ' + e.message);
      console.error(e);
    } finally {
      clearStatus();
    }
  }

  function renderDetail() {
    const d = state.detail;
    if (!d) {
      dom.detailEmpty.hidden = false;
      dom.detailBody.hidden = true;
      return;
    }
    dom.detailEmpty.hidden = true;
    dom.detailBody.hidden = false;
    dom.detailGloss.textContent = d.gloss;
    setStatusPill(dom.detailStatus, d.status);
    dom.detailLang.textContent = d.sign_language;
    dom.detailRegion.textContent = d.regional_variant || '—';
    dom.detailDomain.textContent = d.domain || '—';
    dom.approvalsPresent.textContent = d.qualifying_approvals;
    dom.approvalsRequired.textContent = d.min_approvals_required;
    dom.detailProse.textContent = d.description_prose || '(no description)';
    dom.detailHns.textContent = d.hamnosys || '(no HamNoSys)';
    dom.detailSub.textContent = `Updated ${fmtTime(d.updated_at)}`;
    renderHistory(d.reviewers || []);
    renderActions();
  }

  function renderHistory(records) {
    dom.detailHistory.innerHTML = '';
    if (records.length === 0) {
      const li = document.createElement('li');
      li.className = 'history-item';
      li.textContent = 'No prior reviews.';
      dom.detailHistory.appendChild(li);
      return;
    }
    for (const r of records) {
      const node = dom.tplHistoryItem.content.firstElementChild.cloneNode(true);
      node.classList.add('v-' + r.verdict);
      node.querySelector('.history-verdict').textContent = r.verdict.replace('_', ' ');
      node.querySelector('.history-time').textContent = fmtTime(r.reviewed_at);
      node.querySelector('.history-comment').textContent = r.comment || r.notes || '';
      const meta = [];
      meta.push(r.is_deaf_native ? 'native' : 'non-native');
      if (r.regional_background) meta.push(r.regional_background);
      if (r.category) meta.push('category: ' + r.category);
      if (r.allow_non_native) meta.push('non-native override');
      node.querySelector('.history-meta').textContent = meta.join(' · ');
      dom.detailHistory.appendChild(node);
    }
  }

  function renderActions() {
    const d = state.detail;
    const status = d.status;
    // Default everything visible, then disable per status.
    const actionable = ['draft', 'pending_review'];
    const can = actionable.includes(status);
    dom.actApprove.disabled = !can;
    dom.actReject.disabled  = !can;
    dom.actRevise.disabled  = !can;
    // Flag is allowed from any non-rejected status (rejected is terminal).
    dom.actFlag.disabled = (status === 'rejected');
    // Board-only actions.
    if (state.me && state.me.is_board) {
      dom.actClear.hidden = (status !== 'quarantined');
      dom.actExport.hidden = (status !== 'validated');
    }
  }

  // ---------------------------------------------------------------------
  // Action dialog
  // ---------------------------------------------------------------------
  let pendingAction = null;

  function openDialog(opts) {
    pendingAction = opts.act;
    dom.dialogTitle.textContent = opts.title;
    if (opts.sub) {
      dom.dialogSub.textContent = opts.sub;
      dom.dialogSub.hidden = false;
    } else {
      dom.dialogSub.hidden = true;
    }
    dom.dialogComment.value = '';
    dom.dialogJustify.value = '';
    dom.dialogAllowNonNative.checked = false;
    dom.dialogCategoryWrap.hidden = !opts.category;
    dom.dialogClearWrap.hidden = !opts.clearTarget;
    const showJustify = !!opts.justify;
    dom.dialogJustifyLbl.hidden = !showJustify;
    dom.dialogJustify.hidden = !showJustify;
    dom.dialogAllowNonNativeLbl.hidden = !showJustify;
    dom.dialogCommentLbl.textContent = opts.commentLabel || 'Comment';
    dom.dialog.showModal();
    dom.dialogComment.focus();
  }

  function closeDialog() {
    dom.dialog.close();
    pendingAction = null;
  }

  dom.dialogCancel.addEventListener('click', closeDialog);

  dom.dialogForm.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    if (!pendingAction || !state.detail) return;
    const id = state.detail.id;
    const comment = dom.dialogComment.value.trim();
    let path = '';
    let body = {};
    switch (pendingAction) {
      case 'approve':
        path = `/entries/${id}/approve`;
        body = { comment };
        if (dom.dialogAllowNonNative.checked) {
          body.allow_non_native = true;
          body.justification = dom.dialogJustify.value.trim();
        }
        break;
      case 'reject': {
        if (!comment) return showActionError('Please give a reason.');
        const cat = dom.dialog.querySelector('input[name="rejectCategory"]:checked');
        path = `/entries/${id}/reject`;
        body = { reason: comment, category: cat ? cat.value : 'other' };
        break;
      }
      case 'revise':
        if (!comment) return showActionError('Please describe what to revise.');
        path = `/entries/${id}/request_revision`;
        body = { comment, fields_to_revise: [] };
        break;
      case 'flag':
        if (!comment) return showActionError('Please give a flag reason.');
        path = `/entries/${id}/flag`;
        body = { reason: comment };
        break;
      case 'clear': {
        if (!comment) return showActionError('Please describe why the quarantine is lifted.');
        const tgt = dom.dialog.querySelector('input[name="clearTarget"]:checked');
        path = `/entries/${id}/clear_quarantine`;
        body = {
          comment,
          target_status: tgt ? tgt.value : 'pending_review',
        };
        break;
      }
      default:
        return;
    }
    try {
      setStatus('Submitting…');
      const updated = await api('POST', path, body);
      state.detail = updated;
      state.fatigueCount += 1;
      updateFatigueMeter();
      renderDetail();
      await loadQueue();
      closeDialog();
      announce(`Action ${pendingAction} succeeded.`);
    } catch (e) {
      showActionError(`${e.code || 'error'}: ${e.message}`);
    } finally {
      clearStatus();
    }
  });

  // Action button wiring.
  dom.actApprove.addEventListener('click', () => {
    const needsJustify = state.me && !state.me.is_deaf_native;
    openDialog({
      act: 'approve',
      title: 'Approve sign',
      sub: needsJustify
        ? 'You are not registered as a native-Deaf reviewer; tick the override and add a justification to make your approval count.'
        : 'Optional comment for the audit trail.',
      commentLabel: 'Comment (optional)',
      justify: needsJustify,
    });
  });
  dom.actReject.addEventListener('click', () => {
    openDialog({
      act: 'reject',
      title: 'Reject sign',
      sub: 'Rejection is terminal — pick the closest category and explain.',
      commentLabel: 'Reason',
      category: true,
    });
  });
  dom.actRevise.addEventListener('click', () => {
    openDialog({
      act: 'revise',
      title: 'Request revision',
      sub: 'The author re-opens the draft and addresses your comment.',
      commentLabel: 'Comment for the author',
    });
  });
  dom.actFlag.addEventListener('click', () => {
    openDialog({
      act: 'flag',
      title: 'Flag for quarantine',
      sub: 'Quarantines work on any status. Use for cultural concerns or post-publication issues.',
      commentLabel: 'Flag reason',
    });
  });
  dom.actClear.addEventListener('click', () => {
    openDialog({
      act: 'clear',
      title: 'Clear quarantine',
      sub: 'Board only. Choose where the sign goes next.',
      commentLabel: 'Decision rationale',
      clearTarget: true,
    });
  });
  dom.actExport.addEventListener('click', async () => {
    if (!state.detail) return;
    if (!confirm(`Export "${state.detail.gloss}" to the Kozha library?`)) return;
    try {
      setStatus('Exporting…');
      await api('POST', `/entries/${state.detail.id}/export`);
      announce('Exported to library.');
      await loadDetail(state.detail.id);
      await loadQueue();
    } catch (e) {
      showActionError(`export failed (${e.code}): ${e.message}`);
    } finally {
      clearStatus();
    }
  });

  // Filter wiring.
  dom.filterLang.addEventListener('change', loadQueue);
  dom.filterRegion.addEventListener('change', loadQueue);
  dom.filterQuarantine.addEventListener('change', loadQueue);
  dom.refreshBtn.addEventListener('click', loadQueue);

  dom.signOutBtn.addEventListener('click', signOut);

  dom.signinForm.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    dom.signinError.hidden = true;
    const tok = dom.tokenInput.value.trim();
    if (!tok) return;
    try {
      await trySignIn(tok);
    } catch (e) {
      dom.signinError.textContent = `${e.code || 'error'}: ${e.message}`;
      dom.signinError.hidden = false;
    }
  });

  // ---------------------------------------------------------------------
  // Boot
  // ---------------------------------------------------------------------
  (function boot() {
    const stored = window.localStorage.getItem(TOKEN_KEY);
    if (stored) {
      trySignIn(stored).catch(() => {
        // Token rejected — fall back to gate.
        signOut();
      });
    }
  })();
})();

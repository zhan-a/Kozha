/**
 * Convert any thrown value, rejected reason, or error-event payload into a
 * short, human-readable string suitable for display in the UI or in logs.
 *
 * Contract:
 *   - Never returns the raw "[object Event]" / "[object Object]" coercion.
 *   - Always returns a non-empty string.
 *   - Decodes the common shapes seen in browser pipelines:
 *       Error (incl. subclasses), DOMException, AbortError, ErrorEvent,
 *       MessageEvent from a worker, generic Event (with MediaError-bearing
 *       target), plain strings, plain objects with .message, null/undefined.
 *
 * Usage (ESM):
 *   import { describeError } from "/lib/describe-error.js";
 *
 * Usage (legacy script): the module also assigns itself to
 *   window.describeError so non-module scripts can use it after the
 *   importing <script type="module"> has run.
 *
 * @param {unknown} err
 * @returns {string}
 */
export function describeError(err) {
  if (err === null || err === undefined) return 'unknown error';

  if (typeof err === 'string') return err || 'unknown error';

  if (typeof DOMException !== 'undefined' && err instanceof DOMException) {
    if (err.name === 'AbortError') return 'operation aborted';
    return (err.name || 'DOMException') + ': ' + (err.message || '(no message)');
  }

  if (err && typeof err === 'object' && err.name === 'AbortError') {
    return 'operation aborted';
  }

  if (typeof ErrorEvent !== 'undefined' && err instanceof ErrorEvent) {
    return describeErrorEvent(err);
  }

  if (typeof MessageEvent !== 'undefined' && err instanceof MessageEvent) {
    var workerMsg = describeWorkerMessageEvent(err);
    if (workerMsg) return workerMsg;
  }

  if (typeof Event !== 'undefined' && err instanceof Event) {
    return describeEvent(err);
  }

  if (err instanceof Error) {
    return err.message || err.name || 'error';
  }

  if (typeof err === 'object' && typeof err.message === 'string' && err.message) {
    return err.message;
  }

  var tag = Object.prototype.toString.call(err); // "[object XYZ]"
  if (tag.indexOf('[object ') === 0 && tag.charAt(tag.length - 1) === ']') {
    return tag.slice(8, -1).toLowerCase();
  }
  return 'unknown error';
}

function describeErrorEvent(ev) {
  var msg = ev.message || (ev.error && ev.error.message) || '';
  var hasLoc = !!(ev.filename || ev.lineno);
  if (msg && hasLoc) {
    var loc = (ev.filename || '?') + ':' + (ev.lineno != null ? ev.lineno : '?');
    return msg + ' (at ' + loc + ')';
  }
  if (msg) return msg;
  return 'error event';
}

function describeWorkerMessageEvent(ev) {
  var data = ev.data;
  if (data && typeof data === 'object'
      && data.type === 'error' && typeof data.message === 'string' && data.message) {
    return data.message;
  }
  return '';
}

function describeEvent(ev) {
  var target = ev.target;
  if (target && typeof target === 'object') {
    var mediaErr = target.error;
    if (mediaErr && typeof mediaErr === 'object' && typeof mediaErr.code === 'number') {
      return decodeMediaError(mediaErr);
    }
    var isMediaEl = (typeof HTMLMediaElement !== 'undefined' && target instanceof HTMLMediaElement);
    var isAudioCtx = (typeof AudioContext !== 'undefined' && target instanceof AudioContext);
    if ((isMediaEl || isAudioCtx) && ev.type) {
      return 'media ' + ev.type + ' event';
    }
  }
  if (ev.type) return ev.type + ' event';
  return 'unknown event';
}

function decodeMediaError(mediaErr) {
  var phrase;
  switch (mediaErr.code) {
    case 1: phrase = 'media load aborted'; break;
    case 2: phrase = 'media network error'; break;
    case 3: phrase = 'media decode failed (codec unsupported?)'; break;
    case 4: phrase = 'media source not supported'; break;
    default: phrase = 'media error code ' + mediaErr.code; break;
  }
  if (mediaErr.message) phrase += ': ' + mediaErr.message;
  return phrase;
}

if (typeof window !== 'undefined') {
  window.describeError = describeError;
}

/**
 * Plain-language user copy for translator errors.
 *
 * The two mappers here translate the developer-facing string produced by
 * describeError() into a one-sentence, sentence-case English explanation
 * with a concrete next step. They intentionally drop class names, codec
 * names, and module paths so nothing like "AbortError",
 * "decodeAudioData", or "/onnxruntime-web/..." reaches the UI.
 *
 * The dev string is what gets logged to the console; the mapper output is
 * what the user sees. Keep them in sync with the categories listed in
 * docs/translator-fix.
 *
 * Usage (ESM):
 *   import { userMessageForUpload, userMessageForMic } from '/lib/error-copy.js';
 *
 * Usage (legacy script): the module also assigns itself to
 *   window.userMessageForUpload / window.userMessageForMic so non-module
 *   scripts can use it after the importing <script type="module"> has run.
 */

export function userMessageForUpload(internalMsg) {
  var m = String(internalMsg == null ? '' : internalMsg).toLowerCase();

  if (m.indexOf('user cancelled') !== -1
      || m.indexOf('operation aborted') !== -1
      || m.indexOf('aborterror') !== -1
      || m === 'aborted') {
    return 'Upload cancelled.';
  }

  if (m.indexOf('decode timeout') !== -1 || m.indexOf('timeout') !== -1) {
    return 'Decoding took too long. Try a shorter clip (under 60 seconds).';
  }

  if (m.indexOf('too large') !== -1
      || m.indexOf('upload limit') !== -1
      || m.indexOf('file size') !== -1) {
    return 'This file is over the 50 MB upload limit. Try a shorter clip or compress it first.';
  }

  if (m.indexOf('codec') !== -1
      || m.indexOf('decode failed') !== -1
      || m.indexOf('media decode') !== -1
      || m.indexOf('media format') !== -1
      || m.indexOf('source not supported') !== -1
      || m.indexOf('unsupported') !== -1
      || m.indexOf('format unsupported') !== -1) {
    return "This file uses a codec your browser can't decode in-page. Try MP4 or WebM, or upload an audio-only file like MP3.";
  }

  if (m.indexOf('/models/') !== -1
      || m.indexOf('/onnxruntime-web/') !== -1
      || m.indexOf('speech model') !== -1
      || m.indexOf('audio decoder') !== -1
      || m.indexOf('failed to fetch') !== -1
      || m.indexOf('failed to load') !== -1) {
    return "We couldn't load the speech model. Check your connection and try again.";
  }

  if (m.indexOf('worker') !== -1
      || m.indexOf('terminated') !== -1
      || m.indexOf('unreachable') !== -1) {
    return 'The browser worker that processes audio crashed. Reload the page and try again.';
  }

  if (m.indexOf('network') !== -1 || m.indexOf('offline') !== -1) {
    return 'Network error while uploading. Check your connection and retry.';
  }

  return "Couldn't process this file. Try a different format or reload the page.";
}

export function userMessageForMic(internalMsg) {
  var m = String(internalMsg == null ? '' : internalMsg).toLowerCase();

  if (m.indexOf('notallowed') !== -1
      || m.indexOf('permission') !== -1
      || m.indexOf('denied') !== -1) {
    return "Microphone access was blocked. Allow microphone access in your browser's site settings and reload.";
  }

  if (m.indexOf('notfound') !== -1
      || m.indexOf('no device') !== -1
      || m.indexOf('overconstrained') !== -1
      || m.indexOf('requested device') !== -1) {
    return 'No microphone found. Connect a microphone and reload the page.';
  }

  if (m.indexOf('aborted by user') !== -1
      || m.indexOf('user cancelled') !== -1
      || m.indexOf('operation aborted') !== -1
      || m === 'aborted') {
    return 'Recording cancelled.';
  }

  if (m.indexOf('/models/') !== -1
      || m.indexOf('/onnxruntime-web/') !== -1
      || m.indexOf('speech model') !== -1
      || m.indexOf('failed to fetch') !== -1
      || m.indexOf('failed to load') !== -1
      || m.indexOf('model load') !== -1) {
    return "We couldn't load the speech model. Check your connection and try again.";
  }

  if (m.indexOf('transcrib') !== -1) {
    return "Couldn't transcribe this clip. Try recording again, or upload an audio file instead.";
  }

  return 'Microphone capture failed. Reload the page and try again.';
}

if (typeof window !== 'undefined') {
  window.userMessageForUpload = userMessageForUpload;
  window.userMessageForMic = userMessageForMic;
}

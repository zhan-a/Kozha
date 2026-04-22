(function () {
  'use strict';

  const CONTAINER_TAGS = new Set([
    'hamnosys_manual',
    'hamnosys_nonmanual',
    'hamgestural_sign',
    'hns_sign',
    'sigml',
  ]);

  let _known = null;

  function getKnownTags() {
    if (_known) return _known;
    try {
      const env = (typeof window !== 'undefined' && typeof window.getCWAEnv === 'function')
        ? window.getCWAEnv()
        : null;
      const defs = env && typeof env.get === 'function' ? env.get('HNSDefs') : null;
      if (defs && defs.hamMap) {
        _known = new Set(Object.keys(defs.hamMap));
        return _known;
      }
    } catch (_) {}
    return null;
  }

  function extractHamTags(outerHTML) {
    const out = [];
    const re = /<(ham[A-Za-z][\w]*)/g;
    let m;
    while ((m = re.exec(outerHTML)) !== null) out.push(m[1]);
    return out;
  }

  function validateHnsSignXml(outerHTML) {
    if (typeof outerHTML !== 'string') {
      return { valid: false, reason: 'not-a-string', unknownTags: [], receivedType: typeof outerHTML };
    }
    if (!outerHTML.trim()) {
      return { valid: false, reason: 'empty', unknownTags: [] };
    }
    if (outerHTML.indexOf('[object Object]') !== -1) {
      return { valid: false, reason: 'object-literal-in-xml', unknownTags: [] };
    }
    const known = getKnownTags();
    const tags = extractHamTags(outerHTML);
    if (!known) {
      // CWASA not yet loaded — we cannot confirm the tag set. Allow, and let the
      // post-build literal-object check catch any runtime breakage.
      return { valid: true, unknownTags: [], unchecked: true };
    }
    const unknown = [];
    for (const t of tags) {
      if (CONTAINER_TAGS.has(t)) continue;
      if (!known.has(t)) unknown.push(t);
    }
    const uniq = Array.from(new Set(unknown));
    return { valid: uniq.length === 0, unknownTags: uniq };
  }

  function sigmlHasObjectLiteral(str) {
    return typeof str === 'string' && str.indexOf('[object Object]') !== -1;
  }

  function coerceTranslationResult(value, fallback) {
    if (typeof value === 'string') return value;
    if (value == null) return fallback;
    if (typeof value === 'object') {
      for (const k of ['translated', 'text', 'translation', 'translatedText']) {
        if (typeof value[k] === 'string') return value[k];
      }
    }
    return fallback;
  }

  const api = {
    validateHnsSignXml,
    sigmlHasObjectLiteral,
    coerceTranslationResult,
    getKnownTags,
    _containerTags: CONTAINER_TAGS,
  };

  if (typeof window !== 'undefined') window.SigmlValidator = api;
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})();

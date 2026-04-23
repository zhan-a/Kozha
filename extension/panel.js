var glossToSign = new Map();
var letterToSign = new Map();
var conceptToGloss = new Map();
var dbLoaded = false;
var currentLang = "bsl";
var playbackSpeed = 1.0;
var cwasaLoading = false;
var cwasaScriptLoaded = false;
var pendingAction = null;

function loadCwasaScript(callback) {
  if (cwasaScriptLoaded) { callback(); return; }
  if (cwasaLoading) { pendingAction = callback; return; }
  cwasaLoading = true;
  var s = document.createElement("script");
  s.src = "cwa/allcsa.js";
  s.onload = function() {
    cwasaScriptLoaded = true;
    cwasaLoading = false;
    callback();
    if (pendingAction) { pendingAction(); pendingAction = null; }
  };
  s.onerror = function() {
    cwasaLoading = false;
    window.parent.postMessage({ type: "cwasa_failed" }, "*");
    loadDatabase();
    if (pendingAction) { pendingAction(); pendingAction = null; }
  };
  document.head.appendChild(s);
}

var SIGN_LANG_DB = {
  bsl:      { sigml: ['/data/hamnosys_bsl_version1.sigml'], csv: '/data/hamnosys_bsl.csv', alphabet: '/data/bsl_alphabet_sigml.sigml' },
  asl:      { sigml: ['/data/hamnosys_bsl_version1.sigml'], csv: '/data/hamnosys_bsl.csv', alphabet: '/data/asl_alphabet_sigml.sigml' },
  dgs:      { sigml: ['/data/German_SL_DGS.sigml'], csv: '/data/hamnosys_dgs.csv', alphabet: '/data/dgs_alphabet_sigml.sigml' },
  lsf:      { sigml: ['/data/French_SL_LSF.sigml'], csv: '/data/hamnosys_lsf.csv', alphabet: '/data/lsf_alphabet_sigml.sigml' },
  pjm:      { sigml: ['/data/Polish_SL_PJM.sigml'], csv: '/data/hamnosys_pjm.csv', alphabet: '/data/pjm_alphabet_sigml.sigml' },
  gsl:      { sigml: ['/data/Greek_SL_GSL.sigml'], csv: '/data/hamnosys_gsl.csv', alphabet: null },
  ngt:      { sigml: ['/data/Dutch_SL_NGT.sigml'], csv: '/data/hamnosys_ngt.csv', alphabet: '/data/ngt_alphabet_sigml.sigml' },
  algerian: { sigml: ['/data/Algerian_SL.sigml'], csv: null, alphabet: null },
  bangla:   { sigml: ['/data/Bangla_SL.sigml'], csv: null, alphabet: null },
  fsl:      { sigml: ['/data/Filipino_SL.sigml'], csv: null, alphabet: null },
  isl:      { sigml: ['/data/Indian_SL.sigml'], csv: null, alphabet: null },
  kurdish:  { sigml: ['/data/Kurdish_SL.sigml'], csv: null, alphabet: null },
  vsl:      { sigml: ['/data/Vietnamese_SL.sigml'], csv: null, alphabet: null },
};

var SIGN_LANG_LABELS = {
  bsl: "BSL", asl: "ASL", dgs: "DGS (German)", lsf: "LSF (French)",
  pjm: "PJM (Polish)", gsl: "GSL (Greek)", ngt: "NGT (Dutch)",
  algerian: "Algerian SL", bangla: "Bangla SL", fsl: "Filipino SL",
  isl: "Indian SL", kurdish: "Kurdish SL", vsl: "Vietnamese SL",
};

function glossBase(gloss) {
  return String(gloss).toLowerCase().replace(/\(.*?\)/g,'').replace(/#\d+$/g,'').replace(/\d+[a-z]?\^?$/g,'').replace(/^_num-/g,'').replace(/_\(.*?\)/g,'').replace(/[^a-z0-9À-ɏͰ-ϿЀ-ӿĀ-ſ]+/g,' ').trim();
}

function loadSigmlXml(xmlText) {
  var doc = new DOMParser().parseFromString(xmlText, "application/xml");
  var signs = Array.from(doc.querySelectorAll("hns_sign"));
  for (var i = 0; i < signs.length; i++) {
    var s = signs[i];
    var gloss = (s.getAttribute("gloss") || "").trim().toLowerCase();
    if (!gloss) continue;
    glossToSign.set(gloss, s.outerHTML);
    var base = glossBase(gloss);
    if (base && base !== gloss) glossToSign.set(base, s.outerHTML);
  }
}

function loadAlphabetXml(xmlText) {
  var doc = new DOMParser().parseFromString(xmlText, "application/xml");
  var signs = Array.from(doc.querySelectorAll("hns_sign"));
  for (var i = 0; i < signs.length; i++) {
    var s = signs[i];
    var gloss = (s.getAttribute("gloss") || "").trim();
    if (gloss.length === 1 && /[A-Z]/.test(gloss)) {
      letterToSign.set(gloss, s.outerHTML);
    }
  }
}

function parseTable(txt) {
  var lines = txt.split("\n").filter(function(l) { return l.trim(); });
  if (!lines.length) return { header: [], rows: [] };
  var sep = lines[0].indexOf("\t") >= 0 ? "\t" : ",";
  var header = lines[0].split(sep).map(function(h) { return h.trim().toLowerCase(); });
  var rows = [];
  for (var i = 1; i < lines.length; i++) {
    var cols = lines[i].split(sep);
    var row = {};
    for (var j = 0; j < header.length; j++) {
      row[header[j]] = (cols[j] || "").trim();
    }
    rows.push(row);
  }
  return { header: header, rows: rows };
}

function loadConceptCsv(txt) {
  var parsed = parseTable(txt);
  for (var i = 0; i < parsed.rows.length; i++) {
    var r = parsed.rows[i];
    var concept = String(r.concept || "").toLowerCase().trim();
    var gloss = String(r.gloss || "").toLowerCase().trim();
    if (concept && gloss) {
      conceptToGloss.set(glossBase(concept), gloss);
    }
  }
}

var API_BASE = "https://kozha-translate.com";

function loadDatabase(lang) {
  lang = lang || currentLang;
  var db = SIGN_LANG_DB[lang];
  if (!db) return;

  glossToSign.clear();
  letterToSign.clear();
  conceptToGloss.clear();
  dbLoaded = false;

  var fetches = db.sigml.map(function(url) {
    return fetch(API_BASE + url).then(function(r) { return r.text(); });
  });

  if (db.csv) {
    fetches.push(fetch(API_BASE + db.csv).then(function(r) { return r.text(); }));
  }

  if (db.alphabet) {
    fetches.push(fetch(API_BASE + db.alphabet).then(function(r) { return r.text(); }));
  }

  Promise.all(fetches).then(function(results) {
    var idx = 0;
    for (var i = 0; i < db.sigml.length; i++) {
      loadSigmlXml(results[idx++]);
    }
    if (db.csv) loadConceptCsv(results[idx++]);
    if (db.alphabet) loadAlphabetXml(results[idx++]);
    dbLoaded = true;
    console.log("[Kozha panel] database loaded:", lang, "signs:", glossToSign.size, "letters:", letterToSign.size);
    window.parent.postMessage({ type: "db_ready", lang: lang }, "*");
    tryPlayPending();
  }).catch(function(err) {
    console.error("[Kozha panel] database load failed:", err);
    dbLoaded = true;
    window.parent.postMessage({ type: "db_ready", lang: lang }, "*");
    tryPlayPending();
  });
}

function resolveGloss(token) {
  var t = token.toLowerCase().trim();
  if (t === ".") return null;
  if (glossToSign.has(t)) return glossToSign.get(t);
  var base = glossBase(t);
  if (glossToSign.has(base)) return glossToSign.get(base);
  var mapped = conceptToGloss.get(base);
  if (mapped && glossToSign.has(mapped)) return glossToSign.get(mapped);
  return null;
}

function fingerspellWord(word) {
  var blocks = [];
  var upper = word.toUpperCase();
  for (var i = 0; i < upper.length; i++) {
    var ch = upper[i];
    if (letterToSign.has(ch)) blocks.push(letterToSign.get(ch));
  }
  return blocks;
}

function buildSigml(glosses) {
  var blocks = [];
  for (var i = 0; i < glosses.length; i++) {
    var resolved = resolveGloss(glosses[i]);
    if (resolved) {
      blocks.push(resolved);
    } else if (letterToSign.size > 0) {
      var spelled = fingerspellWord(glosses[i]);
      if (spelled.length > 0) blocks.push.apply(blocks, spelled);
    }
  }
  if (!blocks.length) return null;
  return '<?xml version="1.0" encoding="utf-8"?>\n<sigml>\n' + blocks.join("\n") + "\n</sigml>";
}

var cwasaAvailable = false;
var signQueue = [];
var playingQueue = false;

function playSignQueue() {
  if (playingQueue || signQueue.length === 0) return;
  playingQueue = true;
  playNextSign();
}

function playNextSign() {
  if (signQueue.length === 0) {
    playingQueue = false;
    return;
  }
  var sigml = signQueue.shift();
  if (cwasaAvailable) {
    try { CWASA.playSiGMLText(sigml, 0); } catch(err) {}
  }
  var delay = Math.round(1200 / playbackSpeed);
  setTimeout(playNextSign, delay);
}

function playGlossesWithSpeed(glosses) {
  signQueue = [];
  playingQueue = false;

  if (playbackSpeed === 1.0) {
    var sigml = buildSigml(glosses);
    if (sigml && cwasaAvailable) {
      try { CWASA.playSiGMLText(sigml, 0); } catch(err) {}
    }
    return;
  }

  for (var i = 0; i < glosses.length; i++) {
    var resolved = resolveGloss(glosses[i]);
    if (resolved) {
      signQueue.push('<?xml version="1.0" encoding="utf-8"?>\n<sigml>\n' + resolved + "\n</sigml>");
    } else if (letterToSign.size > 0) {
      var spelled = fingerspellWord(glosses[i]);
      for (var j = 0; j < spelled.length; j++) {
        signQueue.push('<?xml version="1.0" encoding="utf-8"?>\n<sigml>\n' + spelled[j] + "\n</sigml>");
      }
    }
  }
  playSignQueue();
}

function initCwasa() {
  loadCwasaScript(function() {
    if (typeof CWASA === "undefined") {
      window.parent.postMessage({ type: "cwasa_failed" }, "*");
      loadDatabase();
      return;
    }

    setTimeout(function() {
      try {
        CWASA.init({
          useClientConfig: false,
          useCwaConfig: true,
          avSettings: [{
            width: 260,
            height: 220,
            avList: "avs",
            initAv: "luna",
            ambIdle: true,
            allowFrameSteps: false,
            allowSiGMLText: false,
          }],
        });

        CWASA.ready.then(function() {
          cwasaAvailable = true;
          console.log("[Kozha panel] CWASA ready");
          window.parent.postMessage({ type: "cwasa_ready" }, "*");
          loadDatabase();
          tryPlayPending();

          var canvas = document.querySelector("canvas");
          if (canvas) {
            canvas.addEventListener("webglcontextlost", function(e) {
              e.preventDefault();
              cwasaAvailable = false;
              window.parent.postMessage({ type: "cwasa_failed" }, "*");
            });
            canvas.addEventListener("webglcontextrestored", function() {
              cwasaAvailable = true;
              window.parent.postMessage({ type: "cwasa_ready" }, "*");
            });
          }
        });
      } catch(err) {
        window.parent.postMessage({ type: "cwasa_failed" }, "*");
        loadDatabase();
      }
    }, 150);
  });
}

var cwasaInitStarted = false;
var pendingGlosses = null;

function tryPlayPending() {
  if (!pendingGlosses) return;
  if (!cwasaAvailable || !dbLoaded) return;
  var g = pendingGlosses;
  pendingGlosses = null;
  console.log("[Kozha panel] playing queued glosses:", g);
  playGlossesWithSpeed(g);
}

window.addEventListener("message", function(e) {
  if (!e.data || !e.data.type) return;

  if (e.data.type === "play_glosses") {
    console.log("[Kozha panel] received play_glosses:", e.data.glosses, "cwasaAvailable:", cwasaAvailable, "dbLoaded:", dbLoaded);
    if (!cwasaInitStarted) {
      cwasaInitStarted = true;
      initCwasa();
    }
    if (!e.data.glosses || e.data.glosses.length === 0) return;
    if (cwasaAvailable && dbLoaded) {
      try { CWASA.stop(0); } catch(err) {}
      signQueue = [];
      playingQueue = false;
      playGlossesWithSpeed(e.data.glosses);
    } else {
      console.log("[Kozha panel] queuing glosses until ready");
      pendingGlosses = e.data.glosses;
    }
  }

  if (e.data.type === "play_sigml") {
    if (cwasaAvailable) {
      try { CWASA.stop(0); } catch(err) {}
      try { CWASA.playSiGMLText(e.data.sigml, 0); } catch(err) {}
    }
  }

  if (e.data.type === "stop") {
    signQueue = [];
    playingQueue = false;
    if (cwasaAvailable) {
      try { CWASA.stop(0); } catch(err) {}
    }
  }

  if (e.data.type === "switch_language") {
    if (!cwasaInitStarted) {
      cwasaInitStarted = true;
      initCwasa();
    }
    currentLang = e.data.lang;
    loadDatabase(e.data.lang);
  }

  if (e.data.type === "init_cwasa") {
    if (!cwasaInitStarted) {
      cwasaInitStarted = true;
      initCwasa();
    }
  }

  if (e.data.type === "set_speed") {
    playbackSpeed = e.data.speed;
  }
});

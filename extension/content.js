let panelIframe = null;
let translationCache = {};
let currentSegmentIndex = -1;
let segments = [];
let avatarReady = false;
let dbReady = false;
let cwasaFailed = false;
let videoListeners = [];
let isAutoCaption = false;
let currentSignLang = "bsl";
let currentCaptionLang = "";
let playbackSpeed = 1.0;
let settingsOpen = false;
let theaterObservers = [];
let trackedDocListeners = [];
let isWindowTranslating = false;

var SIGN_LANG_LABELS = {
  bsl: "BSL", asl: "ASL", dgs: "DGS (German)", lsf: "LSF (French)",
  pjm: "PJM (Polish)", gsl: "GSL (Greek)", ngt: "NGT (Dutch)",
  algerian: "Algerian SL", bangla: "Bangla SL", fsl: "Filipino SL",
  isl: "Indian SL", kurdish: "Kurdish SL", vsl: "Vietnamese SL",
};

var SEGMENT_THRESHOLD = 200;
var WINDOW_SIZE = 100;

function getVideoId() {
  var params = new URLSearchParams(window.location.search);
  return params.get("v");
}

function isWatchPage() {
  try {
    var url = new URL(location.href);
    return url.pathname === "/watch" && url.searchParams.has("v");
  } catch (e) {
    return false;
  }
}

function isLiveStream() {
  var scripts = document.querySelectorAll("script");
  for (var i = 0; i < scripts.length; i++) {
    var text = scripts[i].textContent;
    var match = text.match(/ytInitialPlayerResponse\s*=\s*(\{.+?\});/s);
    if (!match) continue;
    try {
      var data = JSON.parse(match[1]);
      var details = data?.videoDetails;
      if (details && details.isLiveContent && details.isLive) return true;
    } catch (e) {}
  }
  return false;
}

function extractCaptionTracks() {
  var scripts = document.querySelectorAll("script");
  for (var i = 0; i < scripts.length; i++) {
    var text = scripts[i].textContent;
    var match = text.match(/ytInitialPlayerResponse\s*=\s*(\{.+?\});/s);
    if (!match) continue;
    try {
      var data = JSON.parse(match[1]);
      var tracks =
        data?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
      if (tracks) return tracks;
    } catch (e) {}
  }
  return null;
}

function pickBestTrack(tracks) {
  var manual = tracks.filter(function(t) { return t.kind !== "asr"; });
  var auto = tracks.filter(function(t) { return t.kind === "asr"; });
  isAutoCaption = manual.length === 0;
  var pool = manual.length > 0 ? manual : auto;
  var en = pool.find(function(t) { return t.languageCode === "en"; });
  return en || pool[0];
}

async function fetchTranscript(track) {
  var resp = await fetch(track.baseUrl);
  var xml = await resp.text();
  var parser = new DOMParser();
  var doc = parser.parseFromString(xml, "text/xml");
  var nodes = doc.querySelectorAll("text");
  return Array.from(nodes).map(function(node) {
    return {
      text: node.textContent.replace(/&amp;/g, "&").replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">"),
      start: parseFloat(node.getAttribute("start")),
      duration: parseFloat(node.getAttribute("dur") || "0"),
    };
  });
}

var BATCH_SIZE = 50;

async function translateSegments(segs, sourceLang, videoId) {
  var results = [];
  var total = segs.length;

  for (var i = 0; i < total; i += BATCH_SIZE) {
    var chunk = segs.slice(i, i + BATCH_SIZE);
    var chunkEnd = Math.min(i + BATCH_SIZE, total);
    if (total > BATCH_SIZE) {
      setStatus("Translating " + chunkEnd + "/" + total + "...");
    }

    var chunkResult = await new Promise(function(resolve, reject) {
      chrome.runtime.sendMessage(
        {
          type: "translate_batch",
          segments: chunk.map(function(s) {
            return { text: s.text, start: s.start, duration: s.duration };
          }),
          source_lang: sourceLang,
          video_id: videoId,
        },
        function(resp) {
          if (chrome.runtime.lastError) {
            reject(new Error(chrome.runtime.lastError.message));
            return;
          }
          if (resp && resp.ok) resolve(resp.data);
          else reject(new Error(resp?.error || "Translation failed"));
        }
      );
    });

    results.push.apply(results, chunkResult.results);
  }

  return { results: results };
}

async function translateSlice(startIdx, endIdx, sourceLang) {
  var slice = segments.slice(startIdx, endIdx);
  if (slice.length === 0) return;
  var result = await translateSegments(slice, sourceLang, null);
  result.results.forEach(function(r, i) {
    translationCache[startIdx + i] = r;
  });
}

function evictOutsideWindow(centerIdx) {
  var lo = Math.max(0, centerIdx - WINDOW_SIZE);
  var hi = Math.min(segments.length, centerIdx + WINDOW_SIZE);
  var keys = Object.keys(translationCache);
  for (var k = 0; k < keys.length; k++) {
    var idx = parseInt(keys[k]);
    if (idx < lo || idx >= hi) delete translationCache[idx];
  }
}

async function ensureWindowTranslated(centerIdx) {
  if (isWindowTranslating || centerIdx < 0) return;
  var halfWin = Math.floor(WINDOW_SIZE / 2);
  var lo = Math.max(0, centerIdx - halfWin);
  var hi = Math.min(segments.length, centerIdx + halfWin);

  var needStart = -1;
  var needEnd = -1;
  for (var i = lo; i < hi; i++) {
    if (!translationCache[i]) {
      if (needStart === -1) needStart = i;
      needEnd = i + 1;
    }
  }
  if (needStart === -1) return;

  isWindowTranslating = true;
  try {
    await translateSlice(needStart, needEnd, currentCaptionLang);
    evictOutsideWindow(centerIdx);
  } catch (e) {
    setStatus("Translation error");
  } finally {
    isWindowTranslating = false;
  }
}

function isOfflineError(msg) {
  return msg === "Request timed out" || msg === "Failed to fetch" ||
    msg.indexOf("NetworkError") >= 0 || msg.indexOf("network") >= 0;
}

function setStatus(text) {
  var el = document.getElementById("kozha-status-text");
  if (el) el.textContent = text;
  updateSegmentCounter();
}

function updateSegmentCounter() {
  var el = document.getElementById("kozha-status-lang");
  if (!el) return;
  var parts = [];
  if (currentCaptionLang) parts.push(currentCaptionLang.toUpperCase());
  if (segments.length > 0 && currentSegmentIndex >= 0) {
    parts.push((currentSegmentIndex + 1) + "/" + segments.length);
  } else if (segments.length > 0) {
    parts.push(segments.length + " segs");
  }
  el.textContent = parts.join(" \u00B7 ");
}

function setSubtitle(text) {
  var el = document.getElementById("kozha-subtitle");
  if (el) el.textContent = text;
}

function injectPanel() {
  if (document.getElementById("kozha-panel")) return;

  var panel = document.createElement("div");
  panel.id = "kozha-panel";

  var header = document.createElement("div");
  header.id = "kozha-header";

  var title = document.createElement("span");
  title.id = "kozha-title";
  title.textContent = "Kozha";

  var controls = document.createElement("span");
  controls.id = "kozha-controls";

  var settingsBtn = document.createElement("button");
  settingsBtn.id = "kozha-settings-btn";
  settingsBtn.innerHTML = "&#9881;";
  settingsBtn.addEventListener("click", function(e) {
    e.stopPropagation();
    settingsOpen = !settingsOpen;
    var drawer = document.getElementById("kozha-settings-drawer");
    if (drawer) {
      drawer.classList.toggle("open", settingsOpen);
    }
    settingsBtn.classList.toggle("active", settingsOpen);
  });

  var minBtn = document.createElement("button");
  minBtn.className = "kozha-ctrl-btn";
  minBtn.textContent = "\u2013";
  minBtn.addEventListener("click", function(e) {
    e.stopPropagation();
    var body = document.getElementById("kozha-panel-body");
    body.style.display = body.style.display === "none" ? "flex" : "none";
  });

  var closeBtn = document.createElement("button");
  closeBtn.className = "kozha-ctrl-btn";
  closeBtn.textContent = "\u00D7";
  closeBtn.addEventListener("click", function(e) {
    e.stopPropagation();
    panel.style.display = "none";
    var toggle = document.getElementById("kozha-toggle");
    if (toggle) toggle.style.display = "block";
  });

  controls.appendChild(settingsBtn);
  controls.appendChild(minBtn);
  controls.appendChild(closeBtn);
  header.appendChild(title);
  header.appendChild(controls);

  var body = document.createElement("div");
  body.id = "kozha-panel-body";

  var settingsDrawer = document.createElement("div");
  settingsDrawer.id = "kozha-settings-drawer";

  var langRow = document.createElement("div");
  langRow.className = "kozha-setting-row";
  var langLabel = document.createElement("span");
  langLabel.className = "kozha-setting-label";
  langLabel.textContent = "Sign lang";
  var langSelect = document.createElement("select");
  langSelect.id = "kozha-lang-select";
  Object.keys(SIGN_LANG_LABELS).forEach(function(key) {
    var opt = document.createElement("option");
    opt.value = key;
    opt.textContent = SIGN_LANG_LABELS[key];
    if (key === currentSignLang) opt.selected = true;
    langSelect.appendChild(opt);
  });
  langSelect.addEventListener("change", function() {
    currentSignLang = langSelect.value;
    if (panelIframe && panelIframe.contentWindow) {
      panelIframe.contentWindow.postMessage({ type: "switch_language", lang: currentSignLang }, "*");
    }
  });
  langRow.appendChild(langLabel);
  langRow.appendChild(langSelect);

  var speedRow = document.createElement("div");
  speedRow.className = "kozha-setting-row";
  var speedLabel = document.createElement("span");
  speedLabel.className = "kozha-setting-label";
  speedLabel.textContent = "Speed";
  var speedSelect = document.createElement("select");
  speedSelect.id = "kozha-speed-select";
  [
    { value: "0.5", label: "0.5x" },
    { value: "0.75", label: "0.75x" },
    { value: "1", label: "1x" },
    { value: "1.25", label: "1.25x" },
    { value: "1.5", label: "1.5x" },
    { value: "2", label: "2x" },
  ].forEach(function(item) {
    var opt = document.createElement("option");
    opt.value = item.value;
    opt.textContent = item.label;
    if (parseFloat(item.value) === playbackSpeed) opt.selected = true;
    speedSelect.appendChild(opt);
  });
  speedSelect.addEventListener("change", function() {
    playbackSpeed = parseFloat(speedSelect.value);
    if (panelIframe && panelIframe.contentWindow) {
      panelIframe.contentWindow.postMessage({ type: "set_speed", speed: playbackSpeed }, "*");
    }
  });
  speedRow.appendChild(speedLabel);
  speedRow.appendChild(speedSelect);

  settingsDrawer.appendChild(langRow);
  settingsDrawer.appendChild(speedRow);

  var iframe = document.createElement("iframe");
  iframe.id = "kozha-avatar-frame";
  iframe.src = chrome.runtime.getURL("panel.html");
  iframe.allow = "autoplay";
  panelIframe = iframe;

  var subtitle = document.createElement("div");
  subtitle.id = "kozha-subtitle";
  subtitle.textContent = "Loading captions...";

  var status = document.createElement("div");
  status.id = "kozha-status";

  var statusText = document.createElement("span");
  statusText.id = "kozha-status-text";
  statusText.textContent = "Initializing";

  var statusLang = document.createElement("span");
  statusLang.id = "kozha-status-lang";

  status.appendChild(statusText);
  status.appendChild(statusLang);

  body.appendChild(settingsDrawer);
  body.appendChild(iframe);
  body.appendChild(subtitle);
  body.appendChild(status);
  panel.appendChild(header);
  panel.appendChild(body);

  var toggle = document.createElement("button");
  toggle.id = "kozha-toggle";
  toggle.textContent = "K";
  toggle.style.display = "none";
  toggle.addEventListener("click", function() {
    panel.style.display = "block";
    toggle.style.display = "none";
  });

  document.body.appendChild(panel);
  document.body.appendChild(toggle);

  makeDraggable(panel, header);
}

function clampToViewport(panel) {
  var rect = panel.getBoundingClientRect();
  var vw = window.innerWidth;
  var vh = window.innerHeight;
  var left = rect.left;
  var top = rect.top;
  var changed = false;

  if (left < 0) { left = 0; changed = true; }
  if (top < 0) { top = 0; changed = true; }
  if (left + rect.width > vw) { left = vw - rect.width; changed = true; }
  if (top + rect.height > vh) { top = vh - rect.height; changed = true; }

  if (changed) {
    panel.style.left = left + "px";
    panel.style.top = top + "px";
    panel.style.right = "auto";
    panel.style.bottom = "auto";
  }
}

function addDocListener(target, event, handler) {
  target.addEventListener(event, handler);
  trackedDocListeners.push({ target: target, event: event, handler: handler });
}

function removeDocListeners() {
  trackedDocListeners.forEach(function(entry) {
    entry.target.removeEventListener(entry.event, entry.handler);
  });
  trackedDocListeners = [];
}

function makeDraggable(panel, handle) {
  var dragging = false;
  var offsetX, offsetY;

  handle.addEventListener("mousedown", function(e) {
    if (e.button !== 0) return;
    dragging = true;
    offsetX = e.clientX - panel.getBoundingClientRect().left;
    offsetY = e.clientY - panel.getBoundingClientRect().top;
    e.preventDefault();
  });

  addDocListener(document, "mousemove", function(e) {
    if (!dragging) return;
    var vw = window.innerWidth;
    var vh = window.innerHeight;
    var rect = panel.getBoundingClientRect();
    var newLeft = e.clientX - offsetX;
    var newTop = e.clientY - offsetY;

    newLeft = Math.max(0, Math.min(newLeft, vw - rect.width));
    newTop = Math.max(0, Math.min(newTop, vh - rect.height));

    panel.style.right = "auto";
    panel.style.bottom = "auto";
    panel.style.left = newLeft + "px";
    panel.style.top = newTop + "px";
  });

  addDocListener(document, "mouseup", function() {
    dragging = false;
  });
}

function findSegmentIndex(time) {
  var lo = 0;
  var hi = segments.length - 1;
  while (lo <= hi) {
    var mid = (lo + hi) >> 1;
    var seg = segments[mid];
    if (time < seg.start) {
      hi = mid - 1;
    } else if (time >= seg.start + seg.duration) {
      lo = mid + 1;
    } else {
      return mid;
    }
  }
  return -1;
}

function sendToAvatar(glosses) {
  if (cwasaFailed) {
    showTextGlosses(glosses);
    return;
  }
  if (panelIframe && panelIframe.contentWindow) {
    panelIframe.contentWindow.postMessage(
      { type: "play_glosses", glosses: glosses },
      "*"
    );
  }
}

function showTextGlosses(glosses) {
  var el = document.getElementById("kozha-gloss-text");
  if (el) el.textContent = glosses.join(" ");
}

function stopAvatar() {
  if (panelIframe && panelIframe.contentWindow) {
    panelIframe.contentWindow.postMessage({ type: "stop" }, "*");
  }
}

function addVideoListener(video, event, handler) {
  video.addEventListener(event, handler);
  videoListeners.push({ video: video, event: event, handler: handler });
}

function removeVideoListeners() {
  videoListeners.forEach(function(entry) {
    entry.video.removeEventListener(entry.event, entry.handler);
  });
  videoListeners = [];
}

function startVideoSync() {
  var video = document.querySelector("video");
  if (!video) return;
  var useWindow = segments.length > SEGMENT_THRESHOLD;

  addVideoListener(video, "timeupdate", function() {
    var time = video.currentTime;
    var idx = findSegmentIndex(time);
    if (idx === currentSegmentIndex) return;
    currentSegmentIndex = idx;
    updateSegmentCounter();

    if (idx < 0) {
      setSubtitle("");
      return;
    }

    var seg = segments[idx];
    setSubtitle(seg.text);

    var cached = translationCache[idx];
    if (cached) sendToAvatar(cached.glosses);

    if (useWindow) ensureWindowTranslated(idx);
  });

  addVideoListener(video, "pause", function() {
    stopAvatar();
  });

  addVideoListener(video, "seeking", function() {
    currentSegmentIndex = -1;
    stopAvatar();
  });

  addVideoListener(video, "play", function() {
    currentSegmentIndex = -1;
  });
}

function detectTextDirection(langCode) {
  var rtlLangs = ["ar", "he", "fa", "ur", "yi", "ps", "sd"];
  return rtlLangs.indexOf(langCode) >= 0 ? "rtl" : "ltr";
}

function repositionPanel() {
  var panel = document.getElementById("kozha-panel");
  if (!panel || panel.style.display === "none") return;
  clampToViewport(panel);
}

function observeTheaterAndFullscreen() {
  var ytApp = document.querySelector("ytd-app");
  if (ytApp) {
    var obs1 = new MutationObserver(repositionPanel);
    obs1.observe(ytApp, { attributes: true, attributeFilter: ["class", "masthead-hidden"] });
    theaterObservers.push(obs1);
  }

  var player = document.getElementById("movie_player");
  if (player) {
    var obs2 = new MutationObserver(repositionPanel);
    obs2.observe(player, { attributes: true, attributeFilter: ["class"] });
    theaterObservers.push(obs2);
  }

  addDocListener(document, "fullscreenchange", function() {
    var panel = document.getElementById("kozha-panel");
    var toggle = document.getElementById("kozha-toggle");
    if (document.fullscreenElement) {
      if (panel) document.fullscreenElement.appendChild(panel);
      if (toggle) document.fullscreenElement.appendChild(toggle);
    } else {
      if (panel) document.body.appendChild(panel);
      if (toggle) document.body.appendChild(toggle);
    }
    setTimeout(repositionPanel, 100);
  });
}

window.addEventListener("message", function(e) {
  if (!e.data || !e.data.type) return;
  if (e.data.type === "cwasa_ready") avatarReady = true;
  if (e.data.type === "cwasa_failed") {
    cwasaFailed = true;
    var frame = document.getElementById("kozha-avatar-frame");
    if (frame) frame.style.display = "none";
    var body = document.getElementById("kozha-panel-body");
    if (body) {
      var glossDiv = document.createElement("div");
      glossDiv.id = "kozha-gloss-text";
      body.insertBefore(glossDiv, body.firstChild);
    }
    setStatus("Text-only mode");
  }
  if (e.data.type === "db_ready") {
    dbReady = true;
    setStatus(isAutoCaption ? "Ready (auto captions)" : "Ready");
  }
});

window.addEventListener("resize", repositionPanel);

async function init() {
  if (!isWatchPage()) return;

  avatarReady = false;
  dbReady = false;
  cwasaFailed = false;
  currentSegmentIndex = -1;
  translationCache = {};
  segments = [];
  isAutoCaption = false;
  currentCaptionLang = "";
  settingsOpen = false;
  isWindowTranslating = false;

  injectPanel();
  observeTheaterAndFullscreen();

  if (isLiveStream()) {
    setStatus("Live streams not supported");
    setSubtitle("Live streams are not supported");
    return;
  }

  var tracks = extractCaptionTracks();

  if (!tracks || tracks.length === 0) {
    setStatus("No captions available");
    setSubtitle("This video has no captions");
    return;
  }

  var track = pickBestTrack(tracks);
  currentCaptionLang = track.languageCode;
  var dir = detectTextDirection(track.languageCode);
  var subtitleEl = document.getElementById("kozha-subtitle");
  if (subtitleEl) subtitleEl.dir = dir;

  setStatus(isAutoCaption ? "Fetching captions (auto)..." : "Fetching captions...");

  try {
    segments = await fetchTranscript(track);
  } catch (e) {
    setStatus("Failed to load captions");
    return;
  }

  updateSegmentCounter();
  setStatus("Translating...");

  try {
    if (segments.length > SEGMENT_THRESHOLD) {
      var initialEnd = Math.min(WINDOW_SIZE, segments.length);
      await translateSlice(0, initialEnd, track.languageCode);
    } else {
      var videoId = segments.length <= BATCH_SIZE ? getVideoId() : null;
      var result = await translateSegments(segments, track.languageCode, videoId);
      result.results.forEach(function(r, i) {
        translationCache[i] = r;
      });
      if (segments.length > BATCH_SIZE) {
        var vid = getVideoId();
        if (vid) {
          chrome.runtime.sendMessage({
            type: "cache_results",
            video_id: vid,
            data: { results: result.results },
          });
        }
      }
    }
    setStatus(dbReady
      ? (isAutoCaption ? "Ready (auto captions)" : "Ready")
      : "Loading signs...");
  } catch (e) {
    if (isOfflineError(e.message)) {
      setStatus("Cannot reach server");
      setSubtitle("Check your internet connection and try again");
    } else {
      setStatus("Translation failed");
      setSubtitle(e.message);
    }
    return;
  }

  startVideoSync();
}

function cleanup() {
  removeVideoListeners();
  removeDocListeners();
  theaterObservers.forEach(function(obs) { obs.disconnect(); });
  theaterObservers = [];
  currentSegmentIndex = -1;
  translationCache = {};
  segments = [];
  avatarReady = false;
  dbReady = false;
  cwasaFailed = false;
  isAutoCaption = false;
  currentCaptionLang = "";
  settingsOpen = false;
  isWindowTranslating = false;
  var panel = document.getElementById("kozha-panel");
  if (panel) panel.remove();
  var toggle = document.getElementById("kozha-toggle");
  if (toggle) toggle.remove();
  panelIframe = null;
}

var lastUrl = location.href;
var navObserver = new MutationObserver(function() {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    cleanup();
    if (isWatchPage()) {
      setTimeout(init, 500);
    }
  }
});
navObserver.observe(document.body, { childList: true, subtree: true });

document.addEventListener("yt-navigate-finish", function() {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    cleanup();
    if (isWatchPage()) {
      setTimeout(init, 500);
    }
  }
});

init();

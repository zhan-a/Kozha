let translationCache = {};
let currentSegmentIndex = -1;
let segments = [];
let videoListeners = [];
let isAutoCaption = false;
let currentCaptionLang = "";
let theaterObservers = [];
let isWindowTranslating = false;
let pendingInit = null;

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

function getPlayerResponseFromScripts() {
  var vid = getVideoId();
  var scripts = document.querySelectorAll("script");
  for (var i = 0; i < scripts.length; i++) {
    var text = scripts[i].textContent;
    var match = text.match(/ytInitialPlayerResponse\s*=\s*(\{.+?\});/s);
    if (!match) continue;
    try {
      var data = JSON.parse(match[1]);
      if (vid && data.videoDetails && data.videoDetails.videoId !== vid) continue;
      return data;
    } catch (e) {}
  }
  return null;
}

function getPlayerResponseFromPage() {
  return new Promise(function(resolve) {
    var eid = "kozha-" + Math.random().toString(36).slice(2);
    var handled = false;
    function done(val) {
      if (handled) return;
      handled = true;
      resolve(val);
    }
    window.addEventListener(eid, function h(e) {
      window.removeEventListener(eid, h);
      try { done(JSON.parse(e.detail)); } catch(x) { done(null); }
    });
    var s = document.createElement("script");
    s.textContent = '!function(){try{window.dispatchEvent(new CustomEvent("' + eid + '",{detail:JSON.stringify(window.ytInitialPlayerResponse||null)}))}catch(e){window.dispatchEvent(new CustomEvent("' + eid + '",{detail:"null"}))}}()';
    document.documentElement.appendChild(s);
    s.remove();
    setTimeout(function() { done(null); }, 1000);
  });
}

async function getPlayerResponse(retries) {
  var vid = getVideoId();
  for (var attempt = 0; attempt <= retries; attempt++) {
    if (attempt > 0) {
      setStatus("Waiting for page data...");
      await new Promise(function(r) { setTimeout(r, attempt * 800); });
    }

    var data = getPlayerResponseFromScripts();
    if (data && (!vid || !data.videoDetails || data.videoDetails.videoId === vid)) {
      return data;
    }

    data = await getPlayerResponseFromPage();
    if (data && (!vid || !data.videoDetails || data.videoDetails.videoId === vid)) {
      return data;
    }
  }
  return null;
}

function extractTracksFromResponse(data) {
  if (!data) return null;
  var tracks = data && data.captions && data.captions.playerCaptionsTracklistRenderer &&
    data.captions.playerCaptionsTracklistRenderer.captionTracks;
  return tracks && tracks.length > 0 ? tracks : null;
}

function isLiveFromResponse(data) {
  if (!data || !data.videoDetails) return false;
  return !!(data.videoDetails.isLiveContent && data.videoDetails.isLive);
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
  var url = track.baseUrl;
  var jsonUrl = url + (url.indexOf("?") >= 0 ? "&" : "?") + "fmt=json3";
  console.log("[Kozha YT] fetching transcript from:", jsonUrl.substring(0, 150));
  var resp = await fetch(jsonUrl);
  console.log("[Kozha YT] transcript response:", resp.status, resp.headers.get("content-type"));
  var text = await resp.text();
  console.log("[Kozha YT] response length:", text.length, "first 200 chars:", text.substring(0, 200));

  if (text.trim().charAt(0) === "{") {
    try {
      var data = JSON.parse(text);
      var events = data.events || [];
      console.log("[Kozha YT] json events:", events.length);
      var segments = [];
      for (var i = 0; i < events.length; i++) {
        var ev = events[i];
        if (!ev.segs) continue;
        var txt = ev.segs.map(function(s) { return s.utf8 || ""; }).join("").trim();
        if (!txt) continue;
        segments.push({
          text: txt,
          start: (ev.tStartMs || 0) / 1000,
          duration: (ev.dDurationMs || 0) / 1000,
        });
      }
      if (segments.length > 0) return segments;
    } catch (e) { console.error("[Kozha YT] json parse failed:", e); }
  }

  var parser = new DOMParser();
  var doc = parser.parseFromString(text, "text/xml");
  var nodes = doc.querySelectorAll("text, p");
  console.log("[Kozha YT] xml fallback nodes found:", nodes.length);
  return Array.from(nodes).map(function(node) {
    var start = node.getAttribute("start") || node.getAttribute("t");
    var dur = node.getAttribute("dur") || node.getAttribute("d") || "0";
    var startSec = parseFloat(start);
    if (start && start.length > 5 && !isNaN(startSec)) {
      if (startSec > 100000) startSec = startSec / 1000;
    }
    var durSec = parseFloat(dur);
    if (dur && dur.length > 5 && !isNaN(durSec)) {
      if (durSec > 100000) durSec = durSec / 1000;
    }
    return {
      text: node.textContent.replace(/&amp;/g, "&").replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">"),
      start: startSec,
      duration: durSec,
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
          else reject(new Error(resp && resp.error ? resp.error : "Translation failed"));
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
  Kozha.setStatus(text);
  updateSegmentCounter();
}

function updateSegmentCounter() {
  var parts = [];
  if (currentCaptionLang) parts.push(currentCaptionLang.toUpperCase());
  if (segments.length > 0 && currentSegmentIndex >= 0) {
    parts.push((currentSegmentIndex + 1) + "/" + segments.length);
  } else if (segments.length > 0) {
    parts.push(segments.length + " segs");
  }
  Kozha.setStatusLang(parts.join(" \u00B7 "));
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
  var video = document.querySelector("#movie_player video") ||
    document.querySelector("video");
  if (!video) {
    console.log("[Kozha YT] no video element found, retrying in 500ms");
    setTimeout(startVideoSync, 500);
    return;
  }
  console.log("[Kozha YT] startVideoSync attached to video. segments:", segments.length, "cached:", Object.keys(translationCache).length, "video currentTime:", video.currentTime);
  var useWindow = segments.length > SEGMENT_THRESHOLD;

  addVideoListener(video, "timeupdate", function() {
    var time = video.currentTime;
    var idx = findSegmentIndex(time);
    if (idx === currentSegmentIndex) return;
    currentSegmentIndex = idx;
    updateSegmentCounter();

    if (idx < 0) {
      Kozha.setSubtitle("");
      return;
    }

    var seg = segments[idx];
    Kozha.setSubtitle(seg.text);

    var cached = translationCache[idx];
    if (cached) {
      console.log("[Kozha YT] segment", idx, "→ glosses:", cached.glosses);
      Kozha.sendToAvatar(cached.glosses);
    } else {
      console.log("[Kozha YT] segment", idx, "has no cached glosses yet");
    }

    if (useWindow) ensureWindowTranslated(idx);
  });

  addVideoListener(video, "pause", function() {
    Kozha.stopAvatar();
    setStatus("Paused");
  });

  addVideoListener(video, "seeking", function() {
    currentSegmentIndex = -1;
    Kozha.stopAvatar();
  });

  addVideoListener(video, "play", function() {
    currentSegmentIndex = -1;
    setStatus(isAutoCaption ? "Synced (auto captions)" : "Synced");
  });
}

function detectTextDirection(langCode) {
  var rtlLangs = ["ar", "he", "fa", "ur", "yi", "ps", "sd"];
  return rtlLangs.indexOf(langCode) >= 0 ? "rtl" : "ltr";
}

function observeTheaterAndFullscreen() {
  var ytApp = document.querySelector("ytd-app");
  if (ytApp) {
    var obs1 = new MutationObserver(Kozha.repositionPanel);
    obs1.observe(ytApp, { attributes: true, attributeFilter: ["class", "masthead-hidden"] });
    theaterObservers.push(obs1);
  }

  var player = document.getElementById("movie_player");
  if (player) {
    var obs2 = new MutationObserver(Kozha.repositionPanel);
    obs2.observe(player, { attributes: true, attributeFilter: ["class"] });
    theaterObservers.push(obs2);
  }

  Kozha.addDocListener(document, "fullscreenchange", function() {
    var panel = document.getElementById("kozha-panel");
    var toggle = document.getElementById("kozha-toggle");
    if (document.fullscreenElement) {
      if (panel) document.fullscreenElement.appendChild(panel);
      if (toggle) document.fullscreenElement.appendChild(toggle);
    } else {
      if (panel) document.body.appendChild(panel);
      if (toggle) document.body.appendChild(toggle);
    }
    setTimeout(Kozha.repositionPanel, 100);
  });
}

window.addEventListener("message", function(e) {
  if (!e.data || !e.data.type) return;
  if (e.data.type === "cwasa_failed") setStatus("Text-only mode");
  if (e.data.type === "db_ready") {
    setStatus(isAutoCaption ? "Ready (auto captions)" : "Ready");
  }
});

async function init() {
  if (!isWatchPage()) { console.log("[Kozha YT] not a watch page, skipping init"); return; }
  console.log("[Kozha YT] init starting for:", window.location.href);

  currentSegmentIndex = -1;
  translationCache = {};
  segments = [];
  isAutoCaption = false;
  currentCaptionLang = "";
  isWindowTranslating = false;

  Kozha.avatarReady = false;
  Kozha.dbReady = false;
  Kozha.cwasaFailed = false;
  Kozha._settingsOpen = false;

  Kozha.injectPanel();
  Kozha.setSubtitle("Loading captions...");
  setStatus("Extracting captions...");
  observeTheaterAndFullscreen();

  var playerResponse = await getPlayerResponse(2);

  if (isLiveFromResponse(playerResponse)) {
    setStatus("Live — not supported");
    Kozha.setSubtitle("Live streams are not supported");
    return;
  }

  var tracks = extractTracksFromResponse(playerResponse);
  console.log("[Kozha YT] caption tracks found:", tracks ? tracks.length : 0);

  if (!tracks) {
    setStatus("No captions found");
    Kozha.setSubtitle("This video has no captions");
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
    console.log("[Kozha YT] fetched", segments.length, "transcript segments");
  } catch (e) {
    console.error("[Kozha YT] fetchTranscript failed:", e);
    setStatus("Failed to load captions");
    return;
  }

  updateSegmentCounter();
  setStatus("Translating...");

  try {
    if (segments.length > SEGMENT_THRESHOLD) {
      var initialEnd = Math.min(WINDOW_SIZE, segments.length);
      console.log("[Kozha YT] translating initial window 0-" + initialEnd);
      await translateSlice(0, initialEnd, track.languageCode);
    } else {
      var videoId = segments.length <= BATCH_SIZE ? getVideoId() : null;
      console.log("[Kozha YT] translating all", segments.length, "segments");
      var result = await translateSegments(segments, track.languageCode, videoId);
      console.log("[Kozha YT] translation returned", result.results.length, "results");
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
    console.log("[Kozha YT] translationCache has", Object.keys(translationCache).length, "entries");
    setStatus(Kozha.dbReady
      ? (isAutoCaption ? "Ready (auto captions)" : "Ready")
      : "Loading signs...");
  } catch (e) {
    console.error("[Kozha YT] translation error:", e);
    if (isOfflineError(e.message)) {
      setStatus("Cannot reach server");
      Kozha.setSubtitle("Check your internet connection and try again");
    } else {
      setStatus("Translation failed");
      Kozha.setSubtitle(e.message);
    }
    return;
  }

  startVideoSync();
}

function cleanup() {
  removeVideoListeners();
  Kozha.removeDocListeners();
  theaterObservers.forEach(function(obs) { obs.disconnect(); });
  theaterObservers = [];
  currentSegmentIndex = -1;
  translationCache = {};
  segments = [];
  isAutoCaption = false;
  currentCaptionLang = "";
  isWindowTranslating = false;
  Kozha.avatarReady = false;
  Kozha.dbReady = false;
  Kozha.cwasaFailed = false;
  Kozha._settingsOpen = false;
  Kozha.removePanel();
}

function onNavigate() {
  if (location.href === lastUrl) return;
  lastUrl = location.href;
  if (pendingInit) clearTimeout(pendingInit);
  cleanup();
  if (isWatchPage()) {
    pendingInit = setTimeout(function() {
      pendingInit = null;
      init();
    }, 500);
  }
}

var lastUrl = location.href;
var navObserver = new MutationObserver(onNavigate);
navObserver.observe(document.body, { childList: true, subtree: true });

document.addEventListener("yt-navigate-finish", onNavigate);

init();

let panelIframe = null;
let translationCache = {};
let currentSegmentIndex = -1;
let segments = [];
let avatarReady = false;
let dbReady = false;

function getVideoId() {
  var params = new URLSearchParams(window.location.search);
  return params.get("v");
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

async function translateSegments(segs, sourceLang) {
  return new Promise(function(resolve, reject) {
    chrome.runtime.sendMessage(
      {
        type: "translate_batch",
        segments: segs.map(function(s) {
          return { text: s.text, start: s.start, duration: s.duration };
        }),
        source_lang: sourceLang,
        video_id: getVideoId(),
      },
      function(resp) {
        if (resp && resp.ok) resolve(resp.data);
        else reject(new Error(resp?.error || "Translation failed"));
      }
    );
  });
}

function setStatus(text) {
  var el = document.getElementById("kozha-status");
  if (el) el.textContent = text;
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

  controls.appendChild(minBtn);
  controls.appendChild(closeBtn);
  header.appendChild(title);
  header.appendChild(controls);

  var body = document.createElement("div");
  body.id = "kozha-panel-body";

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
  status.textContent = "Initializing";

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

  document.addEventListener("mousemove", function(e) {
    if (!dragging) return;
    panel.style.right = "auto";
    panel.style.bottom = "auto";
    panel.style.left = e.clientX - offsetX + "px";
    panel.style.top = e.clientY - offsetY + "px";
  });

  document.addEventListener("mouseup", function() {
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
  if (panelIframe && panelIframe.contentWindow) {
    panelIframe.contentWindow.postMessage(
      { type: "play_glosses", glosses: glosses },
      "*"
    );
  }
}

function stopAvatar() {
  if (panelIframe && panelIframe.contentWindow) {
    panelIframe.contentWindow.postMessage({ type: "stop" }, "*");
  }
}

function startVideoSync() {
  var video = document.querySelector("video");
  if (!video) return;

  video.addEventListener("timeupdate", function() {
    var time = video.currentTime;
    var idx = findSegmentIndex(time);
    if (idx === currentSegmentIndex) return;
    currentSegmentIndex = idx;

    if (idx < 0) {
      setSubtitle("");
      return;
    }

    var seg = segments[idx];
    setSubtitle(seg.text);

    var cached = translationCache[idx];
    if (cached) sendToAvatar(cached.glosses);
  });

  video.addEventListener("pause", function() {
    stopAvatar();
  });

  video.addEventListener("seeking", function() {
    currentSegmentIndex = -1;
    stopAvatar();
  });

  video.addEventListener("play", function() {
    currentSegmentIndex = -1;
  });
}

window.addEventListener("message", function(e) {
  if (!e.data || !e.data.type) return;
  if (e.data.type === "cwasa_ready") avatarReady = true;
  if (e.data.type === "db_ready") {
    dbReady = true;
    setStatus("Ready");
  }
});

async function init() {
  var videoId = getVideoId();
  if (!videoId) return;

  avatarReady = false;
  dbReady = false;
  currentSegmentIndex = -1;
  translationCache = {};
  segments = [];

  injectPanel();

  var tracks = extractCaptionTracks();

  if (!tracks || tracks.length === 0) {
    setStatus("No captions available");
    setSubtitle("This video has no captions");
    return;
  }

  var track = pickBestTrack(tracks);
  setStatus("Fetching captions...");

  try {
    segments = await fetchTranscript(track);
  } catch (e) {
    setStatus("Failed to load captions");
    return;
  }

  setStatus("Translating...");

  try {
    var result = await translateSegments(segments, track.languageCode);
    result.results.forEach(function(r, i) {
      translationCache[i] = r;
    });
    setStatus(dbReady ? "Ready" : "Loading signs...");
  } catch (e) {
    setStatus("Translation error");
  }

  startVideoSync();
}

function cleanup() {
  currentSegmentIndex = -1;
  translationCache = {};
  segments = [];
  avatarReady = false;
  dbReady = false;
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
    if (location.href.includes("youtube.com/watch")) {
      setTimeout(init, 500);
    }
  }
});
navObserver.observe(document.body, { childList: true, subtree: true });

document.addEventListener("yt-navigate-finish", function() {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    cleanup();
    if (location.href.includes("youtube.com/watch")) {
      setTimeout(init, 500);
    }
  }
});

init();

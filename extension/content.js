let panelIframe = null;
let translationCache = {};
let currentSegmentIndex = -1;
let segments = [];

function getVideoId() {
  const params = new URLSearchParams(window.location.search);
  return params.get("v");
}

function extractCaptionTracks() {
  const scripts = document.querySelectorAll("script");
  for (const script of scripts) {
    const text = script.textContent;
    const match = text.match(/ytInitialPlayerResponse\s*=\s*(\{.+?\});/s);
    if (!match) continue;
    try {
      const data = JSON.parse(match[1]);
      const tracks =
        data?.captions?.playerCaptionsTracklistRenderer?.captionTracks;
      if (tracks) return tracks;
    } catch {}
  }
  return null;
}

async function fetchTranscript(track) {
  const url = track.baseUrl;
  const resp = await fetch(url);
  const xml = await resp.text();
  const parser = new DOMParser();
  const doc = parser.parseFromString(xml, "text/xml");
  const nodes = doc.querySelectorAll("text");
  return Array.from(nodes).map((node) => ({
    text: node.textContent.replace(/&amp;/g, "&").replace(/&#39;/g, "'"),
    start: parseFloat(node.getAttribute("start")),
    duration: parseFloat(node.getAttribute("dur") || "0"),
  }));
}

async function translateSegments(segs, sourceLang) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage(
      {
        type: "translate_batch",
        segments: segs.map((s) => ({
          text: s.text,
          start: s.start,
          duration: s.duration,
        })),
        source_lang: sourceLang,
      },
      (resp) => {
        if (resp && resp.ok) resolve(resp.data);
        else reject(new Error(resp?.error || "Translation failed"));
      }
    );
  });
}

function injectPanel() {
  if (document.getElementById("kozha-panel")) return;

  const panel = document.createElement("div");
  panel.id = "kozha-panel";

  const toggle = document.createElement("button");
  toggle.id = "kozha-toggle";
  toggle.textContent = "K";
  toggle.addEventListener("click", () => {
    const body = document.getElementById("kozha-panel-body");
    body.style.display = body.style.display === "none" ? "flex" : "none";
  });

  const body = document.createElement("div");
  body.id = "kozha-panel-body";

  const iframe = document.createElement("iframe");
  iframe.id = "kozha-avatar-frame";
  iframe.src = chrome.runtime.getURL("panel.html");
  panelIframe = iframe;

  const subtitle = document.createElement("div");
  subtitle.id = "kozha-subtitle";
  subtitle.textContent = "Loading captions...";

  const status = document.createElement("div");
  status.id = "kozha-status";
  status.textContent = "Initializing";

  body.appendChild(iframe);
  body.appendChild(subtitle);
  body.appendChild(status);
  panel.appendChild(toggle);
  panel.appendChild(body);
  document.body.appendChild(panel);

  makeDraggable(panel, toggle);
}

function makeDraggable(panel, handle) {
  let dragging = false;
  let offsetX, offsetY;

  handle.addEventListener("mousedown", (e) => {
    if (e.button !== 0) return;
    dragging = true;
    offsetX = e.clientX - panel.getBoundingClientRect().left;
    offsetY = e.clientY - panel.getBoundingClientRect().top;
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    panel.style.right = "auto";
    panel.style.bottom = "auto";
    panel.style.left = e.clientX - offsetX + "px";
    panel.style.top = e.clientY - offsetY + "px";
  });

  document.addEventListener("mouseup", () => {
    dragging = false;
  });
}

function startVideoSync() {
  const video = document.querySelector("video");
  if (!video) return;

  function onTimeUpdate() {
    const time = video.currentTime;
    let idx = -1;
    for (let i = 0; i < segments.length; i++) {
      const seg = segments[i];
      if (time >= seg.start && time < seg.start + seg.duration) {
        idx = i;
        break;
      }
    }
    if (idx === currentSegmentIndex) return;
    currentSegmentIndex = idx;

    const subtitle = document.getElementById("kozha-subtitle");
    if (idx < 0) {
      if (subtitle) subtitle.textContent = "";
      return;
    }

    const seg = segments[idx];
    if (subtitle) subtitle.textContent = seg.text;

    const cached = translationCache[idx];
    if (cached && panelIframe) {
      panelIframe.contentWindow.postMessage(
        { type: "play_glosses", glosses: cached.glosses },
        "*"
      );
    }
  }

  video.addEventListener("timeupdate", onTimeUpdate);
}

async function init() {
  const videoId = getVideoId();
  if (!videoId) return;

  injectPanel();

  const tracks = extractCaptionTracks();
  const status = document.getElementById("kozha-status");

  if (!tracks || tracks.length === 0) {
    if (status) status.textContent = "No captions available";
    return;
  }

  const track = tracks.find((t) => t.languageCode === "en") || tracks[0];
  if (status) status.textContent = "Fetching captions...";

  segments = await fetchTranscript(track);
  if (status) status.textContent = "Translating...";

  try {
    const result = await translateSegments(segments, track.languageCode);
    result.results.forEach((r, i) => {
      translationCache[i] = r;
    });
    if (status) status.textContent = "Ready";
  } catch (err) {
    if (status) status.textContent = "Translation error";
  }

  startVideoSync();
}

let lastUrl = location.href;
const observer = new MutationObserver(() => {
  if (location.href !== lastUrl) {
    lastUrl = location.href;
    currentSegmentIndex = -1;
    translationCache = {};
    segments = [];
    const existing = document.getElementById("kozha-panel");
    if (existing) existing.remove();
    panelIframe = null;
    init();
  }
});
observer.observe(document.body, { childList: true, subtree: true });

init();

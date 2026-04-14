var MAX_SELECTION_LENGTH = 10000;
var _kozhaSigningAbort = null;

var _readerState = {
  active: false,
  paused: false,
  stopRequested: false,
  skipRequested: false,
  segments: [],
  currentIndex: 0,
  highlightedEl: null,
};

function extractPageText() {
  var elements = document.querySelectorAll("p, h1, h2, h3, h4, h5, h6, li, td, th, blockquote, figcaption");
  var segments = [];
  var seen = new Set();
  for (var i = 0; i < elements.length; i++) {
    var el = elements[i];
    var style = window.getComputedStyle(el);
    if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") continue;
    var text = el.textContent.trim();
    if (text.length <= 10 || text.length > 2000) continue;
    if (seen.has(text)) continue;
    seen.add(text);
    segments.push({ text: text, element: el });
  }
  return segments;
}

function highlightElement(el) {
  removeCurrentHighlight();
  el.classList.add("kozha-reading-highlight");
  _readerState.highlightedEl = el;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
}

function removeCurrentHighlight() {
  if (_readerState.highlightedEl) {
    _readerState.highlightedEl.classList.remove("kozha-reading-highlight");
    _readerState.highlightedEl = null;
  }
}

function removeAllHighlights() {
  var highlighted = document.querySelectorAll(".kozha-reading-highlight");
  for (var i = 0; i < highlighted.length; i++) {
    highlighted[i].classList.remove("kozha-reading-highlight");
  }
  _readerState.highlightedEl = null;
}

function setReaderControlsVisible(visible) {
  var el = document.getElementById("kozha-reader-controls");
  if (el) el.classList.toggle("open", visible);
  var btn = document.getElementById("kozha-read-btn");
  if (btn) btn.classList.toggle("active", visible);
}

function updatePauseButton() {
  var btn = document.getElementById("kozha-reader-pause");
  if (btn) btn.textContent = _readerState.paused ? "Resume" : "Pause";
}

function signSegmentAndWait(text) {
  return new Promise(function(resolve) {
    if (_kozhaSigningAbort) _kozhaSigningAbort.aborted = true;
    var session = { aborted: false };
    _kozhaSigningAbort = session;

    Kozha.stopAvatar();

    chrome.runtime.sendMessage(
      { type: "translate", text: text, source_lang: "en" },
      function(resp) {
        if (session.aborted || _readerState.stopRequested) { resolve(); return; }

        if (chrome.runtime.lastError || !resp || !resp.ok) {
          resolve();
          return;
        }

        var glosses = resp.data && resp.data.glosses;
        if (!glosses || glosses.length === 0) {
          resolve();
          return;
        }

        Kozha.sendToAvatar(glosses);

        var estimatedMs = Math.max(glosses.length * 1200, 1500);
        setTimeout(resolve, estimatedMs);
      }
    );
  });
}

function waitWhilePaused() {
  return new Promise(function(resolve) {
    function check() {
      if (_readerState.stopRequested) { resolve(); return; }
      if (!_readerState.paused) { resolve(); return; }
      setTimeout(check, 200);
    }
    check();
  });
}

async function readPage() {
  var segments = extractPageText();
  if (segments.length === 0) {
    Kozha.setStatus("No readable text found");
    return;
  }

  _readerState.active = true;
  _readerState.paused = false;
  _readerState.stopRequested = false;
  _readerState.skipRequested = false;
  _readerState.segments = segments;
  _readerState.currentIndex = 0;

  setReaderControlsVisible(true);

  var total = segments.length;
  if (total > 50) total = 50;

  for (var i = 0; i < total; i++) {
    if (_readerState.stopRequested) break;
    _readerState.currentIndex = i;
    _readerState.skipRequested = false;

    highlightElement(segments[i].element);

    var displayText = segments[i].text;
    if (displayText.length > 200) displayText = displayText.substring(0, 200) + "...";
    Kozha.setSubtitle(displayText);
    Kozha.setStatus("Signing " + (i + 1) + "/" + total);

    await signSegmentAndWait(segments[i].text);

    if (_readerState.stopRequested) break;

    await waitWhilePaused();

    if (_readerState.skipRequested) continue;
  }

  removeAllHighlights();
  setReaderControlsVisible(false);
  _readerState.active = false;
  Kozha.setStatus("Done");
  Kozha.setSubtitle("");
}

function startPageReader() {
  if (_readerState.active) {
    stopPageReader();
    return;
  }
  Kozha.injectPanel();
  Kozha.showPanel();
  readPage();
}

function togglePageReaderPause() {
  if (!_readerState.active) return;
  _readerState.paused = !_readerState.paused;
  updatePauseButton();
  if (_readerState.paused) {
    Kozha.setStatus("Paused — " + (_readerState.currentIndex + 1) + "/" + Math.min(_readerState.segments.length, 50));
  } else {
    Kozha.setStatus("Signing " + (_readerState.currentIndex + 1) + "/" + Math.min(_readerState.segments.length, 50));
  }
}

function skipPageReaderSegment() {
  if (!_readerState.active) return;
  _readerState.skipRequested = true;
  _readerState.paused = false;
  updatePauseButton();
}

function stopPageReader() {
  _readerState.stopRequested = true;
  _readerState.paused = false;
  _readerState.active = false;
  removeAllHighlights();
  setReaderControlsVisible(false);
  Kozha.stopAvatar();
  Kozha.setStatus("Stopped");
  Kozha.setSubtitle("");
}

function signText(text) {
  if (!text || !text.trim()) {
    Kozha.injectPanel();
    Kozha.showPanel();
    Kozha.setStatus("No text selected");
    Kozha.setSubtitle("Select text and right-click to sign it");
    return;
  }

  if (text.length > MAX_SELECTION_LENGTH) {
    text = text.substring(0, MAX_SELECTION_LENGTH);
  }

  if (_kozhaSigningAbort) {
    _kozhaSigningAbort.aborted = true;
  }
  var session = { aborted: false };
  _kozhaSigningAbort = session;

  Kozha.stopAvatar();
  Kozha.injectPanel();
  Kozha.showPanel();
  Kozha.setSubtitle(text);
  Kozha.setStatus("Translating...");

  chrome.runtime.sendMessage(
    { type: "translate", text: text, source_lang: "en" },
    function(resp) {
      if (session.aborted) return;

      if (chrome.runtime.lastError) {
        Kozha.setStatus("Connection error");
        return;
      }

      if (!resp || !resp.ok) {
        var errMsg = (resp && resp.error) || "Translation failed";
        if (errMsg === "Request timed out" || errMsg === "Failed to fetch" ||
            errMsg.indexOf("NetworkError") >= 0) {
          Kozha.setStatus("Cannot reach server");
        } else {
          Kozha.setStatus("Translation failed");
        }
        return;
      }

      var glosses = resp.data && resp.data.glosses;
      if (!glosses || glosses.length === 0) {
        Kozha.setStatus("No signs found");
        return;
      }

      Kozha.sendToAvatar(glosses);
      Kozha.setStatus(Kozha.dbReady ? "Signing" : "Loading signs...");
    }
  );
}

if (!window._kozhaUniversalInit) {
  window._kozhaUniversalInit = true;

  chrome.runtime.onMessage.addListener(function(msg) {
    if (msg.type === "sign_selection") {
      signText(msg.text);
    }
  });

  window.addEventListener("message", function(e) {
    if (!e.data || !e.data.type) return;
    if (e.data.type === "cwasa_failed") Kozha.setStatus("Text-only mode");
    if (e.data.type === "db_ready") {
      if (document.getElementById("kozha-panel")) {
        Kozha.setStatus("Ready");
      }
    }
  });
}

var API_BASE = "https://kozha-translate.com";
var CACHE_TTL_MS = 24 * 60 * 60 * 1000;
var FETCH_TIMEOUT_MS = 30000;

chrome.runtime.onInstalled.addListener(function() {
  chrome.contextMenus.create({
    id: "kozha-sign-selection",
    title: "Sign this text",
    contexts: ["selection"]
  });
});

chrome.contextMenus.onClicked.addListener(function(info, tab) {
  if (info.menuItemId === "kozha-sign-selection" && info.selectionText) {
    chrome.scripting.insertCSS({
      target: { tabId: tab.id },
      files: ["panel.css"]
    }).then(function() {
      return chrome.scripting.executeScript({
        target: { tabId: tab.id },
        files: ["content-shared.js", "content-universal.js"]
      });
    }).then(function() {
      chrome.tabs.sendMessage(tab.id, {
        type: "sign_selection",
        text: info.selectionText
      });
    }).catch(function() {
      chrome.storage.local.set({ kozha_pending_text: info.selectionText });
      chrome.action.openPopup();
    });
  }
});

function getCacheKey(videoId) {
  return "transcript_" + videoId;
}

function fetchWithTimeout(url, options) {
  return new Promise(function(resolve, reject) {
    var controller = new AbortController();
    var timer = setTimeout(function() {
      controller.abort();
      reject(new Error("Request timed out"));
    }, FETCH_TIMEOUT_MS);

    options.signal = controller.signal;
    fetch(url, options)
      .then(function(r) {
        clearTimeout(timer);
        resolve(r);
      })
      .catch(function(err) {
        clearTimeout(timer);
        reject(err);
      });
  });
}

chrome.runtime.onMessage.addListener(function(msg, sender, sendResponse) {
  if (msg.type === "translate") {
    fetchWithTimeout(API_BASE + "/api/translate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: msg.text, source_lang: msg.source_lang || "en" }),
    })
      .then(function(r) { return r.json(); })
      .then(function(data) { sendResponse({ ok: true, data: data }); })
      .catch(function(err) { sendResponse({ ok: false, error: err.message }); });
    return true;
  }

  if (msg.type === "translate_batch") {
    var videoId = msg.video_id;
    var cacheKey = videoId ? getCacheKey(videoId) : null;

    if (cacheKey) {
      chrome.storage.local.get(cacheKey, function(stored) {
        var cached = stored[cacheKey];
        if (cached && cached.timestamp && (Date.now() - cached.timestamp) < CACHE_TTL_MS) {
          sendResponse({ ok: true, data: cached.data });
        } else {
          doBatchTranslate(msg, cacheKey, sendResponse);
        }
      });
    } else {
      doBatchTranslate(msg, null, sendResponse);
    }
    return true;
  }

  if (msg.type === "cache_results") {
    var key = getCacheKey(msg.video_id);
    var toStore = {};
    toStore[key] = { data: msg.data, timestamp: Date.now() };
    chrome.storage.local.set(toStore);
    return false;
  }
});

function doBatchTranslate(msg, cacheKey, sendResponse) {
  fetchWithTimeout(API_BASE + "/api/translate/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      segments: msg.segments,
      source_lang: msg.source_lang || "en",
    }),
  })
    .then(function(r) {
      if (!r.ok) throw new Error("Server error: " + r.status);
      return r.json();
    })
    .then(function(data) {
      if (cacheKey) {
        var toStore = {};
        toStore[cacheKey] = { data: data, timestamp: Date.now() };
        chrome.storage.local.set(toStore);
      }
      sendResponse({ ok: true, data: data });
    })
    .catch(function(err) {
      var msg = err.message;
      if (msg === "Failed to fetch" || msg.indexOf("NetworkError") >= 0) {
        msg = "Failed to fetch";
      }
      sendResponse({ ok: false, error: msg });
    });
}

var API_BASE = "https://kozha-translate.com";

function getCacheKey(videoId) {
  return "transcript_" + videoId;
}

chrome.runtime.onMessage.addListener(function(msg, sender, sendResponse) {
  if (msg.type === "translate") {
    fetch(API_BASE + "/api/translate", {
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
        if (stored[cacheKey]) {
          sendResponse({ ok: true, data: stored[cacheKey] });
        } else {
          doBatchTranslate(msg, cacheKey, sendResponse);
        }
      });
    } else {
      doBatchTranslate(msg, null, sendResponse);
    }
    return true;
  }
});

function doBatchTranslate(msg, cacheKey, sendResponse) {
  fetch(API_BASE + "/api/translate/batch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      segments: msg.segments,
      source_lang: msg.source_lang || "en",
    }),
  })
    .then(function(r) { return r.json(); })
    .then(function(data) {
      if (cacheKey) {
        var toStore = {};
        toStore[cacheKey] = data;
        chrome.storage.local.set(toStore);
      }
      sendResponse({ ok: true, data: data });
    })
    .catch(function(err) { sendResponse({ ok: false, error: err.message }); });
}

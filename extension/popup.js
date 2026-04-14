var API_BASE = "https://kozha-translate.com";

var inputEl = document.getElementById("popup-input");
var inputLangEl = document.getElementById("popup-input-lang");
var signLangEl = document.getElementById("popup-sign-lang");
var signBtn = document.getElementById("popup-sign-btn");
var avatarFrame = document.getElementById("popup-avatar-frame");
var statusEl = document.getElementById("popup-status");

var avatarReady = false;
var dbReady = false;
var cwasaFailed = false;

chrome.storage.local.get(["popup_input_lang", "kozha_sign_lang", "kozha_last_input", "kozha_pending_text"], function(stored) {
  if (stored.popup_input_lang) inputLangEl.value = stored.popup_input_lang;
  if (stored.kozha_sign_lang) signLangEl.value = stored.kozha_sign_lang;
  if (stored.kozha_pending_text) {
    inputEl.value = stored.kozha_pending_text;
    chrome.storage.local.remove("kozha_pending_text");
  } else if (stored.kozha_last_input) {
    inputEl.value = stored.kozha_last_input;
  }
});

inputEl.addEventListener("input", function() {
  chrome.storage.local.set({ kozha_last_input: inputEl.value });
});

inputLangEl.addEventListener("change", function() {
  chrome.storage.local.set({ popup_input_lang: inputLangEl.value });
});

signLangEl.addEventListener("change", function() {
  chrome.storage.local.set({ kozha_sign_lang: signLangEl.value });
  dbReady = false;
  statusEl.textContent = "Loading sign language...";
  avatarFrame.contentWindow.postMessage({
    type: "switch_language",
    lang: signLangEl.value
  }, "*");
});

window.addEventListener("message", function(e) {
  if (!e.data || !e.data.type) return;

  if (e.data.type === "cwasa_ready") {
    avatarReady = true;
    statusEl.textContent = "Ready";
  }

  if (e.data.type === "cwasa_failed") {
    avatarReady = false;
    cwasaFailed = true;
    avatarFrame.style.display = "none";
    var glossArea = document.getElementById("popup-gloss-text");
    if (!glossArea) {
      glossArea = document.createElement("div");
      glossArea.id = "popup-gloss-text";
      glossArea.style.cssText = "padding:10px;font-size:15px;font-weight:600;color:#e8843e;background:#000;min-height:50px;text-align:center;line-height:1.6;word-spacing:4px;border-radius:6px;";
      avatarFrame.parentNode.appendChild(glossArea);
    }
    statusEl.textContent = "Text mode";
  }

  if (e.data.type === "db_ready") {
    dbReady = true;
    if (avatarReady) statusEl.textContent = "Ready";
  }
});

signBtn.addEventListener("click", function() {
  var text = inputEl.value.trim();
  if (!text) return;

  signBtn.disabled = true;
  statusEl.textContent = "Processing...";

  var inputLang = inputLangEl.value;

  function sendToPlan(finalText) {
    fetch(API_BASE + "/api/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: finalText })
    })
      .then(function(r) {
        if (!r.ok) throw new Error("Server error: " + r.status);
        return r.json();
      })
      .then(function(data) {
        var glosses = data.glosses || data.plan || [];
        if (glosses.length === 0) {
          statusEl.textContent = "No signs found";
          signBtn.disabled = false;
          return;
        }
        if (cwasaFailed) {
          var glossArea = document.getElementById("popup-gloss-text");
          if (glossArea) glossArea.textContent = glosses.join(" ");
          statusEl.textContent = "Text mode";
        } else {
          avatarFrame.contentWindow.postMessage({
            type: "play_glosses",
            glosses: glosses
          }, "*");
          statusEl.textContent = "Signing...";
        }
        signBtn.disabled = false;
      })
      .catch(function(err) {
        var msg = err.message;
        if (msg === "Failed to fetch" || msg.indexOf("NetworkError") >= 0) {
          statusEl.textContent = "Cannot reach server — check connection";
        } else {
          statusEl.textContent = "Error: " + msg;
        }
        signBtn.disabled = false;
      });
  }

  if (inputLang === "en") {
    sendToPlan(text);
  } else {
    chrome.runtime.sendMessage({
      type: "translate",
      text: text,
      source_lang: inputLang
    }, function(resp) {
      if (chrome.runtime.lastError) {
        statusEl.textContent = "Cannot reach server — check connection";
        signBtn.disabled = false;
        return;
      }
      if (resp && resp.ok && resp.data && resp.data.translated) {
        sendToPlan(resp.data.translated);
      } else {
        var errMsg = (resp && resp.error) || "Translation failed";
        if (errMsg === "Failed to fetch" || errMsg === "Request timed out") {
          statusEl.textContent = "Cannot reach server — check connection";
        } else {
          statusEl.textContent = "Translation failed";
        }
        signBtn.disabled = false;
      }
    });
  }
});

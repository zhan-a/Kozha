var API_BASE = "https://kozha-translate.com";

var inputEl = document.getElementById("popup-input");
var inputLangEl = document.getElementById("popup-input-lang");
var signLangEl = document.getElementById("popup-sign-lang");
var signBtn = document.getElementById("popup-sign-btn");
var avatarFrame = document.getElementById("popup-avatar-frame");
var statusEl = document.getElementById("popup-status");

var avatarReady = false;
var dbReady = false;

chrome.storage.local.get(["popup_input_lang", "kozha_sign_lang"], function(stored) {
  if (stored.popup_input_lang) inputLangEl.value = stored.popup_input_lang;
  if (stored.kozha_sign_lang) signLangEl.value = stored.kozha_sign_lang;
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
    statusEl.textContent = "Avatar unavailable — text mode";
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
      .then(function(r) { return r.json(); })
      .then(function(data) {
        var glosses = data.glosses || data.plan || [];
        if (glosses.length === 0) {
          statusEl.textContent = "No signs found";
          signBtn.disabled = false;
          return;
        }
        avatarFrame.contentWindow.postMessage({
          type: "play_glosses",
          glosses: glosses
        }, "*");
        statusEl.textContent = "Signing...";
        signBtn.disabled = false;
      })
      .catch(function(err) {
        statusEl.textContent = "Error: " + err.message;
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
      if (resp && resp.ok && resp.data && resp.data.translated) {
        sendToPlan(resp.data.translated);
      } else {
        statusEl.textContent = "Translation failed";
        signBtn.disabled = false;
      }
    });
  }
});

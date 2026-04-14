var MAX_SELECTION_LENGTH = 10000;
var _kozhaSigningAbort = null;

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

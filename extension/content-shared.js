if (!window.Kozha) {
  window.Kozha = {
    panelIframe: null,
    avatarReady: false,
    dbReady: false,
    cwasaFailed: false,
    currentSignLang: "bsl",
    playbackSpeed: 1.0,
    _settingsOpen: false,
    _trackedDocListeners: [],
    _storageLoaded: false,

    SIGN_LANG_LABELS: {
      bsl: "BSL", asl: "ASL", dgs: "DGS (German)", lsf: "LSF (French)",
      pjm: "PJM (Polish)", gsl: "GSL (Greek)", ngt: "NGT (Dutch)",
      algerian: "Algerian SL", bangla: "Bangla SL", fsl: "Filipino SL",
      isl: "Indian SL", kurdish: "Kurdish SL", vsl: "Vietnamese SL",
    },

    setStatus: function(text) {
      var el = document.getElementById("kozha-status-text");
      if (el) el.textContent = text;
    },

    setStatusLang: function(text) {
      var el = document.getElementById("kozha-status-lang");
      if (el) el.textContent = text;
    },

    setSubtitle: function(text) {
      var el = document.getElementById("kozha-subtitle");
      if (el) el.textContent = text;
    },

    sendToAvatar: function(glosses) {
      if (Kozha.cwasaFailed) {
        Kozha.showTextGlosses(glosses);
        return;
      }
      if (Kozha.panelIframe && Kozha.panelIframe.contentWindow) {
        Kozha.panelIframe.contentWindow.postMessage(
          { type: "play_glosses", glosses: glosses }, "*"
        );
      }
    },

    showTextGlosses: function(glosses) {
      var el = document.getElementById("kozha-gloss-text");
      if (el) el.textContent = glosses.join(" ");
    },

    stopAvatar: function() {
      if (Kozha.panelIframe && Kozha.panelIframe.contentWindow) {
        Kozha.panelIframe.contentWindow.postMessage({ type: "stop" }, "*");
      }
    },

    addDocListener: function(target, event, handler) {
      target.addEventListener(event, handler);
      Kozha._trackedDocListeners.push({ target: target, event: event, handler: handler });
    },

    removeDocListeners: function() {
      Kozha._trackedDocListeners.forEach(function(entry) {
        entry.target.removeEventListener(entry.event, entry.handler);
      });
      Kozha._trackedDocListeners = [];
    },

    clampToViewport: function(panel) {
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
    },

    makeDraggable: function(panel, handle) {
      var dragging = false;
      var offsetX, offsetY;

      handle.addEventListener("mousedown", function(e) {
        if (e.button !== 0) return;
        dragging = true;
        offsetX = e.clientX - panel.getBoundingClientRect().left;
        offsetY = e.clientY - panel.getBoundingClientRect().top;
        e.preventDefault();
      });

      Kozha.addDocListener(document, "mousemove", function(e) {
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

      Kozha.addDocListener(document, "mouseup", function() {
        dragging = false;
      });
    },

    injectPanel: function() {
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

      var readBtn = document.createElement("button");
      readBtn.id = "kozha-read-btn";
      readBtn.textContent = "\u25B6";
      readBtn.title = "Read page";

      var retryBtn = document.createElement("button");
      retryBtn.id = "kozha-retry-btn";
      retryBtn.className = "kozha-ctrl-btn";
      retryBtn.textContent = "\u21BB";
      retryBtn.title = "Retry caption detection";
      retryBtn.addEventListener("click", function(e) {
        e.stopPropagation();
        if (typeof window._kozhaManualRetry === "function") {
          Kozha.setStatus("Retrying...");
          window._kozhaManualRetry();
        }
      });
      readBtn.addEventListener("click", function(e) {
        e.stopPropagation();
        if (typeof startPageReader === "function") {
          startPageReader();
        }
      });

      var settingsBtn = document.createElement("button");
      settingsBtn.id = "kozha-settings-btn";
      settingsBtn.innerHTML = "&#9881;";
      settingsBtn.addEventListener("click", function(e) {
        e.stopPropagation();
        Kozha._settingsOpen = !Kozha._settingsOpen;
        var drawer = document.getElementById("kozha-settings-drawer");
        if (drawer) drawer.classList.toggle("open", Kozha._settingsOpen);
        settingsBtn.classList.toggle("active", Kozha._settingsOpen);
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
        Kozha.stopAvatar();
        panel.style.display = "none";
        var toggle = document.getElementById("kozha-toggle");
        if (toggle) toggle.style.display = "block";
      });

      controls.appendChild(readBtn);
      controls.appendChild(retryBtn);
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
      Object.keys(Kozha.SIGN_LANG_LABELS).forEach(function(key) {
        var opt = document.createElement("option");
        opt.value = key;
        opt.textContent = Kozha.SIGN_LANG_LABELS[key];
        if (key === Kozha.currentSignLang) opt.selected = true;
        langSelect.appendChild(opt);
      });
      langSelect.addEventListener("change", function() {
        Kozha.currentSignLang = langSelect.value;
        chrome.storage.local.set({ kozha_sign_lang: langSelect.value });
        if (Kozha.panelIframe && Kozha.panelIframe.contentWindow) {
          Kozha.panelIframe.contentWindow.postMessage(
            { type: "switch_language", lang: Kozha.currentSignLang }, "*"
          );
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
        if (parseFloat(item.value) === Kozha.playbackSpeed) opt.selected = true;
        speedSelect.appendChild(opt);
      });
      speedSelect.addEventListener("change", function() {
        Kozha.playbackSpeed = parseFloat(speedSelect.value);
        if (Kozha.panelIframe && Kozha.panelIframe.contentWindow) {
          Kozha.panelIframe.contentWindow.postMessage(
            { type: "set_speed", speed: Kozha.playbackSpeed }, "*"
          );
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
      iframe.addEventListener("load", function() {
        if (iframe.contentWindow) {
          iframe.contentWindow.postMessage({ type: "init_cwasa" }, "*");
        }
      });
      Kozha.panelIframe = iframe;

      var subtitle = document.createElement("div");
      subtitle.id = "kozha-subtitle";

      var status = document.createElement("div");
      status.id = "kozha-status";

      var statusText = document.createElement("span");
      statusText.id = "kozha-status-text";

      var statusLang = document.createElement("span");
      statusLang.id = "kozha-status-lang";

      status.appendChild(statusText);
      status.appendChild(statusLang);

      var readerControls = document.createElement("div");
      readerControls.id = "kozha-reader-controls";

      var pauseBtn = document.createElement("button");
      pauseBtn.className = "kozha-reader-btn";
      pauseBtn.id = "kozha-reader-pause";
      pauseBtn.textContent = "Pause";
      pauseBtn.addEventListener("click", function() {
        if (typeof togglePageReaderPause === "function") togglePageReaderPause();
      });

      var skipBtn = document.createElement("button");
      skipBtn.className = "kozha-reader-btn";
      skipBtn.id = "kozha-reader-skip";
      skipBtn.textContent = "Skip \u25B6\u25B6";
      skipBtn.addEventListener("click", function() {
        if (typeof skipPageReaderSegment === "function") skipPageReaderSegment();
      });

      var stopBtn = document.createElement("button");
      stopBtn.className = "kozha-reader-btn stop";
      stopBtn.id = "kozha-reader-stop";
      stopBtn.textContent = "Stop";
      stopBtn.addEventListener("click", function() {
        if (typeof stopPageReader === "function") stopPageReader();
      });

      readerControls.appendChild(pauseBtn);
      readerControls.appendChild(skipBtn);
      readerControls.appendChild(stopBtn);

      body.appendChild(settingsDrawer);
      body.appendChild(readerControls);
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

      Kozha.makeDraggable(panel, header);
    },

    showPanel: function() {
      var panel = document.getElementById("kozha-panel");
      if (panel) {
        panel.style.display = "block";
        var toggle = document.getElementById("kozha-toggle");
        if (toggle) toggle.style.display = "none";
        var body = document.getElementById("kozha-panel-body");
        if (body) body.style.display = "flex";
      }
    },

    removePanel: function() {
      var panel = document.getElementById("kozha-panel");
      if (panel) panel.remove();
      var toggle = document.getElementById("kozha-toggle");
      if (toggle) toggle.remove();
      Kozha.panelIframe = null;
    },

    repositionPanel: function() {
      var panel = document.getElementById("kozha-panel");
      if (!panel || panel.style.display === "none") return;
      Kozha.clampToViewport(panel);
    },
  };

  window.addEventListener("message", function(e) {
    if (!e.data || !e.data.type) return;
    if (e.data.type === "cwasa_ready") Kozha.avatarReady = true;
    if (e.data.type === "cwasa_failed") {
      Kozha.cwasaFailed = true;
      var frame = document.getElementById("kozha-avatar-frame");
      if (frame) frame.style.display = "none";
      var body = document.getElementById("kozha-panel-body");
      if (body && !document.getElementById("kozha-gloss-text")) {
        var glossDiv = document.createElement("div");
        glossDiv.id = "kozha-gloss-text";
        body.insertBefore(glossDiv, body.firstChild);
      }
    }
    if (e.data.type === "db_ready") Kozha.dbReady = true;
  });

  window.addEventListener("resize", function() {
    Kozha.repositionPanel();
  });

  chrome.storage.local.get("kozha_sign_lang", function(stored) {
    if (stored.kozha_sign_lang) {
      Kozha.currentSignLang = stored.kozha_sign_lang;
    }
    Kozha._storageLoaded = true;
  });
}

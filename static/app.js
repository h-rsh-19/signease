const state = {
  ws: null,
  cameraStream: null,
  cameraTicker: null,
  sessionId: `sess_${Math.random().toString(36).slice(2, 10)}`,
  playlist: [],
  playlistIndex: -1,
  cwasaReady: false,
  recognitionBusy: false,
  lastTrace: null,
  lastTranslation: null
};

const el = {
  tabRec: document.getElementById("tabRec"),
  tabText: document.getElementById("tabText"),
  recPanel: document.getElementById("recPanel"),
  textPanel: document.getElementById("textPanel"),
  themeToggle: document.getElementById("themeToggle"),
  cam: document.getElementById("cam"),
  frameCanvas: document.getElementById("frameCanvas"),
  topLabel: document.getElementById("topLabel"),
  topConfidence: document.getElementById("topConfidence"),
  backendState: document.getElementById("backendState"),
  backendNote: document.getElementById("backendNote"),
  committedSeq: document.getElementById("committedSeq"),
  startCam: document.getElementById("startCam"),
  stopCam: document.getElementById("stopCam"),
  textInput: document.getElementById("textInput"),
  renderMode: document.getElementById("renderMode"),
  runText: document.getElementById("runText"),
  diagBox: document.getElementById("diagBox"),
  hamnosysBox: document.getElementById("hamnosysBox"),
  transformationFlow: document.getElementById("transformationFlow"),
  clipStrip: document.getElementById("clipStrip"),
  tokenList: document.getElementById("tokenList"),
  mainPlayer: document.getElementById("mainPlayer"),
  avatarStage: document.getElementById("avatarStage"),
  mainPlayerMeta: document.getElementById("mainPlayerMeta"),
  playAllBtn: document.getElementById("playAllBtn"),
  stopAllBtn: document.getElementById("stopAllBtn"),
  restartAllBtn: document.getElementById("restartAllBtn"),
  presets: Array.from(document.querySelectorAll(".chip-preset")),
};

function showTab(which) {
  const rec = which === "rec";
  el.recPanel.classList.toggle("hidden", !rec);
  el.textPanel.classList.toggle("hidden", rec);
  el.tabRec.classList.toggle("active", rec);
  el.tabText.classList.toggle("active", !rec);
}

function setTheme(theme) {
  const dark = theme !== "light";
  document.body.setAttribute("data-theme", dark ? "dark" : "light");
  el.themeToggle.checked = !dark;
  localStorage.setItem("trial3_theme", dark ? "dark" : "light");
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.detail || data.message || "Request failed");
  return data;
}

function renderChips(container, tokens) {
  container.innerHTML = "";
  tokens.forEach((tok) => {
    const chip = document.createElement("span");
    chip.textContent = tok;
    container.appendChild(chip);
  });
}

function updateRecognitionUI(data) {
  el.topLabel.textContent = data.top_prediction?.label || "-";
  el.topConfidence.textContent = (data.top_prediction?.confidence ?? 0).toFixed(3);
  el.backendState.textContent = `${data.diagnostics?.backend || "-"} | loaded=${data.diagnostics?.model_loaded}`;
  el.backendNote.textContent = data.diagnostics?.note || "";
  renderChips(el.committedSeq, data.committed_sequence || []);
}

function connectWs() {
  if (state.ws && state.ws.readyState <= 1) return;
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/api/recognition/stream`);
  state.ws.onmessage = (ev) => {
    const data = JSON.parse(ev.data);
    if (data.error) return;
    updateRecognitionUI(data);
  };
}

function captureFrameBase64() {
  const c = el.frameCanvas;
  const ctx = c.getContext("2d");
  ctx.drawImage(el.cam, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.8);
}

async function startCamera() {
  if (state.cameraStream) return;
  state.cameraStream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  el.cam.srcObject = state.cameraStream;
  connectWs();
  state.cameraTicker = setInterval(() => {
    const image_base64 = captureFrameBase64();
    if (state.ws && state.ws.readyState === 1) {
      state.ws.send(JSON.stringify({ image_base64, session_id: state.sessionId }));
      return;
    }
    if (state.recognitionBusy) return;
    state.recognitionBusy = true;
    postJson("/api/recognition/frame", { image_base64, session_id: state.sessionId })
      .then(updateRecognitionUI)
      .catch(() => {})
      .finally(() => {
        state.recognitionBusy = false;
      });
  }, 350);
}

function stopCamera() {
  if (state.cameraTicker) clearInterval(state.cameraTicker);
  state.cameraTicker = null;
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((t) => t.stop());
    state.cameraStream = null;
  }
  el.cam.srcObject = null;
}

function clearPlaylist() {
  state.playlist = [];
  state.playlistIndex = -1;
  el.mainPlayer.pause();
  el.mainPlayer.removeAttribute("src");
  el.mainPlayer.load();
  el.mainPlayerMeta.textContent = "No sequence loaded.";
}

function setMainClip(index, autoplay = false) {
  if (index < 0 || index >= state.playlist.length) return;
  el.mainPlayer.classList.remove("hidden");
  el.avatarStage.classList.add("hidden");
  const item = state.playlist[index];
  state.playlistIndex = index;
  el.mainPlayer.src = item.path;
  el.mainPlayer.load();
  el.mainPlayerMeta.textContent = `Clip ${index + 1}/${state.playlist.length}: ${item.token}`;
  
  const applySpeed = () => {
    el.mainPlayer.playbackRate = (item.source === "letter_fallback") ? 2.0 : 1.0;
    if (autoplay) el.mainPlayer.play().catch(() => {});
  };
  el.mainPlayer.onloadedmetadata = applySpeed;
}

async function initCWASAIfNeeded() {
  if (state.cwasaReady) return true;
  if (!window.CWASA || typeof window.CWASA.init !== "function") return false;
  try {
    const cfg = {
      jasBase: `${location.origin}/jas/loc2021/`,
      cwaBase: "cwa",
      avSettings: { avList: "avsfull", initAv: "luna" },
    };
    window.CWASA.init(cfg);
    state.cwasaReady = true;
    // Set speed after a short delay
    setTimeout(() => {
      const input = document.querySelector('input[class*="Speed"], input[class*="speed"]');
      if (input) {
        input.value = "-0.5";
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }
    }, 1500);
    return true;
  } catch (e) {
    console.error("CWASA Init Error:", e);
    return false;
  }
}

async function playSigmlCombined(fromIndex = 0) {
  const sigmlItems = state.playlist.slice(fromIndex);
  if (!sigmlItems.length) return;

  el.mainPlayerMeta.textContent = "Loading signs...";
  let combined = "<sigml>\n";
  let count = 0;
  for (const item of sigmlItems) {
    try {
      const res = await fetch(item.path);
      if (!res.ok) continue;
      const text = await res.text();
      const inner = text.replace(/<\/?sigml[^>]*>/gi, "").replace(/<\?xml[^>]*\?>/gi, "").trim();
      if (inner) {
        combined += `${inner}\n`;
        count++;
      }
    } catch (e) { console.error("Fetch error:", e); }
  }
  combined += "</sigml>";

  const ok = await initCWASAIfNeeded();
  if (!ok) {
    el.mainPlayerMeta.textContent = "Avatar engine failed to start.";
    return;
  }

  el.mainPlayer.classList.add("hidden");
  el.avatarStage.classList.remove("hidden");
  el.mainPlayerMeta.textContent = `Playing ${count} signs...`;
  
  try {
    window.CWASA.playSiGMLText(combined);
  } catch (e) {
    el.mainPlayerMeta.textContent = "Error playing signs.";
    console.error(e);
  }
}

function playAll() {
  if (!state.playlist.length) return;
  if (el.renderMode.value === "sigml_avatar") {
    playSigmlCombined(0);
  } else {
    setMainClip(0, true);
  }
}

function renderArtifacts(result) {
  el.clipStrip.innerHTML = "";
  renderTransformationFlow(result);
  clearPlaylist();

  const playable = (result.artifacts || []).filter((item) =>
    (item.kind === "video" || item.kind === "sigml") && item.path
  );
  state.playlist = playable.map((item) => ({ token: item.token, path: item.path, source: item.source }));
  const confidenceMap = new Map((state.lastTranslation?.token_confidence || []).map(item => [item.token, item]));
  const tokenLabels = state.playlist.map((item) => {
    const conf = confidenceMap.get(item.token);
    return conf ? `${item.token} (${Number(conf.confidence).toFixed(2)})` : item.token;
  });
  renderChips(el.tokenList, tokenLabels);

  (result.artifacts || []).forEach((item) => {
    const card = document.createElement("article");
    card.className = `clip-card ${item.kind === "missing" ? "missing" : ""}`;
    const conf = confidenceMap.get(item.token);
    const confText = conf ? ` (${Number(conf.confidence).toFixed(2)})` : "";
    card.innerHTML = `<h4>${item.token}${confText}</h4>`;

    if (item.kind === "video" && item.path) {
      const v = document.createElement("video");
      v.src = item.path;
      v.controls = true;
      v.addEventListener("loadedmetadata", () => { if (item.source === "letter_fallback") v.playbackRate = 2.0; });
      card.appendChild(v);
    } else if (item.kind === "sigml" && item.path) {
      const btn = document.createElement("button");
      btn.className = "ghost";
      btn.textContent = "Play in Avatar";
      btn.addEventListener("click", () => {
        const idx = state.playlist.findIndex(p => p.path === item.path);
        if (idx >= 0) playSigmlCombined(idx);
      });
      card.appendChild(btn);
    } else {
      card.innerHTML += "<p>No mapping</p>";
    }
    el.clipStrip.appendChild(card);
  });

  if (state.playlist.length && result.render_mode !== "sigml_avatar") {
    setMainClip(0, false);
  } else if (result.render_mode === "sigml_avatar") {
    el.mainPlayer.classList.add("hidden");
    el.avatarStage.classList.remove("hidden");
    el.mainPlayerMeta.textContent = "Signs loaded. Click Play All.";
    initCWASAIfNeeded();
  }
}

function translate(text, mode) {
  postJson("/api/translation/text", { text })
    .then(translateData => {
      state.lastTranslation = translateData;
      state.lastTrace = translateData;
      return postJson("/api/render/sequence", {
        tokens: translateData.tokens,
        token_confidence: translateData.token_confidence || [],
        render_mode: mode
      });
    })
    .then(renderData => {
      el.diagBox.textContent = JSON.stringify({ pipeline: state.lastTrace, render: renderData }, null, 2);
      renderArtifacts(renderData);
      updateHamNoSysDisplay(renderData);
    })
    .catch(err => {
      el.diagBox.textContent = `Error: ${err.message}`;
    });
}

function updateHamNoSysDisplay(result) {
  if (!el.hamnosysBox) return;
  el.hamnosysBox.innerHTML = "";
  const sigmlItems = result.artifacts.filter(a => a.kind === "sigml" && a.path);
  for (const item of sigmlItems) {
    fetch(item.path)
      .then(res => res.text())
      .then(xml => {
        const hns = extractHamNoSys(xml);
        const div = document.createElement("div");
        div.style.marginBottom = "8px";
        div.innerHTML = `<span style="color:var(--muted)">// ${item.token}</span><br><span style="color:var(--primary); font-family: monospace; font-size: 0.85em;">${hns || "N/A"}</span>`;
        el.hamnosysBox.appendChild(div);
      }).catch(() => {});
  }
}

function extractHamNoSys(sigml) {
  const regex = /<(ham[a-z0-9]+)\/?>/gi;
  let matches = [];
  let m;
  while ((m = regex.exec(sigml)) !== null) { matches.push(m[1]); }
  return matches.length ? matches.join(" ") : null;
}

function renderTransformationFlow(result) {
  if (!el.transformationFlow) return;
  el.transformationFlow.innerHTML = "";
  const trace = state.lastTrace?.trace || [];
  addFlowStep("1", "Sentence Input", el.textInput.value);
  const tokens = trace.find(t => t.stage === "tokenize")?.output || [];
  addFlowStep("2", "Grammar & Tokenization", `Tokens: ${tokens.join(", ") || "None"}`);
  const phrases = trace.find(t => t.stage === "phrase_match")?.output || [];
  const matches = phrases.filter(p => p.includes("_"));
  addFlowStep("3", "Phrase Matching", matches.length ? `Matched: ${matches.join(", ")}` : "No phrases.");
  const solved = result.artifacts?.filter(a => a.kind !== "missing").length || 0;
  addFlowStep("4", "Word Mapping", `Resolved ${solved} of ${result.artifacts?.length || 0} tokens.`);
  const d = result.diagnostics || {};
  const mean = d.mean_confidence == null ? "N/A" : Number(d.mean_confidence).toFixed(2);
  addFlowStep("5", "ISL Playback", `Mode: ${result.render_mode} | Mean Confidence: ${mean}`, "isl-highlight");
}

function addFlowStep(num, label, value, extraClass = "") {
  const row = document.createElement("div");
  row.className = "flow-step-row";
  row.innerHTML = `<div class="flow-step-connector"></div><div class="flow-step-icon">${num}</div><div class="flow-step-content"><div class="flow-step-label">${label}</div><div class="flow-step-value ${extraClass}">${value}</div></div>`;
  el.transformationFlow.appendChild(row);
}

function bind() {
  el.tabRec.addEventListener("click", () => showTab("rec"));
  el.tabText.addEventListener("click", () => showTab("text"));
  el.themeToggle.addEventListener("change", () => setTheme(el.themeToggle.checked ? "light" : "dark"));
  el.startCam.addEventListener("click", startCamera);
  el.stopCam.addEventListener("click", stopCamera);
  el.runText.addEventListener("click", () => {
    const text = el.textInput.value.trim();
    if (text) translate(text, el.renderMode.value);
  });
  el.playAllBtn.addEventListener("click", playAll);
  el.stopAllBtn.addEventListener("click", () => {
    if (el.renderMode.value === "sigml_avatar") window.CWASA?.stopSiGML();
    else el.mainPlayer.pause();
  });
  el.restartAllBtn.addEventListener("click", () => {
    if (el.renderMode.value === "sigml_avatar") playSigmlCombined(0);
    else setMainClip(0, true);
  });
  el.mainPlayer.addEventListener("ended", () => {
    const next = state.playlistIndex + 1;
    if (next < state.playlist.length) setMainClip(next, true);
  });
  el.presets.forEach((node) => {
    node.addEventListener("click", () => {
      el.textInput.value = node.textContent;
      el.runText.click();
    });
  });
}

const storedTheme = localStorage.getItem("trial3_theme");
setTheme(storedTheme || "dark");
bind();
initCWASAIfNeeded();

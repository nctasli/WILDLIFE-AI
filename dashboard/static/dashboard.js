const systemPill = document.querySelector("#system-pill");
const systemState = document.querySelector("#system-state");
const cameraState = document.querySelector("#camera-state");
const cameraSource = document.querySelector("#camera-source");
const detectorState = document.querySelector("#detector-state");
const detectorName = document.querySelector("#detector-name");
const deterrenceState = document.querySelector("#deterrence-state");
const cooldownState = document.querySelector("#cooldown-state");
const cooldownNote = document.querySelector("#cooldown-note");
const video = document.querySelector("#live-video");
const videoPlaceholder = document.querySelector("#video-placeholder");
const frameCaption = document.querySelector("#frame-caption");
const eventsBody = document.querySelector("#events-body");
const actionMessage = document.querySelector("#action-message");
const lastUpdated = document.querySelector("#last-updated");
const startButton = document.querySelector("#start-button");
const stopButton = document.querySelector("#stop-button");
const deterrenceButton = document.querySelector("#deterrence-button");
const soundButton = document.querySelector("#sound-button");

function showMessage(message, isError = false) {
  actionMessage.textContent = message;
  actionMessage.classList.toggle("is-error", isError);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `İstek başarısız (${response.status}).`);
  return data;
}

function readableTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("tr-TR");
}

function applyStatus(data) {
  const state = data.state;
  const names = {
    stopped: "Durduruldu",
    starting: "Başlatılıyor",
    running: "Çalışıyor",
    stopping: "Durduruluyor",
    error: "Hata",
  };
  systemState.textContent = names[state] || state;
  systemPill.classList.toggle("is-running", state === "running");
  systemPill.classList.toggle("is-error", state === "error");
  cameraState.textContent = data.camera_connected ? "Bağlı" : "Bağlı değil";
  cameraSource.textContent = `Kaynak: ${data.source}`;
  detectorState.textContent = {
    stopped: "Beklemede",
    loading: "Yükleniyor",
    ready: "Hazır",
    error: "Hata",
  }[data.detector_state] || data.detector_state;
  detectorName.textContent = `Model: ${data.detector}`;
  deterrenceState.textContent = data.deterrence_enabled ? "Aktif" : "Pasif";
  deterrenceButton.textContent = data.deterrence_enabled
    ? "Caydırmayı devre dışı bırak"
    : "Caydırmayı etkinleştir";
  deterrenceButton.setAttribute("aria-pressed", String(data.deterrence_enabled));
  deterrenceButton.disabled = state !== "running";

  const remaining = Math.ceil(data.cooldown_remaining_seconds || 0);
  cooldownState.textContent = remaining > 0 ? `${remaining} sn` : "Hazır";
  cooldownNote.textContent = remaining > 0
    ? `Yeniden tetiklemeye ${remaining} saniye`
    : `Bekleme süresi: ${data.cooldown_seconds} saniye`;

  const running = ["starting", "running", "stopping"].includes(state);
  startButton.disabled = running;
  stopButton.disabled = !running;
  soundButton.disabled = running;
  const hasStreamStatus = Object.prototype.hasOwnProperty.call(data, "stream_available");
  const streamAvailable = state === "running" && (
    hasStreamStatus ? Boolean(data.stream_available) : Boolean(data.camera_connected)
  );
  video.classList.toggle("is-visible", streamAvailable);
  videoPlaceholder.hidden = streamAvailable;
  frameCaption.textContent = data.camera_connected ? "Görüntü alınıyor" : "Akış durduruldu";
  if (data.last_error) showMessage(data.last_error, true);
  lastUpdated.textContent = `Son güncelleme: ${new Date().toLocaleTimeString("tr-TR")}`;
}

function appendCell(row, value) {
  const cell = document.createElement("td");
  cell.textContent = value;
  row.append(cell);
  return cell;
}

async function refreshEvents() {
  const { events } = await requestJson("/api/events?limit=50");
  eventsBody.replaceChildren();
  if (!events.length) {
    const row = document.createElement("tr");
    const cell = appendCell(row, "Henüz olay kaydı yok.");
    cell.colSpan = 6;
    cell.className = "empty-state";
    eventsBody.append(row);
    return;
  }
  for (const event of events) {
    const row = document.createElement("tr");
    appendCell(row, readableTime(event.last_seen_at));
    appendCell(row, event.species);
    appendCell(row, `${(Number(event.confidence) * 100).toFixed(1)}%`);
    const targetCell = appendCell(row, event.target_confirmed ? "Onaylandı" : "Hayır");
    if (event.target_confirmed) {
      targetCell.textContent = "";
      const badge = document.createElement("span");
      badge.className = "badge badge-target";
      badge.textContent = "Onaylandı";
      targetCell.append(badge);
    }
    appendCell(row, event.audio_triggered ? "Çalındı" : event.trigger_reason);
    const imageCell = document.createElement("td");
    if (event.snapshot_url) {
      const link = document.createElement("a");
      link.className = "snapshot-link";
      link.href = event.snapshot_url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = "Görüntüle";
      imageCell.append(link);
    } else {
      imageCell.textContent = "—";
    }
    row.append(imageCell);
    eventsBody.append(row);
  }
}

async function refreshStatus() {
  try {
    const { status, source, detector, cooldown_seconds } = await requestJson("/api/status");
    applyStatus({ ...status, source, detector, cooldown_seconds });
  } catch (error) {
    showMessage(error.message, true);
  }
}

async function postAction(url, body, successMessage) {
  try {
    await requestJson(url, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    showMessage(successMessage);
    await refreshStatus();
    await refreshEvents();
  } catch (error) {
    showMessage(error.message, true);
  }
}

startButton.addEventListener("click", () => postAction("/api/system/start", undefined, "Sistem başlatıldı."));
stopButton.addEventListener("click", () => postAction("/api/system/stop", undefined, "Sistem durduruldu."));
document.querySelector("#emergency-button").addEventListener("click", () => {
  postAction("/api/emergency-stop", undefined, "Acil durdurma uygulandı; ses kapatıldı ve caydırma devre dışı bırakıldı.");
});
deterrenceButton.addEventListener("click", () => {
  postAction("/api/deterrence", {
    enabled: deterrenceButton.getAttribute("aria-pressed") !== "true",
  }, "Caydırma ayarı güncellendi.");
});
soundButton.addEventListener("click", () => postAction("/api/test-sound", undefined, "Ses testi başlatıldı."));
document.querySelector("#refresh-button").addEventListener("click", () => {
  refreshEvents().catch((error) => showMessage(error.message, true));
});

refreshStatus();
refreshEvents().catch((error) => showMessage(error.message, true));
window.setInterval(refreshStatus, 1500);
window.setInterval(() => refreshEvents().catch((error) => showMessage(error.message, true)), 10000);

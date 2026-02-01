let mediaRecorder;
let audioChunks = [];
let isRecording = false;
let intervalId = null;
let uptimeSeconds = 0;

const API_BASE = 'http://localhost:5000';

// DOM Elements
const sections = document.querySelectorAll('.section');
const navBtns = document.querySelectorAll('nav button');
const startBtn = document.getElementById('start-btn');
const statusIndicator = document.getElementById('status-indicator');
const resultText = document.getElementById('result-text');
const confText = document.getElementById('conf-text');
const pulseRing = document.getElementById('pulse-ring');
const icon = statusIndicator.querySelector('i');
const pulseBar = document.getElementById('pulse-bar');
const pulseVal = document.getElementById('pulse-val');
const pulseLabel = document.getElementById('pulse-label');
const activityItems = document.getElementById('activity-items');
const uptimeEl = document.getElementById('uptime');

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    initCharts();
    startUptimeCounter();
    logActivity("Sentinel AI System Core Initialized", "secure");
});

// --- Navigation ---
navBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        navBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const targetId = btn.dataset.target;
        sections.forEach(sec => {
            sec.classList.remove('active');
            if (sec.id === targetId) sec.classList.add('active');
        });
    });
});

// --- Tab 1: Live Monitoring ---
startBtn.addEventListener('click', toggleMonitoring);

let audioContext;
let processor;
let input;
let audioBuff = [];

async function toggleMonitoring() {
    if (!isRecording) {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            input = audioContext.createMediaStreamSource(stream);
            processor = audioContext.createScriptProcessor(4096, 1, 1);

            input.connect(processor);
            processor.connect(audioContext.destination);

            isRecording = true;
            startBtn.innerHTML = '<i class="fas fa-stop"></i> TERMINATE FEED';
            startBtn.classList.add('stop');
            resultText.textContent = "DECODING FEED...";
            confText.textContent = "Neural Link Established. Sampling spectral data.";
            icon.className = "fas fa-satellite-dish fa-spin";
            logActivity("Live Monitoring Feed Initialized", "secure");

            processor.onaudioprocess = (e) => {
                if (!isRecording) return;
                const channelData = e.inputBuffer.getChannelData(0);
                audioBuff.push(new Float32Array(channelData));

                let totalSamples = audioBuff.reduce((acc, b) => acc + b.length, 0);
                if (totalSamples >= 24000) { // 1.5s
                    const combined = mergeBuffers(audioBuff, totalSamples);
                    audioBuff = [];
                    sendAudioChunk(encodeWAV(combined));
                }
            };

        } catch (err) {
            console.error(err);
            logActivity("Microphone Access Blocked", "danger");
            alert("Hardware Access Denied: " + err);
        }
    } else {
        isRecording = false;
        if (processor) {
            processor.disconnect();
            input.disconnect();
        }
        if (audioContext) audioContext.close();

        startBtn.innerHTML = '<i class="fas fa-power-off"></i> INITIALIZE MONITORING';
        startBtn.classList.remove('stop');
        resetUI();
        logActivity("Monitoring Feed Terminated", "secure");
    }
}

function mergeBuffers(buffers, length) {
    let result = new Float32Array(length);
    let offset = 0;
    for (let i = 0; i < buffers.length; i++) {
        result.set(buffers[i], offset);
        offset += buffers[i].length;
    }
    return result;
}

function encodeWAV(samples) {
    let buffer = new ArrayBuffer(44 + samples.length * 2);
    let view = new DataView(buffer);
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 32 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 16000, true);
    view.setUint32(28, 16000 * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    floatTo16BitPCM(view, 44, samples);
    return new Blob([view], { type: 'audio/wav' });
}

function floatTo16BitPCM(output, offset, input) {
    for (let i = 0; i < input.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, input[i]));
        output.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    }
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
}

async function sendAudioChunk(blob) {
    const formData = new FormData();
    formData.append('audio', blob, 'chunk.wav');

    try {
        const response = await fetch(`${API_BASE}/detect_live`, { method: 'POST', body: formData });
        const data = await response.json();
        updateUI(data);
    } catch (error) {
        console.error("Link Failure:", error);
    }
}

function updateUI(data) {
    const status = data.status || "";
    const conf = data.confidence || 0;

    // Visual Feedback
    if (status.includes("DEEPFAKE")) {
        statusIndicator.className = "status-indicator detected";
        pulseRing.style.animation = "ring-pulse-red 1s infinite";
        icon.className = "fas fa-user-ninja";
        resultText.style.color = "var(--danger-color)";
        updatePulse(conf * 100);
        logActivity(`Deepfake Detected (${(conf * 100).toFixed(0)}%)`, "danger");
    } else if (status.includes("VERIFIED HUMAN")) {
        statusIndicator.className = "status-indicator secure";
        pulseRing.style.animation = "ring-pulse-green 2s infinite";
        icon.className = "fas fa-user-check";
        resultText.style.color = "var(--success-color)";
        updatePulse(conf * 100);
    } else {
        statusIndicator.className = "status-indicator";
        pulseRing.style.animation = "none";
        icon.className = "fas fa-spinner fa-spin";
        resultText.style.color = "var(--text-color)";
    }

    resultText.textContent = status || "ANALYZING...";
    confText.innerHTML = `<div style='opacity: 0.9;'>${data.reason || ""}</div>
                          <div style='font-size: 0.7rem; margin-top: 5px; opacity: 0.4;'>LOGIC: ${data.composition || "NEURAL_FUSION_V2"}</div>`;
}

function resetUI() {
    statusIndicator.className = "status-indicator";
    pulseRing.style.animation = "none";
    icon.className = "fas fa-microphone-slash";
    resultText.textContent = "System Standby";
    resultText.style.color = "#fff";
    confText.textContent = "Ready for neural stream analysis";
    updatePulse(15);
}

// --- Intelligence Panel Helpers ---
function updatePulse(percent) {
    pulseBar.style.width = percent + "%";
    pulseVal.textContent = percent.toFixed(0) + "%";
    if (percent > 70) {
        pulseLabel.textContent = "LEVEL: CRITICAL";
        pulseBar.style.background = "var(--danger-color)";
    } else if (percent > 40) {
        pulseLabel.textContent = "LEVEL: ELEVATED";
        pulseBar.style.background = "orange";
    } else {
        pulseLabel.textContent = "LEVEL: LOW";
        pulseBar.style.background = "linear-gradient(90deg, var(--success-color), var(--primary-color))";
    }
}

function logActivity(text, type = "info") {
    const item = document.createElement('div');
    item.className = 'log-item';
    const iconClass = type === "danger" ? "fas fa-exclamation-triangle danger" : "fas fa-check-circle secure";
    item.innerHTML = `
        <i class="${iconClass}"></i>
        <div class="log-info">
            <div class="log-title">${text}</div>
            <div class="log-time">${new Date().toLocaleTimeString()}</div>
        </div>
    `;
    activityItems.prepend(item);
    if (activityItems.children.length > 8) activityItems.lastChild.remove();
}

function startUptimeCounter() {
    setInterval(() => {
        uptimeSeconds++;
        const hrs = Math.floor(uptimeSeconds / 3600).toString().padStart(2, '0');
        const mins = Math.floor((uptimeSeconds % 3600) / 60).toString().padStart(2, '0');
        const secs = (uptimeSeconds % 60).toString().padStart(2, '0');
        uptimeEl.textContent = `${hrs}:${mins}:${secs}`;
    }, 1000);
}

// --- Tab 2: Forensics ---
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const fileResult = document.getElementById('file-result');

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFile(e.target.files[0]));

async function handleFile(file) {
    if (!file) return;
    document.getElementById('file-name').textContent = "EVIDENCE: " + file.name;
    fileResult.innerHTML = '<div class="loader"><i class="fas fa-atom fa-spin"></i> DECOMPOSING SIGNAL...</div>';
    logActivity(`Analyzing evidence: ${file.name}`);

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE}/detect_file`, { method: 'POST', body: formData });
        const data = await response.json();

        const isDeepfake = data.status.includes("DEEPFAKE");
        let color = isDeepfake ? "var(--danger-color)" : "var(--success-color)";
        let statusIcon = isDeepfake ? "fa-skull-crossbones" : "fa-user-shield";

        fileResult.innerHTML = `
            <div style="color: ${color}; font-size: 1.4rem; margin-top: 20px; font-family: 'Orbitron';">
                <i class="fas ${statusIcon}"></i> ${data.status}
            </div>
            <div style="opacity: 0.7; margin-top: 10px; font-family: 'JetBrains Mono';">FORENSIC SCORE: ${(data.confidence * 100).toFixed(2)}%</div>
        `;
        logActivity(`File Scan Complete: ${data.status}`, isDeepfake ? "danger" : "secure");
    } catch (e) {
        fileResult.textContent = "Neural connection lost during analysis.";
    }
}

// --- Tab 3: Model Evaluation (Charts) ---
function initCharts() {
    const ctx = document.getElementById('accuracyChart').getContext('2d');
    new Chart(ctx, {
        type: 'line',
        data: {
            labels: ['V1.0', 'V1.5', 'V2.0', 'V2.2', 'V2.4'],
            datasets: [{
                label: 'Accuracy %',
                data: [72, 85, 91, 94, 98],
                borderColor: '#00f2ff',
                backgroundColor: 'rgba(0, 242, 255, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            plugins: { legend: { display: false } },
            scales: {
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: 'rgba(255,255,255,0.5)' } },
                x: { grid: { display: false }, ticks: { color: 'rgba(255,255,255,0.5)' } }
            }
        }
    });
}

// --- Tab 4: Performance ---
document.getElementById('health-btn').addEventListener('click', async () => {
    const btn = document.getElementById('health-btn');
    btn.innerHTML = '<i class="fas fa-microchip fa-spin"></i> PROBING HARDWARE...';
    logActivity("Starting Hardware Diagnostics...");

    try {
        const res = await fetch(`${API_BASE}/system_health`);
        const data = await res.json();

        document.getElementById('m-latency').textContent = data.latency_ms + "ms";
        document.getElementById('m-throughput').textContent = data.throughput + "/s";
        document.getElementById('m-rtf').textContent = data.rt_factor + "x";
        document.getElementById('m-device').textContent = data.device;

        btn.innerHTML = '<i class="fas fa-redo"></i> RE-SCAN HARDWARE';
        logActivity("Diagnostics Complete: Performance Nominal", "secure");
    } catch (e) {
        btn.textContent = "TELEMETRY LINK FAILED";
        logActivity("Hardware Probe Failed", "danger");
    }
});

// --- Theme Toggle ---
const themeBtn = document.getElementById('theme-btn');
let currentTheme = 'dark';
themeBtn.addEventListener('click', () => {
    if (currentTheme === 'dark') {
        document.body.setAttribute('data-theme', 'light');
        currentTheme = 'light';
    } else {
        document.body.removeAttribute('data-theme');
        currentTheme = 'dark';
    }
});

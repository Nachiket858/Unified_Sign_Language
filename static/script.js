/**
 * Unified Sign Language Detection - Frontend Logic
 * Handles mode switching, data polling, controls, and UI updates
 */

// ── State ──
let currentMode = 'ASL';
let isStreaming = false;
let pollInterval = null;
let lastPrediction = '';

// ── DOM Elements ──
const videoFeed = document.getElementById('video-feed');
const predictionValue = document.getElementById('prediction-value');
const sentenceText = document.getElementById('sentence-text');
const modeBtnASL = document.getElementById('mode-asl');
const modeBtnISL = document.getElementById('mode-isl');
const modeBadge = document.getElementById('mode-badge');
const streamStatus = document.getElementById('stream-status');
const streamDot = document.getElementById('stream-dot');
const gestureGuide = document.getElementById('gesture-guide');
const islKeysInfo = document.getElementById('isl-keys-info');

// ── Toast ──
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    const icons = { success: '✓', error: '✗', info: 'ℹ' };
    toast.innerHTML = `<span>${icons[type] || 'ℹ'}</span> ${message}`;
    container.appendChild(toast);
    setTimeout(() => toast.remove(), 3000);
}

// ── Mode Switching ──
function switchMode(mode) {
    if (mode === currentMode) return;

    // Show loading
    const videoContainer = document.querySelector('.video-container');
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'loading-overlay';
    loadingDiv.id = 'loading-overlay';
    loadingDiv.innerHTML = '<div class="spinner"></div>';
    videoContainer.appendChild(loadingDiv);

    fetch('/switch_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode })
    })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                currentMode = data.mode;
                updateModeUI();
                restartStream();
                showToast(`Switched to ${currentMode} mode`, 'success');
            }
        })
        .catch(err => {
            showToast('Failed to switch mode', 'error');
            console.error(err);
        })
        .finally(() => {
            const overlay = document.getElementById('loading-overlay');
            if (overlay) overlay.remove();
        });
}

function updateModeUI() {
    // Update buttons
    modeBtnASL.classList.toggle('active', currentMode === 'ASL');
    modeBtnISL.classList.toggle('active', currentMode === 'ISL');

    // Update badge
    modeBadge.textContent = currentMode;
    modeBadge.className = `card-badge badge-${currentMode.toLowerCase()}`;

    // Update gesture guide
    if (currentMode === 'ASL') {
        gestureGuide.src = '/static/images/asl_signs.jpeg';
        gestureGuide.alt = 'ASL Gesture Reference';
        islKeysInfo.style.display = 'none';
    } else {
        gestureGuide.src = '/static/images/isl_gestures.png';
        gestureGuide.alt = 'ISL Gesture Reference';
        islKeysInfo.style.display = 'block';
    }
}

// ── Video Stream ──
function startStream() {
    videoFeed.src = '/video_feed?' + new Date().getTime();
    isStreaming = true;
    streamStatus.textContent = 'Live';
    streamDot.classList.remove('inactive');
    streamDot.classList.add('active');
    startPolling();
}

function stopStream() {
    fetch('/stop', { method: 'POST' })
        .then(res => res.json())
        .then(() => {
            isStreaming = false;
            streamStatus.textContent = 'Stopped';
            streamDot.classList.remove('active');
            streamDot.classList.add('inactive');
            stopPolling();
            showToast('Stream stopped', 'info');
        });
}

function restartStream() {
    stopPolling();
    setTimeout(() => {
        videoFeed.src = '/video_feed?' + new Date().getTime();
        isStreaming = true;
        streamStatus.textContent = 'Live';
        streamDot.classList.remove('inactive');
        streamDot.classList.add('active');
        startPolling();
    }, 800);
}

// ── Data Polling ──
function startPolling() {
    stopPolling();
    pollInterval = setInterval(fetchData, 250);
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
    }
}

function fetchData() {
    fetch('/get_data')
        .then(res => res.json())
        .then(data => {
            // Update prediction
            const pred = data.prediction || '';
            if (pred !== lastPrediction) {
                predictionValue.textContent = pred || '—';
                predictionValue.classList.remove('pop');
                void predictionValue.offsetWidth; // Trigger reflow
                predictionValue.classList.add('pop');
                lastPrediction = pred;
            }

            // Update sentence
            const sentence = data.sentence || '';
            if (sentence) {
                sentenceText.innerHTML = sentence + '<span class="cursor"></span>';
            } else {
                sentenceText.innerHTML = '<span class="placeholder">Start signing to build a sentence...</span>';
            }
        })
        .catch(() => { }); // Silently handle errors during polling
}

// ── Controls ──
function clearSentence() {
    fetch('/clear_sentence', { method: 'POST' })
        .then(res => res.json())
        .then(() => {
            sentenceText.innerHTML = '<span class="placeholder">Start signing to build a sentence...</span>';
            predictionValue.textContent = '—';
            showToast('Sentence cleared', 'success');
        });
}

function speakText() {
    fetch('/speak', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                showToast('Speaking text...', 'info');
            } else {
                showToast('Nothing to speak', 'error');
            }
        })
        .catch(() => showToast('TTS unavailable', 'error'));
}

function deleteLast() {
    fetch('/delete_last', { method: 'POST' })
        .then(res => res.json())
        .then(() => showToast('Deleted last character', 'info'));
}

function addSpace() {
    fetch('/add_space', { method: 'POST' })
        .then(res => res.json())
        .then(() => showToast('Space added', 'info'));
}

function correctWithAI() {
    showToast('Correcting with Gemini AI...', 'info');
    fetch('/correct', { method: 'POST' })
        .then(res => res.json())
        .then(data => {
            if (data.status === 'success') {
                sentenceText.innerHTML = data.corrected + '<span class="cursor"></span>';
                showToast(`Corrected: "${data.original}" → "${data.corrected}"`, 'success');
            } else {
                showToast(data.message || 'Correction failed', 'error');
            }
        })
        .catch(() => showToast('AI correction unavailable', 'error'));
}

function toggleStream() {
    if (isStreaming) {
        stopStream();
    } else {
        startStream();
    }
}

// ── Model Status ──
function fetchModelStatus() {
    fetch('/model_status')
        .then(res => res.json())
        .then(data => {
            document.getElementById('asl-model-status').textContent = data.asl_model ? 'Loaded' : 'Error';
            document.getElementById('asl-model-status').className = `model-badge ${data.asl_model ? 'loaded' : 'error'}`;

            document.getElementById('isl-model-status').textContent = data.isl_model ? 'Loaded' : 'Error';
            document.getElementById('isl-model-status').className = `model-badge ${data.isl_model ? 'loaded' : 'error'}`;

            document.getElementById('tts-status').textContent = data.tts_engine ? 'Ready' : 'N/A';
            document.getElementById('tts-status').className = `model-badge ${data.tts_engine ? 'loaded' : 'error'}`;

            document.getElementById('gemini-status').textContent = data.gemini ? 'Ready' : 'No Key';
            document.getElementById('gemini-status').className = `model-badge ${data.gemini ? 'loaded' : 'error'}`;
        });
}

// ── Open gesture image in new tab ──
function openGestureGuide() {
    window.open(gestureGuide.src, '_blank');
}

// ── Initialize ──
document.addEventListener('DOMContentLoaded', () => {
    updateModeUI();
    startStream();
    fetchModelStatus();
});

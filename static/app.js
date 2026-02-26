const form = document.getElementById('generateForm');
const textArea = document.getElementById('text');
const voiceSelect = document.getElementById('voice');
const generateBtn = document.getElementById('generateBtn');
const stopBtn = document.getElementById('stopBtn');
const progressDiv = document.getElementById('progress');
const errorDiv = document.getElementById('error');
const audioPlayerDiv = document.getElementById('audioPlayer');
const themeToggle = document.getElementById('themeToggle');

// Custom audio player control refs
const playPauseBtn = document.getElementById('playPauseBtn');
const playPauseIcon = document.getElementById('playPauseIcon');
const timeDisplay = document.getElementById('timeDisplay');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const volumeIcon = document.getElementById('volumeIcon');
const volumeSlider = document.getElementById('volumeSlider');

const toastContainer = document.getElementById('toastContainer');

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    toastContainer.appendChild(toast);

    // Remove after animation completes
    setTimeout(() => {
        toast.remove();
    }, 4000);
}

// Add voice to dropdown (maintaining sorted order)
function addVoiceToDropdown(voiceName) {
    // Check if already exists
    const existing = Array.from(voiceSelect.options).find(opt => opt.value === voiceName);
    if (existing) return;

    const option = document.createElement('option');
    option.value = voiceName;
    option.textContent = voiceName;

    // Insert in sorted position
    const options = Array.from(voiceSelect.options);
    const insertIndex = options.findIndex(opt => opt.value > voiceName);

    if (insertIndex === -1) {
        voiceSelect.appendChild(option);
    } else {
        voiceSelect.insertBefore(option, voiceSelect.options[insertIndex]);
    }
}

// Remove voice from dropdown
function removeVoiceFromDropdown(voiceName) {
    const option = Array.from(voiceSelect.options).find(opt => opt.value === voiceName);
    if (option) {
        option.remove();
    }
}

// Voice event source for cleanup
let voiceEventSource = null;

// Subscribe to voice events
function subscribeToVoiceEvents() {
    if (voiceEventSource) {
        voiceEventSource.close();
    }
    voiceEventSource = new EventSource('/voice-events');

    voiceEventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'processing') {
            showToast(`Processing '${data.voice}'...`, 'info');
        } else if (data.type === 'ready') {
            showToast(`Voice '${data.voice}' ready`, 'success');
            addVoiceToDropdown(data.voice);
        } else if (data.type === 'error') {
            showToast(`Failed to load '${data.voice}': ${data.reason}`, 'error');
        } else if (data.type === 'removed') {
            showToast(`Voice '${data.voice}' removed`, 'info');
            removeVoiceFromDropdown(data.voice);
        }
    };

    voiceEventSource.onerror = (error) => {
        console.error('Voice events connection error:', error);
        voiceEventSource.close();
        // Reconnect after 5 seconds
        setTimeout(subscribeToVoiceEvents, 5000);
    };
}

let audioBuffers = [];
let audioContext = null;
let processingChunk = false;
let chunkQueue = [];
let currentAbortController = null;
let currentGenerationId = null;  // Track generation ID for stop requests

// AudioBufferSourceNode scheduling state
let nextStartTime = 0;
let gainNode = null;
let isStreaming = false;

// Playback state management
let isPlaying = false;
let playbackStartTime = 0;        // audioContext.currentTime when play started
let playbackOffset = 0;           // where in the audio we started from
let totalDuration = 0;            // sum of all buffer durations
let currentSource = null;         // current AudioBufferSourceNode
let animationFrameId = null;      // for requestAnimationFrame loop
let previousVolume = 100;         // for mute/unmute toggle

// Theme handling
function setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    themeToggle.textContent = theme === 'light' ? '\u2600\uFE0F' : '\uD83C\uDF19';
    localStorage.setItem('theme', theme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    setTheme(currentTheme === 'light' ? 'dark' : 'light');
}

// Load saved theme or default to dark
const savedTheme = localStorage.getItem('theme') || 'dark';
setTheme(savedTheme);

themeToggle.addEventListener('click', toggleTheme);

// Format time in MM:SS or H:MM:SS for times >= 1 hour
function formatTime(seconds) {
    if (!isFinite(seconds) || seconds < 0) seconds = 0;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (hours > 0) {
        return `${hours}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

// Get current playback time
function getCurrentTime() {
    if (!audioContext) return 0;
    if (isPlaying) {
        return playbackOffset + (audioContext.currentTime - playbackStartTime);
    }
    return playbackOffset;
}

// Update time display and progress bar
function updateTimeDisplay() {
    const current = getCurrentTime();
    const duration = totalDuration;

    timeDisplay.textContent = `${formatTime(current)} / ${formatTime(duration)}`;

    if (duration > 0) {
        const percentage = Math.min((current / duration) * 100, 100);
        progressBar.style.width = `${percentage}%`;
        progressContainer.setAttribute('aria-valuenow', Math.round(percentage));
    } else {
        progressBar.style.width = '0%';
        progressContainer.setAttribute('aria-valuenow', 0);
    }
}

// Start the playback animation loop
function startPlaybackLoop() {
    function loop() {
        updateTimeDisplay();

        // Check if we've reached the end
        const current = getCurrentTime();
        if (current >= totalDuration && totalDuration > 0 && !isStreaming) {
            // Playback ended naturally
            handlePlaybackEnded();
            return;
        }

        if (isPlaying) {
            animationFrameId = requestAnimationFrame(loop);
        }
    }
    animationFrameId = requestAnimationFrame(loop);
}

// Stop the playback animation loop
function stopPlaybackLoop() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
}

// Handle when playback reaches the end naturally
function handlePlaybackEnded() {
    isPlaying = false;
    playbackOffset = 0;
    currentSource = null;
    playPauseIcon.textContent = '\u25B6';
    playPauseIcon.dataset.state = 'play';
    stopPlaybackLoop();
    updateTimeDisplay();
}

// Reset all playback state for a new generation
function resetPlaybackState() {
    // Stop current playback if playing
    if (isPlaying) {
        stopPlayback();
    }

    // Stop any existing source
    if (currentSource) {
        try {
            currentSource.onended = null;
            currentSource.stop();
        } catch (e) {
            // Source may already be stopped
        }
        currentSource = null;
    }

    // Cancel animation frames
    stopPlaybackLoop();

    // Reset playback state variables
    isPlaying = false;
    playbackStartTime = 0;
    playbackOffset = 0;
    totalDuration = 0;

    // Reset UI
    playPauseIcon.textContent = '\u25B6';
    playPauseIcon.dataset.state = 'play';
    timeDisplay.textContent = '0:00 / 0:00';
    progressBar.style.width = '0%';
    progressContainer.setAttribute('aria-valuenow', 0);

    // Hide audio player
    audioPlayerDiv.classList.remove('visible');
}

// Start playback from a given offset
function startPlayback(fromOffset = playbackOffset) {
    if (audioBuffers.length === 0) return;

    // Stop any existing source
    if (currentSource) {
        try {
            currentSource.onended = null;
            currentSource.stop();
        } catch (e) {
            // Source may already be stopped
        }
        currentSource = null;
    }

    // Concatenate all buffers
    const concatenatedBuffer = concatenateAudioBuffers(audioBuffers);
    if (!concatenatedBuffer) return;

    // Clamp offset to valid range
    fromOffset = Math.max(0, Math.min(fromOffset, concatenatedBuffer.duration));

    // Create new AudioBufferSourceNode
    const source = audioContext.createBufferSource();
    source.buffer = concatenatedBuffer;
    source.connect(gainNode);

    // Start at the specified offset
    source.start(0, fromOffset);

    // Update state
    currentSource = source;
    isPlaying = true;
    playbackStartTime = audioContext.currentTime;
    playbackOffset = fromOffset;

    // Update UI
    playPauseIcon.textContent = '\u23F8';
    playPauseIcon.dataset.state = 'pause';
    audioPlayerDiv.classList.add('visible');

    // Start animation loop
    startPlaybackLoop();

    // Handle natural end of playback
    source.onended = () => {
        // Only handle if this is still the current source and we're still "playing"
        if (currentSource === source && isPlaying) {
            // Calculate where we should be in the audio
            const elapsedTime = audioContext.currentTime - playbackStartTime;
            const endedAtPosition = fromOffset + elapsedTime;

            // Check if more audio has arrived since we started
            // (totalDuration may have grown while we were playing)
            if (isStreaming && endedAtPosition < totalDuration) {
                // More audio available - restart from where we left off
                console.log(`Continuing playback: ended at ${endedAtPosition.toFixed(2)}s, total now ${totalDuration.toFixed(2)}s`);
                startPlayback(endedAtPosition);
            } else if (!isStreaming && endedAtPosition < totalDuration - 0.1) {
                // Not streaming but still more audio (edge case: final chunks arrived just before end)
                console.log(`Final continuation: ended at ${endedAtPosition.toFixed(2)}s, total ${totalDuration.toFixed(2)}s`);
                startPlayback(endedAtPosition);
            } else {
                // Truly finished
                handlePlaybackEnded();
            }
        }
    };
}

// Stop/pause playback
function stopPlayback() {
    if (!isPlaying) return;

    // Record current position
    playbackOffset = getCurrentTime();

    // Stop the source
    if (currentSource) {
        try {
            currentSource.onended = null;
            currentSource.stop();
        } catch (e) {
            // Source may already be stopped
        }
        currentSource = null;
    }

    // Update state
    isPlaying = false;
    playPauseIcon.textContent = '\u25B6';
    playPauseIcon.dataset.state = 'play';

    // Stop animation loop
    stopPlaybackLoop();
    updateTimeDisplay();
}

// Toggle play/pause
function togglePlayPause() {
    if (isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

// Seek to a position (0-1 ratio)
function seekTo(ratio) {
    const targetTime = ratio * totalDuration;
    playbackOffset = targetTime;

    if (isPlaying) {
        // Restart playback from new position
        startPlayback(targetTime);
    } else {
        // Just update display
        updateTimeDisplay();
    }
}

// Event listeners for audio controls
playPauseBtn.addEventListener('click', togglePlayPause);

// Volume slider control
volumeSlider.addEventListener('input', () => {
    const volume = volumeSlider.value / 100;
    if (gainNode) {
        gainNode.gain.value = volume;
    }

    // Update volume icon
    if (volume === 0) {
        volumeIcon.textContent = '\uD83D\uDD07';
    } else if (volume < 0.5) {
        volumeIcon.textContent = '\uD83D\uDD09';
    } else {
        volumeIcon.textContent = '\uD83D\uDD0A';
    }
});

// Volume icon click to toggle mute
volumeIcon.addEventListener('click', () => {
    if (volumeSlider.value > 0) {
        // Mute: store current volume and set to 0
        previousVolume = volumeSlider.value;
        volumeSlider.value = 0;
    } else {
        // Unmute: restore previous volume
        volumeSlider.value = previousVolume || 100;
    }
    // Trigger input event to update gainNode
    volumeSlider.dispatchEvent(new Event('input'));
});

// Progress bar click to seek
progressContainer.addEventListener('click', (e) => {
    if (totalDuration <= 0) return;

    const rect = progressContainer.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    seekTo(Math.max(0, Math.min(1, ratio)));
});

// Progress bar drag to seek
let isDraggingProgress = false;
let dragRatio = 0;

function updateDragVisual(ratio) {
    // Update visual without actually seeking (to avoid creating many AudioBufferSourceNodes)
    const clampedRatio = Math.max(0, Math.min(1, ratio));
    const percentage = clampedRatio * 100;
    progressBar.style.width = `${percentage}%`;
    progressContainer.setAttribute('aria-valuenow', Math.round(percentage));

    // Update time display to show where we'd seek to
    const targetTime = clampedRatio * totalDuration;
    timeDisplay.textContent = `${formatTime(targetTime)} / ${formatTime(totalDuration)}`;
}

progressContainer.addEventListener('mousedown', (e) => {
    if (totalDuration <= 0) return;

    isDraggingProgress = true;
    const rect = progressContainer.getBoundingClientRect();
    dragRatio = (e.clientX - rect.left) / rect.width;
    updateDragVisual(dragRatio);

    // Prevent text selection while dragging
    e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
    if (!isDraggingProgress) return;

    const rect = progressContainer.getBoundingClientRect();
    dragRatio = (e.clientX - rect.left) / rect.width;
    updateDragVisual(dragRatio);
});

document.addEventListener('mouseup', (e) => {
    if (!isDraggingProgress) return;

    isDraggingProgress = false;

    // Finalize seek to the drag position
    const finalRatio = Math.max(0, Math.min(1, dragRatio));
    seekTo(finalRatio);
});

// Keyboard support for progress bar
progressContainer.addEventListener('keydown', (e) => {
    if (totalDuration <= 0) return;

    const currentRatio = getCurrentTime() / totalDuration;
    let newRatio = currentRatio;

    switch (e.key) {
        case 'ArrowLeft':
            newRatio = Math.max(0, currentRatio - 0.05);
            break;
        case 'ArrowRight':
            newRatio = Math.min(1, currentRatio + 0.05);
            break;
        case 'Home':
            newRatio = 0;
            break;
        case 'End':
            newRatio = 1;
            break;
        default:
            return;
    }

    e.preventDefault();
    seekTo(newRatio);
});

// Load voices on page load
async function loadVoices() {
    try {
        const response = await fetch('/voices');
        const data = await response.json();

        // Clear existing options using DOM methods
        while (voiceSelect.firstChild) {
            voiceSelect.removeChild(voiceSelect.firstChild);
        }

        if (data.voices.length === 0) {
            const placeholder = document.createElement('option');
            placeholder.value = '';
            placeholder.textContent = 'No voices available';
            voiceSelect.appendChild(placeholder);
            return;
        }

        data.voices.forEach(voice => {
            const option = document.createElement('option');
            option.value = voice;
            option.textContent = voice;
            voiceSelect.appendChild(option);
        });
    } catch (error) {
        console.error('Failed to load voices:', error);
        while (voiceSelect.firstChild) {
            voiceSelect.removeChild(voiceSelect.firstChild);
        }
        const errorOption = document.createElement('option');
        errorOption.value = '';
        errorOption.textContent = 'Error loading voices';
        voiceSelect.appendChild(errorOption);
    }
}

// Handle form submission
form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const text = textArea.value.trim();
    const voice = voiceSelect.value;

    if (!text || !voice) {
        showError('Please enter text and select a voice');
        return;
    }

    // Reset all playback state before starting new generation
    resetPlaybackState();

    // Close existing AudioContext if present
    if (audioContext) {
        try {
            audioContext.close();
        } catch (e) {}
    }

    // Reset audio data state
    audioBuffers = [];
    chunkQueue = [];
    processingChunk = false;

    // Create fresh AudioContext
    audioContext = new (window.AudioContext || window.webkitAudioContext)();

    // Create GainNode for volume control and connect to destination
    gainNode = audioContext.createGain();
    gainNode.connect(audioContext.destination);
    // Apply current volume slider value
    gainNode.gain.value = volumeSlider.value / 100;

    // Reset scheduling state
    nextStartTime = 0;
    isStreaming = true;

    // Clear UI messages
    progressDiv.classList.remove('visible');
    errorDiv.classList.remove('visible');
    generateBtn.disabled = true;

    // Start generation
    try {
        currentAbortController = new AbortController();
        const response = await fetch('/generate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, voice }),
            signal: currentAbortController.signal,
        });
        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.detail || `Server error: ${response.status}`);
        }
        stopBtn.disabled = false;

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const data = JSON.parse(line.slice(6));

                if (data.type === 'start') {
                    currentGenerationId = data.generation_id;
                    showProgress(`Generating ${data.chunks} chunks...`);
                } else if (data.type === 'progress') {
                    showProgress(data.message);
                } else if (data.type === 'chunk') {
                    handleAudioChunk(data.data);
                } else if (data.type === 'complete') {
                    currentGenerationId = null;
                    isStreaming = false;
                    showProgress('Generation complete!');
                    generateBtn.disabled = false;
                    stopBtn.disabled = true;
                } else if (data.type === 'error') {
                    currentGenerationId = null;
                    isStreaming = false;
                    showError(data.message);
                    generateBtn.disabled = false;
                    stopBtn.disabled = true;
                }
            }
        }
    } catch (error) {
        if (error.name !== 'AbortError') {
            console.error('Generation error:', error);
            showError(error.message || 'Failed to start generation');
        }
        currentAbortController = null;
        currentGenerationId = null;
        isStreaming = false;
        generateBtn.disabled = false;
        stopBtn.disabled = true;
    }
});

// Stop generation
function stopGeneration() {
    if (currentAbortController) {
        // Request backend to stop this specific generation with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const stopUrl = currentGenerationId !== null
            ? `/stop?generation_id=${currentGenerationId}`
            : '/stop';
        fetch(stopUrl, { method: 'POST', signal: controller.signal })
            .catch(e => console.error('Failed to send stop request:', e))
            .finally(() => clearTimeout(timeoutId));
        currentAbortController.abort();
        currentAbortController = null;
        currentGenerationId = null;
        isStreaming = false;
        showProgress('Generation stopped');
        generateBtn.disabled = false;
        stopBtn.disabled = true;

        // Stop audio playback but keep buffers for replay
        if (isPlaying) {
            stopPlayback();
        }

        // Reset playback position to beginning for replay
        playbackOffset = 0;
        updateTimeDisplay();

        // Note: We don't clear audioBuffers or hide the player,
        // so the user can still replay what was generated
    }
}

// Stop button click handler
stopBtn.addEventListener('click', stopGeneration);

function showProgress(message) {
    progressDiv.textContent = message;
    progressDiv.classList.add('visible');
}

function showError(message) {
    errorDiv.textContent = message;
    errorDiv.classList.add('visible');
}

async function handleAudioChunk(base64Data) {
    // Add to queue
    chunkQueue.push(base64Data);

    // Process queue if not already processing
    if (!processingChunk) {
        await processChunkQueue();
    }
}

async function processChunkQueue() {
    if (chunkQueue.length === 0) {
        processingChunk = false;
        return;
    }

    processingChunk = true;
    const base64Data = chunkQueue.shift();

    console.log(`Processing chunk ${audioBuffers.length + 1}, ${chunkQueue.length} remaining in queue`);

    // Decode base64 to array buffer
    const binaryString = atob(base64Data);
    const bytes = new Uint8Array(binaryString.length);
    for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }

    // Decode audio data using Web Audio API
    let audioBuffer;
    try {
        audioBuffer = await audioContext.decodeAudioData(bytes.buffer);
    } catch (error) {
        console.error('Failed to decode audio chunk:', error);
        showError('Failed to decode audio chunk. The stream may be corrupted.');
        processingChunk = false;
        return;
    }
    audioBuffers.push(audioBuffer);

    // Update total duration
    totalDuration += audioBuffer.duration;

    console.log(`Decoded chunk ${audioBuffers.length}: ${audioBuffer.duration.toFixed(2)}s, total duration: ${totalDuration.toFixed(2)}s`);

    // Show audio player and update time display
    audioPlayerDiv.classList.add('visible');
    updateTimeDisplay();

    // Auto-start playback on first chunk
    if (audioBuffers.length === 1 && !isPlaying) {
        startPlayback(0);
    }

    // Process next chunk in queue
    await processChunkQueue();
}

function concatenateAudioBuffers(buffers) {
    if (buffers.length === 0) return null;
    if (buffers.length === 1) return buffers[0];

    const sampleRate = buffers[0].sampleRate;
    const numberOfChannels = buffers[0].numberOfChannels;

    // Calculate total length
    const totalLength = buffers.reduce((sum, buf) => sum + buf.length, 0);

    // Create new buffer
    const result = audioContext.createBuffer(numberOfChannels, totalLength, sampleRate);

    // Copy data from each buffer
    let offset = 0;
    for (const buffer of buffers) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
            result.getChannelData(channel).set(buffer.getChannelData(channel), offset);
        }
        offset += buffer.length;
    }

    return result;
}

// Encode AudioBuffer as WAV (PCM 16-bit)
function audioBufferToWav(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const bitsPerSample = 16;
    const bytesPerSample = bitsPerSample / 8;
    const blockAlign = numChannels * bytesPerSample;
    const numSamples = buffer.length;
    const dataSize = numSamples * blockAlign;
    const headerSize = 44;
    const totalSize = headerSize + dataSize;

    const arrayBuffer = new ArrayBuffer(totalSize);
    const view = new DataView(arrayBuffer);

    // WAV header
    writeString(view, 0, 'RIFF');
    view.setUint32(4, totalSize - 8, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);              // fmt chunk size
    view.setUint16(20, 1, true);               // PCM format
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true); // byte rate
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitsPerSample, true);
    writeString(view, 36, 'data');
    view.setUint32(40, dataSize, true);

    // Interleave channels and convert Float32 to Int16
    let offset = headerSize;
    for (let i = 0; i < numSamples; i++) {
        for (let ch = 0; ch < numChannels; ch++) {
            const sample = buffer.getChannelData(ch)[i];
            const clamped = Math.max(-1, Math.min(1, sample));
            view.setInt16(offset, clamped < 0 ? clamped * 0x8000 : clamped * 0x7FFF, true);
            offset += 2;
        }
    }

    return new Blob([arrayBuffer], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
    for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
    }
}

// Encode AudioBuffer as MP3 using lamejs
function audioBufferToMp3(buffer) {
    const numChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const kbps = 128;

    // Convert Float32 to Int16 arrays
    const left = floatTo16BitPCM(buffer.getChannelData(0));
    const right = numChannels > 1 ? floatTo16BitPCM(buffer.getChannelData(1)) : null;

    const encoder = new lamejs.Mp3Encoder(numChannels, sampleRate, kbps);
    const mp3Data = [];
    const blockSize = 1152;

    for (let i = 0; i < left.length; i += blockSize) {
        const leftChunk = left.subarray(i, i + blockSize);
        let mp3buf;
        if (numChannels === 1) {
            mp3buf = encoder.encodeBuffer(leftChunk);
        } else {
            const rightChunk = right.subarray(i, i + blockSize);
            mp3buf = encoder.encodeBuffer(leftChunk, rightChunk);
        }
        if (mp3buf.length > 0) {
            mp3Data.push(mp3buf);
        }
    }

    const end = encoder.flush();
    if (end.length > 0) {
        mp3Data.push(end);
    }

    return new Blob(mp3Data, { type: 'audio/mpeg' });
}

function floatTo16BitPCM(float32Array) {
    const int16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        int16[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return int16;
}

// Trigger file download from a Blob
function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Generate filename: longecho-{voice}-{YYYYMMDD-HHmmss}.{ext}
function generateFilename(ext) {
    const voice = voiceSelect.value || 'audio';
    const now = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    const timestamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}-${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    return `longecho-${voice}-${timestamp}.${ext}`;
}

// Download button refs
const downloadBtn = document.getElementById('downloadBtn');
const downloadBtnText = document.getElementById('downloadBtnText');
const downloadFormat = document.getElementById('downloadFormat');

downloadBtn.addEventListener('click', async () => {
    if (audioBuffers.length === 0) return;

    const format = downloadFormat.value;
    const concatenated = concatenateAudioBuffers(audioBuffers);
    if (!concatenated) return;

    if (format === 'wav') {
        const blob = audioBufferToWav(concatenated);
        downloadBlob(blob, generateFilename('wav'));
    } else if (format === 'mp3') {
        downloadBtn.disabled = true;
        downloadBtnText.textContent = 'Encoding...';
        // Yield to UI so the button text updates
        await new Promise(r => setTimeout(r, 0));
        try {
            const blob = audioBufferToMp3(concatenated);
            downloadBlob(blob, generateFilename('mp3'));
        } finally {
            downloadBtn.disabled = false;
            downloadBtnText.textContent = 'Download';
        }
    }
});

// Load voices on page load
loadVoices();
subscribeToVoiceEvents();

document.addEventListener("DOMContentLoaded", () => {
    
    const chatContainer = document.getElementById('chat-container');
    const bottomGap = document.querySelector('.bottom-gap');
    const widget = document.getElementById('detection-widget');

    // Spawn Widget Handlers
    document.getElementById('btn-spawn-widget').addEventListener('click', () => {
        widget.classList.remove('hidden');
        chatContainer.insertBefore(widget, bottomGap);
        appendMessage('user', 'I want to run a change detection scan.', false);
        setTimeout(() => { 
            chatContainer.scrollTop = chatContainer.scrollHeight;
            map.invalidateSize(); 
        }, 200);
    });

    const vlmSpawnBtn = document.getElementById('btn-spawn-vlm');
    if (vlmSpawnBtn) {
        vlmSpawnBtn.addEventListener('click', () => {
            widget.classList.remove('hidden');
            chatContainer.insertBefore(widget, bottomGap);
            appendMessage('user', 'I want to analyze a single satellite image.', false);
            document.getElementById('nav-vlm-btn').click();
            setTimeout(() => { 
                chatContainer.scrollTop = chatContainer.scrollHeight; 
            }, 200);
        });
    }

    // Tab Navigation Buttons
    const mapBtn = document.getElementById('nav-map-btn');
    const manualBtn = document.getElementById('nav-manual-btn');
    const vlmBtn = document.getElementById('nav-vlm-btn');
    const crossmodalBtn = document.getElementById('nav-crossmodal-btn');
    
    // Tab View Containers
    const mapContainer = document.getElementById('map-mode-container');
    const manualContainer = document.getElementById('manual-mode-container');
    const vlmContainer = document.getElementById('vlm-mode-container');
    const crossmodalContainer = document.getElementById('crossmodal-mode-container');
    
    const submitBtn = document.getElementById('submit-btn');
    let currentMode = 'map'; 

    function resetTabViews() {
        mapBtn.classList.remove('active');
        manualBtn.classList.remove('active');
        if (vlmBtn) vlmBtn.classList.remove('active');
        if (crossmodalBtn) crossmodalBtn.classList.remove('active');
        
        mapContainer.classList.add('hidden');
        manualContainer.classList.add('hidden');
        if (vlmContainer) vlmContainer.classList.add('hidden');
        if (crossmodalContainer) crossmodalContainer.classList.add('hidden');
    }

    mapBtn.addEventListener('click', () => {
        resetTabViews();
        currentMode = 'map';
        mapBtn.classList.add('active');
        mapContainer.classList.remove('hidden');
        setTimeout(() => { map.invalidateSize(); }, 200);
    });

    manualBtn.addEventListener('click', () => {
        resetTabViews();
        currentMode = 'manual';
        manualBtn.classList.add('active');
        manualContainer.classList.remove('hidden');
    });

    if (vlmBtn) {
        vlmBtn.addEventListener('click', () => {
            resetTabViews();
            currentMode = 'vlm';
            vlmBtn.classList.add('active');
            vlmContainer.classList.remove('hidden');
        });
    }

    if (crossmodalBtn) {
        crossmodalBtn.addEventListener('click', () => {
            resetTabViews();
            currentMode = 'crossmodal';
            crossmodalBtn.classList.add('active');
            crossmodalContainer.classList.remove('hidden');
        });
    }

    // Leaflet Setup
    const map = L.map('map').setView([28.367, 76.977], 14); 
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
        maxZoom: 19,
        attribution: '&copy; OpenStreetMap'
    }).addTo(map);

    const drawnItems = new L.FeatureGroup();
    map.addLayer(drawnItems);
    const drawControl = new L.Control.Draw({
        edit: { featureGroup: drawnItems },
        draw: { polygon: false, polyline: false, circle: false, marker: false, circlemarker: false }
    });
    map.addControl(drawControl);

    let currentBbox = null;
    let estimatedSeconds = 30;

    function getBBoxAreaKm2(bbox) {
        const [minLon, minLat, maxLon, maxLat] = bbox;
        const latDist = (maxLat - minLat) * 111; 
        const lonDist = (maxLon - minLon) * 111 * Math.cos(((minLat + maxLat) / 2) * (Math.PI / 180));
        return Math.abs(latDist * lonDist);
    }

    async function updateEstimate() {
        if (!currentBbox) return;
        const t1 = document.getElementById('t1-date').value;
        const t2 = document.getElementById('t2-date').value;
        const modality = document.getElementById('modality-select') ? document.getElementById('modality-select').value : 'optical';
        
        try {
            const res = await fetch('/api/estimate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ bbox: currentBbox, t1_date: t1, t2_date: t2, modality: modality })
            });
            const data = await res.json();
            estimatedSeconds = data.estimated_seconds || 30;
            document.getElementById('time-estimate').innerText = `~${estimatedSeconds} seconds`;
        } catch(e) { console.error("Estimate failed", e); }
    }

    document.getElementById('modality-select')?.addEventListener('change', updateEstimate);

    map.on(L.Draw.Event.CREATED, function (e) {
        drawnItems.clearLayers();
        const layer = e.layer;
        drawnItems.addLayer(layer);
        
        const bounds = layer.getBounds();
        currentBbox = [bounds.getWest(), bounds.getSouth(), bounds.getEast(), bounds.getNorth()];
        
        const area = getBBoxAreaKm2(currentBbox);
        document.getElementById('area-info').innerText = `Area: ${area.toFixed(2)} km²`;
        
        const warning = document.getElementById('warning-msg');
        if (area > 500) {
            submitBtn.disabled = true;
            warning.innerText = "Too large (>500 km²).";
            document.getElementById('time-estimate').innerText = "";
        } else {
            submitBtn.disabled = false;
            warning.innerText = area > 100 ? "Large area." : "";
            document.getElementById('time-estimate').innerText = "Calculating...";
            updateEstimate();
        }
    });

    // Image Upload Preview Helper
    function setupImagePreview(inputId, previewId, textId) {
        const input = document.getElementById(inputId);
        const preview = document.getElementById(previewId);
        const text = document.getElementById(textId);
        if (!input || !preview) return;

        input.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.classList.remove('hidden');
                    if (text) text.classList.add('hidden');
                }
                reader.readAsDataURL(file);
            } else {
                preview.src = '';
                preview.classList.add('hidden');
                if (text) text.classList.remove('hidden');
            }
        });
    }

    // Attach Previews
    setupImagePreview('pre_image', 'prePreview', 'preText');
    setupImagePreview('post_image', 'postPreview', 'postText');
    setupImagePreview('vlm_image', 'vlmPreview', 'vlmPlaceholder');
    setupImagePreview('cm_opt_file', 'cmOptPreview', 'cmOptText');
    setupImagePreview('cm_sar_file', 'cmSarPreview', 'cmSarText');

    document.querySelectorAll('.quick-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            const promptInput = document.getElementById('vlm_prompt');
            if (promptInput) promptInput.value = chip.getAttribute('data-vlm-prompt');
        });
    });

    function appendMessage(role, contentText, isRawHtml = false) {
        const wrapper = document.createElement('div');
        wrapper.className = role === 'user' ? 'user-chat' : 'ai-chat';
        
        let finalHtml = contentText;
        if (role === 'ai' && !isRawHtml) {
            finalHtml = marked.parse(contentText);
        }

        wrapper.innerHTML = `<div class="chat-content">${finalHtml}</div>`;
        chatContainer.insertBefore(wrapper, bottomGap);
        
        setTimeout(() => {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }, 150);
    }
    
    // UI HELPER: Generates the HTML for individual event patches
    function generateEventsHtml(events) {
        let html = '<div class="events-slider">';
        events.forEach(ev => {
            const sev = (ev.severity || ev.Severity || 'LOW').toUpperCase();
            let sevColor = sev === 'HIGH' ? '#f87171' : sev === 'MEDIUM' ? '#fbbf24' : '#34d399';
            let bgSev = sev === 'HIGH' ? 'rgba(248,113,113,0.1)' : sev === 'MEDIUM' ? 'rgba(251,191,36,0.1)' : 'rgba(52,211,153,0.1)';
            
            const activityType = ev.activity_type || ev.Type || 'Unclassified Change';
            const area = ev.area_sq_m || ev.Area || '-';
            const confidence = ev.confidence || ev.Confidence || '-';

            html += `
                <div class="result-card">
                    <div class="result-header">
                        <div>
                            <div class="result-title">Event #${ev.id || '-'} - ${activityType}</div>
                            <div class="result-meta">Area: ${area} m² | Confidence: ${confidence}%</div>
                        </div>
                        <div class="severity-badge" style="color: ${sevColor}; border: 1px solid ${sevColor}; background: ${bgSev};">
                            ${sev}
                        </div>
                    </div>
            `;
            
            if (ev.t1_patch && ev.t2_patch) {
                html += `
                    <div class="comparison-grid" style="margin-top:auto;">
                        <div class="img-wrapper">
                            <span class="img-label">T1 Patch</span>
                            <img src="${ev.t1_patch}" alt="T1 Patch">
                        </div>
                        <div class="img-wrapper">
                            <span class="img-label">T2 Patch</span>
                            <img src="${ev.t2_patch}" alt="T2 Patch">
                        </div>
                    </div>
                `;
            }
            html += `</div>`;
        });
        html += '</div>';
        return html;
    }

    // Submit Routing
    submitBtn.addEventListener('click', async () => {
        if (currentMode === 'map') {
            await runMapDetection();
        } else if (currentMode === 'manual') {
            await runManualDetection();
        } else if (currentMode === 'vlm') {
            await runVlmDetection();
        } else if (currentMode === 'crossmodal') {
            await runCrossModalDetection();
        }
    });

    async function runCrossModalDetection() {
        const optFile = document.getElementById('cm_opt_file').files[0];
        const sarFile = document.getElementById('cm_sar_file').files[0];
        const query = document.getElementById('cm_query').value.trim();

        if (!optFile || !sarFile) {
            return alert("Please upload both an Optical/Multispectral image and a co-registered SAR image.");
        }

        const progContainer = document.getElementById('cm-mode-progress-container');
        if (progContainer) progContainer.classList.remove('hidden');
        submitBtn.disabled = true;

        appendMessage('user', `Analyzing Optical-SAR Pair with query: <i>"${query}"</i>`, false);

        const formData = new FormData();
        formData.append('optical_image', optFile);
        formData.append('sar_image', sarFile);
        formData.append('query', query);

        try {
            const response = await fetch('/api/analyze-crossmodal', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.status === 'success') {
                let resultHtml = `
                    <div class="results-header">
                        <h3><i class="fa-solid fa-layer-group"></i> Optical-SAR Cross-Modal Analysis</h3>
                    </div>
                    <div style="margin: 10px 0; text-align: center;">
                        <img src="${data.annotated_image}" style="max-width: 100%; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                        <p style="font-size: 11px; color: #94a3b8; margin-top: 4px;">Verified Built-Up Areas Highlighted via SAR Structural Backscatter</p>
                    </div>
                    <div style="background: #0f172a; padding: 12px; border-radius: 6px; font-size: 13px; color: #f8fafc; border-left: 4px solid #38bdf8;">
                        ${marked.parse(data.response)}
                    </div>
                `;

                if (data.execution_trace) {
                    resultHtml += formatExecutionTrace(data.execution_trace);
                }

                appendMessage('ai', resultHtml, true);
            } else {
                appendMessage('ai', `**Analysis Error:** ${data.error || 'Failed to analyze pair.'}`, false);
            }
        } catch (err) {
            appendMessage('ai', '**Network Error:** Failed to connect to cross-modal engine.', false);
        } finally {
            submitBtn.disabled = false;
            widget.classList.add('hidden');
            if (progContainer) progContainer.classList.add('hidden');
        }
    }

    async function runMapDetection() {
        if (!currentBbox) return alert("Please draw a bounding box on the map first.");
        const progContainer = document.getElementById('map-mode-progress-container');
        const progBar = document.getElementById('map-progress-bar');
        const statusTxt = document.getElementById('map-status-text');
        const percentTxt = document.getElementById('map-progress-percent');
        
        submitBtn.disabled = true;
        progContainer.classList.remove('hidden');
        
        const modality = document.getElementById('modality-select') ? document.getElementById('modality-select').value : 'optical';
        appendMessage('user', `Initiating scan for the selected coordinates...`, false);
        
        let elapsed = 0;
        let progressTimer = setInterval(() => {
            elapsed += 0.5;
            let percentage = Math.min(Math.round((elapsed / estimatedSeconds) * 100), 95);
            progBar.style.width = percentage + "%";
            percentTxt.innerText = percentage + "%";
        }, 500);

        try {
            const response = await fetch('/api/detect-satellite', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    bbox: currentBbox, 
                    t1_date: document.getElementById('t1-date').value, 
                    t2_date: document.getElementById('t2-date').value,
                    modality: modality
                })
            });
            const data = await response.json();
            clearInterval(progressTimer);
            progBar.style.width = "100%";
            percentTxt.innerText = "100%";
            statusTxt.innerText = "✅ Done!";

            if (data.status === "error") {
                appendMessage('ai', `**Error:** ${data.error}`, false);
                return;
            }

            let resultHtml = `
                <div class="results-header">
                    <h3><i class="fa-solid fa-satellite"></i> Satellite Scan Complete</h3>
                    <p style="font-size: 13px; color: #cbd5e1;">Detected <b>${data.events_count}</b> events.</p>
                </div>
            `;
            if (data.t1_full_url && data.t2_full_url) {
                resultHtml += `
                    <div class="comparison-grid">
                        <div class="img-wrapper"><span class="img-label">T1 Baseline</span><img src="${data.t1_full_url}"></div>
                        <div class="img-wrapper"><span class="img-label">T2 Current</span><img src="${data.t2_full_url}"></div>
                    </div>
                `;
            }
            
            // Restored patch renderer
            if (data.events_count > 0) {
                resultHtml += `<p style="font-size:12px; margin-top:12px; margin-bottom:8px; font-weight:bold; color:#94a3b8; text-transform:uppercase;">Detected Violation Patches</p>`;
                resultHtml += generateEventsHtml(data.events);
            } else {
                resultHtml += `<em style="color:#94a3b8; display:block; margin-top:10px;">No significant structural or vegetation changes detected in this area.</em>`;
            }
            
            appendMessage('ai', resultHtml, true); 
        } catch (error) {
            clearInterval(progressTimer);
            appendMessage('ai', '**Network Error:** Map scan failed.', false);
        } finally {
            submitBtn.disabled = false;
            widget.classList.add('hidden');
            progContainer.classList.add('hidden');
        }
    }

    async function runManualDetection() {
        const file1 = document.getElementById('pre_image').files[0];
        const file2 = document.getElementById('post_image').files[0];
        if (!file1 || !file2) return alert("Please upload both Pre and Post change imagery.");

        const progContainer = document.getElementById('manual-mode-progress-container');
        submitBtn.disabled = true;
        progContainer.classList.remove('hidden');
        appendMessage('user', 'Uploaded imagery. Running change detection...', false);

        const formData = new FormData();
        formData.append('pre_image', file1);
        formData.append('post_image', file2);

        try {
            const response = await fetch('/api/detect', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.status === 'success') {
                let resultHtml = `
                    <div class="results-header">
                        <h3><i class="fa-solid fa-microchip"></i> AI Analysis Complete</h3>
                        <p style="font-size: 13px; color: #cbd5e1;">Flagged <b>${data.events_count}</b> events.</p>
                    </div>
                `;
                if (data.annotated_image) {
                     resultHtml += `<div style="text-align:center;"><img src="${data.annotated_image}" style="max-width:100%; border-radius:6px;"></div>`;
                }
                
                // Restored patch renderer
                if (data.events && data.events.length > 0) {
                    resultHtml += `<p style="font-size:12px; margin-top:12px; margin-bottom:8px; font-weight:bold; color:#94a3b8; text-transform:uppercase;">Detected Violation Patches</p>`;
                    resultHtml += generateEventsHtml(data.events);
                } else {
                    resultHtml += `<em style="color:#94a3b8; display:block; margin-top:10px;">No violations detected in the provided images.</em>`;
                }
                
                appendMessage('ai', resultHtml, true);
            } else {
                appendMessage('ai', `**Error:** ${data.error}`, false);
            }
        } catch (err) {
            appendMessage('ai', '**Network Error:** Failed to execute change detection.', false);
        } finally {
            submitBtn.disabled = false;
            widget.classList.add('hidden');
            progContainer.classList.add('hidden');
        }
    }

    async function runVlmDetection() {
        const file = document.getElementById('vlm_image').files[0];
        const prompt = document.getElementById('vlm_prompt').value.trim();
        if (!file || !prompt) return alert("Please upload an image and specify a prompt.");

        const progContainer = document.getElementById('vlm-mode-progress-container');
        submitBtn.disabled = true;
        progContainer.classList.remove('hidden');
        appendMessage('user', `Analyzing image: <i>"${prompt}"</i>`, false);

        const formData = new FormData();
        formData.append('image', file);
        formData.append('prompt', prompt);

        try {
            const response = await fetch('/api/vlm/predict', { method: 'POST', body: formData });
            const data = await response.json();
            if (data.success) {
                let resultHtml = `
                    <div class="results-header">
                        <h3><i class="fa-solid fa-brain"></i> PaliGemma 2 Output</h3>
                    </div>
                    <div style="text-align:center; margin: 10px 0;"><img src="${data.image}" style="max-width:100%; border-radius:6px;"></div>
                    <div style="background:#0f172a; padding:12px; border-radius:6px; color:#f8fafc; border-left:4px solid #38bdf8;">
                        <strong>Answer:</strong> ${data.answer}
                    </div>
                `;
                appendMessage('ai', resultHtml, true);
            } else {
                appendMessage('ai', `**VLM Error:** ${data.error}`, false);
            }
        } catch (err) {
            appendMessage('ai', '**Network Error:** Failed to reach VLM backend.', false);
        } finally {
            submitBtn.disabled = false;
            widget.classList.add('hidden');
            progContainer.classList.add('hidden');
        }
    }

    // Copilot Chat Handler
    const sendBtn = document.getElementById('chat-send-btn');
    const msgInput = document.getElementById('message');

    async function sendTextQuery() {
        const query = msgInput.value.trim();
        if (!query) return;

        appendMessage('user', query, false);
        msgInput.value = '';
        msgInput.style.height = 'auto'; 

        const loadingId = 'loading-' + Date.now();
        appendMessage('ai', `<span id="${loadingId}" style="opacity:0.7"><i class="fa-solid fa-circle-notch fa-spin"></i> Analyzing telemetry...</span>`, true);

        try {
            const response = await fetch('/api/copilot', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: query })
            });
            const data = await response.json();

            const loaderEl = document.getElementById(loadingId);
            if (loaderEl) {
                let responseContent = marked.parse(data.response);
                if (data.execution_trace) {
                    responseContent += formatExecutionTrace(data.execution_trace);
                }
                loaderEl.parentElement.innerHTML = responseContent;
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        } catch (err) {
            const loaderEl = document.getElementById(loadingId);
            if (loaderEl) {
                loaderEl.parentElement.innerHTML = '<span style="color:#f87171">❌ Connection error to Copilot backend.</span>';
            }
        }
    }

    sendBtn.addEventListener('click', sendTextQuery);
    msgInput.addEventListener('keypress', (e) => { 
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTextQuery(); 
        } 
    });
});

function formatExecutionTrace(trace) {
    if (!trace) return '';
    return `
        <details class="execution-trace-accordion" style="margin-top: 10px; background: rgba(15, 23, 42, 0.9); border: 1px solid #334155; border-radius: 6px; padding: 8px 12px; font-family: monospace; font-size: 11px;">
            <summary style="cursor: pointer; color: #38bdf8; font-weight: bold;">
                🛠️ Auditable Execution Trace (${trace.selected_task || 'ROUTED'})
            </summary>
            <div style="margin-top: 8px; color: #cbd5e1; line-height: 1.6;">
                <div><b>Task Selected:</b> <span style="color: #a78bfa;">${trace.selected_task}</span></div>
                <div><b>Models Invoked:</b> ${JSON.stringify(trace.models_invoked || trace.models_or_tools)}</div>
                <div><b>Sequence:</b> ${(trace.execution_sequence || []).join(' ➔ ') || 'Direct Pipeline'}</div>
                <div><b>Validation Status:</b> <span style="color: #4ade80;">${trace.validation_status || 'VERIFIED'}</span></div>
                <div><b>Parameters:</b> ${JSON.stringify(trace.parameters || trace.permitted_parameters || {})}</div>
            </div>
        </details>
    `;
}
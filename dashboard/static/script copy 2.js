document.addEventListener("DOMContentLoaded", () => {
    
    const chatContainer = document.getElementById('chat-container');
    const bottomGap = document.querySelector('.bottom-gap');
    const widget = document.getElementById('detection-widget');

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

    const mapBtn = document.getElementById('nav-map-btn');
    const manualBtn = document.getElementById('nav-manual-btn');
    const vlmBtn = document.getElementById('nav-vlm-btn');
    
    const mapContainer = document.getElementById('map-mode-container');
    const manualContainer = document.getElementById('manual-mode-container');
    const vlmContainer = document.getElementById('vlm-mode-container');
    
    const submitBtn = document.getElementById('submit-btn');
    let currentMode = 'map'; 

    mapBtn.addEventListener('click', () => {
        currentMode = 'map';
        mapBtn.classList.add('active');
        manualBtn.classList.remove('active');
        if (vlmBtn) vlmBtn.classList.remove('active');
        
        mapContainer.classList.remove('hidden');
        manualContainer.classList.add('hidden');
        if (vlmContainer) vlmContainer.classList.add('hidden');
        
        setTimeout(() => { map.invalidateSize(); }, 200);
    });

    manualBtn.addEventListener('click', () => {
        currentMode = 'manual';
        manualBtn.classList.add('active');
        mapBtn.classList.remove('active');
        if (vlmBtn) vlmBtn.classList.remove('active');
        
        manualContainer.classList.remove('hidden');
        mapContainer.classList.add('hidden');
        if (vlmContainer) vlmContainer.classList.add('hidden');
    });

    if (vlmBtn) {
        vlmBtn.addEventListener('click', () => {
            currentMode = 'vlm';
            vlmBtn.classList.add('active');
            mapBtn.classList.remove('active');
            manualBtn.classList.remove('active');
            
            vlmContainer.classList.remove('hidden');
            mapContainer.classList.add('hidden');
            manualContainer.classList.add('hidden');
        });
    }

    const map = L.map('map').setView([28.5333, 77.2750], 14); // NIT DELHI 
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

    function setupImagePreview(inputId, previewId, textId) {
        const input = document.getElementById(inputId);
        const preview = document.getElementById(previewId);
        const text = document.getElementById(textId);
        if (!input) return;

        input.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.classList.remove('hidden');
                    text.classList.add('hidden');
                }
                reader.readAsDataURL(file);
            } else {
                preview.src = '';
                preview.classList.add('hidden');
                text.classList.remove('hidden');
            }
        });
    }

    setupImagePreview('pre_image', 'prePreview', 'preText');
    setupImagePreview('post_image', 'postPreview', 'postText');
    setupImagePreview('vlm_image', 'vlmPreview', 'vlmPlaceholder');

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

    submitBtn.addEventListener('click', async () => {
        if (currentMode === 'map') {
            await runMapDetection();
        } else if (currentMode === 'manual') {
            await runManualDetection();
        } else if (currentMode === 'vlm') {
            await runVlmDetection();
        }
    });

    async function runMapDetection() {
        if (!currentBbox) return alert("Please draw a bounding box on the map first.");
        
        const progContainer = document.getElementById('map-mode-progress-container');
        const progBar = document.getElementById('map-progress-bar');
        const statusTxt = document.getElementById('map-status-text');
        const percentTxt = document.getElementById('map-progress-percent');
        
        submitBtn.disabled = true;
        progContainer.classList.remove('hidden');
        
        const modality = document.getElementById('modality-select') ? document.getElementById('modality-select').value : 'optical';
        
        let scanTypeLabel = 'Optical (Sentinel-2)';
        if (modality === 'sar') scanTypeLabel = 'SAR (Sentinel-1)';
        else if (modality === 'fusion') scanTypeLabel = 'Multi-Modal Fusion';
        
        appendMessage('user', `Initiating ${scanTypeLabel} scan for the selected coordinates...`, false);
        
        let elapsed = 0;
        let progressTimer = setInterval(() => {
            elapsed += 0.5;
            let percentage = Math.min(Math.round((elapsed / estimatedSeconds) * 100), 95);
            progBar.style.width = percentage + "%";
            percentTxt.innerText = percentage + "%";
        }, 500);

        const payload = { 
            bbox: currentBbox, 
            t1_date: document.getElementById('t1-date').value, 
            t2_date: document.getElementById('t2-date').value,
            modality: modality
        };

        try {
            const response = await fetch('/api/detect-satellite', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
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
                    <p style="font-size: 13px; color: #cbd5e1;">Detected <b>${data.events_count}</b> change events within the bounding box.</p>
                </div>
            `;

            if (data.t1_full_url && data.t2_full_url) {
                resultHtml += `
                    <div class="full-scene-comparison">
                        <p style="font-size:12px; margin-bottom:8px; font-weight:bold; color:#94a3b8; text-transform:uppercase;">Full Scene Analysis (Changes Outlined)</p>
                        <div class="comparison-grid">
                            <div class="img-wrapper"><span class="img-label">T1 Baseline</span><img src="${data.t1_full_url}"></div>
                            <div class="img-wrapper"><span class="img-label">T2 Current</span><img src="${data.t2_full_url}"></div>
                        </div>
                    </div>
                `;
            }
            
            if (data.events_count > 0) {
                resultHtml += `<p style="font-size:12px; margin-bottom:8px; font-weight:bold; color:#94a3b8; text-transform:uppercase;">Detected Violation Patches</p>`;
                resultHtml += generateEventsHtml(data.events);
            } else {
                resultHtml += `<em>No significant structural or vegetation changes detected in this area.</em>`;
            }
            appendMessage('ai', resultHtml, true); 

        } catch (error) {
            clearInterval(progressTimer);
            statusTxt.innerText = "❌ Failed";
            appendMessage('ai', '**Network Error:** Failed to execute map scan. Please check backend logs.', false);
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
        appendMessage('user', 'Uploaded local imagery. Running detection engine...', false);

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
                        <p style="font-size: 13px; color: #cbd5e1;">Engine flagged <b>${data.events_count}</b> local violations.</p>
                    </div>
                `;
                
                const preImgSrc = document.getElementById('prePreview').src;
                if (data.annotated_image) {
                     resultHtml += `
                     <div class="full-scene-comparison">
                        <p style="font-size:12px; margin-bottom:8px; font-weight:bold; color:#94a3b8; text-transform:uppercase;">Full Imagery Output (Changes Outlined)</p>
                        <div class="comparison-grid">
                            <div class="img-wrapper"><span class="img-label">T1 Baseline</span><img src="${preImgSrc}"></div>
                            <div class="img-wrapper"><span class="img-label">T2 AI Overlay</span><img src="${data.annotated_image}"></div>
                        </div>
                     </div>`;
                }

                if (data.events && data.events.length > 0) {
                    resultHtml += `<p style="font-size:12px; margin-bottom:8px; font-weight:bold; color:#94a3b8; text-transform:uppercase;">Detected Violation Patches</p>`;
                    resultHtml += generateEventsHtml(data.events);
                } else {
                    resultHtml += `<em>No violations detected in the provided images.</em>`;
                }
                appendMessage('ai', resultHtml, true);
            } else {
                appendMessage('ai', `**Error:** ${data.error || 'Unknown error occurred'}`, false);
            }
        } catch (err) {
            appendMessage('ai', '**Network Error:** Failed to connect to the backend engine.', false);
        } finally {
            submitBtn.disabled = false;
            widget.classList.add('hidden');
            progContainer.classList.add('hidden');
        }
    }

    async function runVlmDetection() {
        const file = document.getElementById('vlm_image').files[0];
        const prompt = document.getElementById('vlm_prompt').value.trim();
        
        if (!file) return alert("Please upload a single satellite image.");
        if (!prompt) return alert("Please enter a prompt or select a preset.");

        const progContainer = document.getElementById('vlm-mode-progress-container');
        const statusTxt = document.getElementById('vlm-status-text');
        
        submitBtn.disabled = true;
        progContainer.classList.remove('hidden');
        
        appendMessage('user', `Analyzing single image with prompt: <i>"${prompt}"</i>`, false);

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
                        <p style="font-size: 13px; color: #cbd5e1;">Analyzed resolution: ${data.original_width}x${data.original_height}</p>
                    </div>
                    <div class="vlm-result-container" style="background: #1e293b; padding: 15px; border-radius: 8px; margin-top: 10px; border: 1px solid #334155;">
                        <div style="margin-bottom: 15px; text-align: center;">
                            <img src="${data.image}" style="max-width: 100%; border-radius: 6px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);">
                        </div>
                        <div style="background: #0f172a; padding: 12px; border-radius: 6px; font-size: 14px; color: #f8fafc; border-left: 4px solid #38bdf8;">
                            <strong>Answer:</strong><br/>
                            <span style="white-space: pre-wrap;">${data.answer || '(Empty Output)'}</span>
                        </div>
                `;

                if (data.boxes && data.boxes.length > 0) {
                    resultHtml += `
                        <div style="margin-top: 10px; font-size: 12px; color: #94a3b8;">
                            <strong>Detected ${data.boxes.length} bounding box(es):</strong><br/>
                            ${data.boxes.map((b, i) => `Box ${i+1}: [X:${b.xmin} Y:${b.ymin}]`).join(' | ')}
                        </div>
                    `;
                }

                resultHtml += `</div>`;
                appendMessage('ai', resultHtml, true);
            } else {
                appendMessage('ai', `**VLM Error:** ${data.error || 'Unknown API failure'}`, false);
            }
        } catch (err) {
            appendMessage('ai', '**Network Error:** Failed to connect to PaliGemma backend.', false);
        } finally {
            submitBtn.disabled = false;
            widget.classList.add('hidden');
            progContainer.classList.add('hidden');
        }
    }

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
            if(loaderEl) {
                loaderEl.parentElement.innerHTML = marked.parse(data.response);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }
        } catch (err) {
            const loaderEl = document.getElementById(loadingId);
            if(loaderEl) {
                loaderEl.parentElement.innerHTML = '<span style="color:#f87171">❌ Connection error to Groq Copilot.</span>';
            }
        }
    }

    sendBtn.addEventListener('click', sendTextQuery);
    
    msgInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });
    
    msgInput.addEventListener('keypress', (e) => { 
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendTextQuery(); 
        } 
    });
});
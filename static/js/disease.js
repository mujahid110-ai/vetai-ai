// VetAI Pro — Disease Detection JS

function getSymptoms() {
  const checkboxes = document.querySelectorAll('.symptom-check input[type="checkbox"]');
  const symptoms = {};
  checkboxes.forEach(cb => {
    symptoms[cb.name] = cb.checked ? 1 : 0;
  });
  return symptoms;
}

function updateCount() {
  const checked = document.querySelectorAll('.symptom-check input:checked').length;
  document.getElementById('symptomCount').textContent = checked + ' selected';
}

document.querySelectorAll('.symptom-check input').forEach(cb => {
  cb.addEventListener('change', updateCount);
});

function clearAll() {
  document.querySelectorAll('.symptom-check input').forEach(cb => cb.checked = false);
  updateCount();
  document.getElementById('resultsContent').style.display = 'none';
  document.getElementById('resultsPlaceholder').style.display = 'block';
}

function getUrgencyClass(urgency) {
  if (!urgency) return 'urgency-low';
  const u = urgency.toUpperCase();
  if (u.includes('IMMEDIATE') || u.includes('CRITICAL')) return 'urgency-critical';
  if (u.includes('HIGH') || u.includes('24 HOUR')) return 'urgency-high';
  if (u.includes('48') || u.includes('MODERATE')) return 'urgency-moderate';
  return 'urgency-low';
}

function getResultClass(severity, zoonotic) {
  if (!severity) return '';
  const s = severity.toUpperCase();
  if (zoonotic || s.includes('CRITICAL')) return 'danger';
  if (s.includes('HIGH')) return 'warning';
  return '';
}

async function runDiagnosis() {
  const symptoms = getSymptoms();
  const totalSelected = Object.values(symptoms).reduce((a, b) => a + b, 0);

  if (totalSelected < 2) {
    alert('Please select at least 2 symptoms for an accurate diagnosis.');
    return;
  }

  const btn = document.getElementById('diagnoseBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Analyzing...';

  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'block';
  document.getElementById('resultsContent').innerHTML = `
    <div class="loading-spinner">
      <div class="spinner"></div>
      <span>Running AI + Expert System analysis...</span>
    </div>`;

  try {
    const res = await fetch('/api/predict/disease', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ symptoms })
    });
    const data = await res.json();

    if (data.error) {
      document.getElementById('resultsContent').innerHTML = `
        <div style="color:#fca5a5;padding:20px;">⚠️ Error: ${data.error}</div>`;
      return;
    }

    renderDiseaseResults(data);
  } catch (err) {
    document.getElementById('resultsContent').innerHTML = `
      <div style="color:#fca5a5;padding:20px;">⚠️ Connection error. Is the Flask server running?</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = '🔍 Run Diagnosis';
  }
}

function renderDiseaseResults(data) {
  const top = data.top_prediction;
  const alts = data.all_predictions.slice(1);
  const urgClass = getUrgencyClass(top.urgency);
  const cardClass = getResultClass(top.severity, top.zoonotic);

  let zoonoticHtml = '';
  if (top.zoonotic) {
    zoonoticHtml = `<div class="zoonotic-warning">⚠️ ZOONOTIC DISEASE — Can spread to humans. Use protective gear. Notify authorities if required.</div>`;
  }

  let altHtml = '';
  if (alts.length) {
    altHtml = `<div class="alternatives-section">
      <h3>Other Possible Conditions</h3>
      ${alts.map(a => `
        <div class="alt-card">
          <div>
            <div class="alt-name">${a.disease}</div>
            <div class="alt-prob">${a.severity || ''}</div>
          </div>
          <div style="text-align:right;">
            <div style="font-weight:700;color:var(--text-muted)">${a.confidence}%</div>
            <div class="conf-bar-bg" style="width:80px;margin-top:4px;">
              <div class="conf-bar-fill" style="width:${a.confidence}%"></div>
            </div>
          </div>
        </div>`).join('')}
    </div>`;
  }

  document.getElementById('resultsContent').innerHTML = `
    <div class="result-top ${cardClass}">
      <div class="result-header">
        <div>
          <div style="font-size:0.78rem;color:var(--text-dim);text-transform:uppercase;font-weight:700;margin-bottom:4px;">Primary Diagnosis</div>
          <div class="result-name">${top.disease}</div>
          <span class="urgency-tag ${urgClass}">${top.urgency || 'Consult vet'}</span>
        </div>
        <div class="confidence-badge">
          <div class="conf-pct">${top.confidence}%</div>
          <div class="conf-label">${data.confidence_level} Confidence</div>
        </div>
      </div>

      <div class="conf-bar-wrap">
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" style="width:${top.confidence}%"></div>
        </div>
      </div>

      <p class="result-desc">${top.description || ''}</p>

      <div class="result-details">
        <div class="detail-box">
          <h4>💊 Treatment</h4>
          <p>${top.treatment || 'Consult a veterinarian for appropriate treatment.'}</p>
        </div>
        <div class="detail-box">
          <h4>🛡️ Prevention</h4>
          <p>${top.prevention || 'Maintain good farm hygiene and biosecurity.'}</p>
        </div>
      </div>

      ${zoonoticHtml}
    </div>

    <div style="font-size:0.8rem;color:var(--text-dim);margin-bottom:16px;">
      📊 ${data.total_symptoms_entered} symptoms analyzed
    </div>

    ${altHtml}

    <div style="margin-top:20px;background:rgba(0,0,0,0.2);border-radius:10px;padding:14px;font-size:0.82rem;color:var(--text-dim);">
      ⚠️ This AI analysis is a decision-support tool. Always confirm with a licensed veterinarian, especially for critical or notifiable diseases.
    </div>
  `;
}

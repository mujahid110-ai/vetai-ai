// VetAI Pro — Pregnancy Detection JS

const STAGE_ORDER = [
  'Not Pregnant',
  'Early Pregnancy (1-4 weeks)',
  'Mid Pregnancy (4-8 weeks)',
  'Second Trimester (3-6 months)',
  'Third Trimester (7-9 months)',
  'Imminent Calving'
];

function getStageIndex(stage) {
  const idx = STAGE_ORDER.findIndex(s => s === stage);
  return idx === -1 ? 0 : idx;
}

function getObservations() {
  const checkboxes = document.querySelectorAll('.symptom-check input[type="checkbox"]');
  const symptoms = {};
  checkboxes.forEach(cb => {
    symptoms[cb.name] = cb.checked ? 1 : 0;
  });
  return symptoms;
}

function updateCount() {
  const checked = document.querySelectorAll('.symptom-check input:checked').length;
  document.getElementById('symptomCount').textContent = checked + ' observed';
}

document.querySelectorAll('.symptom-check input').forEach(cb => {
  cb.addEventListener('change', updateCount);
});

function clearAll() {
  document.querySelectorAll('.symptom-check input').forEach(cb => cb.checked = false);
  document.getElementById('daysBreeding').value = '';
  updateCount();
  document.getElementById('resultsContent').style.display = 'none';
  document.getElementById('resultsPlaceholder').style.display = 'block';
}

async function runPregnancyCheck() {
  const symptoms = getObservations();
  const days = parseInt(document.getElementById('daysBreeding').value) || 0;
  const totalSelected = Object.values(symptoms).reduce((a, b) => a + b, 0);

  if (totalSelected < 1) {
    alert('Please check at least one observation sign.');
    return;
  }

  document.getElementById('resultsPlaceholder').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'block';
  document.getElementById('resultsContent').innerHTML = `
    <div class="loading-spinner">
      <div class="spinner"></div>
      <span>Analyzing signs with Expert Rule System + ML...</span>
    </div>`;

  try {
    const res = await fetch('/api/predict/pregnancy', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ symptoms, days_since_breeding: days })
    });
    const data = await res.json();

    if (data.error) {
      document.getElementById('resultsContent').innerHTML = `
        <div style="color:#fca5a5;padding:20px;">⚠️ Error: ${data.error}</div>`;
      return;
    }

    renderPregnancyResults(data);
  } catch (err) {
    document.getElementById('resultsContent').innerHTML = `
      <div style="color:#fca5a5;padding:20px;">⚠️ Connection error. Is the Flask server running?</div>`;
  }
}

function getStageColor(stage) {
  if (stage === 'Not Pregnant') return '#64748b';
  if (stage === 'Imminent Calving') return '#f59e0b';
  return '#16a34a';
}

function renderPregnancyResults(data) {
  const stageIdx = getStageIndex(data.stage);
  const color = getStageColor(data.stage);
  const isPregnant = data.stage !== 'Not Pregnant';

  // Stage timeline dots
  let timelineDots = STAGE_ORDER.map((s, i) => {
    let cls = 'stage-dot';
    if (i < stageIdx) cls += ' past';
    if (i === stageIdx) cls += ' active';
    return `<div class="${cls}" title="${s}"></div>`;
  }).join('');

  // Calving box
  let calvingHtml = '';
  if (data.expected_calving_in && isPregnant) {
    calvingHtml = `<div class="calving-box">
      📅 <strong>Expected calving in:</strong> ${data.expected_calving_in} (based on 283-day gestation)
    </div>`;
  }

  // Alternatives
  let altHtml = '';
  if (data.alternatives && data.alternatives.length > 0) {
    altHtml = `<div class="alternatives-section">
      <h3>Alternative Assessments</h3>
      ${data.alternatives.map(a => `
        <div class="alt-card">
          <div class="alt-name">${a.stage}</div>
          <div style="text-align:right;">
            <span style="font-weight:700;color:var(--text-muted)">${a.probability}%</span>
            <div class="conf-bar-bg" style="width:80px;margin-top:4px;">
              <div class="conf-bar-fill" style="width:${a.probability}%"></div>
            </div>
          </div>
        </div>`).join('')}
    </div>`;
  }

  document.getElementById('resultsContent').innerHTML = `
    <div class="result-top" style="border-color:${color};background:linear-gradient(135deg,${color}18,${color}0a);">
      <div class="result-header">
        <div>
          <div style="font-size:0.78rem;color:var(--text-dim);text-transform:uppercase;font-weight:700;margin-bottom:4px;">
            ${isPregnant ? '🤱 Pregnancy Stage Detected' : '🔍 Assessment Result'}
          </div>
          <div class="result-name" style="color:${color};">${data.stage}</div>
          ${data.method ? `<span class="method-badge">${data.method}</span>` : ''}
        </div>
        <div class="confidence-badge">
          <div class="conf-pct" style="color:${color};">${data.confidence}%</div>
          <div class="conf-label">${data.confidence_level} Confidence</div>
        </div>
      </div>

      <!-- Stage Timeline -->
      <div style="margin:12px 0 4px;">
        <div style="font-size:0.72rem;color:var(--text-dim);margin-bottom:4px;">Pregnancy progression</div>
        <div class="stage-timeline">${timelineDots}</div>
        <div style="display:flex;justify-content:space-between;font-size:0.68rem;color:var(--text-dim);">
          <span>Not Pregnant</span><span>Imminent Calving</span>
        </div>
      </div>

      <div class="conf-bar-wrap" style="margin-top:12px;">
        <div class="conf-bar-bg">
          <div class="conf-bar-fill" style="width:${data.confidence}%;background:${color};"></div>
        </div>
      </div>

      <p class="result-desc" style="margin-top:12px;">${data.description || ''}</p>

      ${calvingHtml}
    </div>

    <div class="result-details" style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px;">
      <div class="detail-box">
        <h4>✅ Recommendations</h4>
        <p>${data.recommendations || 'Consult a veterinarian for confirmation.'}</p>
      </div>
      <div class="detail-box">
        <h4>🌿 Care Protocol</h4>
        <p>${data.care || 'Maintain standard cattle management practices.'}</p>
      </div>
    </div>

    ${data.confidence_note ? `
    <div style="margin-top:12px;background:rgba(14,165,233,0.08);border:1px solid rgba(14,165,233,0.2);border-radius:8px;padding:10px 14px;font-size:0.82rem;color:#7dd3fc;">
      💡 <strong>Clinical Note:</strong> ${data.confidence_note}
    </div>` : ''}

    ${altHtml}

    <div style="margin-top:20px;background:rgba(0,0,0,0.2);border-radius:10px;padding:14px;font-size:0.82rem;color:var(--text-dim);">
      ⚠️ Behavioral observation alone gives 70–80% accuracy. For definitive confirmation, use PAG blood/milk test (from day 28) or transrectal ultrasound (95–99% accurate from day 28–30).
    </div>
  `;

  // Animate confidence bar
  setTimeout(() => {
    const fill = document.querySelector('.conf-bar-fill');
    if (fill) fill.style.width = data.confidence + '%';
  }, 100);
}

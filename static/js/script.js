/**
 * CropCare AI – Main JavaScript
 * Handles: upload, drag-drop, predict fetch, results, disease filter, contact form
 */

'use strict';

// ─── Utility ──────────────────────────────────────────────────────────────────

const $ = (selector, ctx = document) => ctx.querySelector(selector);
const $$ = (selector, ctx = document) => [...ctx.querySelectorAll(selector)];
const show = el => el && el.classList.add('visible');
const hide = el => el && el.classList.remove('visible');
const formatSize = bytes => bytes < 1024 * 1024
  ? `${(bytes / 1024).toFixed(1)} KB`
  : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;


// ─── Navbar ───────────────────────────────────────────────────────────────────

function initNavbar() {
  const hamburger = $('#hamburger');
  const navLinks  = $('#nav-links');
  if (!hamburger || !navLinks) return;

  hamburger.addEventListener('click', () => {
    navLinks.classList.toggle('open');
    hamburger.setAttribute('aria-expanded', navLinks.classList.contains('open'));
  });

  // Close on outside click
  document.addEventListener('click', e => {
    if (!hamburger.contains(e.target) && !navLinks.contains(e.target)) {
      navLinks.classList.remove('open');
    }
  });

  // Highlight active nav link
  const path = window.location.pathname;
  $$('.nav-links a').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
}


// ─── Scroll-to-top ────────────────────────────────────────────────────────────

function initScrollTop() {
  const btn = $('#scroll-top-btn');
  if (!btn) return;

  window.addEventListener('scroll', () => {
    btn.classList.toggle('visible', window.scrollY > 300);
  }, { passive: true });

  btn.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
}


// ─── Loading Overlay ──────────────────────────────────────────────────────────

function showLoading(message = 'Analyzing Image...') {
  const overlay = $('#loading-overlay');
  if (!overlay) return;
  $('#loading-text') && ($('#loading-text').textContent = message);
  show(overlay);
}

function hideLoading() {
  hide($('#loading-overlay'));
}


// ─── Image Upload & Preview ───────────────────────────────────────────────────

function initUpload() {
  const dropZone    = $('#drop-zone');
  const fileInput   = $('#file-input');
  const previewPane = $('#preview-pane');
  const previewImg  = $('#preview-img');
  const fileNameEl  = $('#file-name');
  const fileSizeEl  = $('#file-size');
  const analyzeBtn  = $('#analyze-btn');
  const resetBtn    = $('#reset-btn');
  const errorAlert  = $('#upload-error');

  if (!dropZone) return;

  let selectedFile = null;

  // Drag events
  dropZone.addEventListener('dragover', e => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));

  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  });

  fileInput && fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
  });

  function handleFileSelect(file) {
    // Validate type
    const allowed = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowed.includes(file.type)) {
      showError('Invalid file type. Please upload JPG, JPEG, or PNG images.');
      return;
    }

    // Validate size (16 MB)
    if (file.size > 16 * 1024 * 1024) {
      showError('File too large. Maximum size is 16 MB.');
      return;
    }

    clearError();
    selectedFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = e => {
      previewImg.src = e.target.result;
      fileNameEl && (fileNameEl.textContent = file.name);
      fileSizeEl && (fileSizeEl.textContent = formatSize(file.size));
      previewPane && previewPane.classList.add('visible');
      analyzeBtn && (analyzeBtn.disabled = false);
    };
    reader.readAsDataURL(file);

    // Reset any previous results
    hideResults();
  }

  // Analyze button
  analyzeBtn && analyzeBtn.addEventListener('click', () => {
    if (!selectedFile) return;
    submitPrediction(selectedFile);
  });

  // Reset button
  resetBtn && resetBtn.addEventListener('click', () => {
    selectedFile = null;
    fileInput && (fileInput.value = '');
    previewImg && (previewImg.src = '');
    previewPane && previewPane.classList.remove('visible');
    analyzeBtn && (analyzeBtn.disabled = true);
    clearError();
    hideResults();
    // Reset result panel
    const resultPanel = $('#result-panel');
    resultPanel && resultPanel.classList.remove('visible');
  });

  function showError(msg) {
    if (!errorAlert) return;
    errorAlert.style.display = 'flex';
    const msgEl = errorAlert.querySelector('.alert-msg');
    if (msgEl) msgEl.textContent = msg;
  }

  function clearError() {
    errorAlert && (errorAlert.style.display = 'none');
  }
}


// ─── Prediction API Call ──────────────────────────────────────────────────────

async function submitPrediction(file) {
  const formData = new FormData();
  formData.append('image', file);

  showLoading('AI is analyzing your crop image...');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });

    const data = await response.json();
    hideLoading();

    if (!response.ok || !data.success) {
      showPredictError(data.error || 'Prediction failed. Please try again.', data.model_missing);
      return;
    }

    renderResults(data);

  } catch (err) {
    hideLoading();
    showPredictError('Network error. Please check your connection and try again.');
    console.error('Prediction error:', err);
  }
}

function showPredictError(msg, modelMissing = false) {
  const panel = $('#result-panel');
  if (!panel) return;

  panel.classList.add('visible');
  panel.innerHTML = modelMissing
    ? `<div class="model-warning">
        <div class="warn-icon">⚠️</div>
        <div>
          <h4>Model Not Trained Yet</h4>
          <p>${escHtml(msg)}</p>
          <p class="mt-1">Steps to train the model:</p>
          <ol style="margin-top:8px; padding-left:20px; font-size:0.85rem; color:#856404">
            <li>Download the PlantVillage dataset from Kaggle</li>
            <li>Extract to <code>dataset/PlantVillage/</code></li>
            <li>Run: <code>python model/train_model.py</code></li>
            <li>Restart the Flask server</li>
          </ol>
        </div>
      </div>`
    : `<div class="alert alert-error">
        <span class="alert-icon">❌</span>
        <span class="alert-msg">${escHtml(msg)}</span>
      </div>`;

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}


// ─── Render Prediction Results ────────────────────────────────────────────────

function renderResults(data) {
  const panel = $('#result-panel');
  if (!panel) return;

  const isHealthy   = data.is_healthy;
  const statusClass = isHealthy ? 'healthy' : 'diseased';
  const statusText  = isHealthy ? '✅ Healthy' : '🚨 Diseased';
  const confidence  = data.confidence || 0;
  const details     = data.disease_details || {};

  // Confidence ring circumference: 2π × r = 2π × 36 ≈ 226
  const circum = 226;
  const dash   = circum - (confidence / 100) * circum;

  const detailSections = buildDetailSections(details);

  panel.classList.add('visible');
  panel.innerHTML = `
    <div class="result-header">
      <div>
        <h3 style="font-size:1.2rem;font-weight:800;margin-bottom:6px">Detection Result</h3>
        <p style="font-size:0.85rem;color:var(--text-muted)">
          Analyzed: ${data.crop || '—'} leaf
        </p>
      </div>
      <div>
        <span class="status-badge ${statusClass}">${statusText}</span>
        ${data.report_url
          ? `<a href="${escHtml(data.report_url)}" class="btn btn-green" style="margin-left:10px;padding:8px 16px;font-size:0.85rem;" target="_blank">
               📄 Download Report
             </a>`
          : ''}
      </div>
    </div>

    <div style="display:grid;grid-template-columns:auto 1fr;gap:20px;margin-bottom:24px;align-items:start">
      <img src="${escHtml(data.image_url || '')}" alt="Uploaded leaf"
           style="width:160px;height:160px;object-fit:cover;border-radius:12px;border:1px solid var(--border);">
      <div>
        <div class="crop-info">
          <div class="info-item">
            <label>🌱 Crop</label>
            <div class="value">${escHtml(data.crop || 'Unknown')}</div>
          </div>
          <div class="info-item">
            <label>🦠 Disease</label>
            <div class="value">${escHtml(data.disease || 'Unknown')}</div>
          </div>
        </div>
        <div class="confidence-ring">
          <div class="ring-chart" title="${confidence.toFixed(1)}% confidence">
            <svg viewBox="0 0 90 90">
              <circle class="ring-bg" cx="45" cy="45" r="36"/>
              <circle class="ring-fill" cx="45" cy="45" r="36"
                stroke-dasharray="${circum}"
                stroke-dashoffset="${dash}"
                stroke="${isHealthy ? '#27ae60' : '#e74c3c'}"
              />
            </svg>
            <div class="ring-label">${confidence.toFixed(0)}%</div>
          </div>
          <div class="confidence-info">
            <h4>AI Confidence</h4>
            <p>${confidenceLabel(confidence)}</p>
          </div>
        </div>
      </div>
    </div>

    ${details.description
      ? `<p style="font-size:0.9rem;color:var(--text-secondary);line-height:1.8;margin-bottom:20px;padding:16px;background:var(--bg);border-radius:8px;border:1px solid var(--border);">
           📋 ${escHtml(details.description)}
         </p>`
      : ''}

    ${detailSections}

    ${buildTopPredictions(data.top_predictions)}

    <p style="font-size:0.75rem;color:var(--text-muted);margin-top:20px;padding-top:16px;border-top:1px solid var(--border);">
      ⚠️ This AI prediction is a decision-support tool. Always consult a qualified agronomist for diagnosis and treatment.
    </p>
  `;

  panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  // Wire accordion
  $$('.detail-section-header', panel).forEach(header => {
    header.addEventListener('click', () => {
      header.closest('.detail-section').classList.toggle('open');
    });
  });
}

function confidenceLabel(c) {
  if (c >= 90) return 'Very High – strong prediction';
  if (c >= 75) return 'High – reliable prediction';
  if (c >= 60) return 'Moderate – consider expert advice';
  return 'Low – image may be unclear';
}

function buildDetailSections(details) {
  const sections = [
    { key: 'symptoms',   label: '🔴 Symptoms',    icon: '🔴' },
    { key: 'causes',     label: '⚡ Causes',       icon: '⚡' },
    { key: 'prevention', label: '🛡️ Prevention',   icon: '🛡️' },
    { key: 'management', label: '💊 Management',   icon: '💊' },
  ];

  return sections.map(s => {
    const content = details[s.key];
    if (!content || (Array.isArray(content) && !content.length)) return '';
    const items = Array.isArray(content)
      ? content.map(i => `<li>${escHtml(i)}</li>`).join('')
      : `<li>${escHtml(String(content))}</li>`;

    return `
      <div class="detail-section">
        <div class="detail-section-header">
          <span>${s.label}</span>
          <span class="arrow">▼</span>
        </div>
        <div class="detail-section-body">
          <ul class="detail-list">${items}</ul>
          ${details.disclaimer ? `<p style="font-size:0.75rem;color:var(--text-muted);margin-top:12px;font-style:italic;">${escHtml(details.disclaimer)}</p>` : ''}
        </div>
      </div>`;
  }).join('');
}

function buildTopPredictions(preds) {
  if (!preds || !preds.length) return '';
  const bars = preds.map((p, i) => `
    <div class="pred-bar">
      <div class="pred-bar-label">
        <span>${escHtml(p.crop)} – ${escHtml(p.disease)}</span>
        <span>${p.confidence.toFixed(1)}%</span>
      </div>
      <div class="pred-bar-track">
        <div class="pred-bar-fill" style="width:${p.confidence}%;background:${i === 0 ? 'linear-gradient(90deg,var(--primary),var(--accent))' : 'var(--border)'}"></div>
      </div>
    </div>`).join('');

  return `
    <div class="top-predictions">
      <h4>Top 3 Predictions</h4>
      ${bars}
    </div>`;
}

function hideResults() {
  const panel = $('#result-panel');
  panel && panel.classList.remove('visible');
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}


// ─── Disease Library Filter ───────────────────────────────────────────────────

function initDiseaseFilter() {
  const searchInput  = $('#disease-search');
  const cropFilter   = $('#crop-filter');
  const catFilter    = $('#category-filter');
  const cards        = $$('.disease-card');
  const countEl      = $('#disease-count');

  if (!searchInput && !cropFilter) return;

  function applyFilter() {
    const query = (searchInput?.value || '').toLowerCase().trim();
    const crop  = (cropFilter?.value || '').toLowerCase();
    const cat   = (catFilter?.value || '').toLowerCase();

    let visible = 0;
    cards.forEach(card => {
      const name   = (card.dataset.name   || '').toLowerCase();
      const cropD  = (card.dataset.crop   || '').toLowerCase();
      const catD   = (card.dataset.cat    || '').toLowerCase();
      const desc   = (card.dataset.desc   || '').toLowerCase();
      const syms   = (card.dataset.syms   || '').toLowerCase();

      const matchQ = !query || name.includes(query) || desc.includes(query) || syms.includes(query) || cropD.includes(query);
      const matchC = !crop  || cropD === crop;
      const matchA = !cat   || catD  === cat;

      const show = matchQ && matchC && matchA;
      card.style.display = show ? '' : 'none';
      if (show) visible++;
    });

    countEl && (countEl.textContent = `${visible} disease${visible !== 1 ? 's' : ''} found`);
  }

  searchInput && searchInput.addEventListener('input', applyFilter);
  cropFilter  && cropFilter.addEventListener('change', applyFilter);
  catFilter   && catFilter.addEventListener('change', applyFilter);

  // Initial count
  countEl && (countEl.textContent = `${cards.length} diseases found`);
}


// ─── Contact Form ─────────────────────────────────────────────────────────────

function initContactForm() {
  const form = $('#contact-form');
  if (!form) return;

  form.addEventListener('submit', async e => {
    e.preventDefault();
    clearFormErrors(form);

    const name    = form.querySelector('#name')?.value.trim();
    const email   = form.querySelector('#email')?.value.trim();
    const message = form.querySelector('#message')?.value.trim();

    let valid = true;

    if (!name) {
      setFieldError(form, 'name', 'Name is required.');
      valid = false;
    }

    if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      setFieldError(form, 'email', 'Please enter a valid email address.');
      valid = false;
    }

    if (!message || message.length < 10) {
      setFieldError(form, 'message', 'Message must be at least 10 characters.');
      valid = false;
    }

    if (!valid) return;

    const submitBtn = form.querySelector('[type="submit"]');
    submitBtn && (submitBtn.disabled = true);
    submitBtn && (submitBtn.textContent = 'Sending...');

    try {
      const response = await fetch('/contact', {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
        body: new URLSearchParams({ name, email, message })
      });

      const data = await response.json();
      if (data.success) {
        form.reset();
        showFormSuccess(form, data.message || 'Message sent successfully!');
      } else {
        (data.errors || ['Something went wrong.']).forEach(err => {
          showFormAlert(form, err);
        });
      }
    } catch (err) {
      showFormAlert(form, 'Network error. Please try again.');
    } finally {
      submitBtn && (submitBtn.disabled = false);
      submitBtn && (submitBtn.textContent = 'Send Message');
    }
  });
}

function setFieldError(form, fieldId, msg) {
  const field = form.querySelector(`#${fieldId}`);
  if (field) {
    field.classList.add('error');
    const errEl = form.querySelector(`#${fieldId}-error`);
    if (errEl) errEl.textContent = msg;
  }
}

function clearFormErrors(form) {
  form.querySelectorAll('.form-control').forEach(el => el.classList.remove('error'));
  form.querySelectorAll('.form-error').forEach(el => el.textContent = '');
  const alertEl = form.querySelector('#form-alert');
  if (alertEl) alertEl.style.display = 'none';
  const successEl = form.querySelector('#form-success');
  if (successEl) successEl.style.display = 'none';
}

function showFormAlert(form, msg) {
  const alertEl = form.querySelector('#form-alert');
  if (alertEl) {
    alertEl.style.display = 'flex';
    const msgEl = alertEl.querySelector('.alert-msg');
    if (msgEl) msgEl.textContent = msg;
  }
}

function showFormSuccess(form, msg) {
  const el = form.querySelector('#form-success');
  if (el) {
    el.style.display = 'flex';
    const msgEl = el.querySelector('.alert-msg');
    if (msgEl) msgEl.textContent = msg;
  }
}


// ─── Animated counters (hero stats) ──────────────────────────────────────────

function initCounters() {
  const counters = $$('[data-count]');
  if (!counters.length) return;

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      const el  = entry.target;
      const end = parseInt(el.dataset.count);
      const dur = 1500;
      const step = end / (dur / 16);
      let current = 0;

      const timer = setInterval(() => {
        current = Math.min(current + step, end);
        el.textContent = Math.floor(current) + (el.dataset.suffix || '');
        if (current >= end) clearInterval(timer);
      }, 16);

      observer.unobserve(el);
    });
  }, { threshold: 0.5 });

  counters.forEach(c => observer.observe(c));
}


// ─── Hero floating particles ──────────────────────────────────────────────────

function initParticles() {
  const container = $('#hero-particles');
  if (!container) return;

  for (let i = 0; i < 12; i++) {
    const p = document.createElement('div');
    p.className = 'hero-particle';
    const size = Math.random() * 60 + 20;
    Object.assign(p.style, {
      width:  `${size}px`,
      height: `${size}px`,
      left:   `${Math.random() * 100}%`,
      top:    `${Math.random() * 100}%`,
      animationDelay:    `${Math.random() * 6}s`,
      animationDuration: `${Math.random() * 6 + 6}s`,
    });
    container.appendChild(p);
  }
}


// ─── Init ─────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initNavbar();
  initScrollTop();
  initUpload();
  initDiseaseFilter();
  initContactForm();
  initCounters();
  initParticles();
});

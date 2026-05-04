/* ══════════════════════════════════════════════════════
   NEUROSCAN AI — script.js  (all bugs fixed)
   ══════════════════════════════════════════════════════ */

/* ── 1. NEURAL MESH CANVAS ─────────────────────────── */
(function () {
  const canvas = document.getElementById('neural-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H, nodes = [];
  const NODE_COUNT = 70, MAX_DIST = 160, SPEED = 0.25;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  function initNodes() {
    nodes = [];
    for (let i = 0; i < NODE_COUNT; i++) {
      nodes.push({ x: Math.random()*W, y: Math.random()*H,
                   vx: (Math.random()-.5)*SPEED, vy: (Math.random()-.5)*SPEED,
                   r: Math.random()*2+1 });
    }
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    nodes.forEach(n => {
      n.x += n.vx; n.y += n.vy;
      if (n.x < 0 || n.x > W) n.vx *= -1;
      if (n.y < 0 || n.y > H) n.vy *= -1;
    });
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i+1; j < nodes.length; j++) {
        const dx = nodes[i].x - nodes[j].x, dy = nodes[i].y - nodes[j].y;
        const d  = Math.sqrt(dx*dx + dy*dy);
        if (d < MAX_DIST) {
          ctx.beginPath();
          ctx.moveTo(nodes[i].x, nodes[i].y);
          ctx.lineTo(nodes[j].x, nodes[j].y);
          ctx.strokeStyle = `rgba(0,212,232,${(1-d/MAX_DIST)*0.5})`;
          ctx.lineWidth = 0.6;
          ctx.stroke();
        }
      }
    }
    nodes.forEach(n => {
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r, 0, Math.PI*2);
      ctx.fillStyle = 'rgba(0,212,232,0.5)'; ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  window.addEventListener('resize', () => { resize(); initNodes(); });
  resize(); initNodes(); draw();
})();


/* ── 2. NAV SCROLL SHRINK ──────────────────────────── */
window.addEventListener('scroll', () => {
  const nav = document.getElementById('nav');
  if (nav) nav.classList.toggle('scrolled', window.scrollY > 50);
});


/* ── 3. SCROLL REVEAL ──────────────────────────────── */
(function () {
  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) { e.target.classList.add('visible'); obs.unobserve(e.target); }
    });
  }, { threshold: 0.1 });
  document.querySelectorAll('.reveal').forEach(el => obs.observe(el));
})();


/* ── 4. FILE UPLOAD: click, drag-drop, preview ─────── */
(function () {
  const uploadArea   = document.getElementById('upload-area');
  const fileInput    = document.getElementById('file-input');
  const selectedBar  = document.getElementById('file-selected-bar');
  const filenameEl   = document.getElementById('selected-filename');
  const submitBtn    = document.getElementById('submit-btn');
  const previewThumb = document.getElementById('preview-thumb');
  const previewImg   = document.getElementById('preview-img');

  if (!uploadArea || !fileInput) return;

  // ── Drag-over visual class only (input handles the actual drop) ──
  uploadArea.addEventListener('dragenter', e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
  uploadArea.addEventListener('dragover',  e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
  uploadArea.addEventListener('dragleave', ()  => uploadArea.classList.remove('drag-over'));

  // ── Drop: explicitly assign file to <input> then show feedback ──
  uploadArea.addEventListener('drop', e => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const files = e.dataTransfer && e.dataTransfer.files;
    if (!files || !files.length) return;
    // Assign to the real file input so the multipart form sends it
    try {
      const dt = new DataTransfer();
      dt.items.add(files[0]);
      fileInput.files = dt.files;
    } catch (_) { /* very old browser — preview only */ }
    handleFile(files[0]);
  });

  // ── Native browse (click on the transparent input overlay) ──
  fileInput.addEventListener('change', () => {
    if (fileInput.files && fileInput.files[0]) handleFile(fileInput.files[0]);
  });

  // ── Validate → show filename bar + preview thumbnail → enable submit ──
  const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/tiff'];
  const ALLOWED_EXTS  = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'];

  function handleFile(file) {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_TYPES.includes(file.type) && !ALLOWED_EXTS.includes(ext)) {
      alert('Please upload a valid image: JPG, PNG, BMP, or TIFF.');
      fileInput.value = '';
      return;
    }
    // Filename bar
    if (filenameEl)   filenameEl.textContent = file.name;
    if (selectedBar)  selectedBar.classList.add('show');

    // Live preview
    if (previewThumb && previewImg) {
      const reader = new FileReader();
      reader.onload = ev => {
        previewImg.src = ev.target.result;
        previewThumb.classList.add('show');
      };
      reader.readAsDataURL(file);
    }
    // Unlock submit button
    if (submitBtn) submitBtn.disabled = false;
  }
})();


/* ── 5. LOADING ANIMATION ON SUBMIT ───────────────── */
(function () {
  const form     = document.getElementById('upload-form');
  const formWrap = document.getElementById('upload-form-container');
  const loadBox  = document.getElementById('loading-state');
  const label    = document.getElementById('loading-label');
  const steps    = [1,2,3,4,5,6].map(i => document.getElementById('step-' + i));
  const LABELS   = [
    'Preprocessing MRI...',
    'Extracting convolutional features...',
    'Running EfficientNet50 inference...',
    'Computing softmax probabilities...',
    'Finalizing report...',
    'Generating AI attention map...'
  ];

  if (!form) return;

  form.addEventListener('submit', function (e) {
    const fi = document.getElementById('file-input');
    if (!fi || !fi.files || fi.files.length === 0) {
      e.preventDefault();
      alert('Please select an MRI image first.');
      return;
    }
    // Show loading UI
    if (formWrap) formWrap.style.display = 'none';
    if (loadBox)  loadBox.style.display  = 'block';

    // Index for tracking the current loading step
    let idx = 0;
    function tick() {
      if (idx > 0 && steps[idx-1]) {
        steps[idx-1].classList.remove('active');
        steps[idx-1].classList.add('done');
      }
      if (idx < steps.length) {
        steps[idx].classList.add('active');
        if (label) label.textContent = LABELS[idx];
        idx++;
        setTimeout(tick, 950);
      }
      // Form submits normally to Flask — no e.preventDefault() called
    }
    tick();
  });
})();


/* ── 6. ANIMATE RESULT BARS (after Flask renders result) ── */
(function () {
  function animateBars() {
    document.querySelectorAll('[data-width]').forEach(bar => {
      const target = parseFloat(bar.getAttribute('data-width') || '0');
      bar.style.width = '0%';
      // Double rAF ensures the browser paints 0% before transitioning to target
      requestAnimationFrame(() => {
        requestAnimationFrame(() => { bar.style.width = target + '%'; });
      });
    });
  }
  // Only runs when Flask has injected data-width attributes into the DOM
  if (document.querySelector('[data-width]')) animateBars();
})();
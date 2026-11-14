/* GastricSentinel UI: splash, hash-routed tabs, analysis flow, rendering. All dynamic text uses textContent. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const TISSUE = { ADI: 'Adipose tissue', BACK: 'Background', DEB: 'Debris', LYM: 'Lymphocytes', MUC: 'Mucus', MUS: 'Smooth muscle',
    NOR: 'Normal mucosa', NORM: 'Normal mucosa', STR: 'Stroma', TUM: 'Tumour epithelium' };
  const state = { result: null, id: null, ready: false, card: null };
  const store = {
    get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
  };

  function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) if (kid != null) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return e;
  }
  const pct = (x, d = 1) => (x == null ? '-' : (x * 100).toFixed(d) + '%');
  const num = (x, d = 3) => (x == null ? '-' : Number(x).toFixed(d));
  const tissueName = (c) => TISSUE[c] || c;
  function toast(msg) { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove('show'), 3800); }
  function stagger(root) { $$(':scope > *', root).forEach((c, i) => c.style.setProperty('--i', i)); root.classList.add('stagger'); }

  /* ---------- icons ---------- */
  const ICONS = {
    scan: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="3.5"/></svg>',
    eye: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
    doc: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
    upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M7 9l5-5 5 5M4 20h16"/></svg>',
    lock: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="11" width="14" height="9" rx="2"/><path d="M8 11V8a4 4 0 0 1 8 0v3"/></svg>',
  };
  $$('[data-icon]').forEach((e) => { e.innerHTML = ICONS[e.dataset.icon] || ''; });

  /* ---------- splash ---------- */
  function splash() {
    const sp = $('#splash');
    document.body.classList.add('locked');
    let seen = false; try { seen = sessionStorage.getItem('gs-splash') === '1'; } catch { /* ignore */ }
    const finish = () => {
      if (finish.done) return; finish.done = true;
      try { sessionStorage.setItem('gs-splash', '1'); } catch { /* ignore */ }
      sp.classList.add('open'); document.body.classList.remove('locked');
      setTimeout(() => sp.remove(), reduce ? 0 : 1300);
      placeIndicator();
    };
    if (seen || reduce) { sp.remove(); document.body.classList.remove('locked'); return; }
    const title = $('#spTitle'); 'GastricSentinel'.split('').forEach((ch, i) => { const s = h('span', { text: ch }); s.style.animationDelay = (0.35 + i * 0.05) + 's'; title.append(s); });
    const msgs = ['Initialising', 'Loading image model', 'Loading clinical models', 'Checking validation record', 'Ready'];
    msgs.forEach((m, i) => setTimeout(() => { const s = $('#spStatus'); if (s) s.textContent = m; }, 350 + i * 520));
    $('#spSkip').addEventListener('click', finish);
    setTimeout(finish, 3300);
  }

  /* ---------- router ---------- */
  const tabs = $$('#tabs [data-tab]');
  const views = $$('.view');
  function placeIndicator() {
    const cur = $('#tabs [aria-selected=true]'); const ind = $('#tabInd'); if (!cur) return;
    ind.style.width = cur.offsetWidth + 'px'; ind.style.transform = `translateX(${cur.offsetLeft}px)`;
    cur.scrollIntoView({ block: 'nearest', inline: 'center', behavior: reduce ? 'auto' : 'smooth' });
  }
  function route() {
    let name = (location.hash.replace(/^#\/?/, '') || 'overview');
    if (!$('#v-' + name)) name = 'overview';
    const btn = $(`#tabs [data-tab=${name}]`);
    if (btn && btn.dataset.needs && !state.result) { toast('Run an analysis first.'); name = 'analyze'; }
    views.forEach((v) => v.classList.toggle('active', v.id === 'v-' + name));
    tabs.forEach((t) => { const on = t.dataset.tab === name; t.setAttribute('aria-selected', on); t.tabIndex = on ? 0 : -1; if (on) t.classList.remove('fresh'); });
    document.title = 'GastricSentinel · ' + $('#v-' + name).dataset.title;
    const v = $('#v-' + name); if (!v.dataset.st) { v.dataset.st = 1; $$('.stagger', v).forEach((s) => stagger(s)); }
    scrollTo({ top: 0, behavior: 'auto' });
    placeIndicator();
    if (name === 'history') renderHistory();
    if (name === 'overview') animateTimeline();
    if (name === 'results') requestAnimationFrame(animateResults);
    if (name === 'explain') requestAnimationFrame(animateResults);
    if (name === 'clinical') requestAnimationFrame(animateResults);
  }
  function syncLocks() {
    $$('#tabs [data-needs]').forEach((t) => t.classList.toggle('locked', !state.result));
  }
  addEventListener('hashchange', route);
  addEventListener('resize', placeIndicator);
  $('#tabs').addEventListener('keydown', (e) => {
    const i = tabs.indexOf(document.activeElement); if (i < 0) return;
    const j = e.key === 'ArrowRight' ? (i + 1) % tabs.length : e.key === 'ArrowLeft' ? (i - 1 + tabs.length) % tabs.length : -1;
    if (j >= 0) { tabs[j].focus(); e.preventDefault(); }
  });
  tabs.forEach((t) => t.addEventListener('click', () => { location.hash = '#/' + t.dataset.tab; }));

  function animateTimeline() {
    const li = $$('#timeline li'); li.forEach((x) => x.classList.remove('on'));
    if (reduce) { li.forEach((x) => x.classList.add('on')); $('#timeline').style.setProperty('--p', '100%'); return; }
    li.forEach((x, i) => setTimeout(() => { x.classList.add('on'); $('#timeline').style.setProperty('--p', ((i + 1) / li.length * 100) + '%'); }, 500 + i * 450));
  }

  /* ---------- status + overview ---------- */
  async function loadStatus() {
    const dot = $('#statusDot'), txt = $('#statusTxt');
    try {
      const r = await fetch('/api/status'); const j = await r.json();
      state.ready = !!j.ready;
      dot.className = 'dot' + (j.ready ? ' ok' : ''); txt.textContent = j.ready ? 'Models ready' : 'Models not loaded';
    } catch { txt.textContent = 'Offline'; }
    const s = $('#spStatus'); if (s && !state.ready) s.textContent = 'Models not loaded';
  }
  function countUp(el, to, fmt) {
    if (reduce) { el.textContent = fmt(to); return; }
    const t0 = performance.now(), D = 1100;
    const tick = (t) => { const p = Math.min(1, (t - t0) / D); el.textContent = fmt(to * (1 - Math.pow(1 - p, 3))); if (p < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  async function loadCard() {
    try { state.card = await (await fetch('/api/model-card')).json(); } catch { state.card = { available: false }; }
    const box = $('#statStrip'); box.replaceChildren();
    const c = state.card;
    if (!c.available) { box.append(h('div', { class: 'stat empty', text: 'No validation record found. Train the models to populate this panel.' })); }
    else {
      const m = c.image;
      [['Held-out accuracy', m.accuracy, (v) => pct(v)], ['Macro F1', m.macro_f1, (v) => num(v)],
       ['Calibration error (ECE)', m.ece, (v) => num(v)], ['Test images', m.n_test, (v) => Math.round(v).toLocaleString()]]
        .forEach(([l, v, f]) => { const b = h('b', { text: f(0) }); box.append(h('div', { class: 'stat' }, b, h('span', { text: l }))); if (v != null) countUp(b, v, f); });
    }
    stagger(box); renderMethods();
  }

  /* ---------- methods ---------- */
  function renderMethods() {
    const root = $('#methodsCard'); root.replaceChildren(); const c = state.card;
    if (!c || !c.available) { root.append(h('div', { class: 'card' }, h('p', { class: 'note', text: 'Validation record unavailable.' }))); return; }
    const m = c.image;
    const kv = (rows) => h('table', { class: 'kv' }, rows.filter((r) => r[1] != null).map(([k, v]) => h('tr', null, h('td', { text: k }), h('td', { text: String(v) }))));
    const ci = (a) => (a ? `${pct(a[0])} to ${pct(a[1])}` : null);
    root.append(h('div', { class: 'card' }, h('h3', { text: 'Image model: held-out test set' }), kv([
      ['Test images', m.n_test], ['Accuracy', pct(m.accuracy)], ['Accuracy 95% CI', ci(m.ci95_accuracy)], ['Balanced accuracy', pct(m.balanced_accuracy)],
      ['Macro precision / recall / F1', `${num(m.macro_precision)} / ${num(m.macro_recall)} / ${num(m.macro_f1)}`], ['ECE', num(m.ece)],
      ['Temperature', m.temperature != null ? num(m.temperature, 2) : null], ['Pretrained weights', m.pretrained == null ? null : String(m.pretrained)]])));
    const pc = h('table', { class: 'tbl' }, h('thead', null, h('tr', null, ['Class', 'Precision', 'Recall', 'F1', 'Support'].map((t) => h('th', { text: t })))),
      h('tbody', null, m.per_class.map((r) => h('tr', null, h('td', { text: `${r.class} · ${tissueName(r.class)}` }), h('td', { text: num(r.precision) }), h('td', { text: num(r.recall) }), h('td', { text: num(r.f1) }), h('td', { text: r.support })))));
    root.append(h('div', { class: 'card' }, h('h3', { text: 'Per-class results' }), pc));
    if (c.split) {
      const s = c.split, a = c.leakage_audit && c.leakage_audit.test_vs_train;
      root.append(h('div', { class: 'card' }, h('h3', { text: 'Data split and leakage audit' }), kv([
        ['Split mode', s.mode], ['Patient-level groups', s.n_groups], ['Leakage risk', s.leakage_risk ? 'Yes: split was not patient-level' : 'No known patient overlap'],
        ['Near-duplicates (test vs train)', a ? JSON.stringify(a) : null]])));
    }
    if (c.tabular) root.append(h('div', { class: 'card' }, h('h3', { text: 'Clinical and genomic subtype model' }), kv([
      ['Protocol', c.tabular.protocol], ['Patients', c.tabular.n_patients], ['Accuracy', pct(c.tabular.accuracy)], ['Macro F1', num(c.tabular.macro_f1)],
      ['Excluded rare classes', (() => { const x = c.tabular.excluded_classes; return Array.isArray(x) ? (x.join(', ') || 'none') : (x ? JSON.stringify(x) : 'none'); })()]])));
    if (c.fusion) root.append(h('div', { class: 'card' }, h('h3', { text: 'Fusion (experimental)' }), kv([['Pairing', c.fusion.pairing_mode]]), h('p', { class: 'note', text: c.fusion.disclosure })));
    stagger(root);
  }

  /* ---------- analyze ---------- */
  const STEPS = ['Upload', 'Preprocess', 'Classify', 'Grad-CAM', 'Clinical / genomic', 'Compose'];
  const stepsEl = $('#steps'); STEPS.forEach((s) => stepsEl.append(h('li', { class: 'step', text: s })));
  const drop = $('#drop'), file = $('#image'), prev = $('#prev');
  function showPreview(f) {
    if (!f) return;
    prev.src = URL.createObjectURL(f); drop.classList.add('has');
  }
  file.addEventListener('change', () => showPreview(file.files[0]));
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove('over')));
  $$('.btn').forEach((b) => b.addEventListener('pointerdown', (e) => {
    if (reduce || b.disabled) return; const r = b.getBoundingClientRect(), d = Math.max(r.width, r.height);
    const s = h('span', { class: 'rip' }); s.style.cssText = `width:${d}px;height:${d}px;left:${e.clientX - r.left - d / 2}px;top:${e.clientY - r.top - d / 2}px`; b.append(s); setTimeout(() => s.remove(), 650);
  }));

  function runSteps() {
    const items = $$('.step', stepsEl); let i = 0; items.forEach((x) => x.classList.remove('on', 'done'));
    const tick = () => { items.forEach((x, k) => { x.classList.toggle('done', k < i); x.classList.toggle('on', k === i); }); if (i < items.length - 1) i++; };
    tick(); const iv = setInterval(tick, 700);
    return () => { clearInterval(iv); items.forEach((x) => { x.classList.remove('on'); x.classList.add('done'); }); };
  }

  $('#f').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!file.files[0]) { toast('Choose an image first.'); return; }
    const btn = $('#go'), label = $('span', btn); btn.disabled = true; label.textContent = 'Analysing…'; drop.classList.add('scanning');
    const stop = runSteps();
    try {
      const r = await fetch('/api/predict', { method: 'POST', body: new FormData($('#f')) });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.error || `Request failed (${r.status})`);
      stop(); setResult(j, true); saveHistory(j);
      setTimeout(() => { location.hash = '#/results'; }, reduce ? 0 : 450);
    } catch (err) { stop(); toast(err.message); }
    finally { btn.disabled = false; label.textContent = 'Run analysis'; drop.classList.remove('scanning'); }
  });

  /* ---------- result rendering ---------- */
  function setResult(r, fresh) {
    state.result = r; state.id = r.id; window.GS && (window.GS.id = r.id);
    renderResults(r); renderExplain(r); renderClinical(r); syncLocks();
    if (fresh) ['results', 'explain', 'clinical'].forEach((n) => $(`#tabs [data-tab=${n}]`).classList.add('fresh'));
    document.dispatchEvent(new CustomEvent('gs:result', { detail: r }));
  }
  const empty = (msg) => h('div', { class: 'card empty' }, h('div', { class: 'ico', html: null }), h('h3', { text: 'Nothing here yet' }), h('p', { text: msg }), h('a', { class: 'btn primary', href: '#/analyze', text: 'New analysis' }));

  function gauge(p) {
    const g = h('div', { class: 'gauge' });
    g.innerHTML = '<svg viewBox="0 0 120 120"><circle class="bgc" cx="60" cy="60" r="54" fill="none" stroke-width="9"></circle><circle class="fg" cx="60" cy="60" r="54" fill="none" stroke-width="9"></circle></svg><div class="val">0%</div><div class="cap"></div>';
    g.dataset.p = p; g._fg = $('.fg', g); g._val = $('.val', g); g._C = 339.3; return g;
  }
  function bars(probs, top) {
    const wrap = h('div');
    Object.entries(probs).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
      const f = h('div', { class: 'f' }); f.dataset.w = (v * 100).toFixed(2);
      wrap.append(h('div', { class: 'bar' + (k === top ? ' top' : '') }, h('span', { text: k, title: tissueName(k) }), h('div', { class: 't' }, f), h('span', { class: 'n', text: pct(v) })));
    }); return wrap;
  }

  function renderResults(r) {
    const root = $('#resBody'); root.replaceChildren();
    const p = r.prediction;
    if (r.warnings && r.warnings.length) r.warnings.forEach((w) => root.append(h('div', { class: 'alert', text: w })));
    const g = gauge(p.confidence);
    $('.cap', g).textContent = 'confidence';
    const tags = h('div', { class: 'tags' });
    const used = r.modalities_used || {};
    ['image', 'clinical', 'genomic'].forEach((m) => tags.append(h('span', { class: 'tag ' + (used[m] ? 'on' : 'off'), text: m })));
    tags.append(h('span', { class: 'tag', text: 'prediction: ' + (r.mode === 'fusion' ? 'fusion' : 'image model') }));
    root.append(h('div', { class: 'card summary' }, g, h('div', null,
      h('div', { class: 'klabel', text: 'Predicted tissue class' }), h('div', { class: 'cls', text: p.class }), h('div', { class: 'tissue', text: tissueName(p.class) }),
      h('div', { class: 'note', text: `Tumour probability ${pct(r.tumour_probability)}` }), tags)));
    root.append(h('div', { class: 'card' }, h('h3', { text: 'Class probabilities' }), bars(p.probabilities, p.class)));
    const i = r.inputs || {};
    root.append(h('div', { class: 'card' }, h('h3', { text: 'Inputs used' }), h('table', { class: 'kv' }, [
      ['Age', i.age ?? 'not provided'], ['Sex', i.sex || 'not provided'], ['Stage', i.stage || 'not provided'],
      ['Gene expression', i.genomic_provided ? `provided, ${pct(i.genomic_coverage, 0)} gene coverage` : 'not provided']]
      .map(([k, v]) => h('tr', null, h('td', { text: k }), h('td', { text: String(v) }))))));
    root.append(h('div', { class: 'card next' }, h('a', { class: 'btn primary', href: `/report/${r.id}`, text: 'Download PDF report' }),
      h('a', { class: 'btn ghost', href: '#/explain', text: 'See explainability' }), h('a', { class: 'btn ghost', href: '#/assistant', text: 'Ask the assistant' }),
      h('a', { class: 'btn ghost', href: '#/analyze', text: 'New analysis' })));
    root.append(h('p', { class: 'note', text: r.disclaimer || '' }));
    stagger(root);
  }
  function animateResults() {
    $$('.view.active .gauge').forEach((g) => {
      const p = parseFloat(g.dataset.p); g._fg.style.strokeDashoffset = g._C * (1 - p);
      countUp(g._val, p, (v) => (v * 100).toFixed(1) + '%');
    });
    $$('.view.active .bar .f').forEach((f, i) => setTimeout(() => { f.style.width = f.dataset.w + '%'; }, reduce ? 0 : 150 + i * 70));
  }

  function renderExplain(r) {
    const root = $('#expBody'); root.replaceChildren();
    root.append(h('header', { class: 'vhead' }, h('h2', { text: 'Explainability' }), h('p', { text: 'Where the image model looked, and which inputs moved the subtype estimate.' })));
    const cmp = h('div', { class: 'cmp', style: '--pos:50%' },
      h('img', { src: r.input_url, alt: 'Original image' }),
      h('div', { class: 'top-img' }, h('img', { src: r.gradcam_url, alt: 'Grad-CAM overlay' })),
      h('div', { class: 'line' }), h('span', { class: 'lbl l', text: 'Grad-CAM' }), h('span', { class: 'lbl r', text: 'Original' }));
    const rg = h('input', { type: 'range', min: 0, max: 100, value: 50, 'aria-label': 'Compare original and Grad-CAM' });
    rg.addEventListener('input', () => cmp.style.setProperty('--pos', rg.value + '%')); cmp.append(rg);
    if (!reduce) { let v = 8; cmp.style.setProperty('--pos', '8%'); setTimeout(() => { cmp.style.transition = ''; const t0 = performance.now(); const go = (t) => { const k = Math.min(1, (t - t0) / 900); const x = 8 + 42 * (1 - Math.pow(1 - k, 3)); cmp.style.setProperty('--pos', x + '%'); rg.value = x; if (k < 1) requestAnimationFrame(go); }; requestAnimationFrame(go); }, 400); }
    root.append(h('div', { class: 'card' }, h('h3', { text: 'Grad-CAM' }), cmp,
      h('div', { class: 'legend' }, h('i'), h('div', null, h('span', { text: 'less influence' }), h('span', { text: 'more influence' }))),
      h('p', { class: 'note', text: 'Grad-CAM highlights regions that increased the predicted class score. It describes model behaviour and has not been validated by pathologists.' })));
    const ms = r.molecular_subtype;
    if (ms && ms.top_features && ms.top_features.length) {
      const mx = Math.max(...ms.top_features.map((t) => Math.abs(t.shap))) || 1;
      const sh = h('div');
      ms.top_features.forEach((t) => {
        const f = h('div', { class: 'f' }); f.dataset.w = (Math.abs(t.shap) / mx * 100).toFixed(1);
        sh.append(h('div', { class: 'bar shap ' + (t.shap >= 0 ? 'pos' : 'neg') }, h('span', { class: 'lab', text: t.feature, title: t.feature }), h('div', { class: 't' }, f), h('span', { class: 'n', text: (t.shap >= 0 ? '+' : '') + t.shap.toFixed(3) })));
      });
      root.append(h('div', { class: 'card' }, h('h3', { text: 'SHAP: features behind the subtype estimate' }), sh,
        h('p', { class: 'note', text: 'Teal pushes toward the predicted subtype, rose pushes away. Importance is not causation.' })));
    } else root.append(h('div', { class: 'card' }, h('h3', { text: 'SHAP' }), h('p', { class: 'note', text: 'No SHAP output: no clinical or genomic inputs were usable for this analysis.' })));
    stagger(root);
  }

  function renderClinical(r) {
    const root = $('#cliBody'); root.replaceChildren();
    root.append(h('header', { class: 'vhead' }, h('h2', { text: 'Clinical and genomic' }), h('p', { text: 'Molecular subtype estimate from clinical and gene-expression inputs, independent of the image.' })));
    const ms = r.molecular_subtype;
    if (!ms) {
      root.append(h('div', { class: 'card empty' }, h('h3', { text: 'No subtype estimate' }), h('p', { text: 'Provide age, sex, stage or a gene-expression file to get one.' }), h('a', { class: 'btn primary', href: '#/analyze', text: 'New analysis' })));
    } else {
      root.append(h('div', { class: 'card' }, h('div', { class: 'klabel', text: 'Subtype model: ' + (ms.model || '') }), h('div', { class: 'cls', text: ms.class }),
        ms.note ? h('p', { class: 'note', text: ms.note }) : null));
      if (ms.probabilities) root.append(h('div', { class: 'card' }, h('h3', { text: 'Subtype probabilities' }), bars(ms.probabilities, ms.class)));
    }
    const ef = r.experimental_fusion;
    if (ef) {
      const pr = ef.probabilities || (ef.prediction && ef.prediction.probabilities);
      root.append(h('div', { class: 'card' }, h('h3', { text: 'Fusion (experimental)' }),
        ef.class ? h('p', null, 'Fusion class: ', h('b', { text: ef.class })) : null,
        pr ? bars(pr, ef.class) : null,
        h('p', { class: 'note', text: ef.disclosure || ef.note || 'Experimental. Image and genomic cohorts are not patient-matched, so this is not evidence of multimodal benefit.' })));
    }
    stagger(root);
  }

  /* ---------- history ---------- */
  function saveHistory(r) {
    const hst = store.get('gs-history', []).filter((x) => x.id !== r.id);
    hst.unshift({ id: r.id, cls: r.prediction.class, conf: r.prediction.confidence, t: r.created || new Date().toISOString(), img: r.input_url });
    store.set('gs-history', hst.slice(0, 24));
  }
  function renderHistory() {
    const root = $('#histBody'); root.replaceChildren();
    const hst = store.get('gs-history', []);
    if (!hst.length) { root.append(h('div', { class: 'card empty' }, h('h3', { text: 'No analyses yet' }), h('p', { text: 'Completed analyses from this browser appear here.' }), h('a', { class: 'btn primary', href: '#/analyze', text: 'New analysis' }))); return; }
    const grid = h('div', { class: 'hist' });
    hst.forEach((x, i) => {
      const img = h('img', { src: x.img, alt: '', loading: 'lazy' }); img.onerror = () => { img.style.visibility = 'hidden'; };
      const b = h('button', { class: 'hitem', type: 'button' }, img, h('div', null, h('b', { text: x.cls }), h('small', { text: `${tissueName(x.cls)} · ${pct(x.conf, 0)}` }), h('small', { text: new Date(x.t).toLocaleString() })));
      b.style.setProperty('--i', i);
      b.addEventListener('click', async () => {
        try { const r = await fetch('/api/result/' + x.id); if (!r.ok) throw 0; setResult(await r.json(), false); location.hash = '#/results'; }
        catch { store.set('gs-history', store.get('gs-history', []).filter((y) => y.id !== x.id)); toast('That analysis has expired and was removed.'); renderHistory(); }
      });
      grid.append(b);
    });
    root.append(grid); stagger(grid);
  }

  window.GS = { id: null, TISSUE, state };
  splash(); syncLocks();
  Promise.all([loadStatus(), loadCard()]).finally(() => { route(); });
  route();
})();

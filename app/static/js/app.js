/* GastricSentinel UI: opening sequence, analysis flow, result rendering, evidence. All dynamic text uses textContent. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const TISSUE = { ADI: 'Adipose tissue', BACK: 'Background', DEB: 'Debris', LYM: 'Lymphocytes', MUC: 'Mucus', MUS: 'Smooth muscle',
    NOR: 'Normal mucosa', NORM: 'Normal mucosa', STR: 'Stroma', TUM: 'Tumour epithelium' };
  const state = { result: null, card: null, ready: false };
  const GS = window.GS = { id: null, TISSUE, state };
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
  function toast(msg) { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(toast.t); toast.t = setTimeout(() => t.classList.remove('show'), 4200); }
  GS.toast = toast;
  const reveal = (root) => { if (GS.reveal) GS.reveal(root); };

  /* ---------- icons ---------- */
  const ICONS = {
    scan: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="3.5"/></svg>',
    eye: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/></svg>',
    doc: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/></svg>',
    upload: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 16V4M7 9l5-5 5 5M4 20h16"/></svg>',
  };
  $$('[data-icon]').forEach((e) => { e.innerHTML = ICONS[e.dataset.icon] || ''; });

  /* ---------- greeting ---------- */
  const hr = new Date().getHours();
  $('#greet').textContent = hr < 5 ? 'Still up' : hr < 12 ? 'Good morning' : hr < 17 ? 'Good afternoon' : hr < 22 ? 'Good evening' : 'Late night';

  /* ---------- splash ---------- */
  (function splash() {
    const sp = $('#splash'); document.body.classList.add('locked');
    let seen = false; try { seen = sessionStorage.getItem('gs-splash') === '1'; } catch { /* ignore */ }
    const finish = () => {
      if (finish.done) return; finish.done = true;
      try { sessionStorage.setItem('gs-splash', '1'); } catch { /* ignore */ }
      sp.classList.add('open'); document.body.classList.remove('locked'); setTimeout(() => sp.remove(), reduce ? 0 : 1300);
    };
    if (seen || reduce) { sp.remove(); document.body.classList.remove('locked'); return; }
    const title = $('#spTitle');
    'GastricSentinel'.split('').forEach((ch, i) => { const s = h('span', { text: ch }); s.style.animationDelay = (0.35 + i * 0.05) + 's'; title.append(s); });
    ['Warming up', 'Loading the image model', 'Loading clinical models', 'Checking the validation record', 'Ready'].forEach((m, i) => setTimeout(() => { const s = $('#spStatus'); if (s) s.textContent = m; }, 350 + i * 520));
    $('#spSkip').addEventListener('click', finish); setTimeout(finish, 3300);
  })();

  /* ---------- status + overview ---------- */
  function countUp(el, to, fmt) {
    if (reduce) { el.textContent = fmt(to); return; }
    const t0 = performance.now(), D = 1200;
    const tick = (t) => { const p = Math.min(1, (t - t0) / D); el.textContent = fmt(to * (1 - Math.pow(1 - p, 3))); if (p < 1) requestAnimationFrame(tick); };
    requestAnimationFrame(tick);
  }
  GS.countUp = countUp;
  async function loadStatus() {
    const dot = $('#statusDot'), txt = $('#statusTxt');
    try {
      const j = await (await fetch('/api/status')).json();
      state.ready = !!j.ready; dot.className = 'dot' + (j.ready ? ' ok' : ''); txt.textContent = j.ready ? 'Models ready' : 'Models not loaded';
    } catch { txt.textContent = 'Offline'; }
  }
  async function loadCard() {
    try { state.card = await (await fetch('/api/model-card')).json(); } catch { state.card = { available: false }; }
    const box = $('#statStrip'); box.replaceChildren(); const c = state.card;
    if (!c.available) box.append(h('div', { class: 'stat empty', text: 'No validation record found yet. Train the models and this fills in.' }));
    else {
      const m = c.image;
      [['Held-out accuracy', m.accuracy, (v) => pct(v)], ['Macro F1', m.macro_f1, (v) => num(v)], ['Calibration error (ECE)', m.ece, (v) => num(v)], ['Test images', m.n_test, (v) => Math.round(v).toLocaleString()]]
        .forEach(([l, v, f], i) => { const b = h('b', { text: f(0) }); const s = h('div', { class: 'stat' }, b, h('span', { text: l })); s.style.setProperty('--i', i); box.append(s); if (v != null) setTimeout(() => countUp(b, v, f), 500 + i * 90); });
    }
    renderEvidence();
  }

  /* ---------- evidence ---------- */
  function renderEvidence() {
    const root = $('#methodsCard'); root.replaceChildren(); const c = state.card;
    const card = (...k) => { const d = h('div', { class: 'card', 'data-reveal': '' }, ...k); return d; };
    if (!c || !c.available) { root.append(card(h('p', { class: 'note', text: 'The validation record isn’t available yet.' }))); reveal(root); return; }
    const m = c.image;
    const kv = (rows) => h('table', { class: 'kv' }, rows.filter((r) => r[1] != null).map(([k, v]) => h('tr', null, h('td', { text: k }), h('td', { text: String(v) }))));
    const ci = (a) => (Array.isArray(a) ? `${pct(a[0])} to ${pct(a[1])}` : null);
    root.append(card(h('h3', { text: 'Image model on the held-out test set' }), kv([
      ['Test images', m.n_test], ['Accuracy', pct(m.accuracy)], ['Accuracy, 95% CI', ci(m.ci95_accuracy)], ['Balanced accuracy', pct(m.balanced_accuracy)],
      ['Macro precision / recall / F1', `${num(m.macro_precision)} / ${num(m.macro_recall)} / ${num(m.macro_f1)}`], ['Calibration error (ECE)', num(m.ece)],
      ['Temperature scaling', m.temperature != null ? num(m.temperature, 2) : null]])));
    const pc = h('table', { class: 'tbl' }, h('thead', null, h('tr', null, ['Class', 'Precision', 'Recall', 'F1', 'Support'].map((t) => h('th', { text: t })))),
      h('tbody', null, (m.per_class || []).map((r) => h('tr', null, h('td', { text: `${r.class} · ${tissueName(r.class)}` }), h('td', { text: num(r.precision) }), h('td', { text: num(r.recall) }), h('td', { text: num(r.f1) }), h('td', { text: r.support })))));
    root.append(card(h('h3', { text: 'Per-class results' }), h('div', { class: 'tbl-wrap' }, pc)));
    if (c.split) {
      const a = c.leakage_audit && c.leakage_audit.test_vs_train;
      root.append(card(h('h3', { text: 'Data split and leakage check' }), kv([
        ['Split', c.split.mode], ['Leakage risk', c.split.leakage_risk ? 'Yes: this was not a patient-level split, so scores may be optimistic' : 'No known patient overlap'],
        ['Near-duplicates, test vs train', a ? JSON.stringify(a) : null]])));
    }
    if (c.tabular) root.append(card(h('h3', { text: 'Clinical and gene-expression subtype model' }), kv([
      ['Protocol', c.tabular.protocol], ['Patients', c.tabular.n_patients], ['Accuracy', pct(c.tabular.accuracy)], ['Macro F1', num(c.tabular.macro_f1)],
      ['Left out (too few patients)', (() => { const x = c.tabular.excluded_classes; return Array.isArray(x) ? (x.join(', ') || 'none') : (x && Object.keys(x).length ? JSON.stringify(x) : 'none'); })()]])));
    if (c.fusion) root.append(card(h('h3', { text: 'Fusion (experimental)' }), kv([['Pairing', c.fusion.pairing_mode]]), h('p', { class: 'note', text: c.fusion.disclosure })));
    $$('[data-reveal]', root).forEach((e, i) => e.style.setProperty('--i', i % 2)); reveal(root);
  }

  /* ---------- analyze ---------- */
  const STEPS = ['Upload', 'Preprocess', 'Classify', 'Grad-CAM', 'Clinical / genes', 'Write up'];
  const stepsEl = $('#steps'); STEPS.forEach((s) => stepsEl.append(h('li', { class: 'step', text: s })));
  const drop = $('#drop'), file = $('#image'), prev = $('#prev');
  file.addEventListener('change', () => { const f = file.files[0]; if (!f) return; if (prev.dataset.u) URL.revokeObjectURL(prev.dataset.u); prev.src = prev.dataset.u = URL.createObjectURL(f); drop.classList.add('has'); });
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove('over')));

  function runSteps() {
    const items = $$('.step', stepsEl); let i = 0; items.forEach((x) => x.classList.remove('on', 'done'));
    const tick = () => { items.forEach((x, k) => { x.classList.toggle('done', k < i); x.classList.toggle('on', k === i); }); if (i < items.length - 1) i++; };
    tick(); const iv = setInterval(tick, 750);
    return () => { clearInterval(iv); items.forEach((x) => { x.classList.remove('on'); x.classList.add('done'); }); };
  }
  $('#f').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!file.files[0]) { toast('Pick an image first.'); return; }
    const btn = $('#go'), label = $('span', btn); btn.disabled = true; label.textContent = 'Looking closely…'; drop.classList.add('scanning');
    const stop = runSteps();
    try {
      const r = await fetch('/api/predict', { method: 'POST', body: new FormData($('#f')) });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.error || `Something went wrong (${r.status})`);
      stop(); saveHistory(j); setResult(j, true);
    } catch (err) { stop(); toast(err.message); }
    finally { btn.disabled = false; label.textContent = 'Run analysis'; drop.classList.remove('scanning'); }
  });

  /* ---------- results ---------- */
  function setResult(r, fresh) {
    state.result = r; GS.id = r.id;
    renderResults(r);
    const sec = $('#results'); sec.hidden = false; sec.classList.add('show'); $('#navResults').hidden = false;
    const grid = $('#resBody'); grid.classList.remove('go');
    requestAnimationFrame(() => requestAnimationFrame(() => {
      sec.scrollIntoView({ behavior: reduce ? 'auto' : 'smooth', block: 'start' });
      setTimeout(() => { grid.classList.add('go'); animateResults(); }, reduce ? 0 : 450);
    }));
    renderRecent();
    document.dispatchEvent(new CustomEvent('gs:result', { detail: { result: r, fresh } }));
  }
  function gauge(p) {
    const g = h('div', { class: 'gauge' });
    g.innerHTML = '<svg viewBox="0 0 120 120"><circle class="bgc" cx="60" cy="60" r="54" fill="none" stroke-width="9"></circle><circle class="fg" cx="60" cy="60" r="54" fill="none" stroke-width="9"></circle></svg><div class="val">0%</div><div class="cap">confidence</div>';
    g.dataset.p = p; return g;
  }
  function bars(probs, top, mod) {
    const wrap = h('div');
    Object.entries(probs).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
      const f = h('div', { class: 'f' }); f.dataset.w = (v * 100).toFixed(2);
      wrap.append(h('div', { class: 'bar' + (k === top ? ' top' : '') }, h('span', { text: k, title: tissueName(k) }), h('div', { class: 't' }, f), h('span', { class: 'n', text: pct(v) })));
    }); return wrap;
  }
  function plainRead(r) {
    const p = r.prediction, c = p.confidence;
    const band = c >= 0.85 ? 'quite sure' : c >= 0.6 ? 'fairly sure' : 'not very sure';
    const out = h('p', { class: 'said' }, 'My best read: this patch looks like ', h('b', { text: tissueName(p.class).toLowerCase() }), `, and I'm ${band} (${pct(c, 0)}). `);
    if (p.low_confidence) out.append('That’s low, so treat it as a maybe and get a second look. ');
    if (r.tumour_probability != null) out.append(`The tumour score is ${pct(r.tumour_probability, 0)}. That’s a model score for this one patch, not a diagnosis.`);
    return out;
  }
  function renderResults(r) {
    const root = $('#resBody'); root.replaceChildren(); const p = r.prediction; let i = 0;
    const add = (el, wide) => { el.style.setProperty('--i', i++); if (wide) el.classList.add('wide'); root.append(el); };
    (r.warnings || []).forEach((w) => add(h('div', { class: 'alert', text: w }), true));
    const tags = h('div', { class: 'tags' }); const used = r.modalities_used || {};
    ['image', 'clinical', 'genomic'].forEach((m) => tags.append(h('span', { class: 'tag ' + (used[m] ? 'on' : 'off'), text: m })));
    tags.append(h('span', { class: 'tag', text: 'prediction: ' + (r.mode === 'fusion' ? 'fusion' : 'image model') }));
    add(h('div', { class: 'card summary' }, gauge(p.confidence), h('div', null, h('div', { class: 'klabel', text: 'Predicted tissue' }), h('div', { class: 'cls', text: p.class }), h('div', { class: 'tissue', text: tissueName(p.class) }), plainRead(r), tags)), true);

    const cmp = h('div', { class: 'cmp', style: '--pos:50%' }, h('img', { src: r.input_url, alt: 'Original image' }),
      h('div', { class: 'top-img' }, h('img', { src: r.gradcam_url, alt: 'Grad-CAM overlay' })), h('div', { class: 'line' }), h('span', { class: 'lbl l', text: 'Grad-CAM' }), h('span', { class: 'lbl r', text: 'Original' }));
    const rg = h('input', { type: 'range', min: 0, max: 100, value: 50, 'aria-label': 'Slide between the original image and the Grad-CAM overlay' });
    rg.addEventListener('input', () => cmp.style.setProperty('--pos', rg.value + '%')); cmp.append(rg);
    cmp._intro = () => { if (reduce) return; const t0 = performance.now(); const go = (t) => { const k = Math.min(1, (t - t0) / 1100); const x = 6 + 44 * (1 - Math.pow(1 - k, 3)); cmp.style.setProperty('--pos', x + '%'); rg.value = x; if (k < 1) requestAnimationFrame(go); }; cmp.style.setProperty('--pos', '6%'); requestAnimationFrame(go); };
    root._cmp = cmp;
    add(h('div', { class: 'card' }, h('h3', { text: 'Where it looked' }), cmp,
      h('div', { class: 'legend' }, h('i'), h('div', null, h('span', { text: 'less influence' }), h('span', { text: 'more influence' }))),
      h('p', { class: 'note', text: 'Drag the slider. Warm colours are regions that raised the predicted class score. It describes the model, and pathologists haven’t validated it.' })));
    add(h('div', { class: 'card' }, h('h3', { text: 'How the probability was spread' }), bars(p.probabilities, p.class)));

    const ms = r.molecular_subtype;
    if (ms) {
      const body = [h('div', { class: 'klabel', text: 'Subtype model: ' + (ms.model || '').replace(/_/g, ' ') }), h('div', { class: 'cls', style: 'font-size:clamp(28px,6vw,40px)', text: ms.class })];
      if (ms.probabilities) body.push(bars(ms.probabilities, ms.class));
      if (ms.note) body.push(h('p', { class: 'note', text: ms.note }));
      add(h('div', { class: 'card' }, h('h3', { text: 'Molecular subtype estimate' }), ...body));
      if (ms.top_features && ms.top_features.length) {
        const mx = Math.max(...ms.top_features.map((t) => Math.abs(t.shap))) || 1; const sh = h('div');
        ms.top_features.forEach((t) => { const f = h('div', { class: 'f' }); f.dataset.w = (Math.abs(t.shap) / mx * 100).toFixed(1);
          sh.append(h('div', { class: 'bar shap ' + (t.shap >= 0 ? 'pos' : 'neg') }, h('span', { class: 'lab', text: t.feature, title: t.feature }), h('div', { class: 't' }, f), h('span', { class: 'n', text: (t.shap >= 0 ? '+' : '') + t.shap.toFixed(3) }))); });
        add(h('div', { class: 'card' }, h('h3', { text: 'What pushed that estimate' }), sh, h('p', { class: 'note', text: 'Teal pushes toward the predicted subtype, rose pushes away. Importance isn’t causation.' })));
      }
    } else add(h('div', { class: 'card' }, h('h3', { text: 'Molecular subtype' }), h('p', { class: 'note', text: 'No age, sex, stage or gene file was given, so there’s no subtype estimate. Add some context above and run it again for one.' })));

    const ef = r.experimental_fusion;
    if (ef) { const pr = ef.probabilities || (ef.prediction && ef.prediction.probabilities);
      add(h('div', { class: 'card' }, h('h3', { text: 'Fusion (experimental)' }), ef.class ? h('p', null, 'Fusion class: ', h('b', { text: ef.class })) : null, pr ? bars(pr, ef.class) : null,
        h('p', { class: 'note', text: ef.disclosure || ef.note || 'Experimental. The image and gene cohorts aren’t patient-matched, so this is no evidence of multimodal benefit.' }))); }

    const i2 = r.inputs || {};
    add(h('div', { class: 'card' }, h('h3', { text: 'What I was given' }), h('table', { class: 'kv' }, [
      ['Age', i2.age ?? 'not provided'], ['Sex', i2.sex || 'not provided'], ['Stage', i2.stage || 'not provided'],
      ['Gene expression', i2.genomic_provided ? `provided, ${pct(i2.genomic_coverage, 0)} gene coverage` : 'not provided']]
      .map(([k, v]) => h('tr', null, h('td', { text: k }), h('td', { text: String(v) }))))));
    add(h('div', { class: 'card next' },
      h('a', { class: 'btn primary', href: `/report/${r.id}`, text: 'Download the PDF report' }),
      h('button', { class: 'btn ghost', type: 'button', text: 'Ask the assistant', onclick: () => GS.chat && GS.chat.open('What does this result mean?') }),
      h('a', { class: 'btn ghost', href: '#analyze', text: 'Try another image' })), true);
    if (r.disclaimer) add(h('p', { class: 'note', text: r.disclaimer }), true);
  }
  function animateResults() {
    const root = $('#resBody');
    $$('.gauge', root).forEach((g) => { const p = parseFloat(g.dataset.p); $('.fg', g).style.strokeDashoffset = 339.3 * (1 - p); countUp($('.val', g), p, (v) => (v * 100).toFixed(1) + '%'); });
    $$('.bar .f', root).forEach((f, i) => setTimeout(() => { f.style.width = f.dataset.w + '%'; }, reduce ? 0 : 200 + i * 60));
    if (root._cmp && root._cmp._intro) setTimeout(root._cmp._intro, 400);
  }

  /* ---------- recent ---------- */
  function saveHistory(r) {
    const hst = store.get('gs-history', []).filter((x) => x.id !== r.id);
    hst.unshift({ id: r.id, cls: r.prediction.class, conf: r.prediction.confidence, t: r.created || new Date().toISOString(), img: r.input_url });
    store.set('gs-history', hst.slice(0, 12));
  }
  function renderRecent() {
    const hst = store.get('gs-history', []); const box = $('#recent'), row = $('#recentRow'); row.replaceChildren();
    box.hidden = !hst.length;
    hst.forEach((x, i) => {
      const img = h('img', { src: x.img, alt: '', loading: 'lazy' }); img.onerror = () => { img.style.visibility = 'hidden'; };
      const b = h('button', { class: 'hitem', type: 'button' }, img, h('div', null, h('b', { text: x.cls }), h('small', { text: `${tissueName(x.cls)} · ${pct(x.conf, 0)}` }), h('small', { text: new Date(x.t).toLocaleString() })));
      b.style.setProperty('--i', i);
      b.addEventListener('click', async () => {
        try { const r = await fetch('/api/result/' + x.id); if (!r.ok) throw 0; setResult(await r.json(), false); }
        catch { store.set('gs-history', store.get('gs-history', []).filter((y) => y.id !== x.id)); toast('That one has expired and was removed.'); renderRecent(); }
      });
      row.append(b);
    });
  }

  renderRecent(); loadStatus(); loadCard();
})();

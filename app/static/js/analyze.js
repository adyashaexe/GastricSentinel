/* Analyze view: upload form, result page. Also shared helpers (bars, history, openResult). */
(() => {
  'use strict';
  const GS = window.GS, { $, $$, h, pct, num, tissueName } = GS;

  /* ---------- shared helpers ---------- */
  GS.bars = (probs, top, opts = {}) => {
    const wrap = h('div');
    Object.entries(probs).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
      const f = h('div', { class: 'f' }); f.dataset.w = (v * 100).toFixed(2); if (opts.color) f.style.background = GS.color(k);
      wrap.append(h('div', { class: 'bar' + (k === top ? ' lead' : '') }, h('span', { text: k, title: tissueName(k) }), h('div', { class: 't' }, f), h('span', { class: 'n', text: pct(v) })));
    });
    return wrap;
  };
  GS.grow = (root) => requestAnimationFrame(() => requestAnimationFrame(() => $$('.f[data-w]', root).forEach((f) => { f.style.width = f.dataset.w + '%'; })));
  GS.history = {
    all: () => GS.store.get('gs-history', []),
    add(r) {
      const l = this.all().filter((x) => x.id !== r.id);
      l.unshift({ id: r.id, cls: r.prediction.class, conf: r.prediction.confidence, t: r.created || new Date().toISOString(), mode: r.mode, img: r.input_url });
      GS.store.set('gs-history', l.slice(0, 60));
    },
    remove(id) { GS.store.set('gs-history', this.all().filter((x) => x.id !== id)); },
  };
  GS.kv = (rows) => h('table', { class: 'kv' }, rows.filter((r) => r[1] != null).map(([k, v]) => h('tr', null, h('td', { text: k }), h('td', { text: String(v) }))));
  GS.confBand = (c) => (c >= 0.85 ? 'quite sure' : c >= 0.6 ? 'fairly sure' : 'not very sure');

  GS.openResult = (r, fresh) => {
    GS.id = r.id; GS.state = r; GS.history.add(r);
    if (r.mode === 'slide') { GS.views.slide.showResult(r); GS.go('slide'); }
    else { GS.views.analyze.showResult(r); GS.go('analyze'); }
    document.dispatchEvent(new CustomEvent('gs:result', { detail: { result: r, fresh: !!fresh } }));
  };
  GS.loadAndOpen = async (id) => { try { GS.openResult(await GS.api('/api/result/' + id), false); } catch (e) { GS.toast(e.status === 404 ? 'That analysis is no longer available.' : e.message); if (e.status === 404) GS.history.remove(id); } };

  /* ---------- form ---------- */
  const drop = $('#drop'), file = $('#image'), prev = $('#prev'), steps = $('#steps');
  const STEPS = ['Upload', 'Quality check', 'Classify', 'Grad-CAM', 'Clinical / genes', 'Save'];
  STEPS.forEach((s) => steps.append(h('li', { class: 'step', text: s })));
  const setFile = (f) => { if (!f) return; if (prev.dataset.u) URL.revokeObjectURL(prev.dataset.u); prev.src = prev.dataset.u = URL.createObjectURL(f); drop.classList.add('has'); };
  file.addEventListener('change', () => setFile(file.files[0]));
  ['dragenter', 'dragover'].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove('over')));
  function runSteps() {
    const items = $$('.step', steps); let i = 0; steps.classList.add('on'); items.forEach((x) => x.classList.remove('on', 'done'));
    const tick = () => { items.forEach((x, k) => { x.classList.toggle('done', k < i); x.classList.toggle('on', k === i); }); if (i < items.length - 1) i++; };
    tick(); const iv = setInterval(tick, 800);
    return () => { clearInterval(iv); items.forEach((x) => { x.classList.remove('on'); x.classList.add('done'); }); setTimeout(() => steps.classList.remove('on'), 600); };
  }
  $('#f').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!file.files[0]) { GS.toast('Choose an image first.'); return; }
    const btn = $('#go'); GS.busy(btn, true, 'Analysing'); const stop = runSteps();
    try {
      const j = await GS.api('/api/predict', { method: 'POST', body: new FormData($('#f')) });
      stop(); GS.openResult(j, true);
    } catch (err) { stop(); GS.toast(err.message); if (err.status === 401 && GS.openAuth) GS.openAuth('login'); }
    finally { GS.busy(btn, false); }
  });

  /* ---------- result page ---------- */
  function ring(p) {
    const w = h('div', { class: 'ring' }); w.innerHTML = '<svg viewBox="0 0 120 120" aria-hidden="true"><circle class="bg" cx="60" cy="60" r="54"/><circle class="fg" cx="60" cy="60" r="54"/></svg><div class="val"><span>0%</span><small>confidence</small></div>'; w.dataset.p = p; return w;
  }
  function inputCheck(r) {
    const o = r.ood; if (!o) return null;
    const fd = o.feature_distance, q = o.quality || {};
    const card = h('div', { class: 'card' }, h('div', { class: 'card-h' }, h('h3', { text: 'Input check' }), h('span', { class: 'tag ' + (o.flagged ? 'warn' : 'ok'), text: o.flagged ? 'Needs attention' : 'Looks in-distribution' })));
    if (fd) {
      const pos = Math.min(100, fd.ratio * 50);
      card.append(h('div', { class: 'lbl', text: 'Distance from the training images' }),
        h('div', { class: 'meter', style: 'position:relative;margin:8px 0 4px' }, h('i', { style: `width:${pos}%;background:${fd.outside ? 'var(--rose)' : 'var(--ok)'}` }), h('b', { style: 'position:absolute;left:50%;top:-4px;bottom:-4px;width:2px;background:var(--head)', title: 'Flag threshold' })),
        h('p', { class: 'note', text: `Score ${fd.score.toFixed(1)}; flagged above ${fd.threshold.toFixed(1)} (the 99th percentile of held-out images). Typical ${fd.typical.toFixed(1)}.` }));
    } else card.append(h('p', { class: 'note', text: 'The feature-space check is available after retraining with this version of the training script.' }));
    if (q.tissue_fraction != null) card.append(GS.kv([['Stained tissue in view', pct(q.tissue_fraction, 0)], ['H&E-like colour', pct(q.he_like, 0)], ['Sharpness', q.sharpness < 1e-4 ? 'very low' : q.sharpness.toExponential(1)], ['Size', `${q.width} × ${q.height} px`]]));
    return card;
  }
  function viewerCard(r) {
    let mode = 'cam';
    const view = new GS.Viewer({ src: r.input_url, cam: r.gradcam_url, mode, opacity: 0.85, alt: 'Histology patch' });
    const seg = h('div', { class: 'seg' }, [['orig', 'Original'], ['cam', 'Grad-CAM'], ['split', 'Split']].map(([k, l]) => h('button', { type: 'button', 'data-m': k, class: k === mode ? 'on' : '', text: l })));
    const range = h('input', { type: 'range', min: 0, max: 100, value: 85, 'aria-label': 'Grad-CAM strength' });
    const rl = h('label', null, h('span', { text: 'Overlay' }), range);
    range.addEventListener('input', () => { const x = range.value / 100; if (mode === 'split') view.setSplit(x); else view.setOpacity(x); });
    seg.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; mode = b.dataset.m; $$('button', seg).forEach((x) => x.classList.toggle('on', x === b)); view.setMode(mode); range.disabled = mode === 'orig'; range.value = mode === 'split' ? view.split * 100 : view.opacity * 100; rl.firstChild.textContent = mode === 'split' ? 'Divider' : 'Overlay'; });
    const dl = h('a', { class: 'btn ghost sm', href: r.gradcam_url, download: `gradcam_${r.id.slice(0, 8)}.png`, text: 'Save overlay' });
    const bar = h('div', { class: 'vbar' }, seg, rl, h('span', { class: 'grow' }), dl);
    const card = h('div', { class: 'card', 'data-tour': 'viewer' }, h('div', { class: 'card-h' }, h('h3', { text: 'Image and Grad-CAM' })), bar, view.el,
      h('div', { class: 'legend' }, h('span', { text: 'less influence' }), h('i'), h('span', { text: 'more influence' })),
      h('p', { class: 'note', text: 'Warm colours are regions that raised the predicted class score. Grad-CAM describes the model, and pathologists have not validated it.' }));
    card._viewer = view; return card;
  }
  function plain(r) {
    const p = r.prediction;
    const out = h('p', { class: 'said' }, 'This patch looks like ', h('b', { text: tissueName(p.class).toLowerCase() }), `, and the model is ${GS.confBand(p.confidence)} (${pct(p.confidence, 0)}). `);
    if (p.low_confidence) out.append('That is low, so treat it as a maybe and get a second look. ');
    if (r.tumour_probability != null) out.append(`Tumour score ${pct(r.tumour_probability, 0)}: a model score for this patch, not a diagnosis.`);
    return out;
  }
  function render(r) {
    const root = h('div'); const p = r.prediction;
    const created = r.created ? GS.when(r.created) : '';
    const exp = (fmt) => `/api/export?ids=${r.id}&format=${fmt}`;
    root.append(h('div', { class: 'vh' }, h('div', null, h('h1', { text: 'Result' }), h('p', { text: `${created}${r.model_version ? ' · model ' + r.model_version : ''}${r.model_info && r.model_info.arch ? ' · ' + r.model_info.arch : ''}` })),
      h('div', { class: 'acts' },
        h('button', { class: 'btn ghost', type: 'button', text: 'New analysis', onclick: () => GS.views.analyze.showForm() }),
        h('a', { class: 'btn primary', href: `/report/${r.id}`, text: 'PDF report' }),
        h('a', { class: 'btn ghost', href: exp('csv'), text: 'CSV' }), h('a', { class: 'btn ghost', href: exp('json'), text: 'JSON' }),
        h('button', { class: 'btn ghost', type: 'button', text: 'Compare', onclick: () => { GS.compareWith = r.id; GS.go('compare'); } }),
        h('button', { class: 'btn ghost', type: 'button', text: 'Ask assistant', onclick: () => GS.chat && GS.chat.open('What does this result mean?') }))));
    const warn = h('div', { class: 'stack', style: 'margin-bottom:16px' }, (r.warnings || []).map((w) => h('div', { class: 'alert', text: w })));
    if (r.warnings && r.warnings.length) root.append(warn);

    const used = r.modalities_used || {}; const tags = h('div', { class: 'tags' });
    ['image', 'clinical', 'genomic'].forEach((m) => tags.append(h('span', { class: 'tag ' + (used[m] ? 'on' : ''), text: m })));
    tags.append(h('span', { class: 'tag', text: r.mode === 'fusion' ? 'fusion prediction' : 'image-model prediction' }));
    const own = r.viewer && r.viewer.is_owner; tags.append(h('span', { class: 'tag ' + (own ? 'ok' : ''), text: own ? 'saved to your account' : 'saved, not linked to an account' }));
    const summary = h('div', { class: 'card' }, h('div', { class: 'summary' }, ring(p.confidence), h('div', null, h('div', { class: 'lbl mute', text: 'Predicted tissue' }), h('div', { class: 'sum-cls', text: p.class }), h('div', { class: 'sum-name', text: tissueName(p.class) })), plain(r)), tags);
    const probs = h('div', { class: 'card' }, h('h3', { text: 'Probability by class' }), GS.bars(p.probabilities, p.class));
    const vc = viewerCard(r);
    root.append(h('div', { class: 'g-main' }, vc, h('div', { class: 'stack' }, summary, probs, inputCheck(r))));

    const lower = h('div', { class: 'g2', style: 'margin-top:16px' });
    const ms = r.molecular_subtype;
    if (ms) {
      const c1 = h('div', { class: 'card' }, h('h3', { text: 'Molecular subtype estimate' }), h('div', { class: 'lbl mute', text: 'Model: ' + (ms.model || '').replace(/_/g, ' ') }), h('div', { class: 'sum-cls', style: 'font-size:28px;margin:4px 0 10px', text: ms.class }), ms.probabilities ? GS.bars(ms.probabilities, ms.class) : null, ms.note ? h('p', { class: 'note', text: ms.note }) : null);
      lower.append(c1);
      if (ms.top_features && ms.top_features.length) {
        const mx = Math.max(...ms.top_features.map((t) => Math.abs(t.shap))) || 1; const sh = h('div');
        ms.top_features.forEach((t) => { const f = h('div', { class: 'f' }); f.dataset.w = (Math.abs(t.shap) / mx * 100).toFixed(1); sh.append(h('div', { class: 'bar shap ' + (t.shap >= 0 ? 'pos' : 'neg') }, h('span', { text: t.feature, title: t.feature, style: 'overflow:hidden;text-overflow:ellipsis;white-space:nowrap' }), h('div', { class: 't' }, f), h('span', { class: 'n', text: (t.shap >= 0 ? '+' : '') + t.shap.toFixed(3) }))); });
        lower.append(h('div', { class: 'card' }, h('h3', { text: 'What pushed that estimate' }), sh, h('p', { class: 'note', text: 'Teal pushes toward the predicted subtype, red pushes away. Importance is not causation.' })));
      }
    } else lower.append(h('div', { class: 'card' }, h('h3', { text: 'Molecular subtype' }), h('p', { class: 'note', style: 'margin:0', text: 'No age, sex, stage or gene file was given, so there is no subtype estimate. Add some context and run again for one.' })));
    const ef = r.experimental_fusion;
    if (ef) lower.append(h('div', { class: 'card' }, h('h3', { text: 'Fusion (experimental)' }), ef.class ? h('p', null, 'Fusion class: ', h('b', { text: ef.class })) : null, ef.probabilities ? GS.bars(ef.probabilities, ef.class) : null, h('p', { class: 'note', text: ef.patient_level_correspondence ? 'Trained on patient-matched data.' : 'The image and gene cohorts are not patient-matched, so this is a control experiment and no evidence of multimodal benefit.' })));
    const i2 = r.inputs || {};
    lower.append(h('div', { class: 'card' }, h('h3', { text: 'What the model was given' }), GS.kv([['Age', i2.age ?? 'not provided'], ['Sex', i2.sex || 'not provided'], ['Stage', i2.stage || 'not provided'], ['Gene expression', i2.genomic_provided ? `provided, ${pct(i2.genomic_coverage, 0)} gene coverage` : 'not provided']])));
    root.append(lower);
    const rv = h('div', { class: 'card rv-card', style: 'margin-top:16px', id: 'rvHost' }); root.append(rv);
    if (r.disclaimer) root.append(h('p', { class: 'note', style: 'margin-top:16px', text: r.disclaimer }));
    setTimeout(() => { $$('.ring', root).forEach((g) => { $('.fg', g).style.strokeDashoffset = 339.3 * (1 - parseFloat(g.dataset.p)); animate($('.val span', g), parseFloat(g.dataset.p)); }); GS.grow(root); }, 60);
    root._rv = rv; return root;
  }
  function animate(el, to) { const t0 = performance.now(); const tick = (t) => { const k = Math.min(1, (t - t0) / 700); el.textContent = (to * (1 - Math.pow(1 - k, 3)) * 100).toFixed(1) + '%'; if (k < 1) requestAnimationFrame(tick); }; requestAnimationFrame(tick); }

  GS.views.analyze = {
    showForm() { $('#anForm').hidden = false; $('#anResult').hidden = true; window.scrollTo(0, 0); },
    showResult(r) { const box = $('#anResult'); box.replaceChildren(); const el = render(r); box.append(el); $('#anForm').hidden = true; box.hidden = false; GS.currentResultEl = el; if (GS.renderReviewCard) GS.renderReviewCard(r, el._rv); },
    show() { /* keep whichever state we are in */ },
  };
})();

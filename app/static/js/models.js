/* Model performance: overview, interactive confusion matrix, reliability diagram, training curves, versions and statistics. */
(() => {
  'use strict';
  const GS = window.GS, { $, $$, h, pct, num } = GS;
  const NS = 'http://www.w3.org/2000/svg';
  const svg = (tag, attrs, ...kids) => { const e = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs || {})) e.setAttribute(k, v); kids.flat().forEach((k) => k != null && e.append(k.nodeType ? k : document.createTextNode(String(k)))); return e; };
  let tab = 'overview';
  const body = $('#mBody');

  const empty = (what) => h('div', { class: 'card empty' }, h('b', { text: 'No ' + what + ' yet' }), 'Train the image model first, then reload this page.', h('pre', { class: 'cmd', text: 'python -m gastricsentinel.train_image --data-dir data/images/HMU-GC-HE-30K \\\n  --allow-patch-level-split --epochs 20' }));

  /* ---------- overview ---------- */
  function overview(c) {
    const im = c.image, root = h('div', { class: 'stack' });
    const ci = (x) => (x ? `95% CI ${pct(x[0], 1)} to ${pct(x[1], 1)}` : '');
    root.append(h('div', { class: 'kpis' },
      kpi(pct(im.accuracy, 1), 'accuracy', ci(im.ci95_accuracy)), kpi(pct(im.macro_f1, 1), 'macro-F1', ci(im.ci95_macro_f1)), kpi(num(im.ece, 3), 'calibration error (ECE)', 'lower is better'),
      im.brier != null ? kpi(num(im.brier, 3), 'Brier score', 'lower is better') : null, im.roc_auc != null ? kpi(num(im.roc_auc, 3), 'macro ROC-AUC', 'one-vs-rest') : null, kpi(String(im.n_test), 'test images', im.arch)));
    if (c.split && c.split.leakage_risk) root.append(h('div', { class: 'alert', text: 'This model used a patch-level split because the images carry no patient IDs. Patches from one patient can sit in both train and test, so these numbers are probably optimistic. External validation is needed before making any claim.' }));
    else if (c.split) root.append(h('div', { class: 'alert info', text: `Patient-disjoint split (${c.split.mode}); no patient appears in more than one of train, validation and test.` }));
    const rows = im.per_class.map((r) => h('tr', null, h('td', null, h('b', { text: r.class }), ' ', h('span', { class: 'mute', text: GS.tissueName(r.class) })), ['precision', 'recall', 'f1'].map((k) => h('td', { class: 'num', text: r[k].toFixed(3) })), h('td', { class: 'num', text: r.support })));
    root.append(h('div', { class: 'card' }, h('h3', { text: 'Per-class results (test split)' }), h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null, ['Class', 'Precision', 'Recall', 'F1', 'Images'].map((t, i) => h('th', { class: i ? 'num' : '', text: t })))), h('tbody', null, rows)))));
    const g = h('div', { class: 'g2' });
    if (c.tabular) g.append(h('div', { class: 'card' }, h('h3', { text: 'Molecular subtype model (TCGA)' }), GS.kv([['Patients (test)', c.tabular.n_patients], ['Accuracy', pct(c.tabular.accuracy, 1)], ['Macro-F1', pct(c.tabular.macro_f1, 1)], ['Classes', c.tabular.classes.join(', ')], ['Excluded (too rare)', c.tabular.excluded_classes && Object.keys(c.tabular.excluded_classes).length ? Object.keys(c.tabular.excluded_classes).join(', ') : 'none']]), h('p', { class: 'note', text: 'Subtype labels are partly defined by gene expression, so genomic inputs make this task partly circular.' })));
    if (c.fusion) g.append(h('div', { class: 'card' }, h('h3', { text: 'Fusion experiment' }), h('p', { class: 'note', style: 'font-size:14px;color:var(--ink);margin:0', text: c.fusion.disclosure })));
    if (g.children.length) root.append(g);
    return root;
  }
  const kpi = (b, s, small) => h('div', { class: 'kpi' }, h('b', { text: b }), h('span', { text: s }), small ? h('small', { text: small }) : null);

  /* ---------- confusion matrix ---------- */
  function confusion(c) {
    const im = c.image, M = im.confusion_matrix, K = im.classes; if (!M) return empty('confusion matrix');
    let mode = 'recall', sel = null;
    const rowTot = M.map((r) => r.reduce((a, b) => a + b, 0)), colTot = K.map((_, j) => M.reduce((a, r) => a + r[j], 0)), N = rowTot.reduce((a, b) => a + b, 0);
    const val = (i, j) => (mode === 'count' ? M[i][j] / (Math.max(...M.flat()) || 1) : mode === 'recall' ? (rowTot[i] ? M[i][j] / rowTot[i] : 0) : (colTot[j] ? M[i][j] / colTot[j] : 0));
    const root = h('div', { class: 'g-main' }); const grid = h('div', { class: 'cm' }); const detail = h('div', { class: 'detail' });
    const seg = h('div', { class: 'seg' }, [['recall', 'Row % (recall)'], ['precision', 'Column % (precision)'], ['count', 'Counts']].map(([k, l]) => h('button', { type: 'button', 'data-m': k, class: k === mode ? 'on' : '', text: l })));
    seg.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; mode = b.dataset.m; [...seg.children].forEach((x) => x.classList.toggle('on', x === b)); paint(); });
    function explain() {
      detail.replaceChildren();
      if (!sel) { detail.append('Hover or tap a cell for details. Rows are the true class, columns are what the model predicted. The diagonal (green) is correct.'); return; }
      const [i, j] = sel, n = M[i][j];
      if (i === j) detail.append(h('b', { text: `${K[i]} predicted correctly: ${n} images` }), h('p', { style: 'margin:6px 0 0', text: `That is ${pct(rowTot[i] ? n / rowTot[i] : 0, 1)} of all true ${K[i]} images (recall) and ${pct(colTot[j] ? n / colTot[j] : 0, 1)} of everything the model called ${K[j]} (precision).` }));
      else detail.append(h('b', { text: `True ${K[i]} predicted as ${K[j]}: ${n} images` }), h('p', { style: 'margin:6px 0 0', text: `${pct(rowTot[i] ? n / rowTot[i] : 0, 1)} of true ${K[i]} images were mislabelled this way, and ${pct(colTot[j] ? n / colTot[j] : 0, 1)} of the model's ${K[j]} calls were really ${K[i]}.` }));
    }
    const heads = { cols: [], rows: [] }, cells = [];
    function build() {  // cells are created once; hovering only toggles classes, so nothing is replaced under the pointer
      grid.style.gridTemplateColumns = `auto repeat(${K.length}, minmax(52px, 1fr))`; grid.replaceChildren(h('div'));
      K.forEach((k, j) => { const d = h('div', { class: 'h', text: k, title: GS.tissueName(k) }); heads.cols.push(d); grid.append(d); });
      K.forEach((k, i) => {
        const rh = h('div', { class: 'h row', text: k, title: GS.tissueName(k) }); heads.rows.push(rh); grid.append(rh); cells[i] = [];
        K.forEach((_, j) => {
          const cell = h('button', { type: 'button', class: 'c' + (i === j ? ' diag' : ''), 'aria-label': `True ${k}, predicted ${K[j]}: ${M[i][j]} images` }, h('span'), h('small'));
          cell.addEventListener('mouseenter', () => { sel = [i, j]; mark(); }); cell.addEventListener('click', () => { sel = [i, j]; mark(); }); cell.addEventListener('focus', () => { sel = [i, j]; mark(); });
          cells[i][j] = cell; grid.append(cell);
        });
      });
    }
    function paint() {
      K.forEach((_, i) => K.forEach((__, j) => { const v = val(i, j), c = cells[i][j]; c.style.setProperty('--v', v); c.firstChild.textContent = mode === 'count' ? M[i][j] : (v * 100).toFixed(0) + '%'; c.lastChild.textContent = mode === 'count' ? '' : M[i][j]; }));
      mark();
    }
    function mark() {
      K.forEach((_, i) => K.forEach((__, j) => { const c = cells[i][j]; c.classList.toggle('sel', !!sel && sel[0] === i && sel[1] === j); c.classList.toggle('dim', !!sel && !(sel[0] === i || sel[1] === j)); }));
      heads.rows.forEach((d, i) => d.classList.toggle('on', !!sel && sel[0] === i)); heads.cols.forEach((d, j) => d.classList.toggle('on', !!sel && sel[1] === j)); explain();
    }
    build(); paint(); grid.addEventListener('mouseleave', () => { sel = null; mark(); });
    const top = []; M.forEach((r, i) => r.forEach((n, j) => { if (i !== j && n) top.push([i, j, n]); })); top.sort((a, b) => b[2] - a[2]);
    const list = h('div', null, top.slice(0, 6).map(([i, j, n]) => h('div', { class: 'bar', style: 'grid-template-columns:1fr auto;cursor:pointer', onclick: () => { sel = [i, j]; mark(); } }, h('span', null, h('b', { text: K[i] }), ' mistaken for ', h('b', { text: K[j] })), h('span', { class: 'n', text: `${n} (${pct(n / rowTot[i], 1)})` }))));
    const csv = () => GS.download('confusion_matrix.csv', GS.csv([['true \\ predicted', ...K], ...M.map((r, i) => [K[i], ...r])]), 'text/csv');
    root.append(h('div', { class: 'card', 'data-tour': 'cm' }, h('div', { class: 'card-h' }, h('h3', { text: 'Confusion matrix (test split)' }), h('button', { class: 'btn ghost sm', type: 'button', text: 'Download CSV', onclick: csv })), h('div', { class: 'vbar' }, seg), h('div', { class: 'axis', text: 'Predicted class →' }), h('div', { class: 'cm-wrap' }, grid), h('div', { style: 'margin-top:12px' }, detail)),
      h('div', { class: 'stack' }, h('div', { class: 'card' }, h('h3', { text: 'Most common mistakes' }), top.length ? list : h('p', { class: 'note', text: 'No mistakes on the test set.' }), h('p', { class: 'note', text: `${N} test images in total.` })),
        h('div', { class: 'card' }, h('h3', { text: 'How to read it' }), h('p', { class: 'note', style: 'font-size:14px;color:var(--ink);margin:0', text: 'Row % answers "of the real X images, how many did it find?" Column % answers "when it says X, how often is it right?" A high row score with a low column score means the class is over-predicted.' }))));
    return root;
  }

  /* ---------- reliability ---------- */
  function calibration(c) {
    const im = c.image, bins = im.reliability; if (!bins) return empty('reliability data');
    const W = 420, Hh = 360, m = { l: 46, r: 14, t: 14, b: 40 }, pw = W - m.l - m.r, ph = Hh - m.t - m.b;
    const x = (v) => m.l + v * pw, y = (v) => m.t + (1 - v) * ph;
    const s = svg('svg', { class: 'chart', viewBox: `0 0 ${W} ${Hh}`, role: 'img', 'aria-label': 'Reliability diagram: accuracy against confidence' });
    for (let t = 0; t <= 1.001; t += 0.2) { s.append(svg('line', { x1: m.l, x2: m.l + pw, y1: y(t), y2: y(t), class: 'grid' }), svg('text', { x: m.l - 8, y: y(t) + 4, 'text-anchor': 'end' }, (t * 100).toFixed(0) + '%'), svg('text', { x: x(t), y: Hh - 20, 'text-anchor': 'middle' }, (t * 100).toFixed(0) + '%')); }
    s.append(svg('line', { x1: x(0), y1: y(0), x2: x(1), y2: y(1), class: 'ref' }), svg('text', { x: m.l + pw / 2, y: Hh - 4, 'text-anchor': 'middle' }, 'Confidence'), svg('text', { x: 12, y: m.t + ph / 2, transform: `rotate(-90 12 ${m.t + ph / 2})`, 'text-anchor': 'middle' }, 'Accuracy'));
    const maxN = Math.max(...bins.map((b) => b.n)) || 1; const tip = h('div', { class: 'detail', style: 'margin-top:10px', text: 'Hover a bar or point. Bars show how many test images fall in each confidence range.' });
    bins.forEach((b) => {
      const bh = (b.n / maxN) * ph * 0.28; const r = svg('rect', { x: x(b.lo) + 2, y: m.t + ph - bh, width: pw / bins.length - 4, height: bh, fill: 'var(--line2)', rx: 3 });
      const show = () => { tip.textContent = b.n ? `Confidence ${pct(b.lo, 0)} to ${pct(b.hi, 0)}: ${b.n} images; they average ${pct(b.confidence, 1)} confidence and ${pct(b.accuracy, 1)} accuracy.${b.accuracy < b.confidence - 0.05 ? ' Overconfident here.' : ''}` : `Confidence ${pct(b.lo, 0)} to ${pct(b.hi, 0)}: no images.`; };
      r.addEventListener('mouseenter', show); s.append(r);
    });
    const pts = bins.filter((b) => b.n); s.append(svg('polyline', { points: pts.map((b) => `${x(b.confidence)},${y(b.accuracy)}`).join(' '), fill: 'none', stroke: 'var(--accent)', 'stroke-width': 2 }));
    pts.forEach((b) => { const d = svg('circle', { cx: x(b.confidence), cy: y(b.accuracy), r: 5, fill: 'var(--accent)', stroke: 'var(--panel)', 'stroke-width': 2 }); d.addEventListener('mouseenter', () => { tip.textContent = `Confidence ${pct(b.lo, 0)} to ${pct(b.hi, 0)}: ${b.n} images; the model said ${pct(b.confidence, 1)} and was right ${pct(b.accuracy, 1)} of the time.`; }); s.append(d); });
    const root = h('div', { class: 'g-main' });
    root.append(h('div', { class: 'card' }, h('h3', { text: 'Reliability diagram (test split)' }), s, h('div', { class: 'chart-legend' }, h('span', null, h('i', { style: 'background:var(--accent)' }), 'model'), h('span', null, h('i', { style: 'background:var(--mute)' }), 'perfectly calibrated'), h('span', null, h('i', { style: 'background:var(--line2);height:8px' }), 'images per bin')), tip),
      h('div', { class: 'stack' }, h('div', { class: 'kpis' }, kpi(num(im.ece, 3), 'ECE after calibration'), im.ece_before != null ? kpi(num(im.ece_before, 3), 'ECE before', im.temperature ? `temperature ${im.temperature.toFixed(2)}` : '') : null, im.brier != null ? kpi(num(im.brier, 3), 'Brier score') : null),
        h('div', { class: 'card' }, h('h3', { text: 'What this tells you' }), h('p', { class: 'note', style: 'font-size:14px;color:var(--ink);margin:0 0 8px', text: 'If the model says 90% about many images and is right about 90% of them, the points sit on the dashed line. Points below it mean overconfidence, so a high confidence figure should not be taken at face value.' }),
          h('p', { class: 'note', text: 'Temperature scaling is fitted on the validation set only and kept only if it improves validation calibration. Confidence is accurate for images like the test set, not for other scanners or stains.' }))));
    return root;
  }

  /* ---------- training curves ---------- */
  function line(rows, keys, title, fmt) {
    const W = 520, Hh = 280, m = { l: 46, r: 14, t: 12, b: 32 }, pw = W - m.l - m.r, ph = Hh - m.t - m.b;
    const all = keys.flatMap((k) => rows.map((r) => r[k.k])), lo = Math.min(...all), hi = Math.max(...all), pad = (hi - lo) * 0.08 || 0.05, a = lo - pad, b = hi + pad;
    const x = (i) => m.l + (rows.length === 1 ? pw / 2 : i / (rows.length - 1) * pw), y = (v) => m.t + (1 - (v - a) / (b - a)) * ph;
    const s = svg('svg', { class: 'chart', viewBox: `0 0 ${W} ${Hh}`, role: 'img', 'aria-label': title });
    for (let t = 0; t <= 4; t++) { const v = a + (b - a) * t / 4; s.append(svg('line', { x1: m.l, x2: m.l + pw, y1: y(v), y2: y(v), class: 'grid' }), svg('text', { x: m.l - 8, y: y(v) + 4, 'text-anchor': 'end' }, fmt(v))); }
    rows.forEach((r, i) => { if (rows.length <= 12 || i % Math.ceil(rows.length / 10) === 0 || i === rows.length - 1) s.append(svg('text', { x: x(i), y: Hh - 12, 'text-anchor': 'middle' }, r.epoch)); });
    keys.forEach((k) => s.append(svg('polyline', { points: rows.map((r, i) => `${x(i)},${y(r[k.k])}`).join(' '), fill: 'none', stroke: k.c, 'stroke-width': 2.2 }), ...rows.map((r, i) => { const d = svg('circle', { cx: x(i), cy: y(r[k.k]), r: 3.5, fill: k.c }); d.append(svg('title', null, `${k.l}, epoch ${r.epoch}: ${fmt(r[k.k])}`)); return d; })));
    return h('div', { class: 'card' }, h('h3', { text: title }), s, h('div', { class: 'chart-legend' }, keys.map((k) => h('span', null, h('i', { style: 'background:' + k.c }), k.l))));
  }
  function training(c) {
    const hist = c.history; if (!hist || !hist.length) return empty('training history');
    const last = hist[hist.length - 1], best = hist.reduce((p, r) => (r.val_f1 > p.val_f1 ? r : p)), gap = last.train_acc - last.val_acc;
    const root = h('div', { class: 'stack' });
    root.append(h('div', { class: 'kpis' }, kpi(String(hist.length), 'epochs run'), kpi(`epoch ${best.epoch}`, 'best validation', `macro-F1 ${pct(best.val_f1, 1)}`), kpi(pct(last.train_acc, 1), 'final train accuracy'), kpi(pct(last.val_acc, 1), 'final validation accuracy'), kpi((gap * 100).toFixed(1) + ' pts', 'train minus validation', gap > 0.1 ? 'overfitting likely' : 'gap is small')));
    if (gap > 0.1) root.append(h('div', { class: 'alert', text: `Training accuracy is ${(gap * 100).toFixed(0)} points above validation. The model is memorising the training set. Try the default strong augmentation, a smaller backbone or --trainable fc, mixup 0.2, and more data.` }));
    root.append(h('div', { class: 'g2' }, line(hist, [{ k: 'train_loss', l: 'train', c: 'var(--accent)' }, { k: 'val_loss', l: 'validation', c: 'var(--rose)' }], 'Loss', (v) => v.toFixed(2)), line(hist, [{ k: 'train_acc', l: 'train', c: 'var(--accent)' }, { k: 'val_acc', l: 'validation', c: 'var(--rose)' }, { k: 'val_f1', l: 'validation macro-F1', c: 'var(--teal)' }], 'Accuracy and macro-F1', (v) => (v * 100).toFixed(0) + '%')));
    return root;
  }

  /* ---------- versions & statistics ---------- */
  function versions(c) {
    const root = h('div', { class: 'stack' }), vs = c.versions || [], cm = c.comparisons || [];
    if (!vs.length) root.append(h('div', { class: 'card empty' }, h('b', { text: 'No registered versions' }), 'Every training run is saved as a version automatically once you retrain with this release.'));
    else root.append(h('div', { class: 'card' }, h('h3', { text: 'Model versions' }), h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null, ['Version', 'Backbone', 'Accuracy', 'Macro-F1', 'ECE', 'Trained', 'Note', ''].map((t, i) => h('th', { class: i > 1 && i < 5 ? 'num' : '', text: t })))),
      h('tbody', null, vs.map((v) => h('tr', null, h('td', null, h('span', { class: 'mono', text: v.id })), h('td', { text: v.arch || '-' }), h('td', { class: 'num', text: pct(v.accuracy, 1) }), h('td', { class: 'num', text: pct(v.macro_f1, 1) }), h('td', { class: 'num', text: num(v.ece, 3) }), h('td', { text: GS.when(v.created) }), h('td', { text: v.note || '' }), h('td', null, v.active ? h('span', { class: 'tag ok', text: 'active' }) : h('span', { class: 'mono mute', text: 'activate ' + v.id }))))))),
      h('p', { class: 'note', text: 'Every analysis is stamped with the version that produced it. To switch versions run the command below and restart the app.' }), h('pre', { class: 'cmd', text: 'python -m gastricsentinel.registry list\npython -m gastricsentinel.registry activate <version>' })));
    if (!cm.length) root.append(h('div', { class: 'card' }, h('h3', { text: 'Statistical comparison' }),  h('p', { class: 'note', style: 'margin:0', text: 'Train a second model (for example --backbone efficientnet_b0 with the same data and seed) and compare it with the first. This tests whether a difference is bigger than chance.' }),
      h('pre', { class: 'cmd', text: 'python -m gastricsentinel.compare_models <versionA> <versionB>' })));
    cm.forEach((r) => {
      const d = r.diff_b_minus_a, lim = Math.max(0.02, Math.abs(d.accuracy_ci95[0]), Math.abs(d.accuracy_ci95[1])) * 1.15, pos = (v) => (50 + (v / lim) * 50) + '%';
      const sig = r.mcnemar.p_value < 0.05; const forest = h('div', { class: 'forest' + (d.accuracy < 0 ? ' neg' : '') }, h('i', { class: 'axis0', style: 'left:50%' }), h('i', { class: 'ci', style: `left:${pos(d.accuracy_ci95[0])};width:calc(${pos(d.accuracy_ci95[1])} - ${pos(d.accuracy_ci95[0])})` }), h('i', { class: 'pt', style: `left:${pos(d.accuracy)}` }));
      const scale = h('div', { class: 'note', style: 'display:flex;justify-content:space-between;margin:2px 0 0' }, h('span', { text: '-' + (lim * 100).toFixed(0) + ' pts (A better)' }), h('span', { text: '0' }), h('span', { text: '+' + (lim * 100).toFixed(0) + ' pts (B better)' }));
      const f1 = d.macro_f1_ci95;
      root.append(h('div', { class: 'card' }, h('div', { class: 'card-h' }, h('h3', null, r.a.arch || r.a.id, ' vs ', r.b.arch || r.b.id), h('span', { class: 'tag ' + (sig ? 'ok' : ''), text: sig ? 'significant (p < 0.05)' : 'no detectable difference' })),
        h('div', { class: 'g3' }, kpi(pct(r.a.accuracy, 1), `${r.a.arch || 'A'} accuracy`, `F1 ${pct(r.a.macro_f1, 1)}`), kpi(pct(r.b.accuracy, 1), `${r.b.arch || 'B'} accuracy`, `F1 ${pct(r.b.macro_f1, 1)}`), kpi('p = ' + r.mcnemar.p_value.toFixed(4), 'McNemar exact test', `${r.mcnemar.only_a_correct} only A right, ${r.mcnemar.only_b_correct} only B right`)),
        h('div', { class: 'lbl', style: 'margin-top:14px', text: `Accuracy difference (B minus A): ${(d.accuracy * 100).toFixed(2)} pts, 95% CI ${(d.accuracy_ci95[0] * 100).toFixed(2)} to ${(d.accuracy_ci95[1] * 100).toFixed(2)}` }), forest, scale,
        h('p', { class: 'note', text: `Macro-F1 difference ${(d.macro_f1 * 100).toFixed(2)} pts, 95% CI ${(f1[0] * 100).toFixed(2)} to ${(f1[1] * 100).toFixed(2)}. The line is the bootstrap interval; if it crosses the centre mark the difference could be zero.` }),
        h('p', { class: 'note', text: r.verdict + ' ' + r.caveat }), (r.warnings || []).map((w) => h('div', { class: 'alert', text: w }))));
    });
    return root;
  }

  async function paint() {
    $$('#mTabs button').forEach((b) => b.classList.toggle('on', b.dataset.t === tab));
    body.replaceChildren(h('div', { class: 'empty' }, h('span', { class: 'spinner' })));
    const c = GS.card && Date.now() - (GS.cardAt || 0) < 8000 ? GS.card : (GS.cardAt = Date.now(), await GS.loadCard());
    body.replaceChildren();
    if (!c.available || !c.image) { body.append(empty('model evidence')); return; }
    body.append({ overview, confusion, calibration, training, versions }[tab](c));
  }
  $('#mTabs').addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; tab = b.dataset.t; paint(); });
  GS.views.models = { show() { GS.cardAt = 0; paint(); }, setTab(t) { tab = t; } };
})();

/* Whole-slide view: tile map over a zoomable image, summary, and the most suspicious tiles. */
(() => {
  'use strict';
  const GS = window.GS, { $, h, pct, tissueName } = GS;
  let file = null;
  const drop = $('#sDrop'), input = $('#sFile'), go = $('#sGo');
  input.addEventListener('change', () => { file = input.files[0] || null; $('#sName').textContent = file ? `${file.name} (${(file.size / 1048576).toFixed(1)} MB)` : 'Drop a large image, or click to choose'; go.disabled = !file; });
  ['dragenter', 'dragover'].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((e) => drop.addEventListener(e, () => drop.classList.remove('over')));
  go.addEventListener('click', async () => {
    const fd = new FormData(); fd.append('image', file); fd.append('tile', $('#sTile').value);
    GS.busy(go, true, 'Analysing slide'); try { GS.openResult(await GS.api('/api/slide', { method: 'POST', body: fd }), true); }
    catch (e) { GS.toast(e.message); if (e.status === 401 && GS.openAuth) GS.openAuth('login'); } finally { GS.busy(go, false); }
  });

  function render(r) {
    const s = r.slide, root = h('div'); const p = r.prediction; const tum = new Set(['TUM']);
    root.append(h('div', { class: 'vh' }, h('div', null, h('h1', { text: 'Slide result' }), h('p', { text: `${GS.when(r.created)} · ${s.width} × ${s.height} px · ${s.n_tissue_tiles} of ${s.n_tiles} tiles contain tissue · tile ${s.tile} px` })),
      h('div', { class: 'acts' }, h('button', { class: 'btn ghost', type: 'button', text: 'New slide', onclick: () => { $('#slForm').hidden = false; $('#slResult').hidden = true; } }), h('a', { class: 'btn primary', href: `/report/${r.id}`, text: 'PDF report' }),
        h('a', { class: 'btn ghost', href: `/api/export?ids=${r.id}&format=csv`, text: 'CSV' }), h('a', { class: 'btn ghost', href: `/api/export?ids=${r.id}&format=json`, text: 'JSON' }))));
    (r.warnings || []).forEach((w) => root.append(h('div', { class: 'alert', style: 'margin-bottom:12px', text: w })));

    /* map: overlay image + invisible tile grid for hover */
    const view = new GS.Viewer({ src: r.input_url, cam: r.gradcam_url, mode: 'cam', opacity: 1, alt: 'Slide' });
    const grid = h('canvas'); view.stage.append(grid); grid.style.pointerEvents = 'none';
    const tip = h('div', { class: 'tip', hidden: true }); view.el.append(tip);
    const byKey = new Map(s.tiles.map((t) => [t[0] + ',' + t[1], t]));
    let showGrid = false;
    const draw = () => { const w = 1200, hh = Math.round(w * s.height / s.width); grid.width = w; grid.height = hh; const c = grid.getContext('2d'); c.clearRect(0, 0, w, hh); if (!showGrid) return; c.strokeStyle = 'rgba(255,255,255,.55)'; c.lineWidth = 1; const tw = w * s.tile / s.width, th = hh * s.tile / s.height; for (let i = 0; i <= s.cols; i++) { c.beginPath(); c.moveTo(i * tw, 0); c.lineTo(i * tw, s.rows * th); c.stroke(); } for (let j = 0; j <= s.rows; j++) { c.beginPath(); c.moveTo(0, j * th); c.lineTo(s.cols * tw, j * th); c.stroke(); } };
    view.el.addEventListener('pointermove', (e) => {
      const n = view.toNorm(e.clientX, e.clientY); if (!n) { tip.hidden = true; return; }
      const col = Math.floor(n.x * s.width / s.tile), row = Math.floor(n.y * s.height / s.tile), t = byKey.get(row + ',' + col);
      if (!t || col >= s.cols || row >= s.rows) { tip.hidden = true; return; }
      const r0 = view.el.getBoundingClientRect(); tip.hidden = false; tip.style.left = (e.clientX - r0.left) + 'px'; tip.style.top = (e.clientY - r0.top) + 'px';
      tip.textContent = `Tile ${row + 1},${col + 1}: ${s.classes[t[2]]} · tumour ${pct(t[3], 0)} · confidence ${pct(t[4], 0)}`;
    });
    view.el.addEventListener('pointerleave', () => { tip.hidden = true; });
    const seg = h('div', { class: 'seg' }, [['cam', 'Tumour map'], ['orig', 'Slide']].map(([k, l]) => h('button', { type: 'button', 'data-m': k, class: k === 'cam' ? 'on' : '', text: l })));
    seg.addEventListener('click', (e) => { const b = e.target.closest('button'); if (!b) return; [...seg.children].forEach((x) => x.classList.toggle('on', x === b)); view.setMode(b.dataset.m); });
    const gl = h('label', null, h('input', { type: 'checkbox', onchange: (e) => { showGrid = e.target.checked; draw(); } }), 'Tile grid');
    const mapCard = h('div', { class: 'card' }, h('div', { class: 'card-h' }, h('h3', { text: 'Tile map' })), h('div', { class: 'vbar' }, seg, gl, h('span', { class: 'grow' }), h('a', { class: 'btn ghost sm', href: r.gradcam_url, download: 'slide_map.png', text: 'Save map' })), view.el,
      h('div', { class: 'legend' }, h('span', { text: 'tissue, not tumour' }), h('i', { style: 'background:linear-gradient(90deg,rgba(210,50,70,.2),rgba(210,50,70,.95))' }), h('span', { text: 'high tumour probability' })),
      h('p', { class: 'note', text: 'Hover a tile for its class and score. Blank tiles were skipped as background.' }));
    draw();

    const tot = Object.values(s.class_counts).reduce((a, b) => a + b, 0) || 1;
    const side = h('div', { class: 'stack' },
      h('div', { class: 'card' }, h('h3', { text: 'Summary' }), h('div', { class: 'sum-cls', text: p.class }), h('div', { class: 'sum-name', text: tissueName(p.class) + ' (highest mean tile probability)' }),
        h('div', { style: 'margin-top:12px' }, GS.kv([['Tiles predicted tumour', s.tumour_fraction == null ? 'n/a' : pct(s.tumour_fraction, 1)], ['Mean tumour score', pct(r.tumour_probability, 0)], ['Tiles unlike the training images', s.ood_tiles], ['Model', r.model_version]]))),
      h('div', { class: 'card' }, h('h3', { text: 'Tiles by class' }), h('div', { class: 'dist' }, Object.entries(s.class_counts).map(([k, v]) => h('i', { style: `width:${v / tot * 100}%;background:${GS.color(k)}` }))),
        ...Object.entries(s.class_counts).sort((a, b) => b[1] - a[1]).map(([k, v]) => h('div', { class: 'bar', style: 'grid-template-columns:1fr auto' }, h('span', null, h('i', { class: 'sw', style: 'background:' + GS.color(k) }), `${k} ${tissueName(k)}`), h('span', { class: 'n', text: `${v} (${pct(v / tot, 0)})` })))));
    root.append(h('div', { class: 'g-main' }, mapCard, side));

    /* most suspicious tiles, cropped client-side from the preview */
    const top = h('div', { class: 'tiles' });
    const probe = new Image(); probe.crossOrigin = 'anonymous'; probe.src = r.input_url;
    probe.onload = () => s.top_tiles.forEach((t) => {
      const cv = h('canvas', { width: 160, height: 160 }); const sc = probe.naturalWidth / s.width, sz = s.tile * sc;
      cv.getContext('2d').drawImage(probe, t.col * sz, t.row * sz, sz, sz, 0, 0, 160, 160);
      top.append(h('button', { class: 'tile-c', type: 'button', onclick: () => { const cx = (t.col + .5) * s.tile / s.width, cy = (t.row + .5) * s.tile / s.height; view.setView({ s: Math.max(3, s.cols / 2), cx, cy }, false); view.el.scrollIntoView({ behavior: 'smooth', block: 'center' }); } }, cv, h('div', null, h('b', { text: t.class }), ` · tumour ${pct(t.tumour_probability, 0)}`, h('br'), `row ${t.row + 1}, col ${t.col + 1}`)));
    });
    root.append(h('div', { class: 'card', style: 'margin-top:16px' }, h('h3', { text: 'Highest-scoring tiles' }), top, h('p', { class: 'note', text: 'Click a tile to zoom the map to it. These are the model’s top picks, so they are the first place to check for false alarms.' })));
    root.append(h('div', { class: 'card rv-card', id: 'rvHost', style: 'margin-top:16px' }));
    root.append(h('p', { class: 'note', style: 'margin-top:16px', text: r.disclaimer }));
    root._rv = root.lastChild.previousSibling; return root;
  }
  GS.views.slide = {
    showResult(r) { const box = $('#slResult'); box.replaceChildren(); const el = render(r); box.append(el); $('#slForm').hidden = true; box.hidden = false; if (GS.renderReviewCard) GS.renderReviewCard(r, el.querySelector('.rv-card')); },
    show() {},
  };
})();

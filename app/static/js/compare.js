/* Compare two saved analyses side by side with linked zoom and pan. */
(() => {
  'use strict';
  const GS = window.GS, { $, $$, h, pct, tissueName } = GS;
  const A = $('#cA'), B = $('#cB'), link = $('#cLink'), out = $('#cOut');
  let busy = 0;

  async function options() {
    let rows = [];
    if (GS.user) { try { rows = (await GS.api('/api/analyses')).analyses.map((x) => ({ id: x.id, cls: x.pred_class, conf: x.confidence, t: x.created, mode: x.mode })); } catch { /* fall back */ } }
    const seen = new Set(rows.map((r) => r.id)); GS.history.all().forEach((x) => { if (!seen.has(x.id)) rows.push(x); });
    return rows;
  }
  const label = (r) => `${GS.when(r.t)} · ${r.cls} ${pct(r.conf, 0)}${r.mode === 'slide' ? ' · slide' : ''} · ${r.id.slice(0, 6)}`;
  async function fill() {
    const rows = await options(); const prevA = A.value, prevB = B.value;
    for (const s of [A, B]) { s.replaceChildren(h('option', { value: '', text: rows.length ? 'Choose an analysis' : 'No analyses yet' })); rows.forEach((r) => s.append(h('option', { value: r.id, text: label(r) }))); }
    const want = [].concat(GS.compareWith || []); GS.compareWith = null;
    if (want.length) {
      want.forEach((id, k) => { const sel = k ? B : A; if (!rows.find((r) => r.id === id)) sel.append(h('option', { value: id, text: 'Current result' })); sel.value = id; });
    } else { if (prevA) A.value = prevA; if (prevB) B.value = prevB; }
    if (!A.value && rows[0]) A.value = rows[0].id; if (!B.value && rows[1]) B.value = rows[1].id;
    render();
  }
  async function render() {
    const my = ++busy; out.replaceChildren();
    if (!A.value || !B.value) { out.append(h('div', { class: 'card empty' }, h('b', { text: 'Pick two analyses' }), 'Run at least two analyses, then choose them above.')); return; }
    if (A.value === B.value) { out.append(h('div', { class: 'alert info', text: 'Both sides are the same analysis. Pick a different one on the right.' })); }
    let a, b; try { [a, b] = await Promise.all([GS.api('/api/result/' + A.value), GS.api('/api/result/' + B.value)]); } catch (e) { out.replaceChildren(h('div', { class: 'alert bad', text: e.message })); return; }
    if (my !== busy) return;
    const views = []; let syncing = false;
    const col = (r) => {
      const p = r.prediction, v = new GS.Viewer({ src: r.input_url, cam: r.gradcam_url, mode: 'orig', alt: 'Analysis ' + r.id.slice(0, 6), onView: (st) => { if (!link.checked || syncing) return; syncing = true; views.forEach((o) => { if (o !== v) o.setView(st); }); syncing = false; } });
      views.push(v);
      return h('div', { class: 'card' }, h('div', { class: 'card-h' }, h('h3', { text: r.mode === 'slide' ? 'Slide' : 'Patch' }), h('span', { class: 'tag', text: r.model_version || '' })),
        h('div', { style: 'display:flex;align-items:baseline;gap:12px;margin-bottom:10px' }, h('span', { class: 'sum-cls', style: 'font-size:28px', text: p.class }), h('span', { class: 'mute', text: `${tissueName(p.class)} · ${pct(p.confidence, 1)}` })), v.el,
        h('p', { class: 'note', text: GS.when(r.created) }), r.ood && r.ood.flagged ? h('span', { class: 'tag warn', text: 'outside distribution' }) : null);
    };
    const seg = h('div', { class: 'seg' }, [['orig', 'Original'], ['cam', 'Grad-CAM']].map(([k, l]) => h('button', { type: 'button', 'data-m': k, class: k === 'orig' ? 'on' : '', text: l })));
    seg.addEventListener('click', (e) => { const x = e.target.closest('button'); if (!x) return; [...seg.children].forEach((y) => y.classList.toggle('on', y === x)); views.forEach((v) => { v.setMode(x.dataset.m); v.setOpacity(0.9); }); });
    out.append(h('div', { class: 'vbar' }, seg, h('span', { class: 'note', style: 'margin:0', text: 'Scroll to zoom and drag to pan on either image.' })), h('div', { class: 'cmp-cols' }, col(a), col(b)));

    const cls = [...new Set([...Object.keys(a.prediction.probabilities), ...Object.keys(b.prediction.probabilities)])];
    const rows = cls.map((c) => { const x = a.prediction.probabilities[c] ?? 0, y = b.prediction.probabilities[c] ?? 0, d = y - x; return h('tr', null, h('td', null, h('b', { text: c }), ' ', h('span', { class: 'mute', text: tissueName(c) })), h('td', { class: 'num', text: pct(x) }), h('td', { class: 'num', text: pct(y) }), h('td', { class: 'num ' + (Math.abs(d) < 0.005 ? '' : d > 0 ? 'diff-pos' : 'diff-neg'), text: (d >= 0 ? '+' : '') + (d * 100).toFixed(1) + ' pts' })); });
    const same = a.prediction.class === b.prediction.class;
    out.append(h('div', { class: 'card', style: 'margin-top:16px' }, h('div', { class: 'card-h' }, h('h3', { text: 'Difference' }), h('span', { class: 'tag ' + (same ? 'ok' : 'warn'), text: same ? 'Same predicted class' : `${a.prediction.class} vs ${b.prediction.class}` })),
      h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null, ['Class', 'Left', 'Right', 'Right minus left'].map((t, i) => h('th', { class: i ? 'num' : '', text: t })))), h('tbody', null, rows))),
      h('div', { style: 'margin-top:12px' }, GS.kv([['Confidence', `${pct(a.prediction.confidence)} vs ${pct(b.prediction.confidence)}`], ['Tumour score', `${a.tumour_probability == null ? '-' : pct(a.tumour_probability, 0)} vs ${b.tumour_probability == null ? '-' : pct(b.tumour_probability, 0)}`], ['Model version', `${a.model_version} vs ${b.model_version}`], ['Reviews', `${(a.reviews || []).length} vs ${(b.reviews || []).length}`]]))));
  }
  A.addEventListener('change', render); B.addEventListener('change', render);
  GS.views.compare = { show: fill };
})();

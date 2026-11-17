/* Batch view: up to 25 images, one request, a summary and a sortable results table. */
(() => {
  'use strict';
  const GS = window.GS, { $, h, pct, tissueName } = GS;
  const MAX = 25; let files = [], items = [];
  const list = $('#bList'), go = $('#bGo'), clear = $('#bClear'), msg = $('#bMsg'), drop = $('#bDrop'), input = $('#bFiles');
  function paint() {
    list.replaceChildren(...files.map((f, i) => {
      const img = h('img', { alt: '', src: URL.createObjectURL(f) }); img.onload = () => URL.revokeObjectURL(img.src);
      return h('div', { class: 'f' }, img, h('small', { text: f.name, title: f.name }), h('button', { type: 'button', 'aria-label': 'Remove ' + f.name, text: '×', onclick: () => { files.splice(i, 1); paint(); } }));
    }));
    go.disabled = !files.length; clear.hidden = !files.length; msg.textContent = files.length ? `${files.length} image${files.length > 1 ? 's' : ''} selected` : '';
  }
  function add(fl) {
    const ok = [...fl].filter((f) => /\.(png|jpe?g|bmp|tiff?)$/i.test(f.name));
    if (ok.length < fl.length) GS.toast('Some files were skipped (only PNG, JPEG, BMP and TIFF).');
    files = files.concat(ok).slice(0, MAX); if (files.length === MAX && ok.length) GS.toast(`The batch is limited to ${MAX} images.`); paint();
  }
  input.addEventListener('change', () => { add(input.files); input.value = ''; });
  ['dragenter', 'dragover'].forEach((e) => drop.addEventListener(e, (ev) => { ev.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach((e) => drop.addEventListener(e, () => drop.classList.remove('over')));
  drop.addEventListener('drop', (ev) => { ev.preventDefault(); add(ev.dataTransfer.files); });
  clear.addEventListener('click', () => { files = []; paint(); });

  go.addEventListener('click', async () => {
    const fd = new FormData(); files.forEach((f) => fd.append('images', f));
    GS.busy(go, true, `Analysing ${files.length}`); msg.textContent = 'This can take a few seconds per image.';
    try { const j = await GS.api('/api/batch', { method: 'POST', body: fd }); items = j.items; show(j); files = []; paint(); }
    catch (e) { GS.toast(e.message); if (e.status === 401 && GS.openAuth) GS.openAuth('login'); }
    finally { GS.busy(go, false); }
  });

  function show(j) {
    const s = j.summary, out = $('#bOut'); out.replaceChildren();
    const ids = items.filter((i) => i.ok).map((i) => i.result.id);
    ids.forEach((id, k) => GS.history.add(items.filter((i) => i.ok)[k].result));
    const acts = $('#batchActs'); acts.hidden = !ids.length; acts.replaceChildren(
      h('a', { class: 'btn primary', href: `/api/export?ids=${ids.join(',')}&format=csv`, text: 'Export CSV' }), h('a', { class: 'btn ghost', href: `/api/export?ids=${ids.join(',')}&format=json`, text: 'Export JSON' }));
    out.append(h('div', { class: 'kpis' },
      kpi(s.ok + ' / ' + s.n, 'analysed', s.failed ? s.failed + ' failed' : 'all succeeded'), kpi(s.mean_confidence == null ? '-' : pct(s.mean_confidence, 0), 'mean confidence'),
      kpi(String(s.low_confidence), 'low confidence', 'below the review threshold'), kpi(String(s.ood_flagged), 'outside distribution', 'flagged by the input check')));
    const tot = Object.values(s.class_counts).reduce((a, b) => a + b, 0) || 1;
    if (tot) out.append(h('div', { class: 'card', style: 'margin-top:16px' }, h('h3', { text: 'Class mix' }),
      h('div', { class: 'dist' }, Object.entries(s.class_counts).map(([k, v]) => h('i', { style: `width:${v / tot * 100}%;background:${GS.color(k)}`, title: `${k}: ${v}` }))),
      h('div', { class: 'tags' }, Object.entries(s.class_counts).map(([k, v]) => h('span', { class: 'tag' }, h('i', { class: 'sw', style: 'background:' + GS.color(k) }), `${k} ${v}`)))));
    const rows = items.map((it) => {
      if (!it.ok) return h('tr', null, h('td'), h('td', { text: it.filename }), h('td', { colspan: 5 }, h('span', { class: 'tag bad', text: it.error })));
      const r = it.result, p = r.prediction;
      const img = h('img', { class: 'thumb', src: r.input_url, alt: '', loading: 'lazy' });
      const flags = h('div', { class: 'tags' }, p.low_confidence ? h('span', { class: 'tag warn', text: 'low confidence' }) : null, r.ood && r.ood.flagged ? h('span', { class: 'tag warn', text: 'outside distribution' }) : null);
      return h('tr', { class: 'hist-item', onclick: () => GS.openResult(r, false) }, h('td', null, img), h('td', { text: it.filename }), h('td', null, h('b', { text: p.class }), ' ', h('span', { class: 'mute', text: tissueName(p.class) })),
        h('td', { class: 'num', text: pct(p.confidence, 0) }), h('td', { class: 'num', text: r.tumour_probability == null ? '-' : pct(r.tumour_probability, 0) }), h('td', null, flags), h('td', null, h('button', { class: 'btn ghost sm', type: 'button', text: 'Open' })));
    });
    out.append(h('div', { class: 'card', style: 'margin-top:16px' }, h('h3', { text: 'Results' }), h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' },
      h('thead', null, h('tr', null, ['', 'File', 'Predicted', 'Confidence', 'Tumour score', 'Flags', ''].map((t, i) => h('th', { class: i === 3 || i === 4 ? 'num' : '', text: t })))), h('tbody', null, rows))),
      h('p', { class: 'note', text: GS.user ? 'Each image is saved to your account.' : 'Each image is saved. Sign in to keep them in your account.' })));
  }
  const kpi = (b, s, small) => h('div', { class: 'kpi' }, h('b', { text: b }), h('span', { text: s }), small ? h('small', { text: small }) : null);
  GS.views.batch = { show() {} };
})();

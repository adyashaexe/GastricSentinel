/* Library & reports: table of analyses with filters, report/export actions, delete, and compare-two. */
(() => {
  'use strict';
  const GS = window.GS, { $, h, pct, tissueName } = GS;
  let rows = [], q = '', cls = '', tab = 'all', sel = new Set();
  const body = $('#libBody'), acts = $('#libActs');

  async function load() {
    if (GS.user) { try { rows = (await GS.api('/api/analyses')).analyses.map((x) => ({ id: x.id, cls: x.pred_class, conf: x.confidence, t: x.created, mode: x.mode, tum: x.tumour_prob, ver: x.model_version, rev: x.n_reviews, rpt: x.has_report, ext: x.input_ext, mine: true })); } catch { rows = []; } }
    else rows = GS.history.all().map((x) => ({ id: x.id, cls: x.cls, conf: x.conf, t: x.t, mode: x.mode, img: x.img, mine: false }));
    paint();
  }
  function paint() {
    const shown = rows.filter((r) => (!q || (r.id + r.cls + (r.ver || '')).toLowerCase().includes(q)) && (!cls || r.cls === cls) && (tab === 'all' || (tab === 'reports' && r.rpt)));
    const classes = [...new Set(rows.map((r) => r.cls))].sort();
    acts.replaceChildren(
      h('button', { class: 'btn ghost', type: 'button', disabled: sel.size !== 2, text: 'Compare selected', onclick: () => { GS.compareWith = [...sel]; GS.go('compare'); } }),
      h('a', { class: 'btn ghost' + (rows.length ? '' : ' disabled'), href: rows.length ? '/api/export' + (GS.user ? '' : '?ids=' + rows.map((r) => r.id).join(',')) : '#', text: 'Export all CSV' }),
      h('a', { class: 'btn ghost', href: rows.length ? '/api/export?format=json' + (GS.user ? '' : '&ids=' + rows.map((r) => r.id).join(',')) : '#', text: 'JSON' }));
    body.replaceChildren();
    if (!GS.user) body.append(h('div', { class: 'alert info', style: 'margin-bottom:14px' }, 'Showing analyses run in this browser. ', h('a', { href: '#', text: 'Sign in', onclick: (e) => { e.preventDefault(); GS.openAuth('login'); } }), ' to keep a library tied to your account and see it on any device.'));
    const tools = h('div', { class: 'pick', style: 'margin-bottom:12px' },
      h('div', { class: 'seg' }, [['all', 'All analyses'], ['reports', 'Saved reports']].map(([k, l]) => h('button', { type: 'button', class: k === tab ? 'on' : '', text: l, onclick: () => { tab = k; paint(); } }))),
      h('div', { class: 'field' }, h('input', { class: 'plain', type: 'search', placeholder: 'Search by class, id or model', value: q, 'aria-label': 'Search', oninput: (e) => { q = e.target.value.toLowerCase(); paint(); const i = $('#libBody input[type=search]'); i.focus(); i.setSelectionRange(q.length, q.length); } })),
      h('div', { class: 'field', style: 'max-width:200px' }, h('select', { 'aria-label': 'Filter by class', onchange: (e) => { cls = e.target.value; paint(); } }, h('option', { value: '', text: 'All classes' }), classes.map((c) => h('option', { value: c, text: c, selected: c === cls })))));
    body.append(tools);
    if (!shown.length) { body.append(h('div', { class: 'empty' }, h('b', { text: tab === 'reports' ? 'No saved reports yet' : 'Nothing here yet' }), tab === 'reports' ? 'Open an analysis and choose PDF report; it is then kept here.' : 'Run an analysis and it will appear in this list.')); return; }
    const trs = shown.map((r) => {
      const img = h('img', { class: 'thumb', alt: '', loading: 'lazy', src: r.mode === 'slide' || !r.ext ? (r.img || `/outputs/${r.id}_input${r.ext || '.png'}`) : `/outputs/${r.id}_input${r.ext}` }); img.onerror = () => { img.style.visibility = 'hidden'; };
      const cb = h('input', { type: 'checkbox', 'aria-label': 'Select for comparison', checked: sel.has(r.id), onchange: (e) => { if (e.target.checked) { if (sel.size >= 2) { e.target.checked = false; GS.toast('Select two analyses to compare.'); return; } sel.add(r.id); } else sel.delete(r.id); paint(); } });
      return h('tr', null, h('td', null, cb), h('td', null, img), h('td', null, h('b', { text: r.cls }), ' ', h('span', { class: 'mute', text: tissueName(r.cls) }), r.mode === 'slide' ? h('span', { class: 'tag', style: 'margin-left:6px', text: 'slide' }) : null),
        h('td', { class: 'num', text: pct(r.conf, 0) }), h('td', { text: GS.when(r.t) }), h('td', null, h('span', { class: 'mono', text: r.ver || '' })),
        h('td', null, r.mine ? [r.rev ? h('span', { class: 'tag ok', text: `${r.rev} review${r.rev > 1 ? 's' : ''}` }) : null, r.rpt ? h('span', { class: 'tag on', text: 'report saved' }) : null] : null),
        h('td', null, h('div', { class: 'tags' }, h('button', { class: 'btn ghost sm', type: 'button', text: 'Open', onclick: () => GS.loadAndOpen(r.id) }), h('a', { class: 'btn ghost sm', href: `/report/${r.id}`, text: 'PDF' }), h('a', { class: 'btn ghost sm', href: `/api/export?ids=${r.id}&format=csv`, text: 'CSV' }),
          r.mine ? h('button', { class: 'btn danger sm', type: 'button', 'aria-label': 'Delete', text: 'Delete', onclick: async () => { if (!confirm('Delete this analysis, its images, chat and reviews? This cannot be undone.')) return; try { await GS.api('/api/analyses/' + r.id, { method: 'DELETE' }); GS.history.remove(r.id); sel.delete(r.id); GS.toast('Deleted.'); load(); } catch (e) { GS.toast(e.message); } } }) : h('button', { class: 'btn ghost sm', type: 'button', text: 'Forget', onclick: () => { GS.history.remove(r.id); load(); } }))));
    });
    body.append(h('div', { class: 'tbl-wrap' }, h('table', { class: 'tbl' }, h('thead', null, h('tr', null, ['', '', 'Predicted', 'Confidence', 'Date', 'Model', 'Status', ''].map((t, i) => h('th', { class: i === 3 ? 'num' : '', text: t })))), h('tbody', null, trs))));
  }
  GS.views.library = { show: load };
  GS.refreshLibrary = load;
})();

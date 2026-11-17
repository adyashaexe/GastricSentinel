/* Core: helpers, icons, API, router, theme, status. All dynamic text goes through textContent. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const GS = window.GS = { user: null, classes: [], card: null, id: null, views: {}, current: null, version: null };
  GS.$ = $; GS.$$ = $$;
  GS.TISSUE = { ADI: 'Adipose tissue', BACK: 'Background', DEB: 'Debris', LYM: 'Lymphocytes', MUC: 'Mucus', MUS: 'Smooth muscle', NOR: 'Normal mucosa', NORM: 'Normal mucosa', STR: 'Stroma', TUM: 'Tumour epithelium' };
  GS.tissueName = (c) => GS.TISSUE[c] || c;
  GS.PALETTE = ['#1d5fbd', '#0e7c86', '#b4233c', '#9a5f00', '#6b4fbb', '#3f8f2f', '#c2548a', '#5d6b80'];
  GS.color = (cls) => { const i = Math.max(0, GS.classes.indexOf(cls)); return GS.PALETTE[i % GS.PALETTE.length]; };

  GS.h = function h(tag, attrs, ...kids) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v == null || v === false) continue;
      if (k === 'class') e.className = v; else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? '' : v);
    }
    for (const kid of kids.flat()) if (kid != null && kid !== false) e.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
    return e;
  };
  const h = GS.h;
  GS.pct = (x, d = 1) => (x == null ? '-' : (x * 100).toFixed(d) + '%');
  GS.num = (x, d = 3) => (x == null ? '-' : Number(x).toFixed(d));
  GS.when = (t) => { try { return new Date(t).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' }); } catch { return t; } };
  GS.store = {
    get(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode */ } },
  };
  GS.toast = (msg) => { const t = $('#toast'); t.textContent = msg; t.classList.add('show'); clearTimeout(GS.toast.t); GS.toast.t = setTimeout(() => t.classList.remove('show'), 4200); };
  GS.download = (name, text, type = 'text/plain') => { const a = h('a', { href: URL.createObjectURL(new Blob([text], { type })), download: name }); document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); };
  GS.csv = (rows) => rows.map((r) => r.map((v) => { v = v == null ? '' : String(v); if (/^[=+\-@\t\r]/.test(v)) v = "'" + v; return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; }).join(',')).join('\n');

  GS.api = async (path, opts = {}) => {
    const o = { credentials: 'same-origin', headers: {}, ...opts };
    if (o.json !== undefined) { o.method = o.method || 'POST'; o.headers['Content-Type'] = 'application/json'; o.body = JSON.stringify(o.json); delete o.json; }
    const r = await fetch(path, o); const j = await r.json().catch(() => ({}));
    if (!r.ok) { const e = new Error(j.error || `Request failed (${r.status})`); e.status = r.status; throw e; }
    return j;
  };
  GS.busy = (btn, on, label) => { btn.disabled = on; btn.classList.toggle('busy', on); if (label != null) { const s = btn.querySelector('span') || btn; if (on) { btn.dataset.l = s.textContent; s.textContent = label; } else if (btn.dataset.l) s.textContent = btn.dataset.l; } };

  /* ---------- icons ---------- */
  const P = (d) => `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${d}</svg>`;
  GS.ICONS = {
    scan: P('<path d="M4 8V5a1 1 0 0 1 1-1h3M16 4h3a1 1 0 0 1 1 1v3M20 16v3a1 1 0 0 1-1 1h-3M8 20H5a1 1 0 0 1-1-1v-3"/><circle cx="12" cy="12" r="3.2"/>'),
    layers: P('<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5M3 17.5l9 5 9-5" opacity=".6"/>'),
    grid: P('<rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/>'),
    columns: P('<rect x="3" y="4" width="8" height="16" rx="1.5"/><rect x="13" y="4" width="8" height="16" rx="1.5"/>'),
    folder: P('<path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V18a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V7z"/>'),
    check: P('<path d="M5 12.5l4.5 4.5L19 7.5"/><rect x="3" y="3" width="18" height="18" rx="3" opacity=".5"/>'),
    chart: P('<path d="M4 20V4M4 20h16"/><path d="M8 16v-5M12 16V8M16 16v-8"/>'),
    info: P('<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8v.01"/>'),
    upload: P('<path d="M12 16V4M7 9l5-5 5 5M4 20h16"/>'),
    download: P('<path d="M12 4v12M7 11l5 5 5-5M4 20h16"/>'),
    file: P('<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>'),
    user: P('<circle cx="12" cy="8" r="4"/><path d="M4 21c1-4 4.5-6 8-6s7 2 8 6"/>'),
    trash: P('<path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13"/>'),
  };
  GS.icon = (n) => GS.ICONS[n] || '';
  const paintIcons = (root = document) => $$('[data-ic]', root).forEach((e) => { e.innerHTML = GS.icon(e.dataset.ic); e.style.display = 'contents'; });
  GS.paintIcons = paintIcons; paintIcons();

  /* ---------- theme ---------- */
  $('#themeBtn').addEventListener('click', () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next; try { localStorage.setItem('gs-theme', next); } catch { /* ignore */ }
    document.dispatchEvent(new CustomEvent('gs:theme'));
  });

  /* ---------- router ---------- */
  const nav = $('#side'), scrim = $('#navScrim');
  const openNav = (on) => { nav.classList.toggle('open', on); scrim.classList.toggle('on', on); };
  GS.openNav = openNav;
  $('#menuBtn').addEventListener('click', () => openNav(true)); $('#sideX').addEventListener('click', () => openNav(false)); scrim.addEventListener('click', () => openNav(false));
  GS.go = (v) => { if (location.hash !== '#/' + v) location.hash = '#/' + v; else route(); };
  function route() {
    let v = (location.hash.match(/^#\/([a-z]+)/) || [])[1] || 'analyze';
    if (!$('#v-' + v)) v = 'analyze';
    if (v === 'review' && !(GS.user && ['reviewer', 'admin'].includes(GS.user.role))) v = GS.userReady ? 'analyze' : 'review';
    $$('.view').forEach((s) => { s.hidden = s.id !== 'v-' + v; });
    $$('.nav a').forEach((a) => a.classList.toggle('on', a.dataset.view === v));
    const sec = $('#v-' + v); document.title = (sec.dataset.title || 'GastricSentinel') + ' · GastricSentinel'; $('#crumb').textContent = sec.dataset.title || '';
    GS.current = v; openNav(false); window.scrollTo(0, 0);
    const mod = GS.views[v]; if (mod && mod.show) mod.show();
    document.dispatchEvent(new CustomEvent('gs:view', { detail: v }));
  }
  GS.route = route; addEventListener('hashchange', route);

  /* ---------- status + model card ---------- */
  GS.loadStatus = async () => {
    try {
      const r = await fetch('/api/status'); const j = await r.json();
      GS.ready = !!j.ready; GS.classes = j.classes || []; GS.version = j.model_version; GS.statusInfo = j;
      $('#statusDot').className = 'dot' + (j.ready ? ' ok' : ' bad'); $('#statusTxt').textContent = j.ready ? 'Models ready' : 'Models not loaded';
      $('#statusVer').textContent = j.model_version && j.ready ? j.model_version : '';
    } catch { $('#statusTxt').textContent = 'Server unreachable'; $('#statusDot').className = 'dot bad'; }
  };
  GS.loadCard = async () => { try { GS.card = await GS.api('/api/model-card'); } catch { GS.card = { available: false }; } return GS.card; };
})();

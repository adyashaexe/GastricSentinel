/* First-run guided tour: spotlight + card. Shown once automatically; "Take the tour" restarts it. */
(() => {
  'use strict';
  const GS = window.GS, { $, h } = GS;
  const STEPS = [
    { title: 'Welcome to GastricSentinel', text: 'A research tool that classifies stomach histology patches, shows where the model looked, and tells you how far to trust it. This 8-step tour takes about a minute.' },
    { sel: '#nav', side: true, title: 'Your workspace', text: 'Analyze one image, run a batch of up to 25, map a whole slide, or compare two results side by side. Records and evidence are below.' },
    { view: 'analyze', sel: '[data-tour=drop]', title: '1. Add an image', text: 'Drop an H&E patch here (PNG, JPEG, BMP or TIFF). Nothing else is required.' },
    { view: 'analyze', sel: '[data-tour=context]', title: '2. Optional context', text: 'Age, sex, stage and a gene-expression file unlock a molecular-subtype estimate. Skip them and the result is simply image-only.' },
    { view: 'analyze', sel: '[data-tour=run]', title: '3. Run it', text: 'You get a class, a calibrated confidence, a Grad-CAM heat map you can zoom and pan, and an input check that warns when the image looks unlike the training data.' },
    { sel: 'a[data-view=models]', side: true, title: 'How good is it, really?', text: 'Model performance has an interactive confusion matrix, a reliability diagram that shows whether confidence can be trusted, training curves, and statistical comparisons between model versions.' },
    { sel: '#chatFab', title: 'Ask the assistant', text: 'Sentinel explains your result in plain language. It can’t diagnose or advise on treatment.' },
    { sel: '.side-foot', side: true, title: 'Account and theme', text: 'Sign in to keep a library, saved reports and chat history tied to you. Switch between light and dark here. Restart this tour any time.' },
  ];
  let i = 0, mask, hole, card, active = false;
  const done = () => GS.store.set('gs-tour-done', true);
  function end(mark = true) { if (!active) return; active = false; [mask, hole, card].forEach((e) => e && e.remove()); GS.openNav(false); if (mark) done(); removeEventListener('keydown', onKey); removeEventListener('resize', place); }
  function onKey(e) { if (e.key === 'Escape') end(); else if (e.key === 'ArrowRight') go(1); else if (e.key === 'ArrowLeft') go(-1); }
  function go(d) { const n = i + d; if (n < 0) return; if (n >= STEPS.length) { end(); return; } i = n; show(); }
  function show() {
    const s = STEPS[i];
    if (s.view && GS.current !== s.view) { GS.go(s.view); if (GS.views.analyze && s.view === 'analyze') GS.views.analyze.showForm(); }
    if (s.side && matchMedia('(max-width: 860px)').matches) GS.openNav(true); else GS.openNav(false);
    setTimeout(() => { card.replaceChildren(h('h4', { text: s.title }), h('p', { text: s.text }), h('div', { class: 'tour-foot' },
      h('div', { class: 'dots' }, STEPS.map((_, k) => h('i', { class: k === i ? 'on' : '' }))), h('button', { class: 'btn ghost sm', type: 'button', text: 'Skip', onclick: () => end() }),
      i ? h('button', { class: 'btn ghost sm', type: 'button', text: 'Back', onclick: () => go(-1) }) : null, h('button', { class: 'btn primary sm', type: 'button', text: i === STEPS.length - 1 ? 'Done' : 'Next', onclick: () => go(1) }))); place(); const nb = card.querySelector('.btn.primary'); if (nb) nb.focus({ preventScroll: true }); }, s.side ? 280 : 80);
  }
  function place() {
    if (!active) return; const s = STEPS[i]; const el = s.sel && $(s.sel); const vw = innerWidth, vh = innerHeight;
    if (!el || !el.offsetParent && getComputedStyle(el).position !== 'fixed') { hole.style.cssText = 'left:50%;top:50%;width:0;height:0;border-color:transparent'; card.style.left = Math.max(12, (vw - card.offsetWidth) / 2) + 'px'; card.style.top = Math.max(12, (vh - card.offsetHeight) / 2) + 'px'; return; }
    if (!s.side && el.getBoundingClientRect().top > vh - 120) el.scrollIntoView({ block: 'center' });
    const r = el.getBoundingClientRect(), pad = 6;
    hole.style.cssText = `left:${r.left - pad}px;top:${r.top - pad}px;width:${r.width + pad * 2}px;height:${r.height + pad * 2}px`;
    const cw = card.offsetWidth, ch = card.offsetHeight; let x, y;
    if (r.right + cw + 24 < vw) { x = r.right + 16; y = r.top; } else if (r.left - cw - 24 > 0) { x = r.left - cw - 16; y = r.top; } else if (r.bottom + ch + 24 < vh) { x = r.left; y = r.bottom + 14; } else { x = r.left; y = Math.max(12, r.top - ch - 14); }
    card.style.left = Math.min(Math.max(12, x), vw - cw - 12) + 'px'; card.style.top = Math.min(Math.max(12, y), vh - ch - 12) + 'px';
  }
  GS.tour = {
    start() { end(false); active = true; i = 0; mask = h('div', { class: 'tour-mask', onclick: () => {} }); hole = h('div', { class: 'tour-hole' }); card = h('div', { class: 'tour-card', role: 'dialog', 'aria-label': 'Guided tour' }); document.body.append(mask, hole, card); addEventListener('keydown', onKey); addEventListener('resize', place); show(); },
    maybeStart() { if (!GS.store.get('gs-tour-done', false)) setTimeout(() => { if (!active && GS.current === 'analyze' && !GS.state) GS.tour.start(); }, (document.getElementById('splash') ? 4200 : 1200)); },
  };
  $('#tourBtn').addEventListener('click', () => { GS.openNav(false); GS.tour.start(); });
})();

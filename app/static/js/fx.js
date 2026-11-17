/* Motion layer: splash dismissal, shared gradients, card spotlight, count-up, button ripple. Purely cosmetic. */
(() => {
  'use strict';
  const doc = document, body = doc.body, NS = 'http://www.w3.org/2000/svg';

  /* shared gradient for the confidence ring (referenced by CSS as url(#gsGrad)) */
  const svg = doc.createElementNS(NS, 'svg');
  svg.setAttribute('width', '0'); svg.setAttribute('height', '0'); svg.setAttribute('aria-hidden', 'true'); svg.style.cssText = 'position:absolute;width:0;height:0';
  svg.innerHTML = '<defs><linearGradient id="gsGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4ea1ff"/><stop offset=".6" stop-color="#8b6cff"/><stop offset="1" stop-color="#ff7a6b"/></linearGradient></defs>';
  body.appendChild(svg);

  /* splash */
  const splash = doc.getElementById('splash');
  const finish = () => {
    if (!splash || splash.dataset.done) return; splash.dataset.done = '1';
    try { sessionStorage.setItem('gs-splash', '1'); } catch { /* ignore */ }
    splash.classList.add('out'); body.classList.remove('splashing');
    setTimeout(() => splash.remove(), 900);
  };
  if (splash && body.classList.contains('splashing')) {
    const t = setTimeout(finish, 3300);
    splash.addEventListener('click', () => { clearTimeout(t); finish(); });
    doc.addEventListener('keydown', (e) => { if (e.key === 'Escape' || e.key === 'Enter' || e.key === ' ') { clearTimeout(t); finish(); } });
  } else if (splash) splash.remove();

  /* cursor spotlight on cards */
  doc.addEventListener('pointermove', (e) => {
    const c = e.target.closest && e.target.closest('.card, .kpi'); if (!c) return;
    const r = c.getBoundingClientRect(); c.style.setProperty('--mx', (e.clientX - r.left) + 'px'); c.style.setProperty('--my', (e.clientY - r.top) + 'px');
  }, { passive: true });

  /* ripple on buttons */
  doc.addEventListener('pointerdown', (e) => {
    const b = e.target.closest && e.target.closest('.btn'); if (!b || b.disabled) return;
    const r = b.getBoundingClientRect(), s = Math.max(r.width, r.height) * 2, i = doc.createElement('i');
    i.style.cssText = `position:absolute;border-radius:50%;pointer-events:none;width:${s}px;height:${s}px;left:${e.clientX - r.left - s / 2}px;top:${e.clientY - r.top - s / 2}px;background:rgba(255,255,255,.28);transform:scale(0);opacity:1;transition:transform .55s ease-out,opacity .55s ease-out`;
    b.appendChild(i); requestAnimationFrame(() => { i.style.transform = 'scale(1)'; i.style.opacity = '0'; }); setTimeout(() => i.remove(), 600);
  });

  /* count-up for headline numbers (.kpi b) when they appear */
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const seen = new WeakSet();
  const count = (el) => {
    const m = /^(-?\d+(?:\.\d+)?)(.*)$/.exec(el.textContent.trim()); if (!m || seen.has(el)) return; seen.add(el);
    const end = parseFloat(m[1]), dec = (m[1].split('.')[1] || '').length, tail = m[2], t0 = performance.now(), dur = 900;
    const step = (t) => { const k = Math.min(1, (t - t0) / dur), v = end * (1 - Math.pow(1 - k, 3)); el.textContent = v.toFixed(dec) + tail; if (k < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  };
  if (!reduce) new MutationObserver((ms) => ms.forEach((m) => m.addedNodes.forEach((n) => { if (n.nodeType === 1) (n.matches('.kpi b') ? [n] : [...n.querySelectorAll('.kpi b')]).forEach(count); }))).observe(doc.querySelector('.views') || body, { childList: true, subtree: true });
})();

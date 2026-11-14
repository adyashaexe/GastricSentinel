/* Presentation effects: theme toggle (with circular reveal), cursor-lit cards, tilt, hero specimen, marquee, scroll progress. */
(() => {
  'use strict';
  const root = document.documentElement, reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const btn = document.getElementById('themeBtn');
  function apply(t) { root.dataset.theme = t; try { localStorage.setItem('gs-theme', t); } catch { /* ignore */ } document.dispatchEvent(new CustomEvent('gs:theme', { detail: t })); }
  btn.addEventListener('click', () => {
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    if (!document.startViewTransition || reduce) { apply(next); return; }
    const r = btn.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2;
    const end = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    const vt = document.startViewTransition(() => apply(next));
    vt.ready.then(() => root.animate({ clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${end}px at ${x}px ${y}px)`] },
      { duration: 650, easing: 'cubic-bezier(.22,.8,.24,1)', pseudoElement: '::view-transition-new(root)' })).catch(() => {});
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', (e) => { try { if (!localStorage.getItem('gs-theme')) apply(e.matches ? 'dark' : 'light'); } catch { /* ignore */ } });

  // cursor-lit cards + tilt
  document.addEventListener('pointermove', (e) => {
    const c = e.target.closest && e.target.closest('.card'); if (!c) return;
    const r = c.getBoundingClientRect(); c.style.setProperty('--mx', (e.clientX - r.left) + 'px'); c.style.setProperty('--my', (e.clientY - r.top) + 'px');
    if (!reduce && c.classList.contains('feat')) {
      c.classList.add('tilt'); const px = (e.clientX - r.left) / r.width - .5, py = (e.clientY - r.top) / r.height - .5;
      c.style.transform = `perspective(800px) rotateX(${(-py * 6).toFixed(2)}deg) rotateY(${(px * 8).toFixed(2)}deg) translateY(-4px)`;
    }
  }, { passive: true });
  document.addEventListener('pointerout', (e) => { const c = e.target.closest && e.target.closest('.feat'); if (c && !c.contains(e.relatedTarget)) { c.style.transform = ''; c.classList.remove('tilt'); } });

  // hero specimen: drifting cells, parallax
  const cells = document.getElementById('specCells');
  if (cells) {
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < 26; i++) {
      const d = document.createElement('div'); d.className = 'cell'; const s = 9 + rnd() * 15;
      d.style.cssText = `left:${rnd() * 100}%;top:${rnd() * 100}%;width:${s}%;height:${s}%;--d:${14 + rnd() * 18}s;--dl:${-rnd() * 20}s;--dx:${(rnd() - .5) * 40}px;--dy:${(rnd() - .5) * 40}px;--dr:${(rnd() - .5) * 90}deg`;
      cells.append(d);
    }
    const sp = document.getElementById('specimen');
    if (!reduce) addEventListener('pointermove', (e) => { const r = sp.getBoundingClientRect(); const x = (e.clientX - r.left - r.width / 2) / r.width, y = (e.clientY - r.top - r.height / 2) / r.height; cells.style.transform = `translate(${(-x * 18).toFixed(1)}px, ${(-y * 18).toFixed(1)}px)`; }, { passive: true });
  }

  // marquee of tissue classes
  const tr = document.getElementById('marqTrack');
  if (tr) {
    const T = { ADI: 'Adipose tissue', DEB: 'Debris', LYM: 'Lymphocytes', MUC: 'Mucus', MUS: 'Smooth muscle', NOR: 'Normal mucosa', STR: 'Stroma', TUM: 'Tumour epithelium', BACK: 'Background' };
    const once = Object.entries(T).map(([k, v]) => { const s = document.createElement('span'); const b = document.createElement('b'); b.textContent = k; s.append(b, document.createTextNode(v)); return s; });
    [...once, ...once.map((n) => n.cloneNode(true))].forEach((n) => tr.append(n));
  }

  // scroll progress
  const pg = document.getElementById('progress');
  const upd = () => { const h = document.documentElement; const m = h.scrollHeight - innerHeight; pg.style.transform = `scaleX(${m > 0 ? Math.min(1, scrollY / m) : 0})`; };
  addEventListener('scroll', upd, { passive: true }); upd();
  // icons inside the new chips (app.js filled earlier ones before they existed in DOM order)
  document.dispatchEvent(new CustomEvent('gs:fx'));
})();

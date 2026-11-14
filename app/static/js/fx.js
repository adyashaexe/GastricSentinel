/* Presentation effects: theme toggle (circular reveal), scroll reveal, nav, cursor-lit cards, tilt, magnetic buttons, hero specimen + parallax, marquee, progress. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const root = document.documentElement, reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const fine = matchMedia('(hover: hover) and (pointer: fine)').matches;
  const GS = window.GS = window.GS || {};

  /* theme */
  const btn = $('#themeBtn');
  function apply(t) { root.dataset.theme = t; try { localStorage.setItem('gs-theme', t); } catch { /* ignore */ } const m = document.querySelector('meta[name=theme-color]'); if (m) m.content = t === 'dark' ? '#080f1c' : '#0b2a4a'; document.dispatchEvent(new CustomEvent('gs:theme', { detail: t })); }
  btn.addEventListener('click', () => {
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    if (!document.startViewTransition || reduce) { apply(next); return; }
    const r = btn.getBoundingClientRect(), x = r.left + r.width / 2, y = r.top + r.height / 2, end = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    const vt = document.startViewTransition(() => apply(next));
    vt.ready.then(() => root.animate({ clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${end}px at ${x}px ${y}px)`] }, { duration: 650, easing: 'cubic-bezier(.22,.8,.24,1)', pseudoElement: '::view-transition-new(root)' })).catch(() => {});
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener?.('change', (e) => { try { if (!localStorage.getItem('gs-theme')) apply(e.matches ? 'dark' : 'light'); } catch { /* ignore */ } });

  /* scroll reveal (also used by dynamically added content) */
  const io = 'IntersectionObserver' in window ? new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: 0.12, rootMargin: '0px 0px -6% 0px' }) : null;
  GS.reveal = (scope) => { const list = $$('[data-reveal]:not(.in)', scope || document); if (scope && scope.matches && scope.matches('[data-reveal]')) list.push(scope); list.forEach((e) => (io ? io.observe(e) : e.classList.add('in'))); };
  GS.reveal();

  /* header: shadow, progress, mobile menu, active section */
  const bar = $('#topbar'), pg = $('#progress'), burger = $('#burger'), nav = $('#nav');
  const upd = () => { bar.classList.toggle('scrolled', scrollY > 8); const m = root.scrollHeight - innerHeight; pg.style.transform = `scaleX(${m > 0 ? Math.min(1, scrollY / m) : 0})`; };
  addEventListener('scroll', upd, { passive: true }); upd();
  const closeMenu = () => { nav.classList.remove('open'); burger.setAttribute('aria-expanded', 'false'); };
  burger.addEventListener('click', () => { const o = nav.classList.toggle('open'); burger.setAttribute('aria-expanded', String(o)); });
  nav.addEventListener('click', (e) => { if (e.target.closest('a')) closeMenu(); });
  addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenu(); });
  document.addEventListener('click', (e) => { if (!e.target.closest('.topbar')) closeMenu(); });
  if ('IntersectionObserver' in window) {
    const links = $$('.nav a[data-sec]'), secs = links.map((a) => $('#' + a.dataset.sec)).filter(Boolean);
    const so = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a) => a.classList.toggle('on', a.dataset.sec === e.target.id)); }), { rootMargin: '-40% 0px -55% 0px' });
    secs.forEach((s) => so.observe(s));
  }

  /* cursor-lit cards + tilt + magnetic buttons (mouse only) */
  if (fine) {
    document.addEventListener('pointermove', (e) => {
      const c = e.target.closest && e.target.closest('.card'); if (!c) return;
      const r = c.getBoundingClientRect(); c.style.setProperty('--mx', (e.clientX - r.left) + 'px'); c.style.setProperty('--my', (e.clientY - r.top) + 'px');
      if (!reduce && c.classList.contains('feat')) { c.classList.add('tilt'); const px = (e.clientX - r.left) / r.width - .5, py = (e.clientY - r.top) / r.height - .5; c.style.transform = `perspective(800px) rotateX(${(-py * 6).toFixed(2)}deg) rotateY(${(px * 8).toFixed(2)}deg) translateY(-4px)`; }
    }, { passive: true });
    document.addEventListener('pointerout', (e) => { const c = e.target.closest && e.target.closest('.feat'); if (c && !c.contains(e.relatedTarget)) { c.style.transform = ''; c.classList.remove('tilt'); } });
    if (!reduce) $$('.mag').forEach((b) => {
      b.addEventListener('pointermove', (e) => { const r = b.getBoundingClientRect(); b.style.transform = `translate(${((e.clientX - r.left) / r.width - .5) * 10}px, ${((e.clientY - r.top) / r.height - .5) * 8}px)`; });
      b.addEventListener('pointerleave', () => { b.style.transform = ''; });
    });
  }
  /* ripple on every button */
  document.addEventListener('pointerdown', (e) => {
    const b = e.target.closest && e.target.closest('.btn'); if (!b || reduce || b.disabled) return; const r = b.getBoundingClientRect(), d = Math.max(r.width, r.height);
    const s = document.createElement('span'); s.className = 'rip'; s.style.cssText = `width:${d}px;height:${d}px;left:${e.clientX - r.left - d / 2}px;top:${e.clientY - r.top - d / 2}px`; b.append(s); setTimeout(() => s.remove(), 650);
  });

  /* hero specimen: cells, pointer + scroll parallax */
  const cells = $('#specCells'), sp = $('#specimen');
  if (cells) {
    let seed = 7; const rnd = () => (seed = (seed * 16807) % 2147483647) / 2147483647;
    for (let i = 0; i < 26; i++) { const d = document.createElement('div'); d.className = 'cell'; const s = 9 + rnd() * 15;
      d.style.cssText = `left:${rnd() * 100}%;top:${rnd() * 100}%;width:${s}%;height:${s}%;--d:${14 + rnd() * 18}s;--dl:${-rnd() * 20}s;--dx:${(rnd() - .5) * 40}px;--dy:${(rnd() - .5) * 40}px;--dr:${(rnd() - .5) * 90}deg`; cells.append(d); }
    if (!reduce) {
      if (fine) addEventListener('pointermove', (e) => { const r = sp.getBoundingClientRect(); cells.style.transform = `translate(${(-(e.clientX - r.left - r.width / 2) / r.width * 18).toFixed(1)}px, ${(-(e.clientY - r.top - r.height / 2) / r.height * 18).toFixed(1)}px)`; }, { passive: true });
      addEventListener('scroll', () => { if (scrollY < innerHeight) sp.style.transform = `translateY(${(scrollY * -0.08).toFixed(1)}px) rotate(${(scrollY * 0.02).toFixed(2)}deg)`; }, { passive: true });
    }
  }

  /* marquee */
  const tr = $('#marqTrack');
  if (tr) {
    const T = { ADI: 'Adipose tissue', DEB: 'Debris', LYM: 'Lymphocytes', MUC: 'Mucus', MUS: 'Smooth muscle', NOR: 'Normal mucosa', STR: 'Stroma', TUM: 'Tumour epithelium' };
    const once = Object.entries(T).map(([k, v]) => { const s = document.createElement('span'), b = document.createElement('b'); b.textContent = k; s.append(b, document.createTextNode(v)); return s; });
    [...once, ...once.map((n) => n.cloneNode(true))].forEach((n) => tr.append(n));
  }
})();

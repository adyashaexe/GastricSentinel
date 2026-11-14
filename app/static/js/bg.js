/* Background: a slowly drifting histology field - hematoxylin nuclei with eosin cytoplasm halos,
   drawn at very low contrast so it never competes with content. Static under reduced-motion. */
(() => {
  const cv = document.getElementById('bg');
  if (!cv) return;
  const ctx = cv.getContext('2d');
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let W, H, cells = [], raf = 0, last = 0, CE = '214,122,152', CN = '48,52,128', KA = 1;
  function colours() { const cs = getComputedStyle(document.documentElement); CE = cs.getPropertyValue('--bg-eos').trim() || CE; CN = cs.getPropertyValue('--bg-cell').trim() || CN; KA = parseFloat(cs.getPropertyValue('--bg-a')) || 1; }
  colours(); document.addEventListener('gs:theme', () => { colours(); if (reduce) draw(0); });

  function resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    W = innerWidth; H = innerHeight;
    cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + 'px'; cv.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.min(34, Math.max(14, Math.floor(W * H / 42000)));
    cells = Array.from({ length: n }, () => {
      const r = 34 + Math.random() * 70;
      return { x: Math.random() * W, y: Math.random() * H, r, vx: (Math.random() - .5) * 5, vy: (Math.random() - .5) * 5,
        rot: Math.random() * 6.28, vr: (Math.random() - .5) * .03, nr: r * (.28 + Math.random() * .14), sq: .65 + Math.random() * .3,
        eo: .05 + Math.random() * .06, he: .07 + Math.random() * .08 };
    });
  }

  function draw(dt) {
    ctx.clearRect(0, 0, W, H);
    for (const c of cells) {
      if (!reduce) {
        c.x += c.vx * dt; c.y += c.vy * dt; c.rot += c.vr * dt;
        if (c.x < -c.r) c.x = W + c.r; if (c.x > W + c.r) c.x = -c.r;
        if (c.y < -c.r) c.y = H + c.r; if (c.y > H + c.r) c.y = -c.r;
      }
      const g = ctx.createRadialGradient(c.x, c.y, c.nr * .6, c.x, c.y, c.r);
      g.addColorStop(0, `rgba(${CE},${c.eo * KA})`); g.addColorStop(1, `rgba(${CE},0)`);
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(c.x, c.y, c.r, 0, 6.2832); ctx.fill();
      ctx.save(); ctx.translate(c.x, c.y); ctx.rotate(c.rot); ctx.scale(1, c.sq);
      ctx.fillStyle = `rgba(${CN},${c.he * KA})`; ctx.beginPath(); ctx.arc(0, 0, c.nr, 0, 6.2832); ctx.fill();
      ctx.fillStyle = `rgba(${CN},${c.he * .7 * KA})`; ctx.beginPath(); ctx.arc(c.nr * .25, -c.nr * .2, c.nr * .22, 0, 6.2832); ctx.fill();
      ctx.restore();
    }
  }

  function loop(t) { const dt = Math.min(.05, (t - last) / 1000 || 0); last = t; draw(dt); raf = requestAnimationFrame(loop); }
  addEventListener('resize', () => { resize(); if (reduce) draw(0); });
  document.addEventListener('visibilitychange', () => { cancelAnimationFrame(raf); if (!document.hidden && !reduce) raf = requestAnimationFrame(loop); });
  resize();
  if (reduce) draw(0); else raf = requestAnimationFrame(loop);
})();

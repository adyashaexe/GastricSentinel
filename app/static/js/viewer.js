/* Zoom / pan image viewer: wheel, drag, pinch, double-click, keyboard, optional Grad-CAM crossfade or split. */
(() => {
  'use strict';
  const GS = window.GS, h = GS.h;
  const MIN = 1, MAX = 16;

  GS.Viewer = function Viewer(opts) {
    const o = Object.assign({ src: '', cam: null, mode: 'orig', opacity: 0.85, split: 0.5, alt: 'Image', onView: null }, opts);
    const v = { s: 1, tx: 0, ty: 0, fw: 0, fh: 0, bx: 0, by: 0, mode: o.mode, opacity: o.opacity, split: o.split, listeners: [] };
    const base = h('img', { src: o.src, alt: o.alt, draggable: 'false' });
    const cam = o.cam ? h('img', { src: o.cam, alt: o.alt + ' with Grad-CAM overlay', draggable: 'false' }) : null;
    const stage = h('div', { class: 'stage' }, base, cam);
    const line = h('div', { class: 'split-line', hidden: true });
    const zoomLbl = h('div', { class: 'zoom', text: '×1.0' });
    const hint = h('div', { class: 'hint', text: 'Scroll to zoom · drag to pan · double-click to zoom in' });
    const btn = (t, label, fn) => h('button', { type: 'button', 'aria-label': label, title: label, text: t, onclick: fn });
    const tools = h('div', { class: 'vtools' }, btn('+', 'Zoom in', () => v.zoomBy(1.5)), btn('−', 'Zoom out', () => v.zoomBy(1 / 1.5)), btn('⤢', 'Reset view', () => v.reset()));
    const el = v.el = h('div', { class: 'viewer', tabindex: '0', role: 'img', 'aria-label': o.alt + '. Use plus and minus to zoom, arrow keys to pan, zero to reset.' }, stage, line, tools, zoomLbl, hint);
    v.stage = stage; v.base = base; v.cam = cam;

    const W = () => el.clientWidth, H = () => el.clientHeight;
    function layout(keep) {
      const iw = base.naturalWidth, ih = base.naturalHeight; if (!iw || !W()) return;
      const fit = Math.min(W() / iw, H() / ih); v.fw = iw * fit; v.fh = ih * fit; v.bx = (W() - v.fw) / 2; v.by = (H() - v.fh) / 2;
      stage.style.width = v.fw + 'px'; stage.style.height = v.fh + 'px'; if (!keep) { v.s = 1; v.tx = 0; v.ty = 0; } apply();
    }
    function clamp() {
      v.s = Math.min(MAX, Math.max(MIN, v.s));
      const left = v.bx + v.tx, right = left + v.fw * v.s, top = v.by + v.ty, bottom = top + v.fh * v.s;
      if (left > W() * 0.8) v.tx -= left - W() * 0.8; if (right < W() * 0.2) v.tx += W() * 0.2 - right;
      if (top > H() * 0.8) v.ty -= top - H() * 0.8; if (bottom < H() * 0.2) v.ty += H() * 0.2 - bottom;
      if (v.s === MIN) { v.tx = 0; v.ty = 0; }
    }
    function apply(silent) {
      clamp(); stage.style.transform = `translate(${v.bx + v.tx}px, ${v.by + v.ty}px) scale(${v.s})`;
      zoomLbl.textContent = '×' + v.s.toFixed(1); paintMode();
      if (!silent && o.onView) o.onView(v.view());
      v.listeners.forEach((f) => f(v));
    }
    function paintMode() {
      if (!cam) return;
      cam.style.opacity = v.mode === 'orig' ? 0 : v.mode === 'cam' ? v.opacity : 1;
      line.hidden = v.mode !== 'split';
      if (v.mode === 'split') { cam.style.clipPath = `inset(0 0 0 ${v.split * 100}%)`; line.style.left = (v.bx + v.tx + v.fw * v.s * v.split) + 'px'; } else cam.style.clipPath = '';
    }
    v.view = () => ({ s: v.s, cx: ((W() / 2 - v.bx - v.tx) / v.s) / (v.fw || 1), cy: ((H() / 2 - v.by - v.ty) / v.s) / (v.fh || 1) });
    v.setView = ({ s, cx, cy }, silent = true) => { v.s = s; v.tx = W() / 2 - v.bx - cx * v.fw * s; v.ty = H() / 2 - v.by - cy * v.fh * s; apply(silent); };
    v.setMode = (m) => { v.mode = m; paintMode(); };
    v.setOpacity = (x) => { v.opacity = x; paintMode(); };
    v.setSplit = (x) => { v.split = x; paintMode(); };
    v.reset = () => { v.s = 1; v.tx = 0; v.ty = 0; apply(); };
    v.zoomAt = (f, cx, cy) => { const ns = Math.min(MAX, Math.max(MIN, v.s * f)); const k = ns / v.s; v.tx = cx - v.bx - (cx - v.bx - v.tx) * k; v.ty = cy - v.by - (cy - v.by - v.ty) * k; v.s = ns; apply(); };
    v.zoomBy = (f) => v.zoomAt(f, W() / 2, H() / 2);
    v.toNorm = (clientX, clientY) => {
      const r = el.getBoundingClientRect(); const x = ((clientX - r.left - v.bx - v.tx) / v.s) / v.fw, y = ((clientY - r.top - v.by - v.ty) / v.s) / v.fh;
      return x >= 0 && x <= 1 && y >= 0 && y <= 1 ? { x, y } : null;
    };
    v.onChange = (f) => v.listeners.push(f);

    /* pointers: one finger pans, two fingers pinch */
    const ptrs = new Map(); let last = null; let moved = false;
    const hideHint = () => { hint.style.opacity = 0; };
    el.addEventListener('pointerdown', (e) => { if (e.target.closest('.vtools')) return; el.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY }); el.classList.add('drag'); moved = false; last = null; hideHint(); });
    el.addEventListener('pointermove', (e) => {
      if (!ptrs.has(e.pointerId)) return; const p = ptrs.get(e.pointerId); const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY;
      if (ptrs.size === 1) { v.tx += dx; v.ty += dy; if (Math.abs(dx) + Math.abs(dy) > 1) moved = true; apply(); }
      else if (ptrs.size === 2) {
        const [a, b] = [...ptrs.values()]; const d = Math.hypot(a.x - b.x, a.y - b.y); const r = el.getBoundingClientRect(); const mx = (a.x + b.x) / 2 - r.left, my = (a.y + b.y) / 2 - r.top;
        if (last) { v.tx += mx - last.mx; v.ty += my - last.my; v.zoomAt(d / last.d, mx, my); } last = { d, mx, my }; moved = true;
      }
    });
    const up = (e) => { ptrs.delete(e.pointerId); last = null; if (!ptrs.size) el.classList.remove('drag'); };
    el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up);
    el.addEventListener('wheel', (e) => { e.preventDefault(); const r = el.getBoundingClientRect(); v.zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0016)), e.clientX - r.left, e.clientY - r.top); hideHint(); }, { passive: false });
    el.addEventListener('dblclick', (e) => { if (e.target.closest('.vtools')) return; const r = el.getBoundingClientRect(); if (v.s > 1.05) v.reset(); else v.zoomAt(2.5, e.clientX - r.left, e.clientY - r.top); });
    el.addEventListener('keydown', (e) => {
      const k = e.key, step = 40; let used = true;
      if (k === '+' || k === '=') v.zoomBy(1.3); else if (k === '-' || k === '_') v.zoomBy(1 / 1.3); else if (k === '0') v.reset();
      else if (k === 'ArrowLeft') { v.tx += step; apply(); } else if (k === 'ArrowRight') { v.tx -= step; apply(); } else if (k === 'ArrowUp') { v.ty += step; apply(); } else if (k === 'ArrowDown') { v.ty -= step; apply(); } else used = false;
      if (used) e.preventDefault();
    });
    base.addEventListener('load', () => layout(false)); if (base.complete) layout(false);
    if (cam) cam.addEventListener('load', paintMode);
    new ResizeObserver(() => layout(true)).observe(el);
    setTimeout(hideHint, 6000); v.moved = () => moved;
    return v;
  };
})();

/* Floating assistant (bottom-right). textContent only; history kept in memory. */
(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const fab = $('#chatFab'), panel = $('#chatPanel'), msgs = $('#msgs'), form = $('#chatForm'), input = $('#chatIn');
  const sugg = $('#sugg'), note = $('#chatNote'), badge = $('#chatBadge'), scrim = $('#chatScrim');
  if (!fab || !panel) return;
  const history = []; let unread = 0, greeted = false;
  const SUGGEST = ['What does this result mean?', 'How sure is the model?', 'What is Grad-CAM showing?', 'What are the limits?'];

  function add(text, who) { const m = document.createElement('div'); m.className = 'm ' + who; m.textContent = text; msgs.append(m); msgs.scrollTop = msgs.scrollHeight; return m; }
  function typing() { const m = document.createElement('div'); m.className = 'm bot typing'; for (let i = 0; i < 3; i++) m.append(document.createElement('i')); msgs.append(m); msgs.scrollTop = msgs.scrollHeight; return m; }
  function setUnread(n) { unread = n; badge.hidden = !n; badge.textContent = n > 9 ? '9+' : String(n); }
  function open(prefill) {
    panel.classList.add('open'); fab.setAttribute('aria-expanded', 'true'); scrim.classList.add('on'); fab.classList.add('hide-m'); setUnread(0);
    if (!greeted) { greeted = true; add('Hi, I’m Sentinel. I can explain your result, the confidence, the Grad-CAM map and the report in plain language. I can’t diagnose anything or advise on treatment.', 'bot'); }
    if (prefill) send(prefill); else setTimeout(() => { if (matchMedia('(hover: hover)').matches) input.focus(); }, 350);
  }
  function close() { panel.classList.remove('open'); fab.setAttribute('aria-expanded', 'false'); scrim.classList.remove('on'); fab.classList.remove('hide-m'); fab.focus({ preventScroll: true }); }
  async function send(text) {
    text = String(text || '').trim(); if (!text) return;
    add(text, 'me'); input.value = ''; const t = typing();
    try {
      const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, result_id: (window.GS && window.GS.id) || null, history: history.slice(-8) }) });
      const j = await r.json(); t.remove();
      if (!r.ok) { add(j.error || 'Something went wrong. Try again in a moment.', 'bot'); return; }
      add(j.reply, 'bot'); history.push({ role: 'user', content: text }, { role: 'assistant', content: j.reply });
      if (note) note.textContent = j.mode === 'offline' ? 'Built-in assistant. Set an API key on the server for a language-model assistant.' : 'Language-model assistant (' + j.mode + ').';
    } catch { t.remove(); add('I couldn’t reach the server. Check your connection and try again.', 'bot'); }
  }
  fab.addEventListener('click', () => (panel.classList.contains('open') ? close() : open()));
  $('#chatClose').addEventListener('click', close); scrim.addEventListener('click', close);
  addEventListener('keydown', (e) => { if (e.key === 'Escape' && panel.classList.contains('open')) close(); });
  form.addEventListener('submit', (e) => { e.preventDefault(); send(input.value); });
  SUGGEST.forEach((s) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = s; b.addEventListener('click', () => send(s)); sugg.append(b); });
  document.addEventListener('gs:result', (e) => {
    if (!e.detail || !e.detail.fresh) return;
    if (!greeted) { greeted = true; add('Hi, I’m Sentinel. I can explain your result in plain language. I can’t diagnose or advise on treatment.', 'bot'); }
    add('I’ve got your new result loaded. Ask me anything about it.', 'bot');
    if (!panel.classList.contains('open')) setUnread(unread + 1);
  });
  // keep the sheet above the on-screen keyboard on phones
  if (window.visualViewport) visualViewport.addEventListener('resize', () => { panel.style.maxHeight = panel.classList.contains('open') ? visualViewport.height * 0.95 + 'px' : ''; msgs.scrollTop = msgs.scrollHeight; });
  (window.GS = window.GS || {}).chat = { open, close };
})();

/* Floating assistant (bottom-right). textContent only; history kept in memory. */
(() => {
  'use strict';
  const $ = (s) => document.querySelector(s);
  const fab = $('#chatFab'), panel = $('#chatPanel'), msgs = $('#msgs'), form = $('#chatForm'), input = $('#chatIn');
  const sugg = $('#sugg'), note = $('#chatNote'), badge = $('#chatBadge'), scrim = $('#chatScrim');
  const closeButton = $('#chatClose');
  if (!fab || !panel || !msgs || !form || !input || !sugg || !scrim) return;
  const history = []; let unread = 0, greeted = false;
  const SUGGEST = ['Explain my result', 'What does confidence mean?', 'What is Grad-CAM?', 'What are the limits?'];

  function add(text, who) { const m = document.createElement('div'); m.className = 'm ' + who; m.textContent = text; msgs.append(m); msgs.scrollTop = msgs.scrollHeight; return m; }
  function typing() { const m = document.createElement('div'); m.className = 'm bot typing'; for (let i = 0; i < 3; i++) m.append(document.createElement('i')); msgs.append(m); msgs.scrollTop = msgs.scrollHeight; return m; }
  function setUnread(n) { unread = n; if (!badge) return; badge.hidden = !n; badge.textContent = n > 9 ? '9+' : String(n); }
  function open(prefill) {
    panel.classList.add('open'); fab.setAttribute('aria-expanded', 'true'); scrim.classList.add('on'); fab.classList.add('hide-m'); setUnread(0);
    if (!greeted) { greeted = true; add('Hi, I’m Sentinel. I can explain your result, the confidence, the Grad-CAM map and the report in plain language. I can’t diagnose anything or advise on treatment.', 'bot'); }
    if (prefill) send(prefill); else setTimeout(() => { if (matchMedia('(hover: hover)').matches) input.focus(); }, 350);
  }
  function close() { panel.classList.remove('open'); fab.setAttribute('aria-expanded', 'false'); scrim.classList.remove('on'); fab.classList.remove('hide-m'); fab.focus({ preventScroll: true }); }
  async function send(text) {
    text = String(text || '').trim(); if (!text) return;
    if (form.dataset.sending === 'true') return;
    form.dataset.sending = 'true';
    add(text, 'me'); input.value = ''; const t = typing();
    try {
      const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text.slice(0, 1000), result_id: (window.GS && window.GS.id) || null, history: history.slice(-8) }) });
      const j = await r.json().catch(() => ({})); t.remove();
      if (!r.ok) { add(j.detail || j.error || 'Something went wrong. Try again in a moment.', 'bot'); return; }
      const reply = typeof j.reply === 'string' && j.reply.trim() ? j.reply : 'I could not prepare an answer. Please try asking in a different way.';
      add(reply, 'bot'); history.push({ role: 'user', content: text }, { role: 'assistant', content: reply });
      if (note) note.textContent = j.mode === 'offline' ? 'Built-in assistant. Set an API key on the server for a language-model assistant.' : 'Language-model assistant (' + j.mode + ').';
    } catch { t.remove(); add('I couldn’t reach the server. Check your connection and try again.', 'bot'); }
    finally { form.dataset.sending = 'false'; }
  }
  fab.addEventListener('click', () => (panel.classList.contains('open') ? close() : open()));
  if (closeButton) closeButton.addEventListener('click', close); scrim.addEventListener('click', close);
  addEventListener('keydown', (e) => { if (e.key === 'Escape' && panel.classList.contains('open')) close(); });
  form.addEventListener('submit', (e) => { e.preventDefault(); send(input.value); });
  SUGGEST.forEach((s) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = s; b.addEventListener('click', () => send(s)); sugg.append(b); });
  const INTRO = 'Hi, I\u2019m Sentinel. I can explain your result, the confidence, the Grad-CAM map and the report in plain language. I can\u2019t diagnose anything or advise on treatment.';
  document.addEventListener('gs:result', async (e) => {
    const d = e.detail || {}, r = d.result; if (!r) return;
    history.length = 0; msgs.replaceChildren(); greeted = true; add(INTRO, 'bot');
    const owner = r.viewer && r.viewer.is_owner;
    if (owner && !d.fresh && window.GS.api) {
      try {
        const j = await window.GS.api('/api/analyses/' + r.id + '/chat');
        j.messages.forEach((m) => { add(m.content, m.role === 'user' ? 'me' : 'bot'); history.push({ role: m.role, content: m.content }); });
        if (j.messages.length) add('Picking up where we left off. Ask me anything else about this result.', 'bot');
      } catch { /* conversation unavailable; start fresh */ }
    }
    if (d.fresh) { add('I\u2019ve got your new result loaded. Ask me anything about it.', 'bot'); if (!panel.classList.contains('open')) setUnread(unread + 1); }
  });
  // keep the sheet above the on-screen keyboard on phones
  if (window.visualViewport) visualViewport.addEventListener('resize', () => { panel.style.maxHeight = panel.classList.contains('open') ? visualViewport.height * 0.95 + 'px' : ''; msgs.scrollTop = msgs.scrollHeight; });
  (window.GS = window.GS || {}).chat = { open, close };
})();

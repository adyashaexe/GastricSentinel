/* Assistant tab. Uses textContent only; history is kept in memory. */
(() => {
  'use strict';
  const msgs = document.getElementById('msgs'), form = document.getElementById('chatForm'), input = document.getElementById('chatIn');
  const sugg = document.getElementById('sugg'), note = document.getElementById('chatNote');
  if (!msgs || !form) return;
  const history = [];
  const SUGGEST = ['What does this result mean?', 'How confident is the model?', 'What is Grad-CAM showing?', 'What are the limitations?'];

  function add(text, who) {
    const m = document.createElement('div'); m.className = 'm ' + who; m.textContent = text; msgs.append(m);
    msgs.scrollTop = msgs.scrollHeight; return m;
  }
  function typing() {
    const m = document.createElement('div'); m.className = 'm bot typing';
    for (let i = 0; i < 3; i++) m.append(document.createElement('i'));
    msgs.append(m); msgs.scrollTop = msgs.scrollHeight; return m;
  }
  async function send(text) {
    text = text.trim(); if (!text) return;
    add(text, 'me'); input.value = ''; const t = typing();
    try {
      const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, result_id: (window.GS && window.GS.id) || null, history: history.slice(-8) }) });
      const j = await r.json();
      t.remove();
      if (!r.ok) { add(j.error || 'Something went wrong.', 'bot'); return; }
      add(j.reply, 'bot');
      history.push({ role: 'user', content: text }, { role: 'assistant', content: j.reply });
      if (note) note.textContent = j.mode === 'offline' ? 'Built-in assistant. Set an API key on the server for a language-model assistant.' : 'Language-model assistant (' + j.mode + ').';
    } catch { t.remove(); add('Could not reach the server.', 'bot'); }
  }
  form.addEventListener('submit', (e) => { e.preventDefault(); send(input.value); });
  SUGGEST.forEach((s) => { const b = document.createElement('button'); b.type = 'button'; b.textContent = s; b.addEventListener('click', () => send(s)); sugg.append(b); });
  add('Hello. I can explain what the result, the confidence, Grad-CAM, SHAP and the report mean. I cannot diagnose or recommend treatment.', 'bot');
  document.addEventListener('gs:result', () => add('A new result is loaded. Ask me about it.', 'bot'));
})();

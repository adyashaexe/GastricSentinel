/* Accounts: sign-in modal, account menu, reviewer queue and the expert-review card. textContent only. */
(() => {
  'use strict';
  const GS = window.GS, { $, $$, h, pct } = GS;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isReviewer = () => GS.user && ['reviewer', 'admin'].includes(GS.user.role);

  /* ---------- account button + menu ---------- */
  const btn = $('#acctBtn'), menu = $('#acctMenu');
  function renderAccount() {
    const u = GS.user, av = $('#acctAv');
    if (u) { av.textContent = (u.name || u.email)[0].toUpperCase(); $('#acctLabel').textContent = u.name; $('#acctSub').textContent = u.role === 'user' ? u.email : u.role; }
    else { av.innerHTML = GS.icon('user'); $('#acctLabel').textContent = 'Sign in'; $('#acctSub').textContent = 'Keep your work in one place'; }
    btn.classList.toggle('in', !!u); $('#navReview').hidden = !isReviewer();
  }
  const closeMenu = () => { menu.hidden = true; btn.setAttribute('aria-expanded', 'false'); };
  function openMenu() {
    const u = GS.user; menu.replaceChildren(h('div', { class: 'am-head' }, h('b', { text: u.name }), h('small', { text: u.email }), u.role !== 'user' ? h('span', { class: 'tag on', style: 'justify-self:start;margin-top:4px', text: u.role }) : null));
    const item = (t, fn) => h('button', { type: 'button', role: 'menuitem', class: 'am-item', text: t, onclick: () => { closeMenu(); fn(); } });
    menu.append(item('Library & reports', () => GS.go('library')));
    if (isReviewer()) menu.append(item('Review queue', () => GS.go('review')));
    menu.append(item('Sign out', async () => { try { await GS.api('/api/auth/logout', { method: 'POST' }); } catch { /* ignore */ } location.reload(); }));
    menu.hidden = false; btn.setAttribute('aria-expanded', 'true');
  }
  btn.addEventListener('click', (e) => { e.stopPropagation(); if (!GS.user) { openAuth('login'); return; } menu.hidden ? openMenu() : closeMenu(); });
  document.addEventListener('click', (e) => { if (!e.target.closest('.acct-wrap')) closeMenu(); });
  addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenu(); });
  function setUser(u) {
    GS.user = u; GS.userReady = true; renderAccount();
    if (GS.current === 'review' && !isReviewer()) GS.go('analyze'); else if (GS.views[GS.current] && GS.views[GS.current].show && ['library', 'compare', 'review'].includes(GS.current)) GS.views[GS.current].show();
  }

  /* ---------- modal ---------- */
  const modal = $('#authModal'); let mode = 'login', lastFocus = null;
  function setMode(m) {
    mode = m; $$('#authSeg button').forEach((b) => b.classList.toggle('on', b.dataset.mode === m)); const reg = m === 'register';
    $('#fName').hidden = !reg; $('#authTitle').textContent = reg ? 'Create your account' : 'Welcome back';
    $('#authSub').textContent = reg ? 'Keep your analyses, chat history and reviews. Takes ten seconds.' : 'Sign in to keep your analyses, chat history and reviews.';
    $('span', $('#authGo')).textContent = reg ? 'Create account' : 'Sign in'; $('#aPw').autocomplete = reg ? 'new-password' : 'current-password'; $('#authErr').textContent = '';
  }
  function openAuth(m) { lastFocus = document.activeElement; setMode(m || 'login'); modal.hidden = false; requestAnimationFrame(() => modal.classList.add('open')); setTimeout(() => $(mode === 'register' ? '#aName' : '#aEmail').focus(), 80); }
  function closeAuth() { modal.classList.remove('open'); setTimeout(() => { modal.hidden = true; if (lastFocus) lastFocus.focus({ preventScroll: true }); }, reduce ? 0 : 200); }
  GS.openAuth = openAuth;
  $$('#authSeg button').forEach((b) => b.addEventListener('click', () => setMode(b.dataset.mode)));
  modal.addEventListener('click', (e) => { if (e.target.closest('[data-close]')) closeAuth(); });
  addEventListener('keydown', (e) => { if (e.key === 'Escape' && !modal.hidden) closeAuth(); });
  $('#authForm').addEventListener('submit', async (e) => {
    e.preventDefault(); const err = $('#authErr'); err.textContent = '';
    const email = $('#aEmail').value.trim(), pw = $('#aPw').value, name = $('#aName').value.trim();
    if (!email || !pw) { err.textContent = 'Enter your email and password.'; return; }
    if (mode === 'register' && !name) { err.textContent = 'Enter your name.'; return; }
    const go = $('#authGo'); go.disabled = true;
    try { const j = await GS.api('/api/auth/' + mode, { json: mode === 'register' ? { email, password: pw, name } : { email, password: pw } }); $('#aPw').value = ''; closeAuth(); setUser(j.user); GS.toast(mode === 'register' ? `Welcome, ${j.user.name.split(' ')[0]}.` : 'Signed in.'); }
    catch (ex) { err.textContent = ex.message; } finally { go.disabled = false; }
  });

  /* ---------- review queue ---------- */
  let qStatus = 'pending';
  $$('#qSeg button').forEach((b) => b.addEventListener('click', () => { qStatus = b.dataset.st; $$('#qSeg button').forEach((x) => x.classList.toggle('on', x === b)); loadQueue(); }));
  async function loadQueue() {
    const box = $('#queueBody'); if (!isReviewer()) { box.replaceChildren(h('p', { class: 'empty', text: 'Reviewer access is required.' })); return; }
    let items = []; try { items = (await GS.api('/api/review/queue?status=' + qStatus)).items; } catch (ex) { box.replaceChildren(h('p', { class: 'note', text: ex.message })); return; }
    if (!items.length) { box.replaceChildren(h('div', { class: 'empty' }, h('b', { text: qStatus === 'pending' ? 'Nothing waiting' : 'Nothing here yet' }), qStatus === 'pending' ? 'Every saved analysis has a sign-off.' : '')); return; }
    box.replaceChildren(h('table', { class: 'tbl' }, h('thead', null, h('tr', null, ['Submitted', 'By', 'Predicted', 'Confidence', 'Reviews', ''].map((t, i) => h('th', { class: i === 3 ? 'num' : '', text: t })))),
      h('tbody', null, items.map((x) => h('tr', null, h('td', { text: GS.when(x.created) }), h('td', { text: x.owner }), h('td', null, h('b', { text: x.pred_class }), ' ', h('span', { class: 'mute', text: GS.tissueName(x.pred_class) })), h('td', { class: 'num', text: pct(x.confidence, 0) }), h('td', { text: String(x.n_reviews) }), h('td', null, h('button', { class: 'btn ghost sm', type: 'button', text: 'Open', onclick: () => GS.loadAndOpen(x.id) })))))));
  }
  GS.views.review = { show: loadQueue };

  /* ---------- review card ---------- */
  const VERDICT = { agree: 'Agrees', disagree: 'Disagrees', uncertain: 'Unsure' };
  const list = (rv) => (rv.length ? h('div', { class: 'rv-list' }, rv.map((v) => h('div', { class: 'rv ' + v.verdict },
    h('div', { class: 'rv-top' }, h('b', { text: v.reviewer }), h('span', { class: 'tag', text: VERDICT[v.verdict] }), h('small', { text: GS.when(v.created) })),
    v.corrected_class ? h('p', null, 'Suggested class: ', h('b', { text: `${v.corrected_class} (${GS.tissueName(v.corrected_class)})` })) : null, v.note ? h('p', { class: 'rv-note', text: v.note }) : null))) : h('p', { class: 'note', text: 'No reviewer has signed off on this analysis yet.' }));
  function form(r, card) {
    const mine = (r.reviews || []).find((v) => GS.user && v.reviewer_id === GS.user.id) || {}; let verdict = mine.verdict || 'agree';
    const seg = h('div', { class: 'seg' }, Object.entries(VERDICT).map(([k, l]) => h('button', { type: 'button', 'data-v': k, class: k === verdict ? 'on' : '', text: l })));
    const classes = GS.classes.length ? GS.classes : [r.prediction.class];
    const sel = h('select', { id: 'rvClass', 'aria-label': 'Correct class' }, h('option', { value: '', text: 'Choose the correct class' }), classes.map((c) => h('option', { value: c, text: `${c} · ${GS.tissueName(c)}` }))); sel.value = mine.corrected_class || '';
    const corr = h('div', { class: 'field' }, h('label', { for: 'rvClass', text: 'If you disagree, the class you believe is correct' }), sel);
    const note = h('textarea', { id: 'rvNote', rows: 3, maxlength: 1000, placeholder: 'Optional note for the record' }); note.value = mine.note || '';
    const err = h('p', { class: 'form-err', role: 'alert' }); const sync = () => { corr.hidden = verdict !== 'disagree'; $$('button', seg).forEach((b) => b.classList.toggle('on', b.dataset.v === verdict)); };
    seg.addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) { verdict = b.dataset.v; sync(); } }); sync();
    const save = h('button', { class: 'btn primary', type: 'submit' }, h('span', { text: mine.verdict ? 'Update my sign-off' : 'Save sign-off' }));
    const f = h('form', { class: 'rv-form', novalidate: true }, h('h4', { text: 'Your sign-off' }), seg, corr, h('div', { class: 'field' }, h('label', { for: 'rvNote', text: 'Note' }), note), err, save);
    f.addEventListener('submit', async (e) => { e.preventDefault(); err.textContent = ''; save.disabled = true;
      try { const j = await GS.api(`/api/analyses/${r.id}/review`, { json: { verdict, corrected_class: sel.value, note: note.value } }); r.reviews = j.reviews; GS.toast('Sign-off saved.'); fill(r, card); if (GS.current === 'review') loadQueue(); }
      catch (ex) { err.textContent = ex.message; } finally { save.disabled = false; } });
    return f;
  }
  function fill(r, card) {
    card.replaceChildren(h('h3', { text: 'Expert review' })); const v = r.viewer || {};
    if (v.owned) { card.append(list(r.reviews || [])); if (v.can_review) card.append(form(r, card)); }
    else card.append(h('p', { class: 'note', style: 'margin:0', text: 'Expert review is available once an analysis is saved to an account. Sign in before running an analysis to use it.' }), GS.user ? null : h('button', { class: 'btn ghost sm', style: 'margin-top:10px', type: 'button', text: 'Sign in', onclick: () => openAuth('login') }));
  }
  GS.renderReviewCard = (r, host) => { if (host) fill(r, host); };

  GS.refreshUser = () => GS.api('/api/auth/me').then((j) => setUser(j.user)).catch(() => setUser(null));
  renderAccount();
})();

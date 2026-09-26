"use strict";
/* Fly Lab front end: vanilla JS, SVG and canvas. Talks to flyjack/web/server.py. */

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const HIT = 1, STAND = 0;
const CARD = v => (v === 1 ? "A" : String(v));
const fmt = (x, d = 3) => (Math.abs(x) < 0.5 * 10 ** -d ? (0).toFixed(d) : (x > 0 ? "+" : "−") + Math.abs(x).toFixed(d));
const pct = (x, d = 1) => (100 * x).toFixed(d) + "%";
const esc = s => String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

const S = {
  meta: null, mode: "readout", model: "trained", spiking: "not loaded",
  hand: null, busy: false, session: { hands: 0, net: 0, wins: 0, decisions: 0, agree: 0, log: [] },
  policy: {}, last: null, strategySrc: "readout",
};

// ---------------------------------------------------------------- helpers
async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}
function toast(msg, ms = 3500) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), ms);
}
function color(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function rgb(hex) {
  const h = hex.replace("#", "");
  return [0, 2, 4].map(i => parseInt(h.slice(i, i + 2), 16));
}
function mix(a, b, t) {
  const A = rgb(a), B = rgb(b);
  return `rgb(${A.map((v, i) => Math.round(v + (B[i] - v) * t)).join(",")})`;
}
/** diverging STAND (0) -> neutral (0.5) -> HIT (1) */
function diverge(p) {
  const mid = color("--mid");
  return p < 0.5 ? mix(color("--stand"), mid, p * 2) : mix(mid, color("--hit"), (p - 0.5) * 2);
}
const tip = $("#tooltip");
function showTip(html, ev) {
  tip.innerHTML = html; tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
function hideTip() { tip.hidden = true; }
const sleep = ms => new Promise(r => setTimeout(r, ms));
const svgEl = (tag, attrs = {}) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
};
function stateName(o) { return `${o[2] ? "soft" : "hard"} ${o[0]} vs dealer ${CARD(o[1])}`; }

// ---------------------------------------------------------------- init
async function init() {
  try {
    S.meta = await api("/api/meta");
  } catch (e) {
    document.body.innerHTML = `<p style="padding:24px">Could not reach the Fly Lab server: ${esc(e.message)}</p>`;
    return;
  }
  const m = S.meta;
  const trainedHands = m.training ? m.training.config.hands : 0;
  $("#brain-stats").textContent =
    `${m.counts.neurons.toLocaleString()} neurons · ${m.counts.kc.toLocaleString()} Kenyon cells · ` +
    `learned from ${(trainedHands / 1000).toFixed(0)}k hands`;
  setupTabs(); setupHeader(); setupPlay(); setupStrategy(); setupProbe(); setupTournament();
  setupLearning(); renderAbout();
  renderSession();
  const st = await api("/api/spiking"); S.spiking = st.status; renderSpikingPill();
}

function setupTabs() {
  $$(".tabs button").forEach(b => b.addEventListener("click", () => openTab(b.dataset.tab)));
}
function openTab(name) {
  $$(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === name));
  $$(".tab").forEach(t => t.classList.toggle("on", t.id === "tab-" + name));
  if (name === "strategy") renderStrategy();
  if (name === "learning") renderLearning();
}

// ---------------------------------------------------------------- header: mode / model
function setupHeader() {
  $$("#mode-seg button").forEach(b => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $$("#model-seg button").forEach(b => b.addEventListener("click", () => {
    S.model = b.dataset.model;
    $$("#model-seg button").forEach(x => x.classList.toggle("on", x === b));
    toast(S.model === "new" ? "Your newly trained fly is playing now." : "The original trained fly is playing now.");
    if ($("#tab-strategy").classList.contains("on")) renderStrategy();
  }));
}
async function setMode(mode) {
  if (mode === "spiking") {
    if (!S.meta.cuda) { toast("Spiking mode needs a CUDA GPU; this machine has none."); return; }
    if (S.spiking !== "ready") {
      const r = await api("/api/spiking/load", {});
      S.spiking = r.status; renderSpikingPill();
      if (S.spiking.startsWith("unavailable") || S.spiking.startsWith("failed")) { toast(S.spiking); return; }
      pollSpiking();
    }
  }
  S.mode = mode;
  $$("#mode-seg button").forEach(b => b.classList.toggle("on", b.dataset.mode === mode));
  if (mode === "spiking" && S.model === "new")
    toast("The spiking brain has the original fly's synapses installed; readout mode uses your new fly.", 5000);
  updatePlayButtons();
}
async function pollSpiking() {
  while (S.spiking !== "ready" && !S.spiking.startsWith("failed")) {
    await sleep(1000);
    S.spiking = (await api("/api/spiking")).status;
    renderSpikingPill(); updatePlayButtons();
  }
  if (S.spiking === "ready") toast("Spiking brain loaded: 138,639 neurons with the learned synapses on the GPU.");
  else toast(S.spiking, 6000);
}
function renderSpikingPill() {
  const p = $("#spiking-pill");
  p.hidden = S.spiking === "not loaded";
  p.textContent = S.spiking === "ready" ? "● spiking brain ready" : "spiking brain: " + S.spiking;
  p.classList.toggle("ready", S.spiking === "ready");
}
const spikingBlocked = () => S.mode === "spiking" && S.spiking !== "ready";

// ---------------------------------------------------------------- PLAY
function setupPlay() {
  const opts = ['<option value="">?</option>'].concat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(v => `<option value="${v}">${CARD(v)}</option>`));
  ["#c-p1", "#c-p2", "#c-d1", "#c-d2"].forEach(id => ($(id).innerHTML = opts.join("")));
  $("#btn-deal").addEventListener("click", () => deal());
  $("#btn-custom").addEventListener("click", () => {
    const v = id => ($(id).value ? +$(id).value : null);
    deal([v("#c-p1"), v("#c-p2")], [v("#c-d1"), v("#c-d2")]);
  });
  $("#btn-decide").addEventListener("click", () => flyStep());
  $("#btn-auto").addEventListener("click", () => autoplay());
  $("#btn-hit").addEventListener("click", () => humanStep(HIT));
  $("#btn-stand").addEventListener("click", () => humanStep(STAND));
  $("#btn-reset-session").addEventListener("click", () => {
    S.session = { hands: 0, net: 0, wins: 0, decisions: 0, agree: 0, log: [] }; renderSession();
  });
  $("#kc-vote").addEventListener("change", () => S.last && drawKcMap($("#kc-canvas"), S.last, $("#kc-vote").checked));
  $("#btn-replay").addEventListener("click", () => S.last && S.last.raster && animateRaster(S.last));
  window.addEventListener("resize", () => { if (S.last && S.last.raster) drawRasterFrame(S.last, 200); });
}

async function deal(player, dealer) {
  if (S.busy) return;
  try {
    const custom = player && (player.some(Boolean) || dealer.some(Boolean));
    S.hand = await api("/api/hand/new", custom ? { player, dealer } : {});
    S.hand.steps = [];
    renderTable();
    if (S.hand.done) finishHand();
  } catch (e) { toast(e.message); }
  updatePlayButtons();
}

function cardEl(v, back = false) {
  const d = document.createElement("div");
  if (back) { d.className = "pcard back"; return d; }
  const suits = ["♠", "♥", "♦", "♣"];
  const suit = suits[Math.floor(Math.random() * 4)];
  const face = v === 10 ? ["10", "J", "Q", "K"][Math.floor(Math.random() * 4)] : CARD(v);
  d.className = "pcard" + (suit === "♥" || suit === "♦" ? " red" : "");
  d.innerHTML = `<span class="corner">${face}${suit}</span><span class="big">${suit}</span><span class="corner b">${face}${suit}</span>`;
  return d;
}
function syncCards(el, values, hidden) {
  // keep existing card elements (so suits/faces don't reshuffle), append new ones
  const want = values.length + (hidden ? 1 : 0);
  if (el.dataset.hand !== S.hand.id) { el.innerHTML = ""; el.dataset.hand = S.hand.id; }
  const backs = $$(".back", el); backs.forEach(b => b.remove());
  for (let i = el.children.length; i < values.length; i++) el.appendChild(cardEl(values[i]));
  if (hidden && el.children.length < want) el.appendChild(cardEl(0, true));
}
function renderTable() {
  const h = S.hand;
  syncCards($("#player-cards"), h.player, false);
  syncCards($("#dealer-cards"), h.dealer, !h.done);
  $("#player-total").innerHTML = `${h.total}${h.soft ? "<small>soft</small>" : ""}`;
  $("#dealer-total").innerHTML = h.done ? `${h.dealer_total}` : `<small>showing</small>${CARD(h.dealer[0])}`;
  const b = $("#banner");
  b.className = "banner";
  if (!h.done) { b.textContent = "Fly's turn"; return; }
  if (h.natural) b.textContent = h.reward > 0 ? "Blackjack! The fly wins" : h.reward < 0 ? "Dealer blackjack" : "Both blackjack: push";
  else if (h.reward > 0) b.textContent = h.dealer_total > 21 ? "Dealer busts: the fly wins" : "The fly wins";
  else if (h.reward < 0) b.textContent = h.total > 21 ? "Bust: the fly loses" : "The fly loses";
  else b.textContent = "Push";
  b.classList.add(h.reward > 0 ? "win" : h.reward < 0 ? "loss" : "push");
}
function updatePlayButtons() {
  const live = S.hand && !S.hand.done;
  const blocked = spikingBlocked();
  $("#btn-decide").disabled = !live || S.busy || blocked;
  $("#btn-auto").disabled = S.busy || blocked;
  $("#btn-hit").disabled = $("#btn-stand").disabled = !live || S.busy;
  $("#btn-deal").disabled = $("#btn-custom").disabled = S.busy;
  $("#btn-decide").textContent = blocked ? "Loading spiking brain…" : "Let the fly decide";
}

async function flyStep() {
  if (!S.hand || S.hand.done) return;
  S.busy = true; updatePlayButtons();
  try {
    $("#brain-caption").textContent = S.mode === "spiking" ? "simulating 200 ms of the whole brain…" : "reading the mushroom body…";
    const r = await api("/api/sense", { obs: S.hand.obs, mode: S.mode, model: S.model });
    S.last = r;
    renderBrain(r);
    if (r.raster) await animateRaster(r);
    else await sleep(250);
    await applyAction(r.action, "fly", r);
  } catch (e) { toast(e.message); }
  S.busy = false; updatePlayButtons();
}
async function humanStep(action) {
  if (!S.hand || S.hand.done || S.busy) return;
  S.busy = true; updatePlayButtons();
  try { await applyAction(action, "you", null); } catch (e) { toast(e.message); }
  S.busy = false; updatePlayButtons();
}
async function applyAction(action, who, sense) {
  const obs = S.hand.obs;
  const steps = S.hand.steps;
  const next = await api("/api/hand/act", { id: S.hand.id, action });
  S.hand = Object.assign(next, { steps });
  S.hand.steps.push({ obs, action, who, basic: sense ? sense.basic : null });
  if (sense) { S.session.decisions++; if (sense.basic === action) S.session.agree++; }
  renderTable();
  if (S.hand.done) finishHand();
}
function finishHand() {
  const h = S.hand;
  S.session.hands++; S.session.net += h.reward; if (h.reward > 0) S.session.wins++;
  S.session.log.unshift({ steps: h.steps, reward: h.reward, player: h.player, dealer: h.dealer,
                          total: h.total, dealer_total: h.dealer_total, natural: h.natural });
  renderSession(); updatePlayButtons();
}
async function autoplay() {
  if (S.busy) return;
  if (!S.hand || S.hand.done) await deal();
  while (S.hand && !S.hand.done) {
    await flyStep();
    await sleep(350);
  }
}
function renderSession() {
  const s = S.session;
  $("#session-tiles").innerHTML = [
    [s.hands, "hands"],
    [(s.net >= 0 ? "+" : "−") + Math.abs(s.net), "net winnings"],
    [s.hands ? pct(s.wins / s.hands, 0) : "–", "hands won"],
    [s.decisions ? pct(s.agree / s.decisions, 0) : "–", "fly choices = basic strategy"],
  ].map(([v, k]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  $("#hand-log").innerHTML = s.log.slice(0, 60).map((h, i) => {
    const steps = h.steps.map(st => {
      const tag = `<span class="tag ${st.action ? "hit" : "stand"}">${st.action ? "HIT" : "STAND"}</span>`;
      const off = st.who === "you" ? ' <span class="tag warn">you</span>' :
        (st.basic !== null && st.basic !== st.action ? ' <span class="tag warn">≠ basic</span>' : "");
      return `${st.obs[0]}${st.obs[2] ? "s" : ""} ${tag}${off}`;
    }).join(" → ") || (h.natural ? "natural" : "");
    const res = h.reward > 0 ? "+1" : h.reward < 0 ? "−1" : "0";
    return `<li><span class="muted">#${s.hands - i}</span> <span>${steps}</span>
      <span class="muted small">vs ${h.dealer.map(CARD).join(" ")}</span><span class="res">${res}</span></li>`;
  }).join("");
}

// ---------------------------------------------------------------- brain panel
function renderBrain(r) {
  $("#brain-empty").hidden = true; $("#brain").hidden = false;
  const o = r.obs;
  $("#brain-caption").textContent = r.mode === "spiking"
    ? `live spiking simulation · ${r.sim_seconds}s on GPU`
    : `recorded neural trial #${r.trial}${r.held_out ? " (never seen in training)" : ""}`;
  $("#odor-caption").textContent = `· ${stateName(o)}`;
  renderPnGrid(r);
  const nAct = r.kc_active.length;
  $("#kc-caption").textContent = `· ${nAct.toLocaleString()} of ${S.meta.kc.n.toLocaleString()} fired (${pct(r.kc_fraction)})`;
  drawKcMap($("#kc-canvas"), r, $("#kc-vote").checked);
  renderMbon(r);
  $("#raster-stage").hidden = !r.raster;
}

function renderPnGrid(r) {
  const on = new Set(r.channels);
  const featColor = ["--f-sum", "--f-dealer", "--f-ace", "--f-ace", null];
  const maxRate = Math.max(1, ...Object.values(r.pn_rates));
  $("#pn-grid").innerHTML = S.meta.pn_rows.map((row, ri) => {
    const cells = row.channels.map(ch => {
      const rate = r.pn_rates[ch] || 0;
      const active = on.has(ch);
      const c = featColor[ri] ? color(featColor[ri]) : color("--ink-3");
      const bg = active ? c : rate > 0.5 ? mix(color("--surface-2"), c, Math.min(1, rate / maxRate) * .5) : "";
      return `<span class="pn-cell" data-ch="${esc(ch)}" data-rate="${rate.toFixed(0)}" data-on="${active ? 1 : 0}"
        style="${bg ? `background:${bg};border-color:${bg}` : ""}"></span>`;
    }).join("");
    return `<div class="pn-row"><span class="lab">${row.label}</span><span class="pn-cells">${cells}</span></div>`;
  }).join("");
  $$("#pn-grid .pn-cell").forEach(c => {
    c.addEventListener("mousemove", ev => showTip(
      `<b>glomerulus ${esc(c.dataset.ch.replace(/_.*PN$/, ""))}</b> (${esc(c.dataset.ch)})<br>` +
      (c.dataset.on === "1" ? "stimulated by this card odor<br>" : "not stimulated<br>") +
      `projection neurons fire at ${c.dataset.rate} Hz`, ev));
    c.addEventListener("mouseleave", hideTip);
  });
}

// KC map: groups start on fresh rows, one gap row between groups
function kcLayout() {
  if (kcLayout.cache) return kcLayout.cache;
  const cols = S.meta.kc.columns, pos = new Array(S.meta.kc.n);
  let row = 0; const labels = [];
  for (const g of S.meta.kc.groups) {
    labels.push({ label: g.label, row });
    for (let i = 0; i < g.count; i++) pos[g.start + i] = [row + Math.floor(i / cols), i % cols];
    row += Math.ceil(g.count / cols) + 1;
  }
  return (kcLayout.cache = { pos, rows: row - 1, cols, labels });
}
const KC_CELL = 6;
function drawKcMap(canvas, r, byVote, mode = "activity") {
  const L = kcLayout(), pref = S.meta.kc.preference;
  canvas.width = L.cols * KC_CELL; canvas.height = L.rows * KC_CELL;
  const ctx = canvas.getContext("2d");
  const base = color("--border");
  const prefScale = kcPrefScale();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  // background dots
  ctx.fillStyle = base;
  for (const [row, col] of L.pos) ctx.fillRect(col * KC_CELL + 1, row * KC_CELL + 1, KC_CELL - 2, KC_CELL - 2);
  const surf2 = color("--surface-2"), kcC = color("--kc");
  if (mode === "preference") {
    for (let k = 0; k < pref.length; k++) {
      const t = Math.max(-1, Math.min(1, pref[k] / prefScale));
      ctx.fillStyle = diverge(0.5 + t / 2);
      const [row, col] = L.pos[k];
      ctx.fillRect(col * KC_CELL, row * KC_CELL, KC_CELL, KC_CELL);
    }
  } else {
    const spikes = r.kc_spikes, maxS = Math.max(3, ...spikes);
    r.kc_active.forEach((k, i) => {
      const s = spikes[i] / maxS;
      if (byVote) {
        const t = Math.max(-1, Math.min(1, (pref[k] * spikes[i]) / (prefScale * 2)));
        ctx.fillStyle = diverge(0.5 + (t >= 0 ? 0.15 + 0.35 * t : -0.15 + 0.35 * t));
      } else ctx.fillStyle = mix(surf2, kcC, 0.35 + 0.65 * s);
      const [row, col] = L.pos[k];
      ctx.fillRect(col * KC_CELL, row * KC_CELL, KC_CELL, KC_CELL);
    });
  }
  canvas.onmousemove = ev => {
    const b = canvas.getBoundingClientRect();
    const col = Math.floor(((ev.clientX - b.left) / b.width) * L.cols);
    const row = Math.floor(((ev.clientY - b.top) / b.height) * L.rows);
    const k = kcIndexAt(row, col);
    if (k < 0) return hideTip();
    const i = r && r.kc_active ? r.kc_active.indexOf(k) : -1;
    const sp = i >= 0 ? r.kc_spikes[i] : (r && r.kc_mean ? r.kc_mean[k] : 0);
    const g = S.meta.kc.groups.find(g => k >= g.start && k < g.start + g.count);
    const vote = pref[k] > 0 ? "HIT" : "STAND";
    showTip(`<b>Kenyon cell</b> · ${g.label}<br>` +
      (!r ? "" : r.kc_mean ? `mean spikes over trials: ${(+sp).toFixed(2)}<br>` : `spikes in 200 ms: ${sp}<br>`) +
      `learned vote: <b style="color:var(${pref[k] > 0 ? "--hit" : "--stand"})">${vote}</b> (${fmt(pref[k], 3)})`, ev);
  };
  canvas.onmouseleave = hideTip;
  const legend = canvas.id === "kc-canvas" ? $("#kc-legend") : null;
  if (legend) legend.innerHTML = byVote
    ? `<span><span class="sw" style="background:var(--hit)"></span>active, pushes toward HIT</span>
       <span><span class="sw" style="background:var(--stand)"></span>active, pushes toward STAND</span>
       <span><span class="sw" style="background:var(--border)"></span>silent</span>
       <span class="muted">rows: ${S.meta.kc.groups.map(g => g.label).join(" · ")}</span>`
    : `<span>silent<span class="ramp" style="background:linear-gradient(90deg,var(--surface-2),var(--kc))"></span>many spikes</span>
       <span class="muted">rows: ${S.meta.kc.groups.map(g => g.label).join(" · ")}</span>`;
}
function kcIndexAt(row, col) {
  const L = kcLayout();
  for (const g of S.meta.kc.groups) {
    const r0 = L.pos[g.start][0], i = (row - r0) * L.cols + col;
    if (row >= r0 && i >= 0 && i < g.count && col >= 0 && col < L.cols) return g.start + i;
  }
  return -1;
}
function kcPrefScale() {
  if (kcPrefScale.v) return kcPrefScale.v;
  const a = S.meta.kc.preference.map(Math.abs).sort((x, y) => x - y);
  return (kcPrefScale.v = a[Math.floor(a.length * 0.98)] || 1);
}

function renderMbon(r) {
  let rows, note;
  if (r.mode === "spiking") {
    const max = Math.max(20, r.spikes.hit, r.spikes.stand) * 1.1;
    rows = [["HIT", "MBON01 ×2", r.spikes.hit, r.spikes.hit / max, "--hit", `${r.spikes.hit} spikes`],
            ["STAND", "MBON03 ×2", r.spikes.stand, r.spikes.stand / max, "--stand", `${r.spikes.stand} spikes`]];
    note = `The fly hits if its HIT neurons fire more spikes than its STAND neurons.`;
    $("#mbon-bars").innerHTML = rows.map(([n, sub, , w, c, lab]) => `
      <div class="bar-row"><span class="name">${n}<br><span class="muted small">${sub}</span></span>
      <span class="bar-track"><span class="bar-fill" style="left:0;width:${(100 * w).toFixed(1)}%;background:var(${c})"></span></span>
      <span class="val">${lab}</span></div>`).join("");
  } else {
    // readout pools predict the expected winnings of each action (-1..+1)
    rows = [["HIT", r.mbon.hit, "--hit"], ["STAND", r.mbon.stand, "--stand"]];
    note = `Each output pool has learned to predict the winnings of its action (−1 … +1).`;
    $("#mbon-bars").innerHTML = rows.map(([n, v, c]) => {
      const x = Math.max(-1, Math.min(1, v)), left = x < 0 ? 50 + 50 * x : 50, w = Math.abs(50 * x);
      return `<div class="bar-row"><span class="name">${n} pool</span>
        <span class="bar-track"><span style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:var(--ink-3)"></span>
        <span class="bar-fill" style="left:${left}%;width:${w}%;background:var(${c})"></span></span>
        <span class="val">${fmt(v, 3)}</span></div>`;
    }).join("");
  }
  const agree = r.action === r.basic;
  const ro = r.mode === "spiking" && r.readout_action !== r.action
    ? ` <span class="tag warn">readout of the same spikes would ${r.readout_action ? "HIT" : "STAND"}</span>` : "";
  $("#verdict").innerHTML = `<span class="muted">Fly chooses</span>
    <span class="choice" style="color:var(${r.action ? "--hit" : "--stand"})">${r.action ? "HIT" : "STAND"}</span>
    <span class="tag ${agree ? "stand" : "warn"}" style="${agree ? "color:var(--s3);background:color-mix(in srgb,var(--s3) 16%,transparent)" : ""}">
      ${agree ? "✓ same as basic strategy" : `basic strategy says ${r.basic ? "HIT" : "STAND"}`}</span>${ro}
    <div class="muted small" style="flex-basis:100%">${note}</div>`;
}

// ---------------------------------------------------------------- raster (spiking mode)
const RASTER_GROUPS = [
  ["pn", "projection neurons", "--pn", 0.2], ["kc", "Kenyon cells", "--kc", 0.46],
  ["hit", "HIT MBONs", "--hit", 0.08], ["stand", "STAND MBONs", "--stand", 0.08], ["dn", "descending neurons", "--dn", 0.18],
];
function rasterRows(r) {
  if (r._rows) return r._rows;
  const out = {};
  for (const [name] of RASTER_GROUPS) {
    const g = r.raster[name];
    const uniq = name === "hit" || name === "stand" ? [...Array(g.n).keys()] : [...new Set(g.row)].sort((a, b) => a - b);
    const rank = new Map(uniq.map((v, i) => [v, i]));
    out[name] = { n: Math.max(uniq.length, 1), rank, active: uniq.length };
  }
  return (r._rows = out);
}
function drawRasterFrame(r, tNow) {
  const cv = $("#raster-canvas"), dpr = devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr);
  const left = 128, right = 8, top = 4, gap = 6, bottom = 18;
  const plotW = W - left - right, avail = H - top - bottom - gap * (RASTER_GROUPS.length - 1);
  const rows = rasterRows(r);
  let y = top;
  ctx.font = "11px system-ui"; ctx.textBaseline = "middle";
  for (const [name, label, cvar, frac] of RASTER_GROUPS) {
    const h = avail * frac, g = r.raster[name], R = rows[name];
    ctx.fillStyle = color("--surface-2"); ctx.fillRect(left, y, plotW, h);
    ctx.fillStyle = color("--ink-2"); ctx.textAlign = "right";
    ctx.fillText(label, left - 8, y + h / 2 - 6);
    ctx.fillStyle = color("--ink-3");
    ctx.fillText(name === "hit" || name === "stand" ? `${g.t.length} spikes` : `${R.active} of ${g.n} fired`, left - 8, y + h / 2 + 7);
    ctx.fillStyle = color(cvar);
    const big = name === "hit" || name === "stand";
    for (let i = 0; i < g.t.length; i++) {
      if (g.t[i] > tNow) continue;
      const x = left + (g.t[i] / 200) * plotW;
      const yy = y + ((R.rank.get(g.row[i]) + 0.5) / R.n) * h;
      if (big) ctx.fillRect(x - 0.75, yy - h / (2 * R.n) + 1, 1.5, h / R.n - 2);
      else ctx.fillRect(x - 0.6, yy - 0.6, 1.4, Math.max(1.2, h / R.n * 0.8));
    }
    y += h + gap;
  }
  ctx.strokeStyle = color("--ink-3"); ctx.beginPath();
  const xNow = left + (Math.min(tNow, 200) / 200) * plotW;
  if (tNow < 200) { ctx.moveTo(xNow, top); ctx.lineTo(xNow, y - gap); ctx.stroke(); }
  ctx.fillStyle = color("--ink-3"); ctx.textBaseline = "alphabetic";
  for (const t of [0, 50, 100, 150, 200]) {
    ctx.textAlign = t === 0 ? "left" : t === 200 ? "right" : "center";
    ctx.fillText(t + " ms", left + (t / 200) * plotW, H - 4);
  }
  drawRace(r, tNow);
}
function drawRace(r, tNow) {
  const svg = $("#race"); svg.innerHTML = "";
  const W = svg.clientWidth, H = svg.clientHeight, m = { l: 128, r: 8, t: 22, b: 8 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const ts = { hit: r.raster.hit.t.slice().sort((a, b) => a - b), stand: r.raster.stand.t.slice().sort((a, b) => a - b) };
  const ymax = Math.max(10, ts.hit.length, ts.stand.length);
  const X = t => m.l + (t / 200) * pw, Y = v => m.t + ph - (v / ymax) * ph;
  for (const v of [0, Math.round(ymax / 2), ymax]) {
    svg.appendChild(svgEl("line", { x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v), class: "gridline" }));
    const t = svgEl("text", { x: m.l - 6, y: Y(v) + 4, "text-anchor": "end" }); t.textContent = v; svg.appendChild(t);
  }
  const lab = svgEl("text", { x: m.l - 34, y: m.t + ph / 2, "text-anchor": "end" });
  lab.textContent = "spike race"; svg.appendChild(lab);
  const lab2 = svgEl("text", { x: m.l - 34, y: m.t + ph / 2 + 13, "text-anchor": "end" });
  lab2.textContent = "(cumulative)"; svg.appendChild(lab2);
  const counts = {};
  for (const [name, cvar, txt] of [["stand", "--stand", "STAND"], ["hit", "--hit", "HIT"]]) {
    const pts = [[X(0), Y(0)]]; let n = 0;
    for (const t of ts[name]) { if (t > tNow) break; pts.push([X(t), Y(n)]); n++; pts.push([X(t), Y(n)]); }
    pts.push([X(Math.min(tNow, 200)), Y(n)]);
    svg.appendChild(svgEl("polyline", { points: pts.map(p => p.join(",")).join(" "), fill: "none",
      stroke: color(cvar), "stroke-width": 2 }));
    counts[name] = n;
  }
  const head = svgEl("text", { x: m.l, y: 13, "font-weight": 600 });
  head.innerHTML = `<tspan style="fill:${color("--hit")}">HIT pool ${counts.hit}</tspan>` +
    `<tspan dx="14" style="fill:${color("--stand")}">STAND pool ${counts.stand}</tspan>` +
    `<tspan dx="14">${tNow >= 200 ? (counts.hit > counts.stand ? "→ HIT wins the race" : counts.hit === counts.stand ? "→ tie: the fly stands" : "→ STAND wins the race") : ""}</tspan>`;
  svg.appendChild(head);
}
async function animateRaster(r) {
  const dur = 1400, t0 = performance.now();
  return new Promise(resolve => {
    const frame = now => {
      const t = Math.min(200, ((now - t0) / dur) * 200);
      drawRasterFrame(r, t);
      if (t < 200) requestAnimationFrame(frame); else resolve();
    };
    requestAnimationFrame(frame);
  });
}

// ---------------------------------------------------------------- STRATEGY
function setupStrategy() {
  $$("#strategy-src button").forEach(b => b.addEventListener("click", () => {
    S.strategySrc = b.dataset.src;
    $$("#strategy-src button").forEach(x => x.classList.toggle("on", x === b));
    renderStrategy();
  }));
}
async function getPolicy() {
  if (!S.policy[S.model]) S.policy[S.model] = await api("/api/policy?model=" + S.model);
  return S.policy[S.model];
}
async function renderStrategy() {
  let pol;
  try { pol = await getPolicy(); } catch (e) { return toast(e.message); }
  const spiking = S.strategySrc === "spiking";
  if (spiking && pol.expected_return_spiking === null) {
    toast("Spiking-brain strategy is only measured for the original trained fly.");
    S.strategySrc = "readout"; $$("#strategy-src button").forEach(x => x.classList.toggle("on", x.dataset.src === "readout"));
    return renderStrategy();
  }
  const P = s => (spiking ? s.p_hit_spiking : s.p_hit);
  const agree = pol.states.filter(s => (P(s) > 0.5) === (s.basic === 1)).length;
  const noisy = pol.states.filter(s => P(s) > 0 && P(s) < 1).length;
  const ev = spiking ? pol.expected_return_spiking : pol.expected_return;
  $("#strategy-tiles").innerHTML = [
    [fmt(ev, 4), `fly's expected winnings per hand (${spiking ? "spiking brain" : "readout"})`],
    [fmt(pol.basic_return, 4), "basic strategy (optimal)"],
    [`${agree}/280`, "states where the fly's majority choice = basic strategy"],
    [noisy, "states where neural noise makes the choice vary"],
  ].map(([v, k]) => `<div class="tile"><div class="v">${v}</div><div class="k">${k}</div></div>`).join("");
  const by = new Map(pol.states.map(s => [s.obs.join(","), s]));
  const table = (soft, useBasic) => {
    const sums = soft ? [...Array(10)].map((_, i) => 21 - i) : [...Array(18)].map((_, i) => 21 - i);
    let h = `<table><tr><th></th>${[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map(d => `<th>${CARD(d)}</th>`).join("")}</tr>`;
    for (const s of sums) {
      h += `<tr><th>${s}</th>`;
      for (let d = 1; d <= 10; d++) {
        const st = by.get([s, d, soft].join(","));
        const p = useBasic ? st.basic : P(st);
        const miss = !useBasic && (p > 0.5) !== (st.basic === 1);
        h += `<td data-k="${st.obs.join(",")}" class="${miss ? "miss" : ""}" style="background:${diverge(p)}">${(p * 100).toFixed(0)}</td>`;
      }
      h += "</tr>";
    }
    return h + "</table>";
  };
  const who = spiking ? "Fly · spiking brain" : "Fly · readout";
  $("#heatmaps").innerHTML = `
    <div class="heat"><h3>${who}: hard hands</h3>${table(false, false)}</div>
    <div class="heat"><h3>Basic strategy: hard hands</h3>${table(false, true)}</div>
    <div class="heat"><h3>${who}: soft hands</h3>${table(true, false)}</div>
    <div class="heat"><h3>Basic strategy: soft hands</h3>${table(true, true)}</div>`;
  $("#heat-legend").innerHTML = `<span>always STAND<span class="ramp" style="background:linear-gradient(90deg,${diverge(0)},${diverge(.5)},${diverge(1)})"></span>always HIT</span>
    <span>rows: fly's total · columns: dealer's upcard</span><span><span class="sw" style="outline:2px solid var(--ink);outline-offset:-2px"></span>majority choice ≠ basic strategy</span>`;
  $$("#heatmaps td").forEach(td => {
    const st = by.get(td.dataset.k);
    td.addEventListener("mousemove", ev => showTip(
      `<b>${stateName(st.obs)}</b><br>fly hits in ${pct(P(st), 0)} of trials<br>` +
      `fly's predicted winnings: HIT ${fmt(st.q_hit)}, STAND ${fmt(st.q_stand)}<br>` +
      `true values: HIT ${fmt(st.true_q_hit)}, STAND ${fmt(st.true_q_stand)}<br>` +
      `basic strategy: <b>${st.basic ? "HIT" : "STAND"}</b><br><span class="muted">click to probe this state</span>`, ev));
    td.addEventListener("mouseleave", hideTip);
    td.addEventListener("click", () => { hideTip(); probeState(st.obs); });
  });
}

// ---------------------------------------------------------------- PROBE
function setupProbe() {
  $("#p-sum").innerHTML = S.meta.sum_values.map(v => `<option ${v === 16 ? "selected" : ""}>${v}</option>`).join("");
  $("#p-dealer").innerHTML = S.meta.dealer_values.map(v => `<option value="${v}" ${v === 10 ? "selected" : ""}>${CARD(v)}</option>`).join("");
  $("#p-soft").addEventListener("change", () => {
    const soft = $("#p-soft").checked, cur = +$("#p-sum").value;
    $("#p-sum").innerHTML = S.meta.sum_values.filter(v => !soft || v >= 12)
      .map(v => `<option ${v === Math.max(cur, soft ? 12 : 4) ? "selected" : ""}>${v}</option>`).join("");
  });
  $("#btn-probe").addEventListener("click", runProbe);
}
function probeState(obs) {
  openTab("probe");
  $("#p-soft").checked = obs[2]; $("#p-soft").dispatchEvent(new Event("change"));
  $("#p-sum").value = obs[0]; $("#p-dealer").value = obs[1];
  runProbe();
}
async function runProbe() {
  if (spikingBlocked()) return toast("The spiking brain is still loading.");
  const obs = [+$("#p-sum").value, +$("#p-dealer").value, $("#p-soft").checked];
  const trials = +$("#p-trials").value;
  $("#probe-note").textContent = S.mode === "spiking" ? `simulating ${trials} × 200 ms on the GPU…` : "";
  $("#btn-probe").disabled = true;
  try {
    const r = await api("/api/probe", { obs, mode: S.mode, trials, model: S.model });
    renderProbe(r);
    $("#probe-note").textContent = S.mode === "spiking" ? "fresh spiking trials" : "all 32 recorded trials (hollow = used for training)";
  } catch (e) { toast(e.message); $("#probe-note").textContent = ""; }
  $("#btn-probe").disabled = false;
}
function renderProbe(r) {
  $("#probe-out").hidden = false;
  const spk = r.mode === "spiking";
  const held = r.trials.filter(t => spk || t.held_out);
  const pHeld = held.filter(t => t.action === 1).length / held.length;
  $("#probe-tiles").innerHTML = [
    [stateName(r.obs), "state"],
    [pct(pHeld, 0), spk ? "of spiking trials → HIT" : "of held-out trials → HIT"],
    [r.basic ? "HIT" : "STAND", "basic strategy"],
    [(pHeld > 0.5) === (r.basic === 1) ? "✓ agrees" : "✗ disagrees", "fly's majority vs basic"],
  ].map(([v, k]) => `<div class="tile"><div class="v" style="font-size:${String(v).length > 12 ? 15 : 22}px">${v}</div><div class="k">${k}</div></div>`).join("");
  // strip chart of HIT - STAND per trial
  const svg = $("#probe-chart"); svg.innerHTML = "";
  const W = svg.clientWidth, H = svg.clientHeight, m = { l: 56, r: 16, t: 14, b: 32 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const d = r.trials.map(t => t.hit - t.stand);
  const ext = Math.max(...d.map(Math.abs), spk ? 5 : 0.05) * 1.15;
  const X = i => m.l + ((i + 0.5) / d.length) * pw, Y = v => m.t + ph / 2 - (v / ext) * (ph / 2);
  for (const v of [-ext, -ext / 2, 0, ext / 2, ext].map(v => +v.toPrecision(2))) {
    svg.appendChild(svgEl("line", { x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v), class: "gridline",
      "stroke-width": v === 0 ? 1.5 : 1, stroke: v === 0 ? color("--ink-3") : color("--grid") }));
    const t = svgEl("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" });
    t.textContent = spk ? Math.round(v) : v.toFixed(2); svg.appendChild(t);
  }
  const lt = svgEl("text", { x: m.l + 4, y: m.t + 10 }); lt.textContent = "↑ HIT wins"; svg.appendChild(lt);
  const lb = svgEl("text", { x: m.l + 4, y: m.t + ph - 2 }); lb.textContent = "↓ STAND wins"; svg.appendChild(lb);
  const xl = svgEl("text", { x: m.l + pw / 2, y: H - 6, "text-anchor": "middle" });
  xl.textContent = spk ? "spiking trial (HIT − STAND pool spikes)" : "recorded neural trial (HIT − STAND predicted winnings)";
  svg.appendChild(xl);
  r.trials.forEach((t, i) => {
    const c = color(d[i] > 0 ? "--hit" : "--stand");
    const hollow = !spk && !t.held_out;
    const dot = svgEl("circle", { cx: X(i), cy: Y(d[i]), r: 5.5, fill: hollow ? color("--surface") : c,
      stroke: hollow ? c : color("--surface"), "stroke-width": 2 });
    dot.addEventListener("mousemove", ev => showTip(`<b>trial ${t.trial}</b>${spk ? "" : t.held_out ? " (held out)" : " (training trial)"}<br>` +
      (spk ? `HIT pool ${t.hit} spikes · STAND pool ${t.stand} spikes` : `HIT ${fmt(t.hit)} · STAND ${fmt(t.stand)}`) +
      `<br>→ <b>${t.action ? "HIT" : "STAND"}</b>`, ev));
    dot.addEventListener("mouseleave", hideTip);
    svg.appendChild(dot);
  });
  drawKcMap($("#probe-kc"), { kc_mean: r.kc_mean, kc_active: r.kc_mean.map((v, i) => (v > 0 ? i : -1)).filter(i => i >= 0),
    kc_spikes: r.kc_mean.filter(v => v > 0) }, false);
  drawKcMap($("#probe-pref"), null, false, "preference");
  $("#probe-legend").innerHTML = `<span>left: silent<span class="ramp" style="background:linear-gradient(90deg,var(--surface-2),var(--kc))"></span>fires a lot</span>
    <span>right: votes STAND<span class="ramp" style="background:linear-gradient(90deg,${diverge(0)},${diverge(.5)},${diverge(1)})"></span>votes HIT</span>
    <span class="muted">The fly's choice is the sum of the votes of the cells that fire. Hover a cell for details.</span>`;
}

// ---------------------------------------------------------------- charts
const SERIES = ["--s1", "--s2", "--s3", "--s4", "--s5"];
/** Line chart with hover crosshair. series: [{name, color, x, y, width, dash, hidden}] */
function lineChart(svg, { series, refs = [], xFmt = v => v, xTick, yFmt = v => v.toFixed(2), yLabel = "", xLabel = "", yDomain }) {
  xTick = xTick || xFmt;
  svg.innerHTML = "";
  const W = svg.clientWidth, H = svg.clientHeight, m = { l: 62, r: refs.length ? 170 : 20, t: 12, b: 36 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const vis = series.filter(s => !s.hidden);
  const xs = vis.flatMap(s => s.x), ys = vis.flatMap(s => s.y).concat(refs.map(r => r.y));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  let [y0, y1] = yDomain || [Math.min(...ys), Math.max(...ys)];
  if (y0 === y1) { y0 -= 1; y1 += 1; }
  const pad = (y1 - y0) * 0.06; if (!yDomain) { y0 -= pad; y1 += pad; }
  const X = v => m.l + ((v - x0) / (x1 - x0 || 1)) * pw, Y = v => m.t + ph - ((v - y0) / (y1 - y0)) * ph;
  const ticks = (a, b, n) => { const st = niceStep((b - a) / n); const out = []; for (let v = Math.ceil(a / st) * st; v <= b + 1e-9; v += st) out.push(+v.toPrecision(12)); return out; };
  for (const v of ticks(y0, y1, 5)) {
    svg.appendChild(svgEl("line", { x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v), class: "gridline" }));
    const t = svgEl("text", { x: m.l - 8, y: Y(v) + 4, "text-anchor": "end" }); t.textContent = yFmt(v); svg.appendChild(t);
  }
  for (const v of ticks(x0, x1, 6)) {
    const anchor = X(v) > m.l + pw - 20 ? "end" : X(v) < m.l + 20 ? "start" : "middle";
    const t = svgEl("text", { x: X(v), y: m.t + ph + 18, "text-anchor": anchor }); t.textContent = xTick(v); svg.appendChild(t);
  }
  if (xLabel) { const t = svgEl("text", { x: m.l + pw / 2, y: H - 4, "text-anchor": "middle" }); t.textContent = xLabel; svg.appendChild(t); }
  if (yLabel) { const t = svgEl("text", { x: 12, y: m.t + ph / 2, transform: `rotate(-90 12 ${m.t + ph / 2})`, "text-anchor": "middle" }); t.textContent = yLabel; svg.appendChild(t); }
  for (const r of refs) {
    if (r.y < y0 || r.y > y1) continue;
    svg.appendChild(svgEl("line", { x1: m.l, x2: m.l + pw, y1: Y(r.y), y2: Y(r.y), stroke: r.color, "stroke-width": 1.2, "stroke-dasharray": r.dash || "5 4" }));
    const t = svgEl("text", { x: m.l + pw + 6, y: Y(r.y) + 4 }); t.textContent = r.label; svg.appendChild(t);
  }
  for (const s of vis) {
    const pts = s.x.map((x, i) => `${X(x).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join(" ");
    svg.appendChild(svgEl("polyline", { points: pts, fill: "none", stroke: s.color, "stroke-width": s.width || 2,
      "stroke-linejoin": "round", opacity: s.opacity || 1, "stroke-dasharray": s.dash || "" }));
  }
  // hover layer
  const cross = svgEl("line", { y1: m.t, y2: m.t + ph, stroke: color("--ink-3"), visibility: "hidden" });
  svg.appendChild(cross);
  const hit = svgEl("rect", { x: m.l, y: m.t, width: pw, height: ph, fill: "transparent" });
  svg.appendChild(hit);
  const main = vis.filter(s => !s.noTip);
  hit.addEventListener("mousemove", ev => {
    const b = svg.getBoundingClientRect();
    const xv = x0 + ((ev.clientX - b.left - m.l) / pw) * (x1 - x0);
    const ref = main[0]; if (!ref) return;
    let i = 0, best = Infinity;
    ref.x.forEach((x, j) => { const d = Math.abs(x - xv); if (d < best) { best = d; i = j; } });
    cross.setAttribute("x1", X(ref.x[i])); cross.setAttribute("x2", X(ref.x[i])); cross.setAttribute("visibility", "visible");
    showTip(`<b>${xFmt(ref.x[i])}</b><br>` + main.map(s => `<span class="sw" style="display:inline-block;width:9px;height:9px;border-radius:2px;background:${s.color};margin-right:5px"></span>${esc(s.name)}: <b>${yFmt(s.y[Math.min(i, s.y.length - 1)])}</b>`).join("<br>"), ev);
  });
  hit.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); hideTip(); });
}
function niceStep(raw) {
  const p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}
function legend(el, items, onToggle) {
  el.innerHTML = items.map((it, i) => `<span data-i="${i}" style="cursor:${onToggle ? "pointer" : "default"};opacity:${it.hidden ? .4 : 1}">
    <span class="sw" style="background:${it.color}"></span>${esc(it.name)}${it.extra ? ` <b>${it.extra}</b>` : ""}</span>`).join("") +
    (onToggle ? `<span class="muted">click to show/hide</span>` : "");
  if (onToggle) $$("span[data-i]", el).forEach(s => s.addEventListener("click", () => onToggle(+s.dataset.i)));
}
const kFmt = v => (Math.abs(v) >= 1000 ? (v / 1000).toFixed(v % 1000 ? 1 : 0) + "k" : String(Math.round(v)));

// ---------------------------------------------------------------- TOURNAMENT
function setupTournament() { $("#btn-tournament").addEventListener("click", runTournament); }
async function runTournament() {
  const hands = +$("#t-hands").value;
  $("#btn-tournament").disabled = true; $("#t-note").textContent = `dealing ${hands.toLocaleString()} hands to each player…`;
  try {
    const r = await api("/api/tournament", { hands, model: S.model });
    S.tour = r; S.tourHidden = S.tourHidden || { random: true };
    renderTournament();
    $("#t-note").textContent = `seed ${r.seed}`;
  } catch (e) { toast(e.message); $("#t-note").textContent = ""; }
  $("#btn-tournament").disabled = false;
}
function renderTournament() {
  const r = S.tour; $("#t-out").hidden = false;
  const names = Object.keys(r.players);
  const series = names.map((n, i) => ({ name: n === "fly" ? "the fly" : n, key: n, color: color(SERIES[i]), x: r.points,
    y: r.players[n].cumulative, width: n === "fly" ? 3 : 2, hidden: !!S.tourHidden[n] }));
  lineChart($("#t-chart"), { series, xFmt: v => kFmt(v) + " hands", xTick: kFmt, yFmt: v => (v > 0 ? "+" : v < 0 ? "−" : "") + kFmt(Math.abs(v)),
    yLabel: "cumulative winnings (bets)", xLabel: "hands played" });
  legend($("#t-legend"), series.map(s => ({ ...s, extra: fmt(r.players[s.key].mean, 3) + "/hand" })), i => {
    S.tourHidden[names[i]] = !S.tourHidden[names[i]]; renderTournament();
  });
  $("#t-table").innerHTML = `<tr><th>player</th><th class="n">winnings / hand</th><th class="n">95% CI</th><th class="n">won</th><th class="n">push</th><th class="n">lost</th><th class="n">total</th></tr>` +
    names.map(n => { const p = r.players[n];
      return `<tr class="${n === "fly" ? "fly" : ""}"><td>${n === "fly" ? "🪰 the fly" : n}</td><td class="n">${fmt(p.mean, 4)}</td><td class="n">± ${p.ci95.toFixed(4)}</td>
        <td class="n">${pct(p.wins / r.hands)}</td><td class="n">${pct(p.pushes / r.hands)}</td><td class="n">${pct(p.losses / r.hands)}</td>
        <td class="n">${fmt(p.cumulative.at(-1), 0)}</td></tr>`; }).join("");
}

// ---------------------------------------------------------------- LEARNING
function refLines() {
  const t = Object.fromEntries((S.meta.evaluation || []).map(r => [r.name, r.exact]));
  return [
    { y: t["basic strategy (optimal)"], label: "basic strategy " + fmt(t["basic strategy (optimal)"]), color: color("--ink"), dash: "0" },
    { y: t["dealer rule (hit < 17)"], label: "dealer rule " + fmt(t["dealer rule (hit < 17)"]), color: color("--s2") },
    { y: t["never bust"], label: "never bust " + fmt(t["never bust"]), color: color("--s3") },
    { y: t["random"], label: "random " + fmt(t["random"]), color: color("--s5"), dash: "2 3" },
  ].filter(r => r.y !== undefined);
}
function renderLearning() {
  const tr = S.meta.training;
  if (!tr) return;
  const runs = tr.runs;
  const x = runs[0].history.map(h => h.hands);
  const mean = x.map((_, i) => runs.reduce((a, r) => a + r.history[i].expected_return, 0) / runs.length);
  const series = runs.map(r => ({ name: `fly seed ${r.seed}`, color: color("--s1"), x, y: r.history.map(h => h.expected_return),
    width: 1, opacity: 0.3, noTip: true }))
    .concat([{ name: "fly (mean of 5)", color: color("--s1"), x, y: mean, width: 2.5 }]);
  lineChart($("#l-chart"), { series, refs: refLines(), xFmt: v => kFmt(v), yFmt: v => fmt(v, 2), yDomain: [-0.4, 0],
    yLabel: "expected winnings per hand", xLabel: "hands played" });
  legend($("#l-legend"), [{ name: "mean of 5 flies", color: color("--s1") }, { name: "individual flies", color: mix(color("--surface"), color("--s1"), .35) }]);
  renderTrainChart(S.trainStatus);
}
function setupLearning() {
  $("#btn-train").addEventListener("click", async () => {
    try {
      S.trainStatus = await api("/api/train/start", { hands: +$("#n-hands").value, eta: +$("#n-eta").value,
        epsilon: +$("#n-eps").value, seed: +$("#n-seed").value });
      pollTraining();
    } catch (e) { toast(e.message); }
  });
  $("#btn-train-stop").addEventListener("click", () => api("/api/train/stop", {}));
}
async function pollTraining() {
  $("#btn-train").disabled = true; $("#btn-train-stop").disabled = false;
  while (true) {
    const st = await api("/api/train/status");
    S.trainStatus = st; renderTrainChart(st);
    const last = st.history.at(-1);
    $("#n-note").textContent = last ? `${kFmt(last.hands)} hands · winnings ${fmt(last.expected_return, 4)}/hand · ${pct(last.agreement)} = basic strategy` : "starting…";
    if (!st.running) break;
    await sleep(400);
  }
  $("#btn-train").disabled = false; $("#btn-train-stop").disabled = true;
  if (S.trainStatus.error) return toast(S.trainStatus.error);
  if (S.trainStatus.available) {
    delete S.policy.new;
    $("#model-field").hidden = false;
    toast("Your new fly is ready. Choose “your new fly” in the header to play with it.", 5000);
  }
}
function renderTrainChart(st) {
  const svg = $("#n-chart");
  $("#n-wrap").hidden = !st || !st.history.length;
  if (!st || !st.history.length) { svg.innerHTML = ""; $("#n-legend").innerHTML = ""; return; }
  const x = st.history.map(h => h.hands);
  const total = st.config ? st.config.hands : x.at(-1);
  lineChart(svg, { series: [{ name: "your new fly", color: color("--s1"), x: x.concat(x.length < 2 ? [total] : []),
      y: st.history.map(h => h.expected_return).concat(x.length < 2 ? [st.history[0].expected_return] : []), width: 2.5 }],
    refs: refLines(), xFmt: v => kFmt(v), yFmt: v => fmt(v, 2), yDomain: [-0.4, 0],
    yLabel: "expected winnings per hand", xLabel: "hands played" });
  legend($("#n-legend"), [{ name: "your new fly", color: color("--s1") }]);
}

// ---------------------------------------------------------------- ABOUT
function renderAbout() {
  const c = S.meta.counts;
  const steps = [
    ["Cards", "total · dealer card · soft/hard"],
    ["Odor", `${S.meta.pn_rows.slice(0, 4).reduce((a, r) => a + r.channels.length, 0)} glomerulus channels, ${S.meta.encoder.rate_hz} Hz`],
    ["Projection neurons", `${c.pn} uniglomerular PNs`],
    ["Kenyon cells", `${c.kc.toLocaleString()} cells, ~7% fire`],
    ["Output neurons", "HIT pool vs STAND pool"],
    ["Choice", "dopamine taught the weights"],
  ];
  $("#pipeline").innerHTML = steps.map(([b, s], i) => `${i ? '<span class="pipe-arrow">→</span>' : ""}<div class="pipe"><b>${b}</b><span>${s}</span></div>`).join("");
  const rows = (S.meta.evaluation || []).map(r => [r.name, r.exact, r.agreement]);
  const cl = S.meta.closed_loop;
  if (cl) rows.splice(rows.length - 1, 0, ["fly: pure spiking brain (closed loop)", cl.sweep.exact_ev_spiking, cl.sweep.spiking_vs_basic]);
  $("#about-table").innerHTML = `<tr><th>policy</th><th class="n">expected winnings / hand</th><th class="n">same choice as basic strategy</th></tr>` +
    rows.map(([n, e, a]) => `<tr class="${n.startsWith("fly: mushroom") || n.startsWith("fly: pure") ? "fly" : ""}"><td>${esc(n)}</td>
      <td class="n">${fmt(e, 4)}</td><td class="n">${a === null || a === undefined ? "–" : pct(a)}</td></tr>`).join("");
}

init();

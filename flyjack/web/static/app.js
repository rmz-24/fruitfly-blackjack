"use strict";
// Spectator view. All cards and choices come from the existing Fly Lab API.
const $ = selector => document.querySelector(selector);
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const state = {
  ready: false, playing: false, task: false, continuous: false, resume: null,
  hand: null, number: 0, mode: "readout", loadingMode: false, history: [],
};

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.error || `The server returned ${response.status}.`);
  }
  return response.json();
}

function controls() {
  const unavailable = !state.ready || state.loadingMode;
  $("#btn-watch").disabled = unavailable;
  $("#btn-watch").innerHTML = state.loadingMode ? "Loading the brain…" : !state.ready ? "Loading the fly…" :
    state.playing ? '<span aria-hidden="true">Ⅱ</span> Pause' :
    state.task ? '<span aria-hidden="true">▶</span> Resume watching' : '<span aria-hidden="true">▶</span> Watch the fly play';
  $("#btn-step").disabled = unavailable;
  $("#btn-step").innerHTML = state.task ? 'Finish hand <span aria-hidden="true">→</span>' : 'One hand <span aria-hidden="true">→</span>';
  $("#btn-step").title = state.task ? "Finish the current hand, then stop playback" : "Watch just one hand";
  $("#brain-mode").disabled = unavailable || state.task;
  $("#player-status").textContent = state.loadingMode ? "Loading brain" : !state.ready ? "Connecting" :
    state.playing ? "At the table" : state.task ? "Paused" : "Ready to play";
  $("#fly-portrait").classList.toggle("thinking", state.playing && $("#stage-think").classList.contains("active"));
}

function narrate(label, message, tableMessage, stage) {
  $("#thought-label").textContent = label;
  $("#narration").textContent = message;
  $("#table-message").textContent = tableMessage;
  for (const name of ["cards", "think", "act"]) {
    const item = $("#stage-" + name);
    item.classList.toggle("active", name === stage);
    if (name === stage) item.setAttribute("aria-current", "step");
    else item.removeAttribute("aria-current");
  }
  controls();
}

// A single playback task owns the hand. Pausing gates every render and request,
// including the response from an already-running brain simulation.
async function checkpoint() {
  if (!state.playing) await new Promise(resolve => { state.resume = resolve; });
}
async function beat(ms = 1100) {
  await sleep(ms * Number($("#speed").value));
  await checkpoint();
}

function card(value, index, dealer, back = false) {
  const element = document.createElement("div");
  element.className = "playing-card";
  element.setAttribute("role", "img");
  if (back) {
    element.classList.add("back");
    element.setAttribute("aria-label", "Face-down card");
    return element;
  }
  // The engine deals values, not suits. Decorative suits stay stable per card.
  const suits = ["♠", "♥", "♣", "♦"];
  const suitNames = ["spades", "hearts", "clubs", "diamonds"];
  const suit = (state.number + index + (dealer ? 1 : 0)) % suits.length;
  const face = value === 1 ? "A" : String(value);
  element.classList.toggle("red", suit === 1 || suit === 3);
  element.setAttribute("aria-label", `${value === 1 ? "Ace" : value} of ${suitNames[suit]}`);
  element.innerHTML = `<span class="card-corner">${face}</span><span class="card-suit">${suits[suit]}</span><span class="card-corner bottom">${face}</span>`;
  return element;
}

function renderCards(selector, values, dealer, hidden = false) {
  const container = $(selector);
  if (container.dataset.hand !== state.hand.id) {
    container.replaceChildren();
    container.dataset.hand = state.hand.id;
  }
  container.querySelectorAll(".back").forEach(element => element.remove());
  for (let i = container.children.length; i < values.length; i++) {
    container.append(card(values[i], i, dealer));
  }
  if (hidden) container.append(card(0, 0, dealer, true));
}

function total(cards) {
  let sum = cards.reduce((a, b) => a + b, 0);
  if (cards.includes(1) && sum + 10 <= 21) sum += 10;
  return sum;
}

function renderHand(reveal = false, dealerCards = state.hand.dealer) {
  const hand = state.hand;
  renderCards("#player-cards", hand.player, false);
  renderCards("#dealer-cards", reveal ? dealerCards : dealerCards.slice(0, 1), true, !reveal);
  $("#player-total").textContent = `${hand.total}${hand.soft ? " · soft" : ""}`;
  $("#player-total").title = hand.soft ? "An ace counts as 11; it can count as 1 if needed." : "Fly’s total";
  $("#dealer-total").textContent = reveal ? total(dealerCards) : `${dealerCards[0] === 1 ? "A" : dealerCards[0]} showing`;
  $("#hand-number").textContent = `HAND ${String(state.number).padStart(2, "0")}`;
}

async function result() {
  const hand = state.hand;
  narrate("THE REVEAL", "Time to turn over the dealer’s cards.", "The dealer reveals its hand.", null);
  for (let n = Math.min(2, hand.dealer.length); n <= hand.dealer.length; n++) {
    await beat(700);
    renderHand(true, hand.dealer.slice(0, n));
  }
  const won = hand.reward > 0, lost = hand.reward < 0;
  const label = won ? "THE FLY WINS" : lost ? "THE DEALER WINS" : "IT’S A TIE";
  let message;
  if (hand.natural) message = won ? "Blackjack! An ace and a ten. A perfect start for a tiny brain." : lost ? "The dealer has blackjack. This one was decided at the deal." : "Both have blackjack. An even match.";
  else if (hand.total > 21) message = `The fly went over 21 with ${hand.total}. That’s a bust — the dealer takes this one.`;
  else if (hand.dealer_total > 21) message = `The dealer went over 21. The fly wins with ${hand.total}!`;
  else message = won ? `${hand.total} beats ${hand.dealer_total}. A little victory for the fly!` : lost ? `The dealer’s ${hand.dealer_total} beats the fly’s ${hand.total}. There’s always another hand.` : `Both finish on ${hand.total}. No winner this time.`;
  narrate(label, message, won ? "This one goes to the fly." : lost ? "This one goes to the dealer." : "A tie. Honours even.", null);
  state.history.unshift({ number: state.number, outcome: won ? "Win" : lost ? "Loss" : "Tie", className: won ? "win" : lost ? "loss" : "tie" });
  state.history = state.history.slice(0, 5);
  $("#recent-empty").hidden = true;
  $("#recent-hands").innerHTML = state.history.map(h => `<li class="${h.className}" aria-label="Hand ${h.number}: ${h.outcome}">${h.number} · ${h.outcome}</li>`).join("");
}

async function playHand() {
  await checkpoint();
  state.hand = await api("/api/hand/new", {});
  await checkpoint();
  state.number++;
  renderHand();
  narrate("A NEW HAND", `The fly has ${state.hand.total}. The dealer shows ${state.hand.dealer[0] === 1 ? "an ace" : state.hand.dealer[0]}.`, "The cards are on the table.", "cards");
  await beat(1500);
  while (!state.hand.done) {
    narrate("A TINY MOMENT OF THOUGHT", state.mode === "spiking" ? "The simulated brain is responding to these cards…" : "The fly’s learned connections are turning these cards into a choice…", "The fly is thinking…", "think");
    const choice = await api("/api/sense", { obs: state.hand.obs, mode: state.mode, model: "trained" });
    await beat(1000);
    const hit = choice.action === 1;
    narrate(hit ? "THE FLY CHOOSES HIT" : "THE FLY CHOOSES STAND", hit ? `With ${state.hand.total}, the fly chooses to take another card.` : `The fly stays at ${state.hand.total}. Now it’s the dealer’s turn.`, hit ? "Hit. One more card, please." : "Stand. Over to the dealer.", "act");
    await beat(1500);
    state.hand = await api("/api/hand/act", { id: state.hand.id, action: choice.action });
    await checkpoint();
    renderHand();
    if (hit) {
      const drawn = state.hand.player.at(-1);
      narrate("ONE MORE CARD", `A${drawn === 1 || drawn === 8 ? "n" : ""} ${drawn === 1 ? "ace" : drawn}. The fly’s total is now ${state.hand.total}.`, `The fly now has ${state.hand.total}.`, "cards");
      await beat(1400);
    }
  }
  await result();
}

function showError(error, startup = false) {
  $("#error-message").textContent = startup ? `Couldn’t connect to Fly Lab. Make sure the Python server is running, then try again. ${error.message}` : `Playback stopped: ${error.message} You can start a fresh hand when you’re ready.`;
  $("#error").hidden = false;
}

async function run(continuous) {
  if (!state.ready || state.loadingMode || state.task) return;
  state.playing = true;
  state.task = true;
  state.continuous = continuous;
  $("#error").hidden = true;
  controls();
  try {
    do {
      await playHand();
      if (state.continuous) await beat(2800);
    } while (state.continuous);
  } catch (error) {
    // Never retry a move automatically: a failed response may follow an applied move.
    state.hand = null;
    narrate("LET’S TAKE A BREATHER", "Playback stopped. Try again to deal a fresh hand.", "The table is taking a break.", null);
    showError(error);
  } finally {
    state.playing = false;
    state.task = false;
    state.resume = null;
    controls();
  }
}

$("#btn-watch").addEventListener("click", () => {
  if (state.playing) {
    state.playing = false;
    controls();
  } else if (state.task) {
    state.playing = true;
    state.resume?.();
    state.resume = null;
    controls();
  } else run(true);
});
$("#btn-step").addEventListener("click", () => {
  if (!state.task) return void run(false);
  state.continuous = false;
  state.playing = true;
  state.resume?.();
  state.resume = null;
  controls();
});
$("#btn-retry").addEventListener("click", () => {
  if (!state.ready) init();
  else run(false);
});

$("#brain-mode").addEventListener("change", async event => {
  const mode = event.target.value;
  if (mode === "readout") {
    state.mode = mode;
    $("#mode-caption").textContent = "Playing with learned responses from recorded brain activity.";
    $("#mode-note").textContent = "The trained fly chooses using recorded neural activity. Live mode runs a new whole-brain simulation for each choice.";
    $("#error").hidden = true;
    return;
  }
  state.loadingMode = true;
  $("#error").hidden = true;
  controls();
  try {
    let status = (await api("/api/spiking/load", {})).status;
    while (status.startsWith("loading") || status.startsWith("warming")) {
      $("#mode-note").textContent = "Preparing the whole-brain simulation. This can take a little while…";
      await sleep(1000);
      status = (await api("/api/spiking")).status;
    }
    if (status !== "ready") throw new Error(status);
    state.mode = mode;
    $("#mode-caption").textContent = "Each choice runs a live whole-brain spiking simulation.";
    $("#mode-note").textContent = "The live brain is ready. Each decision may take a few seconds.";
  } catch (error) {
    event.target.value = state.mode;
    $("#mode-note").textContent = "Live mode couldn’t start. Recorded responses are still available.";
    showError(error);
  } finally {
    state.loadingMode = false;
    controls();
  }
});

async function init() {
  $("#error").hidden = true;
  $("#btn-retry").disabled = true;
  try {
    const meta = await api("/api/meta");
    state.ready = true;
    narrate("READY WHEN YOU ARE", "A fresh deck. A tiny player. Let’s see what this fly has learned.", "A little brain at a big table.", null);
    if (!meta.cuda) {
      $('#brain-mode option[value="spiking"]').disabled = true;
      $("#mode-note").textContent = "The trained fly chooses using recorded neural activity. Live simulation needs a CUDA GPU, which isn’t available on this machine.";
    }
  } catch (error) {
    narrate("THE FLY ISN’T CONNECTED YET", "Start the Fly Lab server, then use Try again below to reconnect.", "Waiting for the fly…", null);
    showError(error, true);
  }
  finally { $("#btn-retry").disabled = false; controls(); }
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden && state.playing) { state.playing = false; controls(); }
});
init();

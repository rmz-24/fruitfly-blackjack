"""Everything the web interface can ask of the fly, independent of HTTP.

Two ways for the fly to decide:

* ``readout``: the trained mushroom body reads one of the *recorded* Kenyon
  cell responses (held-out Poisson trials of the whole-brain simulation).
  Instant.
* ``spiking``: the whole-brain LIF simulation with the learned KC->MBON
  synapses installed (see :mod:`flyjack.closed_loop`) is run live for 200 ms;
  the fly hits if the HIT MBONs out-spike the STAND MBONs. Needs the GPU brain
  to be loaded (~20 s once).
"""

import itertools
import json
import threading
import time
import uuid

import numpy as np

from ..anatomy import annotations, load_anatomy
from ..blackjack import (
    HIT, STAND, Blackjack, action_values, all_observations, expected_return, hand_value,
    is_natural, play_hand,
)
from ..encoding import DEALER_VALUES, SUM_VALUES, OdorEncoder
from ..mushroom_body import KenyonCode, MushroomBody
from ..record import load_responses
from ..strategies import basic_strategy, dealer_rule, make_random, never_bust
from ..train import RESULTS, train

KC_COLUMNS = 96
KC_GROUPS = [("γ lobe", "KCg"), ("α/β lobes", "KCab"), ("α′/β′ lobes", "KCa")]


def _obs(o):
    return (int(o[0]), int(o[1]), bool(o[2]))


def _json_load(name, default=None):
    path = RESULTS / name
    return json.load(open(path)) if path.exists() else default


class ScriptedDeck:
    """``random.Random`` stand-in: returns scripted cards first, then random ones."""

    def __init__(self, cards, seed):
        import random
        self.cards, self.rng = list(cards), random.Random(seed)

    def randint(self, a, b):
        if self.cards:
            c = self.cards.pop(0)
            return 13 if c == 10 else c          # draw() maps 11..13 -> 10
        return self.rng.randint(a, b)


class FlyLab:
    def __init__(self):
        self.anatomy = load_anatomy()
        self.encoder = OdorEncoder(self.anatomy)
        self.code = KenyonCode()
        self.observations = self.code.observations
        _, d = load_responses()
        self.kc_counts = d["kc"]                       # [obs, trial, kc] raw spike counts
        self.pn_counts = d["pn"]
        self.n_trials = self.kc_counts.shape[1]
        log = _json_load("training.json")
        self.best_seed = log["best_seed"] if log else 0
        self.models = {"trained": MushroomBody.load(RESULTS / f"mushroom_body_seed{self.best_seed}.npz")}
        self.closed = _json_load("closed_loop.json")
        self._kc_layout()
        self._pn_layout()
        self.hands = {}
        self.rng = np.random.default_rng()
        self.gpu_lock = threading.Lock()
        self.spiking = None
        self.spiking_status = "not loaded"
        self.spike_seed = itertools.count(10_000)
        self.training = None
        self._policy_cache = {}

    # ------------------------------------------------------------------ layout
    def _kc_layout(self):
        """Order KCs by lobe system, then hemisphere, for the KC map."""
        ann = annotations().set_index("index")
        types = ann.loc[self.anatomy.kc, "cell_type"].fillna("").to_numpy()
        sides = ann.loc[self.anatomy.kc, "side"].fillna("").to_numpy()
        group = np.array([next((g for g, (_, pre) in enumerate(KC_GROUPS[:2]) if t.startswith(pre)), 2)
                          for t in types])
        self.kc_order = np.lexsort((np.arange(len(types)), sides != "left", group))
        rows, start = [], 0
        for g, (label, _) in enumerate(KC_GROUPS):
            n = int((group == g).sum())
            rows.append({"label": label, "start": start, "count": n,
                         "left": int(((group == g) & (sides == "left")).sum())})
            start += n
        self.kc_groups = rows
        self.kc_types = types[self.kc_order]

    def _pn_layout(self):
        """Glomerulus channels grouped by the feature pool they encode."""
        enc = self.encoder
        used = set(enc.sum_pool) | set(enc.dealer_pool) | set(enc.ace_pool[False]) | set(enc.ace_pool[True])
        unused = [g for g in self.anatomy.glomeruli if g not in used]
        self.pn_rows = [
            {"label": "player sum", "channels": list(enc.sum_pool)},
            {"label": "dealer upcard", "channels": list(enc.dealer_pool)},
            {"label": "hard hand", "channels": list(enc.ace_pool[False])},
            {"label": "soft hand (usable ace)", "channels": list(enc.ace_pool[True])},
            {"label": "not used by the task", "channels": unused},
        ]
        glom = self.anatomy.pn_glomerulus
        self.channel_pn_cols = {g: np.flatnonzero(glom == g) for g in self.anatomy.glomeruli}

    # ------------------------------------------------------------------ meta
    def meta(self):
        mb = self.models["trained"]
        pref = (mb.w[HIT] - mb.w[STAND])[self.kc_order]
        evaluation = _json_load("evaluation.json", {})
        return {
            "observations": [list(o) for o in self.observations],
            "sum_values": SUM_VALUES, "dealer_values": DEALER_VALUES,
            "encoder": {"sum_width": self.encoder.sum_width, "dealer_width": self.encoder.dealer_width,
                        "rate_hz": self.encoder.rate},
            "pn_rows": [{**r, "channels": [str(c) for c in r["channels"]]} for r in self.pn_rows],
            "kc": {"n": len(self.anatomy.kc), "columns": KC_COLUMNS, "groups": self.kc_groups,
                   "preference": np.round(pref, 4).tolist()},
            "counts": {"neurons": 138_639, "pn": len(self.anatomy.pn), "kc": len(self.anatomy.kc),
                       "mbon": len(self.anatomy.mbon), "dan": len(self.anatomy.dan),
                       "dn": len(self.anatomy.dn)},
            "n_trials": self.n_trials, "test_trials": self.code.test_trials.tolist(),
            "evaluation": evaluation.get("table", []),
            "training": _json_load("training.json"),
            "closed_loop": ({k: self.closed[k] for k in ("gain", "pools", "base", "sweep", "play")}
                            if self.closed else None),
            "cuda": self._cuda(),
        }

    @staticmethod
    def _cuda():
        import torch
        return torch.cuda.is_available()

    def policy(self, model="trained"):
        """Per-state choice probabilities and values of the fly vs basic strategy."""
        if model in self._policy_cache:
            return self._policy_cache[model]
        mb = self.models[model]
        p = mb.hit_probability(self.code)
        q = mb.q_table(self.code)
        exact = action_values()
        spk = {}
        if self.closed and model == "trained":
            spk = {_obs(r[:3]): r[3] for r in self.closed["p_hit"]}
        states = []
        for o in self.observations:
            states.append({"obs": list(o), "p_hit": p[o], "q_hit": q[o][HIT], "q_stand": q[o][STAND],
                           "basic": int(basic_strategy(o)), "p_hit_spiking": spk.get(o),
                           "true_q_hit": exact[o][HIT], "true_q_stand": exact[o][STAND]})
        out = {"model": model, "states": states,
               "expected_return": expected_return(lambda o: p[o]),
               "expected_return_spiking": expected_return(lambda o: spk[o]) if spk else None,
               "basic_return": expected_return(basic_strategy)}
        self._policy_cache[model] = out
        return out

    # ------------------------------------------------------------------ sensing
    def _pn_channel_rates(self, pn_counts):
        """Mean firing rate (Hz) per glomerulus channel."""
        return {str(g): float(pn_counts[cols].mean() / 0.2) for g, cols in self.channel_pn_cols.items()}

    def _describe(self, obs, kc, pn, mb):
        """Shared decision payload from one KC/PN spike-count vector."""
        kc = kc.astype(np.float32)
        norm = max(float(np.linalg.norm(kc)), 1e-6)
        m = mb.mbon(kc / norm)
        pref = mb.w[HIT] - mb.w[STAND]
        contrib = pref * kc / norm
        ordered = kc[self.kc_order]
        active = np.flatnonzero(ordered)
        top = np.argsort(-np.abs(contrib))[:12]
        inv = np.empty_like(self.kc_order)
        inv[self.kc_order] = np.arange(len(self.kc_order))
        return {
            "obs": list(obs),
            "channels": [str(c) for c in self.encoder.channels(obs)],
            "pn_rates": self._pn_channel_rates(pn),
            "kc_active": active.tolist(), "kc_spikes": ordered[active].astype(int).tolist(),
            "kc_fraction": float((kc > 0).mean()),
            "mbon": {"hit": float(m[HIT]), "stand": float(m[STAND]),
                     "bias_hit": float(mb.b[HIT]), "bias_stand": float(mb.b[STAND])},
            "readout_action": int(HIT if m[HIT] > m[STAND] else STAND),
            "top_kcs": [{"pos": int(inv[k]), "type": str(self.kc_types[inv[k]]),
                         "spikes": int(kc[k]), "push": float(contrib[k])} for k in top if kc[k] > 0],
            "basic": int(basic_strategy(obs)),
        }

    def sense(self, obs, mode="readout", trial=None, model="trained"):
        obs = _obs(obs)
        if mode == "spiking":
            return self._sense_spiking(obs)
        i = self.observations.index(obs)
        if trial is None:
            trial = int(self.rng.choice(self.code.test_trials))
        out = self._describe(obs, self.kc_counts[i, trial], self.pn_counts[i, trial], self.models[model])
        out.update(mode="readout", trial=int(trial), action=out["readout_action"],
                   held_out=bool(trial in self.code.test_trials))
        return out

    # ------------------------------------------------------------------ spiking brain
    def load_spiking(self):
        if self.spiking is not None or self.spiking_status.startswith("loading"):
            return self.spiking_status
        if not self.closed:
            self.spiking_status = "unavailable: run python -m flyjack.closed_loop first"
            return self.spiking_status
        if not self._cuda():
            self.spiking_status = "unavailable: no CUDA GPU (spiking mode would take minutes per decision)"
            return self.spiking_status

        def work():
            from ..closed_loop import SpikingFly
            try:
                self.spiking_status = "loading connectome (15M synapses) ..."
                base = {HIT: self.closed["base"]["hit"], STAND: self.closed["base"]["stand"]}
                fly = SpikingFly(self.models["trained"], self.closed["gain"], base, "cuda")
                self.spiking_status = "warming up the GPU ..."
                with self.gpu_lock:
                    fly.sense([(12, 2, False)], seed=1)
                self.spiking = fly
                self.spiking_status = "ready"
            except Exception as e:                        # surfaced in the UI
                self.spiking_status = f"failed: {e}"

        self.spiking_status = "loading ..."
        threading.Thread(target=work, daemon=True).start()
        return self.spiking_status

    def _sense_spiking(self, obs, with_raster=True):
        fly = self.spiking
        if fly is None:
            raise RuntimeError(f"spiking brain not ready ({self.spiking_status})")
        groups = {"pn": self.encoder.neurons(obs), "kc": fly.anatomy.kc, "hit": fly.pools[HIT],
                  "stand": fly.pools[STAND], "dn": fly.anatomy.dn}
        idx = np.concatenate(list(groups.values())) if with_raster else None
        t0 = time.time()
        with self.gpu_lock:
            rates = fly.encoder.rates([obs], fly.brain.n)
            counts, ras = fly.brain.run(rates, 200.0, seed=next(self.spike_seed), record_raster=idx)
        sim_s = time.time() - t0
        counts = counts[0].numpy()
        n_hit = int(counts[fly.pools[HIT]].sum())
        n_stand = int(counts[fly.pools[STAND]].sum())
        out = self._describe(obs, counts[fly.anatomy.kc], counts[fly.anatomy.pn], fly.mb)
        out.update(mode="spiking", action=int(HIT if n_hit > n_stand else STAND),
                   spikes={"hit": n_hit, "stand": n_stand}, sim_seconds=round(sim_s, 2))
        # raster rows: neuron -> row within its group (KCs in map order, only active ones)
        raster = {}
        kc_pos = np.empty(len(fly.anatomy.kc), int)
        kc_pos[self.kc_order] = np.arange(len(self.kc_order))
        for name, ids in groups.items():
            m = np.isin(ras[:, 1], ids)
            neuron, t = ras[m, 1].astype(np.int64), ras[m, 2]
            if name == "kc":
                row = kc_pos[np.searchsorted(fly.anatomy.kc, neuron)]
            else:
                row = np.searchsorted(np.sort(ids), neuron)
            raster[name] = {"n": int(len(ids)), "row": row.tolist(), "t": np.round(t, 1).tolist()}
        out["raster"] = raster
        out["dn_active"] = int((counts[fly.anatomy.dn] > 0).sum())
        return out

    def probe(self, obs, mode="readout", trials=16, model="trained"):
        """Many trials of one state: how reliable is the fly's choice?"""
        obs = _obs(obs)
        if mode == "spiking":
            fly = self.spiking
            if fly is None:
                raise RuntimeError(f"spiking brain not ready ({self.spiking_status})")
            with self.gpu_lock:
                r = fly.sense([obs] * trials, seed=next(self.spike_seed) * 1000)
            rows = [{"trial": k, "hit": int(h), "stand": int(s), "action": int(a),
                     "readout_action": int(ra)}
                    for k, (h, s, a, ra) in enumerate(zip(r["n_hit"], r["n_stand"], r["spiking"], r["readout"]))]
        else:
            i = self.observations.index(obs)
            mb = self.models[model]
            x = self.code.x[i]                                  # all recorded trials
            m = mb.mbon(x)
            rows = [{"trial": k, "hit": float(m[k, HIT]), "stand": float(m[k, STAND]),
                     "action": int(m[k, HIT] > m[k, STAND]), "held_out": bool(k in self.code.test_trials)}
                    for k in range(len(x))]
        mean_kc = self.kc_counts[self.observations.index(obs)].mean(0)[self.kc_order]
        return {"obs": list(obs), "mode": mode, "trials": rows, "basic": int(basic_strategy(obs)),
                "kc_mean": np.round(mean_kc, 2).tolist(),
                "p_hit": float(np.mean([r["action"] for r in rows]))}

    # ------------------------------------------------------------------ hands
    def new_hand(self, player=None, dealer=None, seed=None):
        seed = int(self.rng.integers(1 << 31)) if seed is None else int(seed)
        env = Blackjack(seed=seed)
        if player or dealer:
            # deal order in Blackjack.reset(): player, player, dealer up, dealer hole;
            # missing cards (None) stay random
            wanted = [*(list(player or []) + [None] * 2)[:2], *(list(dealer or []) + [None] * 2)[:2]]
            deck = ScriptedDeck([], seed)
            deck.cards = [c if c else min(deck.rng.randint(1, 13), 10) for c in wanted]
            env.rng = deck
        obs = env.reset()
        hid = uuid.uuid4().hex[:12]
        self.hands[hid] = env
        if len(self.hands) > 500:
            self.hands.pop(next(iter(self.hands)))
        return self._hand_state(hid, env, obs)

    def act(self, hid, action):
        env = self.hands[hid]
        obs, _, _ = env.step(int(action))
        return self._hand_state(hid, env, obs)

    @staticmethod
    def _hand_state(hid, env, obs):
        total, soft = hand_value(env.player)
        state = {"id": hid, "player": env.player, "total": total, "soft": soft,
                 "done": env.done, "obs": list(obs) if obs else None,
                 "natural": is_natural(env.player) or is_natural(env.dealer)}
        if env.done:
            state.update(dealer=env.dealer, dealer_total=hand_value(env.dealer)[0], reward=env.reward)
        else:
            state.update(dealer=[env.dealer[0]])
        return state

    # ------------------------------------------------------------------ tournament
    def tournament(self, hands=2000, seed=None, model="trained"):
        """Everyone plays the same number of hands; cumulative winnings."""
        seed = int(self.rng.integers(1 << 31)) if seed is None else int(seed)
        mb = self.models[model]
        rng = np.random.default_rng(seed)
        players = {
            "fly": lambda o: mb.act(self.code.sample(o, rng, "test"), rng),
            "basic strategy": basic_strategy,
            "dealer rule": dealer_rule,
            "never bust": never_bust,
            "random": make_random(seed),
        }
        points = np.unique(np.linspace(0, hands, min(hands, 200) + 1).astype(int))
        out = {}
        for name, pol in players.items():
            env = Blackjack(seed=seed)            # same shoe seed for everyone
            r = np.array([play_hand(env, pol) for _ in range(hands)])
            cum = np.concatenate([[0], np.cumsum(r)])
            out[name] = {"cumulative": cum[points].tolist(), "mean": float(r.mean()),
                         "ci95": float(1.96 * r.std() / np.sqrt(hands)),
                         "wins": int((r > 0).sum()), "pushes": int((r == 0).sum()),
                         "losses": int((r < 0).sum())}
        return {"hands": hands, "seed": seed, "points": points.tolist(), "players": out}

    # ------------------------------------------------------------------ training
    def start_training(self, hands=300_000, eta=0.05, epsilon=0.3, seed=0):
        if self.training and self.training["running"]:
            return self.training_status()
        job = {"running": True, "history": [], "hands": hands, "stop": False,
               "config": dict(hands=hands, eta=eta, epsilon=epsilon, seed=seed), "error": None}
        self.training = job

        def cb(entry, mb):
            job["history"].append(entry)
            return job["stop"]

        def work():
            try:
                mb, _ = train(self.code, hands, eta=eta, epsilon=epsilon, seed=seed,
                              eval_every=max(hands // 100, 500), log=lambda *_: None, callback=cb)
                self.models["new"] = mb
                self._policy_cache.pop("new", None)
            except Exception as e:
                job["error"] = str(e)
            job["running"] = False

        threading.Thread(target=work, daemon=True).start()
        return self.training_status()

    def stop_training(self):
        if self.training:
            self.training["stop"] = True
        return self.training_status()

    def training_status(self):
        t = self.training
        if not t:
            return {"running": False, "history": [], "available": "new" in self.models}
        return {"running": t["running"], "history": t["history"], "config": t["config"],
                "error": t["error"], "available": "new" in self.models}

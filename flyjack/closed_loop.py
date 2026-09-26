"""Closed loop: the learned synapses are written into the connectome and the
fly plays Blackjack purely as a spiking brain.

The readout learned in :mod:`flyjack.train` becomes real synapses from every
Kenyon cell onto two pairs of real mushroom-body output neurons, replacing
their native KC inputs:

* HIT pool:   MBON01 (MBON-gamma5beta'2a, left + right hemisphere)
* STAND pool: MBON03 (MBON-beta'2mp, left + right)

Both types are silent in the task and receive ~85% of their input from KCs, so
the learned synapses dominate them. KCs are excitatory, so all learned weights
are positive: the readout's preference ``D_k = w_hit,k - w_stand,k`` (with the
bias folded into a uniform per-KC term, see ``learned_synapses``) is split
into its positive part onto the HIT pool and its negative part onto the STAND
pool, on top of a baseline weight that keeps both pools in their firing
range. The two MBON types get different native non-KC input (MBON03 more net
excitation), so the HIT pool's baseline is calibrated homeostatically: with
no learned preference, both pools must fire equally on average. For every
decision the card odor is presented for 200 ms and the
fly hits if the HIT pool fires more spikes than the STAND pool.

    python -m flyjack.closed_loop --calibrate      # baseline + gain scan
    python -m flyjack.closed_loop --trials 8 --hands 4000
"""

import argparse
import json
import time

import numpy as np

from .anatomy import load_anatomy
from .blackjack import HIT, STAND, Blackjack, all_observations, expected_return
from .brain import task_brain
from .encoding import OdorEncoder
from .mushroom_body import MushroomBody
from .record import DURATION_MS
from .strategies import basic_strategy
from .train import RESULTS

POOLS = {HIT: "MBON01", STAND: "MBON03"}
BASE_STAND = 4.0      # baseline KC -> STAND pool weight (synapse-count units)


def pool_neurons(anatomy):
    return {a: anatomy.mbon[anatomy.mbon_type == t] for a, t in POOLS.items()}


def kc_preference(mb, kc_counts):
    """Per-KC preference for HIT over STAND, in readout units per spike.

    The readout acts on L2-normalised KC vectors plus a bias:
    ``(w_H - w_S) . c / |c| + (b_H - b_S)``. Synapses only see raw spike counts
    ``c``, so the bias is folded in using ``|c| ~ beta * sum(c)`` (beta fitted
    on the recorded responses; decisions agree with the exact readout on 99.7%
    of held-out trials).
    """
    norm = np.linalg.norm(kc_counts, axis=-1)
    total = kc_counts.sum(-1)
    beta = float((norm / np.maximum(total, 1)).mean())
    return mb.w[HIT] - mb.w[STAND] + (mb.b[HIT] - mb.b[STAND]) * beta


def learned_synapses(pref, anatomy, gain, base):
    """``(pre, post, weight)`` edges KC -> pool neurons (weights in synapses).
    ``base`` maps each action to its pool's baseline weight."""
    pools = pool_neurons(anatomy)
    w = {HIT: base[HIT] + gain * np.maximum(pref, 0),
         STAND: base[STAND] + gain * np.maximum(-pref, 0)}
    pre, post, weight = [], [], []
    for a, neurons in pools.items():
        for m in neurons:
            pre.append(anatomy.kc)
            post.append(np.full(len(anatomy.kc), m))
            weight.append(w[a].astype(np.float32))
    return np.concatenate(pre), np.concatenate(post), np.concatenate(weight)


def closed_loop_brain(pref, gain, base, device="cuda", anatomy=None):
    anatomy = anatomy or load_anatomy()
    is_kc = np.zeros(anatomy.kc.max() + 1, bool)
    is_kc[anatomy.kc] = True
    targets = np.concatenate(list(pool_neurons(anatomy).values()))

    def native_kc_to_pools(pre, post):
        return np.isin(post, targets) & (pre < len(is_kc)) & is_kc[np.minimum(pre, len(is_kc) - 1)]

    return task_brain(extra=learned_synapses(pref, anatomy, gain, base), device=device,
                      drop=native_kc_to_pools)


def calibrate_baseline(base_stand=BASE_STAND, device="cuda", log=print):
    """Homeostatic set point: the HIT-pool baseline at which both pools fire
    equally often, on average over card odors, before any learning."""
    anatomy = load_anatomy()
    encoder, pools = OdorEncoder(anatomy), pool_neurons(anatomy)
    obs = all_observations()[::2]
    zero = np.zeros(len(anatomy.kc), np.float32)

    def imbalance(base_hit):
        brain = closed_loop_brain(zero, 0.0, {HIT: base_hit, STAND: base_stand}, device, anatomy)
        c = brain.run(encoder.rates(obs, brain.n), DURATION_MS, seed=99).numpy()
        h, s = c[:, pools[HIT]].sum(1).mean(), c[:, pools[STAND]].sum(1).mean()
        log(f"  HIT baseline {base_hit:.3f}: pool spikes hit {h:.2f} stand {s:.2f}")
        return h - s

    lo, hi = base_stand, 3 * base_stand
    for _ in range(8):
        mid = (lo + hi) / 2
        if imbalance(mid) < 0:
            lo = mid
        else:
            hi = mid
    return {HIT: (lo + hi) / 2, STAND: base_stand}


class SpikingFly:
    """Plays Blackjack with nothing but the spiking whole-brain model."""

    def __init__(self, mb, gain, base, device="cuda"):
        from .record import load_responses
        _, d = load_responses()
        self.anatomy = load_anatomy()
        self.pref = kc_preference(mb, d["kc"].astype(np.float32))
        self.mb, self.gain, self.base = mb, gain, base
        self.brain = closed_loop_brain(self.pref, gain, base, device, self.anatomy)
        self.encoder = OdorEncoder(self.anatomy)
        self.pools = pool_neurons(self.anatomy)

    def sense(self, observations, seed, raster=None):
        """Present each card odor for 200 ms. Returns spiking decisions, pool
        spike counts, the readout's decision on the same trial's KC spikes,
        and optionally a raster."""
        rates = self.encoder.rates(observations, self.brain.n)
        out = self.brain.run(rates, DURATION_MS, seed=seed, record_raster=raster)
        counts, ras = out if raster is not None else (out, None)
        counts = counts.numpy()
        n_hit = counts[:, self.pools[HIT]].sum(1)
        n_stand = counts[:, self.pools[STAND]].sum(1)
        kc = counts[:, self.anatomy.kc].astype(np.float32)
        x = kc / np.maximum(np.linalg.norm(kc, axis=1, keepdims=True), 1e-6)
        readout = np.where((self.mb.mbon(x)[:, HIT] > self.mb.mbon(x)[:, STAND]), HIT, STAND)
        spiking = np.where(n_hit > n_stand, HIT, STAND)
        return dict(spiking=spiking, readout=readout, n_hit=n_hit, n_stand=n_stand,
                    kc_active=(kc > 0).mean(1), raster=ras)

    def state_sweep(self, trials, seed=0, log=print):
        """Every observation x ``trials`` fresh Poisson trials."""
        obs = all_observations()
        res = []
        for t in range(trials):
            t0 = time.time()
            res.append(self.sense(obs, seed=seed + 7919 * (t + 1)))
            log(f"  sweep trial {t + 1}/{trials}: {time.time() - t0:.0f}s")
        stack = lambda k: np.stack([r[k] for r in res], 1)          # [obs, trials]
        return obs, {k: stack(k) for k in ("spiking", "readout", "n_hit", "n_stand", "kc_active")}

    def play(self, hands, batch=500, seed=0, log=print):
        """Play ``hands`` hands; all hands in a batch make their decisions
        in parallel, each through its own simulated trial."""
        rewards, decisions, agree = [], 0, 0
        step_seed = seed
        for start in range(0, hands, batch):
            envs = [Blackjack(seed=seed * 1_000_003 + start + i) for i in range(min(batch, hands - start))]
            obs = [e.reset() for e in envs]
            live = [i for i, o in enumerate(obs) if o is not None]
            t0 = time.time()
            while live:
                step_seed += 1
                r = self.sense([obs[i] for i in live], seed=step_seed)
                decisions += len(live)
                agree += int((r["spiking"] == r["readout"]).sum())
                nxt = []
                for i, a in zip(live, r["spiking"]):
                    obs[i], _, done = envs[i].step(int(a))
                    if not done:
                        nxt.append(i)
                live = nxt
            rewards += [e.reward for e in envs]
            log(f"  hands {start + len(envs)}/{hands}: mean reward {np.mean(rewards):+.4f} "
                f"({time.time() - t0:.0f}s)")
        rewards = np.array(rewards)
        return dict(hands=hands, mean=float(rewards.mean()),
                    ci95=float(1.96 * rewards.std() / np.sqrt(hands)),
                    wins=float((rewards > 0).mean()), losses=float((rewards < 0).mean()),
                    decisions=decisions, readout_agreement=agree / max(decisions, 1))

    def record_hand(self, seed):
        """Spike raster of the fly playing one full hand (for the figure)."""
        env = Blackjack(seed=seed)
        o = env.reset()
        while o is None:
            o = env.reset()
        rec = {"pn": self.anatomy.pn, "kc": self.anatomy.kc, "hit": self.pools[HIT],
               "stand": self.pools[STAND], "dn": self.anatomy.dn}
        idx = np.concatenate(list(rec.values()))
        steps = []
        while o is not None:
            r = self.sense([o], seed=seed + len(steps), raster=idx)
            a = int(r["spiking"][0])
            steps.append(dict(obs=list(map(int, o)), action=a, n_hit=int(r["n_hit"][0]),
                              n_stand=int(r["n_stand"][0]), raster=r["raster"][:, 1:]))
            o, _, _ = env.step(a)
        return dict(player=env.player, dealer=env.dealer, reward=env.reward, steps=steps,
                    groups={k: v.tolist() for k, v in rec.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gain", type=float, default=200.0,
                    help="synapses per readout unit of KC preference")
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--hands", type=int, default=4000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--calibrate", action="store_true",
                    help="scan gains on one sweep of all states and exit")
    a = ap.parse_args()

    log = json.load(open(RESULTS / "training.json"))
    mb = MushroomBody.load(RESULTS / f"mushroom_body_seed{log['best_seed']}.npz")

    base_file = RESULTS / "closed_loop_baseline.json"
    if a.calibrate or not base_file.exists():
        base = calibrate_baseline(device=a.device)
        json.dump({"hit": base[HIT], "stand": base[STAND]}, open(base_file, "w"))
    b = json.load(open(base_file))
    base = {HIT: b["hit"], STAND: b["stand"]}
    print(f"pool baselines: HIT {base[HIT]:.3f}, STAND {base[STAND]:.3f}")

    if a.calibrate:
        for gain in [5, 10, 20, 50, 100, 200]:
            fly = SpikingFly(mb, gain, base, a.device)
            obs, s = fly.state_sweep(1, log=lambda *_: None)
            sp, ro = s["spiking"][:, 0], s["readout"][:, 0]
            basic = np.array([basic_strategy(o) for o in obs])
            ties = (s["n_hit"] == s["n_stand"]).mean()
            print(f"gain {gain:5.1f}: spiking = readout {np.mean(sp == ro):.3f}, = basic "
                  f"{np.mean(sp == basic):.3f}; pool spikes hit {s['n_hit'].mean():.1f} "
                  f"stand {s['n_stand'].mean():.1f}; ties {ties:.3f}", flush=True)
        return

    fly = SpikingFly(mb, a.gain, base, a.device)
    print(f"state sweep: 280 observations x {a.trials} trials")
    obs, s = fly.state_sweep(a.trials)
    p_hit = {o: float((s["spiking"][i] == HIT).mean()) for i, o in enumerate(obs)}
    basic = np.array([basic_strategy(o) for o in obs])
    sweep = dict(
        trials=a.trials,
        spiking_vs_readout=float((s["spiking"] == s["readout"]).mean()),
        spiking_vs_basic=float((s["spiking"] == basic[:, None]).mean()),
        majority_vs_basic=float(np.mean([(p_hit[o] > 0.5) == bool(basic_strategy(o)) for o in obs])),
        exact_ev_spiking=expected_return(lambda o: p_hit[o]),
        ties=float((s["n_hit"] == s["n_stand"]).mean()),
        pool_spikes=dict(hit=float(s["n_hit"].mean()), stand=float(s["n_stand"].mean())),
        kc_active=float(s["kc_active"].mean()),
    )
    print(json.dumps(sweep, indent=1))

    print(f"playing {a.hands} hands in the spiking brain")
    play = fly.play(a.hands)
    print(json.dumps(play, indent=1))

    hand = fly.record_hand(seed=3)
    np.savez_compressed(RESULTS / "closed_loop_hand.npz",
                        meta=json.dumps({k: v for k, v in hand.items() if k != "steps"} |
                                        {"steps": [{k: v for k, v in st.items() if k != "raster"}
                                                   for st in hand["steps"]]}),
                        **{f"raster{i}": st["raster"] for i, st in enumerate(hand["steps"])})
    json.dump(dict(gain=a.gain, pools=POOLS, base=b, best_seed=log["best_seed"], sweep=sweep,
                   p_hit=[[*o, p] for o, p in p_hit.items()], play=play),
              open(RESULTS / "closed_loop.json", "w"), indent=1)
    print(f"saved {RESULTS / 'closed_loop.json'}")


if __name__ == "__main__":
    main()

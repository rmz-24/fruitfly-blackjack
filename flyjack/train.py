"""Teach the fly Blackjack: dopamine-gated learning in the mushroom body.

The fly plays hands against the environment. Each decision state is sensed as
a card odor, whose Kenyon cell response is one of the recorded noisy trials of
the whole-brain simulation; the MBON pools choose, and dopamine reports the
reward prediction error after every action.

    python -m flyjack.train --hands 300000 --seeds 5
"""

import argparse
import json
import time

import numpy as np

from .anatomy import ROOT
from .blackjack import Blackjack, expected_return
from .mushroom_body import KenyonCode, MushroomBody
from .strategies import basic_strategy

RESULTS = ROOT / "results"


def agreement(p_hit):
    """Fraction of decision states where the fly's majority choice is basic strategy's."""
    return float(np.mean([(p_hit[o] > 0.5) == bool(basic_strategy(o)) for o in p_hit]))


def evaluate(mb, code):
    p = mb.hit_probability(code, "test")
    return {"expected_return": expected_return(lambda o: p[o]), "agreement": agreement(p)}


def train(code, hands=300_000, eta=0.05, eta_final=0.002, epsilon=0.3, epsilon_final=0.02,
          seed=0, eval_every=5_000, log=print, callback=None):
    """Play ``hands`` hands, learning online. Learning rate and exploration
    decay geometrically from their initial to their final values.
    ``callback(entry, mb)`` is called after every evaluation; returning True stops."""
    rng = np.random.default_rng(seed)
    env = Blackjack(seed=seed)
    mb = MushroomBody(code.n_kc, eta=eta)
    history = []
    t0 = time.time()
    for h in range(hands + 1):
        if h % eval_every == 0:
            ev = evaluate(mb, code)
            history.append({"hands": h, **ev})
            if callback is not None and callback(history[-1], mb):
                break
            if h % (eval_every * 10) == 0:
                log(f"  seed {seed} hand {h:7d}: EV {ev['expected_return']:+.4f}  "
                    f"agreement {ev['agreement']:.3f}  ({time.time() - t0:.0f}s)")
        if h == hands:
            break
        frac = h / hands
        eta_t = eta * (eta_final / eta) ** frac
        eps_t = epsilon * (epsilon_final / epsilon) ** frac

        obs = env.reset()
        if obs is None:                       # natural: no decision to make
            continue
        x = code.sample(obs, rng)
        while True:
            a = mb.act(x, rng, eps_t)
            obs, r, done = env.step(a)
            if done:
                target, x_next = r, None
            else:
                x_next = code.sample(obs, rng)
                target = r + mb.mbon(x_next).max()
            mb.dopamine(x, a, target - mb.mbon(x)[a], eta_t)
            if done:
                break
            x = x_next
    return mb, history


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", type=int, default=300_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--eta-final", type=float, default=0.002)
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--epsilon-final", type=float, default=0.02)
    a = ap.parse_args()

    code = KenyonCode()
    RESULTS.mkdir(exist_ok=True)
    runs = []
    for seed in range(a.seeds):
        mb, history = train(code, a.hands, a.eta, a.eta_final, a.epsilon, a.epsilon_final, seed)
        final = history[-1]
        print(f"seed {seed}: EV {final['expected_return']:+.4f}, agreement {final['agreement']:.3f}")
        mb.save(RESULTS / f"mushroom_body_seed{seed}.npz", seed=seed)
        runs.append({"seed": seed, "history": history})
    best = max(runs, key=lambda r: r["history"][-1]["expected_return"])
    with open(RESULTS / "training.json", "w") as f:
        json.dump({"config": vars(a), "best_seed": best["seed"], "runs": runs}, f)
    evs = [r["history"][-1]["expected_return"] for r in runs]
    print(f"final EV over {a.seeds} seeds: {np.mean(evs):+.4f} +- {np.std(evs):.4f} "
          f"(basic strategy {expected_return(basic_strategy):+.4f})")


if __name__ == "__main__":
    main()

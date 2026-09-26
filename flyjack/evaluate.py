"""Evaluate the trained fly against reference strategies.

Every policy is scored two ways:

* **exact** expected reward per hand (dynamic programming over the infinite
  deck; for the fly, its noisy choice probabilities on held-out trials);
* **simulated**: ``--hands`` Monte Carlo hands with a 95% confidence interval,
  the fly sensing each state through a random held-out neural trial.

A control trains the same dopamine rule on the projection-neuron code (the
input *to* the mushroom body), to measure what the Kenyon cell expansion buys.

    python -m flyjack.evaluate --hands 200000
"""

import argparse
import json

import numpy as np

from .blackjack import HIT, STAND, Blackjack, expected_return, play_hand
from .mushroom_body import KenyonCode, MushroomBody
from .strategies import basic_strategy, dealer_rule, never_bust
from .train import RESULTS, agreement, train


def simulate(policy_factory, hands, seed):
    env, policy = Blackjack(seed=seed), policy_factory()
    r = np.array([play_hand(env, policy) for _ in range(hands)])
    return float(r.mean()), float(1.96 * r.std() / np.sqrt(hands))


def fly_policy(mb, code, seed):
    """The fly as a player: each decision uses a fresh held-out neural trial."""
    rng = np.random.default_rng(seed)
    return lambda obs: mb.act(code.sample(obs, rng, "test"), rng)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()

    log = json.load(open(RESULTS / "training.json"))
    code = KenyonCode()
    fly = MushroomBody.load(RESULTS / f"mushroom_body_seed{log['best_seed']}.npz")
    p_fly = fly.hit_probability(code)
    seeds_ev = [r["history"][-1]["expected_return"] for r in log["runs"]]

    # Control: same learning rule and budget, reading projection neurons directly
    pn_code = KenyonCode(population="pn")
    pn_evs, pn_agree = [], []
    for s in range(len(log["runs"])):
        mb, hist = train(pn_code, log["config"]["hands"], seed=s, eval_every=log["config"]["hands"],
                         log=lambda *_: None)
        pn_evs.append(hist[-1]["expected_return"])
        pn_agree.append(hist[-1]["agreement"])

    rows = [
        ("random", lambda: 0.5, lambda: (lambda o: HIT if np.random.random() < 0.5 else STAND)),
        ("always stand", lambda: STAND, lambda: (lambda o: STAND)),
        ("never bust", None, lambda: never_bust),
        ("dealer rule (hit < 17)", None, lambda: dealer_rule),
        ("fly: projection-neuron readout (control)", None, None),
        ("fly: mushroom body", None, lambda: fly_policy(fly, code, a.seed)),
        ("basic strategy (optimal)", None, lambda: basic_strategy),
    ]
    np.random.seed(a.seed)
    table = []
    for name, const, factory in rows:
        if name.startswith("fly: projection"):
            table.append(dict(name=name, exact=float(np.mean(pn_evs)), exact_sd=float(np.std(pn_evs)),
                              agreement=float(np.mean(pn_agree))))
            continue
        if name == "fly: mushroom body":
            exact, agree = expected_return(lambda o: p_fly[o]), agreement(p_fly)
        else:
            pol = (lambda o: const()) if const else factory()
            exact = expected_return(pol)
            agree = None if name == "random" else agreement({o: float(pol(o)) for o in p_fly})
        mean, ci = simulate(factory, a.hands, a.seed)
        row = dict(name=name, exact=exact, agreement=agree, simulated=mean, ci95=ci)
        if name == "fly: mushroom body":
            row.update(exact_sd=float(np.std(seeds_ev)), seeds=len(seeds_ev),
                       mean_over_seeds=float(np.mean(seeds_ev)))
        table.append(row)

    noisy = sum(0 < p < 1 for p in p_fly.values())
    summary = dict(hands=a.hands, best_seed=log["best_seed"], table=table,
                   p_hit=[[*o, p] for o, p in p_fly.items()], noisy_states=noisy)
    json.dump(summary, open(RESULTS / "evaluation.json", "w"), indent=1)

    print(f"{'policy':42s} {'exact EV':>9s} {'simulated (95% CI)':>21s} {'= basic':>8s}")
    for r in table:
        sim = f"{r['simulated']:+.4f} +- {r['ci95']:.4f}" if "simulated" in r else ""
        agree = "-" if r["agreement"] is None else f"{r['agreement']:.1%}"
        print(f"{r['name']:42s} {r['exact']:+9.4f} {sim:>21s} {agree:>8s}")
    print(f"states where neural noise makes the fly's choice vary: {noisy}/280")


if __name__ == "__main__":
    main()

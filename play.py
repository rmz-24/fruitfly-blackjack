"""Watch the trained fly play Blackjack in the terminal.

    python play.py                 # mushroom-body readout on recorded neural trials (instant)
    python play.py --spiking       # live whole-brain spiking simulation (GPU recommended)
    python play.py --hands 20 --seed 3
"""

import argparse
import json

import numpy as np

from flyjack.blackjack import HIT, Blackjack
from flyjack.mushroom_body import KenyonCode, MushroomBody
from flyjack.strategies import basic_strategy
from flyjack.train import RESULTS

CARD = {1: "A", 10: "10"}


def cards(cs):
    return " ".join(CARD.get(c, str(c)) for c in cs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hands", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--spiking", action="store_true", help="decide with the live spiking brain")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    log = json.load(open(RESULTS / "training.json"))
    mb = MushroomBody.load(RESULTS / f"mushroom_body_seed{log['best_seed']}.npz")
    rng = np.random.default_rng(a.seed)
    if a.spiking:
        from flyjack.closed_loop import SpikingFly
        cl = json.load(open(RESULTS / "closed_loop.json"))
        print("building the spiking brain with the learned synapses ...")
        fly = SpikingFly(mb, cl["gain"], {HIT: cl["base"]["hit"], 0: cl["base"]["stand"]}, a.device)
        step = [a.seed * 1000]

        def decide(obs):
            step[0] += 1
            r = fly.sense([obs], seed=step[0])
            return int(r["spiking"][0]), f"HIT MBONs {r['n_hit'][0]:3d} spikes | STAND MBONs {r['n_stand'][0]:3d}"
    else:
        code = KenyonCode()

        def decide(obs):
            x = code.sample(obs, rng, "test")
            m = mb.mbon(x)
            return mb.act(x, rng), f"HIT pool {m[HIT]:+.3f} | STAND pool {m[0]:+.3f}"

    env, total = Blackjack(seed=a.seed), 0.0
    for h in range(1, a.hands + 1):
        obs = env.reset()
        print(f"\nhand {h}: fly {cards(env.player)}  |  dealer shows {cards(env.dealer[:1])}")
        while obs is not None:
            action, detail = decide(obs)
            name = "HIT" if action == HIT else "STAND"
            book = "" if action == basic_strategy(obs) else "   (basic strategy disagrees)"
            print(f"  sum {obs[0]:2d}{' soft' if obs[2] else '     '}  {detail}  ->  {name}{book}")
            obs, _, _ = env.step(action)
            if action == HIT:
                print(f"  draws {cards(env.player[-1:])} -> {cards(env.player)}")
        total += env.reward
        result = {1.0: "WIN", 0.0: "push", -1.0: "loss"}[env.reward]
        print(f"  dealer: {cards(env.dealer)}  =>  {result}   (running total {total:+.0f})")
    print(f"\n{a.hands} hands, net {total:+.0f}")


if __name__ == "__main__":
    main()

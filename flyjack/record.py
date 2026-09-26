"""Record the fly brain's response to every Blackjack observation.

For each of the 280 observations the card odor is presented to the clamped
projection neurons for ``DURATION_MS`` and the whole brain is simulated;
``TRIALS`` independent Poisson trials capture neural variability. Spike
counts of the task-relevant populations are cached to ``data/cache/``.

    python -m flyjack.record --trials 8 --workers 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .anatomy import ROOT, load_anatomy
from .blackjack import all_observations

CACHE = ROOT / "data" / "cache"
RESPONSES = CACHE / "responses.npz"
DURATION_MS = 200.0
CHUNK = 8


def _worker(args):
    chunk_id, observations, trial, seed, out = args
    import torch
    torch.set_num_threads(1)
    from .brain import task_brain
    from .encoding import OdorEncoder
    global _BRAIN, _ENC
    if "_BRAIN" not in globals():
        _BRAIN, _ENC = task_brain(), OdorEncoder()
    t0 = time.time()
    counts = _BRAIN.run(_ENC.rates(observations, _BRAIN.n), DURATION_MS, seed=seed)
    anatomy = load_anatomy()
    np.savez_compressed(out, **{k: counts[:, ix].numpy().astype(np.uint16)
                                for k, ix in anatomy.recorded().items()},
                        n_active=(counts > 0).sum(1).numpy())
    return chunk_id, trial, time.time() - t0


def record(trials=8, workers=4, seed=0):
    from concurrent.futures import ProcessPoolExecutor
    from .encoding import DEFAULT_RATE, SUM_WIDTH, DEALER_WIDTH, ACE_WIDTH, SEED

    obs = all_observations()
    parts = CACHE / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    jobs = []
    for trial in range(trials):
        for c in range(0, len(obs), CHUNK):
            out = parts / f"t{trial:02d}_c{c:03d}.npz"
            if not out.exists():
                jobs.append((c, obs[c:c + CHUNK], trial, seed + 1000 * trial + c, out))
    print(f"{len(jobs)} chunks to simulate ({len(obs)} observations x {trials} trials, "
          f"{DURATION_MS:.0f} ms each) on {workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        for k, (c, trial, dt) in enumerate(ex.map(_worker, jobs), 1):
            el = time.time() - t0
            print(f"  [{k}/{len(jobs)}] trial {trial} obs {c}-{c + CHUNK - 1}: {dt:.0f}s "
                  f"(elapsed {el / 60:.1f} min, eta {el / k * (len(jobs) - k) / 60:.1f} min)",
                  flush=True)

    anatomy = load_anatomy()
    merged = {}
    for name in list(anatomy.recorded()) + ["n_active"]:
        per_trial = []
        for trial in range(trials):
            per_trial.append(np.concatenate(
                [np.load(parts / f"t{trial:02d}_c{c:03d}.npz")[name] for c in range(0, len(obs), CHUNK)]))
        merged[name] = np.stack(per_trial, 1)           # [obs, trial, ...]
    meta = dict(duration_ms=DURATION_MS, trials=trials, rate_hz=DEFAULT_RATE,
                widths=[SUM_WIDTH, DEALER_WIDTH, ACE_WIDTH], encoder_seed=SEED, seed=seed)
    np.savez_compressed(RESPONSES, observations=np.array(obs, dtype=np.int16),
                        meta=json.dumps(meta), **merged,
                        **{f"idx_{k}": v for k, v in anatomy.recorded().items()})
    print(f"saved {RESPONSES}")


def load_responses(path=RESPONSES):
    d = np.load(path)
    obs = [tuple(int(x) for x in o[:2]) + (bool(o[2]),) for o in d["observations"]]
    return obs, d


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    record(a.trials, a.workers, a.seed)

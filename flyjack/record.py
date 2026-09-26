"""Record the fly brain's response to every Blackjack observation.

For each of the 280 observations the card odor is presented to the clamped
projection neurons for ``DURATION_MS`` and the whole brain is simulated;
``TRIALS`` independent Poisson trials capture neural variability. Spike
counts of the task-relevant populations are cached to ``data/cache/``.

    python -m flyjack.record --trials 32 --device cuda   # GPU: one batch of 280 per trial
    python -m flyjack.record --trials 8 --workers 4      # CPU: parallel worker processes
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


def _part(parts, trial, c, size):
    return parts / f"t{trial:02d}_c{c:03d}_n{size:03d}.npz"


def _worker(args, device="cpu"):
    chunk_id, observations, trial, seed, out = args
    import torch
    from .brain import task_brain
    from .encoding import OdorEncoder
    global _BRAIN, _ENC
    if "_BRAIN" not in globals():
        if device == "cpu":
            torch.set_num_threads(1)
        _BRAIN, _ENC = task_brain(device=device), OdorEncoder()
    t0 = time.time()
    counts = _BRAIN.run(_ENC.rates(observations, _BRAIN.n), DURATION_MS, seed=seed)
    anatomy = load_anatomy()
    np.savez_compressed(out, **{k: counts[:, ix].numpy().astype(np.uint16)
                                for k, ix in anatomy.recorded().items()},
                        n_active=(counts > 0).sum(1).numpy())
    return chunk_id, trial, time.time() - t0


def _merge_trial(parts, trial, n_obs, name):
    """Concatenate the chunks of one trial, checking they tile all observations."""
    files = sorted(parts.glob(f"t{trial:02d}_c*_n*.npz"))
    chunks, nxt = [], 0
    for f in files:
        c, size = (int(x[1:]) for x in f.stem.split("_")[1:])
        if c != nxt:
            raise RuntimeError(f"trial {trial}: chunks do not tile the observations at {f.name}")
        chunks.append(np.load(f)[name])
        nxt = c + size
    if nxt != n_obs:
        raise RuntimeError(f"trial {trial}: only {nxt}/{n_obs} observations recorded")
    return np.concatenate(chunks)


def record(trials=8, workers=4, seed=0, device="cpu", chunk=None):
    from concurrent.futures import ProcessPoolExecutor
    from .encoding import DEFAULT_RATE, SUM_WIDTH, DEALER_WIDTH, ACE_WIDTH, SEED

    obs = all_observations()
    parts = CACHE / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    # GPU: whole trial in one batch (sparse ops scale well); CPU: small chunks per worker
    chunk = chunk or (len(obs) if device != "cpu" else 8)
    jobs = []
    for trial in range(trials):
        if any(parts.glob(f"t{trial:02d}_c*_n*.npz")) and not all(
                _part(parts, trial, c, len(obs[c:c + chunk])).exists()
                for c in range(0, len(obs), chunk)):
            print(f"trial {trial}: keeping parts recorded with another chunk size")
            continue
        for c in range(0, len(obs), chunk):
            out = _part(parts, trial, c, len(obs[c:c + chunk]))
            if not out.exists():
                jobs.append((c, obs[c:c + chunk], trial, seed + 1000 * trial + c, out))
    where = device if device != "cpu" else f"{workers} CPU workers"
    print(f"{len(jobs)} chunks to simulate ({len(obs)} observations x {trials} trials, "
          f"{DURATION_MS:.0f} ms each) on {where}")
    t0 = time.time()
    if device == "cpu":
        ex = ProcessPoolExecutor(workers)
        results = ex.map(_worker, jobs)
    else:
        ex, results = None, (_worker(j, device) for j in jobs)
    for k, (c, trial, dt) in enumerate(results, 1):
        el = time.time() - t0
        print(f"  [{k}/{len(jobs)}] trial {trial} obs {c}+: {dt:.0f}s "
              f"(elapsed {el / 60:.1f} min, eta {el / k * (len(jobs) - k) / 60:.1f} min)",
              flush=True)
    if ex is not None:
        ex.shutdown()

    anatomy = load_anatomy()
    merged = {}
    for name in list(anatomy.recorded()) + ["n_active"]:
        merged[name] = np.stack([_merge_trial(parts, t, len(obs), name)
                                 for t in range(trials)], 1)   # [obs, trial, ...]
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
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--chunk", type=int, default=None,
                    help="observations per simulated batch (default: 280 on GPU, 8 on CPU)")
    a = ap.parse_args()
    record(a.trials, a.workers, a.seed, a.device, a.chunk)

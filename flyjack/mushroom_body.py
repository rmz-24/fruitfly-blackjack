"""The mushroom body as a learner: plastic Kenyon cell -> MBON synapses.

Two output pools read the Kenyon cell (KC) population code recorded from the
whole-brain simulation: a HIT MBON pool and a STAND MBON pool. Pool activity
is a weighted sum of KC activity plus a tonic baseline,

    MBON_a(s) = sum_k w[a, k] * kc_k(s) + b[a],

and the fly picks the action whose pool is more active. After every choice a
dopaminergic teaching signal reports the reward prediction error

    delta = r + max_a' MBON_a'(s') - MBON_a(s)        (0 after the hand ends)

and only the synapses of the chosen pool that received KC input change:

    dw[a, k] = eta * delta * kc_k(s),   db[a] = eta * delta.

This is the three-factor rule of the fly mushroom body (presynaptic KC
activity x postsynaptic pool x dopamine), and mathematically it is Q-learning
with a linear readout of connectome-generated features. Because each KC
activity vector is L2-normalised, ``eta`` is directly the fraction of the
prediction error corrected per step.

Neural noise is part of the task: every visit to a state uses one of the
recorded Poisson trials, so the fly never sees the same KC pattern twice in a
row, and it is evaluated on trials it was never trained on.
"""

from dataclasses import dataclass

import numpy as np

from .blackjack import ACTIONS, HIT, STAND
from .record import load_responses


class KenyonCode:
    """L2-normalised KC spike-count vectors per observation and trial."""

    def __init__(self, path=None, n_train_trials=None):
        obs, d = load_responses(path) if path else load_responses()
        kc = d["kc"].astype(np.float32)                         # [obs, trial, kc]
        norm = np.linalg.norm(kc, axis=2, keepdims=True)
        self.x = kc / np.maximum(norm, 1e-6)
        self.observations = obs
        self.index = {o: i for i, o in enumerate(obs)}
        n_trials = kc.shape[1]
        self.n_train = n_train_trials or (3 * n_trials) // 4
        self.train_trials = np.arange(self.n_train)
        self.test_trials = np.arange(self.n_train, n_trials)
        self.n_kc = kc.shape[2]

    def sample(self, obs, rng, split="train"):
        trials = self.train_trials if split == "train" else self.test_trials
        return self.x[self.index[obs], trials[rng.integers(len(trials))]]

    def all(self, split="test"):
        """[obs, trials_in_split, kc]"""
        return self.x[:, self.train_trials if split == "train" else self.test_trials]


@dataclass
class MushroomBody:
    n_kc: int
    eta: float = 0.02

    def __post_init__(self):
        self.w = np.zeros((len(ACTIONS), self.n_kc), np.float32)   # [STAND, HIT] x KC
        self.b = np.zeros(len(ACTIONS), np.float32)

    def mbon(self, x):
        """Output-pool drive ``[..., 2]`` (STAND, HIT) for KC vectors ``x [..., n_kc]``."""
        return x @ self.w.T + self.b

    def act(self, x, rng, epsilon=0.0):
        if rng.random() < epsilon:
            return int(rng.integers(len(ACTIONS)))
        m = self.mbon(x)
        return HIT if m[HIT] > m[STAND] else STAND

    def dopamine(self, x, action, delta, eta=None):
        """Three-factor update of the chosen pool's KC synapses."""
        eta = self.eta if eta is None else eta
        self.w[action] += eta * delta * x
        self.b[action] += eta * delta

    # -- policies read out of the learned synapses -------------------------
    def hit_probability(self, code, split="test"):
        """P(hit | obs) over held-out noisy trials: the fly's actual behaviour."""
        m = self.mbon(code.all(split))                          # [obs, trials, 2]
        p = (m[..., HIT] > m[..., STAND]).mean(1)
        return {o: float(p[i]) for i, o in enumerate(code.observations)}

    def q_table(self, code, split="test"):
        """Mean pool drive per observation: the fly's learned action values."""
        m = self.mbon(code.all(split)).mean(1)
        return {o: {STAND: float(m[i, STAND]), HIT: float(m[i, HIT])}
                for i, o in enumerate(code.observations)}

    def save(self, path, **extra):
        np.savez_compressed(path, w=self.w, b=self.b, eta=self.eta, **extra)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        mb = cls(n_kc=d["w"].shape[1], eta=float(d["eta"]))
        mb.w, mb.b = d["w"], d["b"]
        return mb

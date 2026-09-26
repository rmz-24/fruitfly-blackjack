"""Whole-brain leaky integrate-and-fire simulator of the adult fruit fly.

This is the model of Shiu et al. (Nature 2024) as implemented in the PyTorch
backend of https://github.com/eonsystemspbc/fly-brain (``code/run_pytorch.py``,
GPL-2.0): same parameters, same update order, same alpha-synapse / delay /
refractory handling. Two changes make it fast on a CPU:

* **Event-driven synaptic propagation.** Instead of multiplying the full
  15M-entry weight matrix with the spike vector every 0.1 ms, only the outgoing
  synapses of neurons that actually spiked are gathered (spikes are sparse).
* **Ring-buffer delays** instead of ``torch.roll`` over the delay buffer.

Several independent trials (e.g. different Blackjack observations) are
simulated in parallel along a batch dimension.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .anatomy import PATH_CONN, _require, neuron_ids

# Identical to MODEL_PARAMS in fly-brain/code/run_pytorch.py
MODEL_PARAMS = {
    "tauSyn": 5.0,        # ms
    "tDelay": 1.8,        # ms
    "v0": -52.0,          # mV
    "vReset": -52.0,      # mV
    "vRest": -52.0,       # mV
    "vThreshold": -45.0,  # mV
    "tauMem": 20.0,       # ms
    "tRefrac": 2.2,       # ms
    "scalePoisson": 250,
    "wScale": 0.275,      # mV per synapse
}
DT = 0.1  # ms


def load_connectome():
    """Return ``(pre, post, weight, n_neurons)``; weight = signed synapse count."""
    cols = ["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"]
    df = pd.read_parquet(_require(PATH_CONN), columns=cols)
    return (df[cols[0]].to_numpy(np.int64), df[cols[1]].to_numpy(np.int64),
            df[cols[2]].to_numpy(np.float32), len(neuron_ids()))


class FlyBrain:
    """Event-driven, batched LIF simulation over a connectome.

    Parameters
    ----------
    pre, post, weight, n : connectome edges (weight in synapse counts, signed).
        Defaults to the full FlyWire v783 connectome.
    extra : optional ``(pre, post, weight)`` arrays appended to the connectome
        (used to install learned synapses).
    """

    def __init__(self, pre=None, post=None, weight=None, n=None, extra=None,
                 params=MODEL_PARAMS, dt=DT):
        if pre is None:
            pre, post, weight, n = load_connectome()
        if extra is not None:
            pre = np.concatenate([pre, extra[0]])
            post = np.concatenate([post, extra[1]])
            weight = np.concatenate([weight, np.asarray(extra[2], np.float32)])
        self.n = int(n)
        self.params, self.dt = params, dt

        # CSR by presynaptic neuron: outgoing synapses of j are ptr[j]:ptr[j+1].
        order = np.argsort(pre, kind="stable")
        self.post = torch.from_numpy(np.ascontiguousarray(post[order]))
        self.w = torch.from_numpy(np.ascontiguousarray(weight[order] * params["wScale"]))
        ptr = np.zeros(self.n + 1, np.int64)
        np.cumsum(np.bincount(pre, minlength=self.n), out=ptr[1:])
        self.ptr = torch.from_numpy(ptr)

        self.delay_steps = int(params["tDelay"] / dt) + 1   # = upstream buffer length
        self.refrac_steps = int(round(params["tRefrac"] / dt))
        self.syn_decay = 1 - dt / params["tauSyn"]
        self.mem_factor = dt / params["tauMem"]
        self.poisson_kick = params["scalePoisson"] * params["wScale"]

    # ------------------------------------------------------------------
    def _propagate(self, spk):
        """Synaptic events caused by spikes at flat indices ``b * N + j``.

        Returns ``(target, weight)``: flat postsynaptic indices and weights (mV),
        one entry per synapse (duplicates are summed later by ``index_add_``).
        """
        if spk.numel() == 0:
            return None
        b, j = spk // self.n, spk % self.n
        start, count = self.ptr[j], self.ptr[j + 1] - self.ptr[j]
        total = int(count.sum())
        if total == 0:
            return None
        offset = torch.repeat_interleave(torch.cumsum(count, 0) - count, count)
        syn = torch.repeat_interleave(start, count) + torch.arange(total) - offset
        target = torch.repeat_interleave(b, count) * self.n + self.post[syn]
        return target, self.w[syn]

    @torch.no_grad()
    def run(self, rates, duration_ms, record_raster=None, seed=0):
        """Simulate a batch of trials, all starting from rest.

        Parameters
        ----------
        rates : float tensor/array [B, N] of Poisson stimulation rates (Hz).
        duration_ms : simulated time.
        record_raster : optional index array; spike times of these neurons are
            returned for every trial.

        Returns
        -------
        counts : int32 tensor [B, N] spike counts over the whole run.
        raster : ``[n_spikes, 3]`` array of ``(trial, neuron, time_ms)``, if requested.

        Update order per 0.1 ms step (identical to upstream ``TorchModel``):
        refractory counter from last step's spikes -> gated delayed synaptic
        input into the alpha synapse -> Poisson kick + membrane integration with
        the *previous* conductance -> threshold, reset of v and conductance.
        Only the dense membrane/conductance decay touches every neuron; the
        delay buffer, refractory gating and resets are handled sparsely.
        """
        rates = torch.as_tensor(rates, dtype=torch.float32)
        if rates.dim() == 1:
            rates = rates.unsqueeze(0)
        B, N = rates.shape
        assert N == self.n
        p = self.params
        gen = torch.Generator().manual_seed(seed)

        flat_rates = rates.reshape(-1)
        stim_idx = flat_rates.nonzero(as_tuple=True)[0]
        stim_prob = flat_rates[stim_idx] * self.dt / 1000.0

        # Refractory length per neuron; upstream sets it to 0 for stimulated neurons.
        refrac_len = torch.full((B * N,), self.refrac_steps, dtype=torch.int64)
        refrac_len[stim_idx] = 0
        last_spike = torch.full((B * N,), -(10 ** 9), dtype=torch.int64)

        v = torch.full((B * N,), p["v0"])
        g = torch.zeros(B * N)
        counts = torch.zeros(B * N, dtype=torch.int32)
        ring = [None] * self.delay_steps     # synaptic events in flight
        spk = torch.zeros(0, dtype=torch.int64)
        leak, bias = 1 - self.mem_factor, self.mem_factor * p["vRest"]

        rec_mask, raster = None, []
        if record_raster is not None:
            rec_mask = torch.zeros(N, dtype=torch.bool)
            rec_mask[torch.as_tensor(np.asarray(record_raster), dtype=torch.long)] = True

        n_steps = int(round(duration_ms / self.dt))
        for t in range(n_steps):
            head = t % self.delay_steps
            arriving = ring[head]
            ring[head] = self._propagate(spk)        # delivered delay_steps later

            # Poisson drive: voltage kick on stimulated neurons
            if stim_idx.numel():
                fired = torch.bernoulli(stim_prob, generator=gen)
                v.index_add_(0, stim_idx, fired * self.poisson_kick)
            # v += dt/tau_m * (g - (v - v_rest)), using the previous conductance
            v.mul_(leak).add_(g, alpha=self.mem_factor).add_(bias)

            # alpha synapse: decay + delayed input, blocked while refractory
            g.mul_(self.syn_decay)
            if arriving is not None:
                tgt, w = arriving
                open_ = (t - 1 - last_spike[tgt]) >= refrac_len[tgt]
                g.index_add_(0, tgt, w * open_)

            spk = (v > p["vThreshold"]).nonzero(as_tuple=True)[0]
            if spk.numel():
                v[spk] = p["vReset"]
                g[spk] = 0.0
                last_spike[spk] = t
                counts[spk] += 1
                if rec_mask is not None:
                    keep = spk[rec_mask[spk % N]]
                    if keep.numel():
                        raster.append(np.stack([(keep // N).numpy(), (keep % N).numpy(),
                                                np.full(keep.numel(), (t + 1) * self.dt)], 1))

        counts = counts.view(B, N)
        if rec_mask is not None:
            return counts, (np.concatenate(raster) if raster else np.zeros((0, 3)))
        return counts

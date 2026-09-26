import numpy as np
import pytest
import torch

from flyjack.anatomy import PATH_CONN
from flyjack.brain import DT, FlyBrain


def tiny_brain():
    # 0 -> 1 strongly excitatory, 0 -> 2 inhibitory, 3 isolated
    pre = np.array([0, 0])
    post = np.array([1, 2])
    w = np.array([100.0, -100.0], np.float32)
    return FlyBrain(pre, post, w, n=4)


def test_silent_without_input():
    counts = tiny_brain().run(torch.zeros(1, 4), 50.0)
    assert counts.sum() == 0


def test_stimulation_propagates_with_delay():
    brain = tiny_brain()
    rates = torch.zeros(2, 4)
    rates[:, 0] = 500.0
    counts, raster = brain.run(rates, 100.0, record_raster=[0, 1])
    assert (counts[:, 0] > 20).all()          # driven neuron fires ~ at the stim rate
    assert (counts[:, 1] > 0).all()           # excitation propagates
    assert (counts[:, 2] == 0).all()          # inhibition never causes spikes
    assert (counts[:, 3] == 0).all()
    for trial in range(2):
        r = raster[raster[:, 0] == trial]
        first0 = r[r[:, 1] == 0, 2].min()
        first1 = r[r[:, 1] == 1, 2].min()
        assert first1 - first0 >= brain.delay_steps * DT - 1e-9


def test_batch_trials_are_independent():
    brain = tiny_brain()
    rates = torch.zeros(2, 4)
    rates[1, 0] = 500.0
    counts = brain.run(rates, 50.0)
    assert counts[0].sum() == 0 and counts[1].sum() > 0


@pytest.mark.skipif(not PATH_CONN.exists(), reason="connectome not downloaded")
def test_sugar_neurons_drive_feeding_motor_neurons():
    """Shiu et al.: sugar GRN activation recruits proboscis motor neurons."""
    from flyjack.anatomy import annotations, neuron_ids
    sugar = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
             720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
             720575940621502051, 720575940640649691, 720575940639332736]
    idx = {r: i for i, r in enumerate(neuron_ids())}
    brain = FlyBrain()
    rates = torch.zeros(1, brain.n)
    rates[0, [idx[r] for r in sugar]] = 200.0
    counts = brain.run(rates, 300.0)[0]
    ann = annotations()
    motor = ann.loc[ann.super_class == "motor", "index"].to_numpy(copy=True)
    assert counts[motor].max() >= 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_cuda_matches_cpu_behaviour():
    pre, post = np.array([0, 0]), np.array([1, 2])
    w = np.array([100.0, -100.0], np.float32)
    rates = torch.zeros(2, 4)
    rates[:, 0] = 500.0
    counts, raster = FlyBrain(pre, post, w, n=4, device="cuda").run(rates, 100.0, record_raster=[0, 1])
    assert counts.device.type == "cpu"
    assert (counts[:, 0] > 20).all() and (counts[:, 1] > 0).all()
    assert (counts[:, 2:] == 0).all()
    assert len(raster) == int(counts[:, :2].sum())

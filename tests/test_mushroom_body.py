import numpy as np

from flyjack.blackjack import HIT, STAND
from flyjack.mushroom_body import MushroomBody


def sparse_code(rng, n_kc=500, active=0.1):
    x = (rng.random(n_kc) < active) * rng.random(n_kc)
    return (x / np.linalg.norm(x)).astype(np.float32)


def test_dopamine_rule_learns_action_values_of_a_bandit():
    """Two odors, two actions: MBON drive converges to the expected rewards."""
    rng = np.random.default_rng(0)
    odors = [sparse_code(rng), sparse_code(rng)]
    reward = {(0, STAND): 0.5, (0, HIT): -0.5, (1, STAND): -1.0, (1, HIT): 1.0}
    mb = MushroomBody(n_kc=500, eta=0.1)
    for _ in range(3000):
        s = int(rng.integers(2))
        a = int(rng.integers(2))
        noisy = odors[s] + 0.02 * rng.standard_normal(500).astype(np.float32)
        r = reward[s, a] + 0.3 * rng.standard_normal()
        mb.dopamine(noisy, a, r - mb.mbon(noisy)[a])
    for (s, a), r in reward.items():
        assert abs(mb.mbon(odors[s])[a] - r) < 0.15
    assert mb.act(odors[0], rng) == STAND and mb.act(odors[1], rng) == HIT


def test_only_the_chosen_pool_and_active_kcs_change():
    x = np.zeros(10, np.float32)
    x[[2, 5]] = 1 / np.sqrt(2)
    mb = MushroomBody(n_kc=10, eta=0.5)
    mb.dopamine(x, HIT, 1.0)
    assert np.all(mb.w[STAND] == 0) and mb.b[STAND] == 0
    assert np.flatnonzero(mb.w[HIT]).tolist() == [2, 5]

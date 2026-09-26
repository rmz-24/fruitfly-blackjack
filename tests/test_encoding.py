import numpy as np

from flyjack.anatomy import Anatomy
from flyjack.blackjack import all_observations
from flyjack.encoding import OdorEncoder

EMPTY = np.zeros(0, np.int64)


def fake_anatomy(n_glomeruli=69, pns_per_glomerulus=3):
    names = np.repeat([f"G{k:02d}" for k in range(n_glomeruli)], pns_per_glomerulus)
    return Anatomy(pn=np.arange(len(names)) + 100, pn_glomerulus=names, kc=EMPTY,
                   mbon=EMPTY, mbon_type=np.zeros(0, str), dan=EMPTY, apl=EMPTY, dn=EMPTY)


def test_encoding_is_deterministic():
    a, b = OdorEncoder(fake_anatomy()), OdorEncoder(fake_anatomy())
    for obs in all_observations():
        assert a.channels(obs) == b.channels(obs)
    assert OdorEncoder(fake_anatomy(), seed=1).channels((12, 3, False)) != a.channels((12, 3, False))


def test_feature_pools_are_disjoint_and_graded():
    enc = OdorEncoder(fake_anatomy())
    pools = [set(enc.sum_pool), set(enc.dealer_pool), set(enc.ace_pool[False]), set(enc.ace_pool[True])]
    assert sum(map(len, pools)) == len(set().union(*pools))
    # neighbouring sums share all but one channel; distant sums share none
    s = lambda v: set(enc.channels((v, 5, False))) & set(enc.sum_pool)
    assert len(s(12) & s(13)) == enc.sum_width - 1
    assert not s(4) & s(21)
    # usable ace changes only the ace channels
    hard, soft = set(enc.channels((15, 5, False))), set(enc.channels((15, 5, True)))
    assert hard - soft == set(enc.ace_pool[False])


def test_rates_stimulate_exactly_the_channel_pns():
    anatomy = fake_anatomy()
    enc = OdorEncoder(anatomy, rate=123.0)
    obs = [(4, 1, False), (21, 10, True)]
    r = enc.rates(obs, n_neurons=400)
    assert r.shape == (2, 400)
    for k, o in enumerate(obs):
        on = set(np.flatnonzero(r[k].numpy()))
        expected = {int(i) for c in enc.channels(o) for i in anatomy.pns_of(c)}
        assert on == expected
        assert (r[k, list(on)] == 123.0).all()

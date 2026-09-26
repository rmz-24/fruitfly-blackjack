import numpy as np

from flyjack.anatomy import Anatomy
from flyjack.blackjack import HIT, STAND
from flyjack.closed_loop import learned_synapses


def test_learned_synapses_are_excitatory_and_encode_the_preference():
    kc = np.arange(10, 20)
    anatomy = Anatomy(pn=np.zeros(0, int), pn_glomerulus=np.zeros(0, str), kc=kc,
                      mbon=np.array([1, 2, 3, 4]),
                      mbon_type=np.array(["MBON01", "MBON01", "MBON03", "MBON03"]),
                      dan=np.zeros(0, int), apl=np.zeros(0, int), dn=np.zeros(0, int))
    pref = np.linspace(-1, 1, 10).astype(np.float32)
    base = {HIT: 3.0, STAND: 2.0}
    pre, post, w = learned_synapses(pref, anatomy, gain=10.0, base=base)
    assert len(pre) == 4 * len(kc) and (w > 0).all()
    w_hit = w[post == 1] - base[HIT]
    w_stand = w[post == 3] - base[STAND]
    assert np.allclose(w_hit - w_stand, 10.0 * pref)       # push-pull = gain x preference
    assert np.array_equal(pre[post == 1], kc)

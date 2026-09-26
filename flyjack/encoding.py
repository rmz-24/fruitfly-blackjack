"""Sensory encoding: Blackjack observations as odors.

Each of the 69 uniglomerular projection-neuron types (one per antennal-lobe
glomerulus channel) is a sensory "channel". A fixed, seeded shuffle assigns the
channels to three pools, and every feature value activates a contiguous window
of channels within its pool:

* player sum 4..21  -> window of ``SUM_WIDTH`` channels, sliding by one per point,
  so nearby sums smell alike (graded code, like odor concentration);
* dealer upcard A..10 -> window of ``DEALER_WIDTH`` channels, sliding likewise;
* usable ace yes/no  -> one of two disjoint groups of ``ACE_WIDTH`` channels.

All PNs of an active channel receive Poisson stimulation at ``rate`` Hz. The
mushroom body's random PN->KC wiring then mixes these features non-linearly,
which is exactly what lets a linear KC->MBON readout learn the interactions
between player sum and dealer card.
"""

import numpy as np
import torch

from .anatomy import load_anatomy

SEED = 7
SUM_VALUES = list(range(4, 22))          # 18 values
DEALER_VALUES = list(range(1, 11))       # 10 values
SUM_WIDTH, DEALER_WIDTH, ACE_WIDTH = 6, 5, 4
DEFAULT_RATE = 150.0                     # Hz  (-> ~10% of Kenyon cells respond)


class OdorEncoder:
    def __init__(self, anatomy=None, rate=DEFAULT_RATE, seed=SEED,
                 sum_width=SUM_WIDTH, dealer_width=DEALER_WIDTH, ace_width=ACE_WIDTH):
        self.anatomy = anatomy or load_anatomy()
        self.rate = rate
        self.sum_width, self.dealer_width = sum_width, dealer_width
        channels = list(self.anatomy.glomeruli)
        np.random.default_rng(seed).shuffle(channels)
        n_sum = len(SUM_VALUES) + sum_width - 1
        n_dealer = len(DEALER_VALUES) + dealer_width - 1
        assert n_sum + n_dealer + 2 * ace_width <= len(channels)
        self.sum_pool = channels[:n_sum]
        self.dealer_pool = channels[n_sum:n_sum + n_dealer]
        rest = channels[n_sum + n_dealer:]
        self.ace_pool = {False: rest[:ace_width], True: rest[ace_width:2 * ace_width]}

    def channels(self, obs):
        """Glomerulus channels activated by an observation."""
        player_sum, dealer, usable = obs
        i = SUM_VALUES.index(player_sum)
        j = DEALER_VALUES.index(dealer)
        return (self.sum_pool[i:i + self.sum_width]
                + self.dealer_pool[j:j + self.dealer_width]
                + self.ace_pool[bool(usable)])

    def neurons(self, obs):
        """Simulator indices of the PNs stimulated for ``obs``."""
        return np.concatenate([self.anatomy.pns_of(c) for c in self.channels(obs)])

    def rates(self, observations, n_neurons):
        """[len(observations), n_neurons] Poisson rate tensor (Hz)."""
        r = torch.zeros(len(observations), n_neurons)
        for k, obs in enumerate(observations):
            r[k, self.neurons(obs)] = self.rate
        return r

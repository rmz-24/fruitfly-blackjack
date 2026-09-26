# 🪰🃏 fruitfly-blackjack

Teaching the **whole-brain connectome of the adult fruit fly** (*Drosophila melanogaster*)
to play **Blackjack**.

The brain model is the leaky integrate-and-fire (LIF) emulation of the full
[FlyWire](https://flywire.ai/) v783 connectome (~139k neurons, ~15M connections between neuron pairs)
from [eonsystemspbc/fly-brain](https://github.com/eonsystemspbc/fly-brain), itself based on
Shiu et al., *Nature* 2024.

The fly learns the way real flies learn: cards are presented as "odors" to its olfactory
projection neurons, the signal spreads through the real wiring to the **mushroom body**,
and dopamine-like reward signals reshape the Kenyon cell → output neuron synapses.

> 🚧 Work in progress. See the commit history for the step-by-step build.

## License
GPL-2.0 (the simulator code is adapted from the GPL-2.0 `fly-brain` repository).

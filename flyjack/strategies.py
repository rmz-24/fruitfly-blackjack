"""Reference Blackjack strategies to compare the fly against."""

import random

from .blackjack import HIT, STAND, optimal_policy_table

_OPTIMAL = None


def basic_strategy(obs):
    """Optimal hit/stand play for these rules (exact DP = 'basic strategy')."""
    global _OPTIMAL
    if _OPTIMAL is None:
        _OPTIMAL = optimal_policy_table()
    return _OPTIMAL[obs]


def dealer_rule(obs):
    """Mimic the dealer: hit below 17."""
    return HIT if obs[0] < 17 else STAND


def never_bust(obs):
    """Only hit when a bust is impossible."""
    return HIT if (obs[0] <= 11 or obs[2]) else STAND


def make_random(seed=0):
    rng = random.Random(seed)
    return lambda obs: rng.choice((STAND, HIT))


BASELINES = {
    "random": make_random,
    "never bust": lambda: never_bust,
    "dealer rule (hit < 17)": lambda: dealer_rule,
    "basic strategy (optimal)": lambda: basic_strategy,
}

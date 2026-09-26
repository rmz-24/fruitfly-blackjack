import statistics

from flyjack.blackjack import (
    HIT, STAND, Blackjack, _add, all_observations, dealer_final_given_upcard,
    expected_return, hand_value, play_hand,
)
from flyjack.strategies import basic_strategy, dealer_rule, never_bust


def test_hand_value():
    assert hand_value([1, 10]) == (21, True)
    assert hand_value([1, 1]) == (12, True)
    assert hand_value([1, 5, 10]) == (16, False)
    assert hand_value([10, 9, 5]) == (24, False)


def test_add_soft_to_hard():
    assert _add(18, True, 5) == (13, False)
    assert _add(10, False, 1) == (21, True)


def test_observation_space():
    obs = all_observations()
    assert len(obs) == 280 == len(set(obs))


def test_dealer_distribution_sums_to_one():
    for up in range(1, 11):
        dist = dealer_final_given_upcard(up)
        assert abs(sum(dist.values()) - 1) < 1e-12
        assert set(dist) <= {17, 18, 19, 20, 21, 22}


def test_basic_strategy_textbook_cells():
    # Well-known hit/stand basic strategy decisions.
    assert basic_strategy((16, 10, False)) == HIT
    assert basic_strategy((12, 4, False)) == STAND
    assert basic_strategy((12, 2, False)) == HIT
    assert basic_strategy((13, 2, False)) == STAND
    assert basic_strategy((17, 1, False)) == STAND
    assert basic_strategy((11, 6, False)) == HIT
    assert basic_strategy((18, 9, True)) == HIT
    assert basic_strategy((18, 8, True)) == STAND
    assert basic_strategy((19, 10, True)) == STAND


def test_strategy_ranking_exact():
    opt = expected_return(basic_strategy)
    assert opt > expected_return(dealer_rule) > -0.1
    assert opt > expected_return(never_bust)
    assert -0.06 < opt < -0.02


def test_simulation_matches_exact_value():
    env = Blackjack(seed=1)
    rewards = [play_hand(env, basic_strategy) for _ in range(200_000)]
    mean, sem = statistics.fmean(rewards), statistics.stdev(rewards) / len(rewards) ** 0.5
    assert abs(mean - expected_return(basic_strategy)) < 4 * sem

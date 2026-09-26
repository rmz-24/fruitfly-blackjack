"""Infinite-deck Blackjack with hit/stand only.

Rules (Sutton & Barto style, with the dealer peeking for blackjack):

* Cards are drawn with replacement: 1 (ace) .. 10, face cards count 10.
* A natural (ace + ten-valued card in the first two cards) ends the hand at once:
  player natural vs no dealer natural -> +1, both -> 0, dealer only -> -1.
* Otherwise the player hits until standing or busting (bust -> -1).
* The dealer then draws until reaching 17 or more (stands on soft 17).
* Higher total wins (+1 / -1), equal totals push (0).

An observation is ``(player_sum, dealer_upcard, usable_ace)`` where a usable ace
is an ace currently counted as 11. Actions: ``STAND = 0``, ``HIT = 1``.
"""

import random
from functools import lru_cache

STAND, HIT = 0, 1
ACTIONS = (STAND, HIT)
ACTION_NAMES = {STAND: "stand", HIT: "hit"}

# Probability of drawing each card value from an infinite deck.
CARD_PROBS = {c: (4 / 13 if c == 10 else 1 / 13) for c in range(1, 11)}


def draw(rng):
    return min(rng.randint(1, 13), 10)


def hand_value(cards):
    """Return (total, usable_ace) for a list of card values."""
    total = sum(cards)
    if 1 in cards and total + 10 <= 21:
        return total + 10, True
    return total, False


def is_natural(cards):
    return len(cards) == 2 and sorted(cards) == [1, 10]


def all_observations():
    """Every decision state the player can face (280 of them)."""
    obs = []
    for dealer in range(1, 11):
        for s in range(4, 22):
            obs.append((s, dealer, False))
        for s in range(12, 22):
            obs.append((s, dealer, True))
    return obs


class Blackjack:
    """Minimal episodic environment: ``reset() -> obs | None``, ``step(a)``."""

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    def reset(self):
        """Deal a new hand. Returns the first observation, or ``None`` if a
        natural ended the hand immediately (reward available in ``self.reward``)."""
        self.player = [draw(self.rng), draw(self.rng)]
        self.dealer = [draw(self.rng), draw(self.rng)]
        self.done = False
        self.reward = 0.0
        pn, dn = is_natural(self.player), is_natural(self.dealer)
        if pn or dn:
            self.done = True
            self.reward = 0.0 if (pn and dn) else (1.0 if pn else -1.0)
            return None
        return self.observation()

    def observation(self):
        total, usable = hand_value(self.player)
        return (total, self.dealer[0], usable)

    def step(self, action):
        """Returns ``(obs, reward, done)``."""
        assert not self.done, "hand is over, call reset()"
        if action == HIT:
            self.player.append(draw(self.rng))
            if hand_value(self.player)[0] > 21:
                self.done, self.reward = True, -1.0
                return None, -1.0, True
            return self.observation(), 0.0, False
        # Stand: dealer plays out.
        while hand_value(self.dealer)[0] < 17:
            self.dealer.append(draw(self.rng))
        p = hand_value(self.player)[0]
        d = hand_value(self.dealer)[0]
        self.done = True
        self.reward = 1.0 if (d > 21 or p > d) else (0.0 if p == d else -1.0)
        return None, self.reward, True


def play_hand(env, policy):
    """Play one hand with ``policy(obs) -> action``. Returns the reward."""
    obs = env.reset()
    if obs is None:
        return env.reward
    while True:
        obs, reward, done = env.step(policy(obs))
        if done:
            return reward


# ---------------------------------------------------------------------------
# Exact dynamic programming (infinite deck)
# ---------------------------------------------------------------------------

def _add(total, soft, card):
    """Add a card to a hand given as (total, usable_ace)."""
    total += card
    if card == 1 and total + 10 <= 21:
        return total + 10, True
    if total > 21 and soft:
        return total - 10, False
    return total, soft


@lru_cache(maxsize=None)
def _dealer_final(total, soft):
    """Distribution over the dealer's final total (22 = bust) from a hand state."""
    if total > 21:
        return {22: 1.0}
    if total >= 17:
        return {total: 1.0}
    out = {}
    for card, p in CARD_PROBS.items():
        for final, q in _dealer_final(*_add(total, soft, card)).items():
            out[final] = out.get(final, 0.0) + p * q
    return out


@lru_cache(maxsize=None)
def dealer_final_given_upcard(upcard):
    """Dealer final-total distribution given the upcard and no dealer natural."""
    out, norm = {}, 0.0
    for hole, p in CARD_PROBS.items():
        if is_natural([upcard, hole]):
            continue
        norm += p
        start = _add(*_add(0, False, upcard), hole)
        for final, q in _dealer_final(*start).items():
            out[final] = out.get(final, 0.0) + p * q
    return {k: v / norm for k, v in out.items()}


def stand_value(player_sum, upcard):
    ev = 0.0
    for d, p in dealer_final_given_upcard(upcard).items():
        ev += p * (1.0 if (d > 21 or player_sum > d) else (0.0 if player_sum == d else -1.0))
    return ev


def _successors(obs):
    """[(prob, next_obs or None if bust)] after hitting."""
    s, dealer, soft = obs
    out = []
    for card, p in CARD_PROBS.items():
        t, sft = _add(s, soft, card)
        out.append((p, None if t > 21 else (t, dealer, sft)))
    return out


def action_values(policy=None):
    """Q(obs, a) for every observation.

    With ``policy=None`` returns optimal action values; otherwise the values of
    following ``policy`` after the first action. ``policy(obs)`` returns an
    action or, for a stochastic policy, the probability of hitting (a
    deterministic action is the special case 0/1, since ``STAND=0, HIT=1``).
    """
    q = {}
    # Successor states are always evaluated first:
    # hard >= 11 -> higher hard; soft -> higher soft or hard >= 13;
    # hard <= 10 -> anything above.
    def rank(o):
        s, _, soft = o
        return s + (100 if soft else (200 if s >= 11 else 0))
    order = sorted(all_observations(), key=rank, reverse=True)
    v = {}
    for obs in order:
        hit = 0.0
        for p, nxt in _successors(obs):
            hit += p * (-1.0 if nxt is None else v[nxt])
        st = stand_value(obs[0], obs[1])
        q[obs] = {STAND: st, HIT: hit}
        if policy is None:
            v[obs] = max(st, hit)
        else:
            v[obs] = _mix(q[obs], policy(obs))
    return q


def _mix(q, p_hit):
    return p_hit * q[HIT] + (1 - p_hit) * q[STAND]


def optimal_policy_table():
    q = action_values()
    return {obs: (HIT if vals[HIT] > vals[STAND] else STAND) for obs, vals in q.items()}


def expected_return(policy):
    """Exact expected reward per hand of a (possibly stochastic) policy."""
    q = action_values(policy)
    ev = 0.0
    for c1, p1 in CARD_PROBS.items():
        for c2, p2 in CARD_PROBS.items():
            player = [c1, c2]
            for up, pu in CARD_PROBS.items():
                for hole, ph in CARD_PROBS.items():
                    p = p1 * p2 * pu * ph
                    pn, dn = is_natural(player), is_natural([up, hole])
                    if pn or dn:
                        ev += p * (0.0 if (pn and dn) else (1.0 if pn else -1.0))
                        continue
                    total, usable = hand_value(player)
                    obs = (total, up, usable)
                    ev += p * _mix(q[obs], policy(obs))
    return ev

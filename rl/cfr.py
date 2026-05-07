"""
Chance-Sampled CFR for heads-up Texas Hold'em with hand/action abstraction.

Bucketing (fast, no Monte Carlo in the hot path):
  Pre-flop  — rank/suit formula → bucket 0-4
  Post-flop — evaluator hand class → bucket 0-4

Action abstraction: fold / check-call / bet-half-pot / bet-pot
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from collections import defaultdict

from evaluator import Card, RANKS, SUITS, best_hand, RANK_VAL

# ── Actions ────────────────────────────────────────────────────────────────────
FOLD, CALL, HALF, POT = 0, 1, 2, 3
A_STR = {FOLD: "f", CALL: "c", HALF: "h", POT: "p"}

NUM_BUCKETS  = 5
MAX_RAISES   = 4
STARTING_STACK = 100.0   # big blinds
SB, BB       = 0.5, 1.0


# ── Hand abstraction ───────────────────────────────────────────────────────────

def _preflop_bucket(hole: list) -> int:
    """O(1) pre-flop hand strength bucket (0 = weakest, 4 = strongest)."""
    v1, v2 = sorted([c.val() for c in hole], reverse=True)
    suited = hole[0].suit == hole[1].suit

    if v1 == v2:                         # pocket pair
        if v1 >= 10: return 4            # TT+
        if v1 >= 7:  return 3            # 77-99
        return 2                          # 22-66

    # Non-pair score: weight top card heavily, add suited/connected bonus
    gap   = v1 - v2
    score = v1 + v2 * 0.4 + (1.5 if suited else 0) - max(0, gap - 1) * 0.5

    if score >= 22: return 4             # AKs/AKo, AQs
    if score >= 18: return 3             # AJ–AT, KQs/KQo
    if score >= 14: return 2             # medium broadways, suited connectors
    if score >= 10: return 1             # weak suited / connectors
    return 0                              # trash


def _postflop_bucket(hole: list, board: list) -> int:
    """O(C(n,5)) post-flop hand class bucket (0 = weakest, 4 = strongest)."""
    score, _ = best_hand(hole + board)
    rank = score[0]   # -1 (high card) … 8 (royal flush)
    if rank <= 0: return 0   # high card, one pair
    if rank == 1: return 1   # two pair
    if rank == 2: return 2   # trips
    if rank <= 4: return 3   # straight, flush
    return 4                  # full house, quads, straight/royal flush


def hand_bucket(hole: list, board: list) -> int:
    return _preflop_bucket(hole) if not board else _postflop_bucket(hole, board)


# ── CFR Node (one per info set) ────────────────────────────────────────────────

class Node:
    __slots__ = ("r", "s")

    def __init__(self):
        self.r = [0.0] * 4
        self.s = [0.0] * 4

    def strategy(self, valid: list) -> list:
        pos   = [max(0.0, self.r[a]) if a in valid else 0.0 for a in range(4)]
        total = sum(pos)
        if total > 0:
            return [p / total for p in pos]
        n = len(valid)
        return [1.0 / n if a in valid else 0.0 for a in range(4)]

    def avg_strategy(self, valid: list) -> list:
        total = sum(self.s[a] for a in valid)
        if total > 0:
            return [self.s[a] / total if a in valid else 0.0 for a in range(4)]
        n = len(valid)
        return [1.0 / n if a in valid else 0.0 for a in range(4)]


# ── Game State ─────────────────────────────────────────────────────────────────

class GState:
    """
    Mutable game state. All chip values in big blinds.
    Player 0 = SB / button.  Player 1 = BB.
    """

    def __init__(self):
        self.stacks  = [STARTING_STACK - SB, STARTING_STACK - BB]
        self.pot     = SB + BB
        self.bets    = [SB, BB]   # chips put in so far this street
        self.street  = 0
        self.to_act  = 0          # pre-flop: SB first in heads-up
        self.n_raise = 0
        self.history = []
        self.bb_opt  = True       # BB may check/raise after SB's pre-flop limp
        self.folded  = -1
        self.done    = False

    def clone(self):
        g           = GState.__new__(GState)
        g.stacks    = self.stacks[:]
        g.pot       = self.pot
        g.bets      = self.bets[:]
        g.street    = self.street
        g.to_act    = self.to_act
        g.n_raise   = self.n_raise
        g.history   = self.history[:]
        g.bb_opt    = self.bb_opt
        g.folded    = self.folded
        g.done      = self.done
        return g

    def to_call(self) -> float:
        return max(self.bets) - self.bets[self.to_act]

    def valid(self) -> list:
        tc   = self.to_call()
        acts = [CALL]
        if tc > 0.01:
            acts.append(FOLD)
        if self.n_raise < MAX_RAISES:
            acts += [HALF, POT]
        return sorted(acts)

    def apply(self, action: int):
        tc = self.to_call()

        if action == FOLD:
            self.folded = self.to_act
            self.done   = True
            self.history.append(action)
            return

        if action == CALL:
            self._commit(self.to_act, tc)
            self.history.append(action)
            bets_eq = abs(self.bets[0] - self.bets[1]) < 0.01
            if bets_eq:
                if self.bb_opt and self.to_act == 0:
                    # SB limped pre-flop → give BB the option
                    self.bb_opt = False
                    self.to_act = 1
                else:
                    self._advance()
            else:
                self.to_act = 1 - self.to_act
            return

        # HALF / POT : call + raise
        self._commit(self.to_act, tc)
        raise_size = (self.pot / 2) if action == HALF else self.pot
        self._commit(self.to_act, min(raise_size, self.stacks[self.to_act]))
        self.n_raise += 1
        self.bb_opt   = False
        self.history.append(action)
        self.to_act   = 1 - self.to_act

    def _commit(self, p: int, amount: float):
        actual          = min(amount, self.stacks[p])
        self.stacks[p] -= actual
        self.bets[p]   += actual
        self.pot       += actual

    def _advance(self):
        self.bets    = [0.0, 0.0]
        self.n_raise = 0
        self.history = []
        self.bb_opt  = False
        self.street += 1
        self.to_act  = 1        # OOP (BB) acts first post-flop
        if self.street > 3:
            self.done = True

    def info_key(self, player: int, bkt: int) -> str:
        h = "".join(A_STR[a] for a in self.history)
        return f"{self.street}|{player}|{bkt}|{h}"

    def utility(self, player: int, holes: list, community: list) -> float:
        if self.folded >= 0:
            return self.pot if self.folded != player else -self.pot
        s0, _ = best_hand(holes[0] + community[:5])
        s1, _ = best_hand(holes[1] + community[:5])
        if s0 > s1:
            return  self.pot if player == 0 else -self.pot
        if s1 > s0:
            return -self.pot if player == 0 else  self.pot
        return 0.0


# ── CFR Trainer ────────────────────────────────────────────────────────────────

class CFRTrainer:

    def __init__(self):
        self.nodes: dict[str, Node] = defaultdict(Node)

    @staticmethod
    def _deal():
        deck = [Card(r, s) for r in RANKS for s in SUITS]
        random.shuffle(deck)
        return [deck[:2], deck[2:4]], deck[4:9]

    @staticmethod
    def _precompute_buckets(holes: list, community: list) -> list:
        """Compute hand bucket for each (player, street) once per iteration."""
        boards = [[], community[:3], community[:4], community[:5]]
        return [[hand_bucket(holes[p], b) for b in boards] for p in range(2)]

    def cfr(self, state: GState, player: int, reach: list,
            holes: list, community: list, bkts: list) -> float:
        if state.done:
            return state.utility(player, holes, community)

        cur   = state.to_act
        bkt   = bkts[cur][state.street]
        key   = state.info_key(cur, bkt)
        node  = self.nodes[key]
        valid = state.valid()
        strat = node.strategy(valid)

        # Accumulate strategy sum (for average strategy computation)
        for a in valid:
            node.s[a] += reach[cur] * strat[a]

        # Recurse over every action
        a_vals   = [0.0] * 4
        node_val = 0.0
        for a in valid:
            ns           = state.clone()
            ns.apply(a)
            new_reach    = reach[:]
            new_reach[cur] *= strat[a]
            v            = self.cfr(ns, player, new_reach, holes, community, bkts)
            a_vals[a]    = v
            node_val    += strat[a] * v

        # Update regrets for the acting player
        if cur == player:
            opp_reach = reach[1 - player]
            for a in valid:
                node.r[a] += opp_reach * (a_vals[a] - node_val)

        return node_val

    def train(self, iterations: int, log_every: int = 10_000):
        util = [0.0, 0.0]
        for i in range(1, iterations + 1):
            holes, community = self._deal()
            bkts             = self._precompute_buckets(holes, community)
            for p in range(2):
                s = GState()
                util[p] += self.cfr(s, p, [1.0, 1.0], holes, community, bkts)
            if i % log_every == 0:
                print(f"  iter {i:>8,}  |  avg util p0: {util[0]/i:+.4f}"
                      f"  p1: {util[1]/i:+.4f}  |  nodes: {len(self.nodes):,}")
        return self.nodes

    def strategy_for(self, key: str, valid: list) -> list:
        node = self.nodes.get(key)
        if node is None:
            n = len(valid)
            return [1.0 / n if a in valid else 0.0 for a in range(4)]
        return node.avg_strategy(valid)

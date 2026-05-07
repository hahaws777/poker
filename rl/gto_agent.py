"""
GTO agent that loads a pre-trained CFR strategy and plugs into the main game.

The agent maps real game state → abstract info-set key → CFR strategy → action.
"""

import sys, os, pickle, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.cfr import (hand_bucket, Node, GState,
                    FOLD, CALL, HALF, POT, A_STR, NUM_BUCKETS, SB, BB,
                    STARTING_STACK)
from evaluator import Card, RANKS, SUITS

STRATEGY_PATH = os.path.join(os.path.dirname(__file__), "strategy.pkl")

# ── Strategy loader ────────────────────────────────────────────────────────────

_nodes: dict | None = None

def _load():
    global _nodes
    if _nodes is None:
        if not os.path.exists(STRATEGY_PATH):
            raise FileNotFoundError(
                f"No trained strategy found at {STRATEGY_PATH}.\n"
                "Run:  python rl/train.py"
            )
        with open(STRATEGY_PATH, "rb") as f:
            _nodes = pickle.load(f)
    return _nodes


def _avg_strategy(key: str, valid: list) -> list:
    nodes = _load()
    node  = nodes.get(key)
    if node is None:
        n = len(valid)
        return [1.0/n if a in valid else 0.0 for a in range(4)]
    return node.avg_strategy(valid)


# ── Real-game → abstract mapping ──────────────────────────────────────────────

def _street_from_board(board: list) -> int:
    n = len(board)
    if n == 0: return 0
    if n == 3: return 1
    if n == 4: return 2
    return 3


def _history_from_actions(action_log: list) -> list:
    """Convert real action strings to abstract action codes."""
    codes = []
    for act in action_log:
        act = act.strip().lower()
        if act == "fold":
            codes.append(FOLD)
        elif act in ("check", "call"):
            codes.append(CALL)
        elif act.startswith("bet") or act.startswith("raise"):
            # Classify by size relative to pot (approximation)
            try:
                amt = int(act.split()[-1])
            except ValueError:
                amt = 0
            codes.append(HALF if amt <= 25 else POT)
    return codes


# ── CFR state tracker (per hand) ──────────────────────────────────────────────

class GTOState:
    """
    Tracks the abstract CFR game state in parallel with the real game.
    One instance per hand; reset between hands via reset().
    """

    def __init__(self, player_idx: int):
        self.player_idx = player_idx
        self._g = GState()

    def reset(self):
        self._g = GState()

    def update(self, acting_player: int, action_code: int):
        """Feed an action taken by any player into the abstract state."""
        self._g.to_act = acting_player
        self._g.apply(action_code)

    def query(self, hole: list, board: list) -> tuple[str, list]:
        """Return (info_key, valid_actions) for the current abstract state."""
        bkt   = hand_bucket(hole, board)
        key   = self._g.info_key(self.player_idx, bkt)
        valid = self._g.valid()
        return key, valid


# ── Agent action translation ───────────────────────────────────────────────────

def _abstract_to_real(abstract: int, to_call: int, pot: int) -> str:
    if abstract == FOLD:
        return "fold"
    if abstract == CALL:
        return "call" if to_call > 0 else "check"
    size = max(pot // 2, 1) if abstract == HALF else max(pot, 1)
    return f"raise {size}" if to_call > 0 else f"bet {size}"


# ── Public agent factory ───────────────────────────────────────────────────────

def make_gto_agent(player_idx: int):
    """
    Returns an agent function compatible with the main game engine.
    Call make_gto_agent(0) for player 0, make_gto_agent(1) for player 1.
    """
    state = GTOState(player_idx)
    prev_street = [-1]   # track street changes to detect new streets

    def agent(player, opponent, board, pot, to_call) -> str:
        street = _street_from_board(board)

        # Detect new hand (street reset to 0)
        if street < prev_street[0]:
            state.reset()
        prev_street[0] = street

        key, valid = state.query(player.hole, board)
        strat = _avg_strategy(key, valid)

        # Sample action from strategy distribution
        r   = random.random()
        acc = 0.0
        act = valid[0]
        for a in valid:
            acc += strat[a]
            if r < acc:
                act = a
                break

        # Advance internal abstract state
        state.update(player_idx, act)

        real_action = _abstract_to_real(act, to_call, pot)
        return real_action

    agent.__name__ = f"gto_p{player_idx}"
    return agent


# ── Convenience: named agent for main.py ──────────────────────────────────────

_gto0 = None
_gto1 = None

def gto_agent_0(player, opponent, board, pot, to_call) -> str:
    global _gto0
    if _gto0 is None:
        _gto0 = make_gto_agent(0)
    return _gto0(player, opponent, board, pot, to_call)

def gto_agent_1(player, opponent, board, pot, to_call) -> str:
    global _gto1
    if _gto1 is None:
        _gto1 = make_gto_agent(1)
    return _gto1(player, opponent, board, pot, to_call)

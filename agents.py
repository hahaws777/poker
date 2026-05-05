"""
Built-in agents for the 2-player Texas Hold'em game.

Each agent is a callable:
    agent(player, opponent, board, pot, to_call) -> str

Valid action strings:
    "fold"
    "check"  / "call"
    "bet N"  / "raise N"   (N = amount on top of any call)
"""

import random
from evaluator import best_hand, hand_name, Card


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _hand_strength(player, board) -> float:
    """
    Rough [0,1] strength estimate based on current best hand class.
    Uses Monte-Carlo rollout if board is incomplete (< 5 cards).
    """
    known = player.hole + board
    if len(board) >= 5 or len(known) < 2:
        score, _ = best_hand(known) if len(known) >= 5 else ((-1,), None)
        return (score[0] + 1) / 9.0   # normalize -1..8 → 0..1

    # Monte-Carlo: simulate 200 completions
    from evaluator import Deck, RANKS, SUITS, Card as C
    wins = 0
    trials = 200
    for _ in range(trials):
        deck = [C(r, s) for r in RANKS for s in SUITS
                if C(r, s).__repr__() not in {c.__repr__() for c in known}]
        random.shuffle(deck)
        needed = 5 - len(board)
        simboard = board + deck[:needed]
        score, _ = best_hand(player.hole + simboard)
        wins += (score[0] + 1)
    return wins / (trials * 9.0)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def random_agent(player, opponent, board, pot, to_call) -> str:
    """Picks a random legal action."""
    actions = ["fold", "call", "raise 20"]
    if to_call == 0:
        actions = ["check", "bet 20", "bet 40"]
    return random.choice(actions)


def call_station(player, opponent, board, pot, to_call) -> str:
    """Always calls (or checks), never raises, never folds."""
    return "call" if to_call > 0 else "check"


def tight_agent(player, opponent, board, pot, to_call) -> str:
    """Folds unless holding a pair or better; bets/raises strong hands."""
    if not board:
        # Pre-flop: play only pairs or connected/suited high cards
        v1, v2 = player.hole[0].val(), player.hole[1].val()
        suited = player.hole[0].suit == player.hole[1].suit
        pair = v1 == v2
        high = min(v1, v2) >= 10
        connected = abs(v1 - v2) <= 2
        playable = pair or (high and suited) or (high and connected)
        if not playable and to_call > 0:
            return "fold"
        return "raise 30" if pair and to_call == 0 else "call"

    score, _ = best_hand(player.hole + board)
    rank = score[0]

    if rank >= 5:       # full house or better
        return "raise 80"
    if rank >= 3:       # straight or flush
        return f"raise {min(40, player.stack)}"
    if rank >= 0:       # pair or two-pair or trips
        return "call" if to_call > 0 else "bet 20"
    # high card
    if to_call > pot * 0.3:
        return "fold"
    return "call" if to_call > 0 else "check"


def aggressive_agent(player, opponent, board, pot, to_call) -> str:
    """Bets and raises frequently; folds only on very weak hands facing large bets."""
    score, _ = best_hand(player.hole + board) if board else ((-1,), None)
    rank = score[0] if score else -1

    if to_call > player.stack * 0.5 and rank < 2:
        return "fold"
    if rank >= 4:
        return f"raise {min(pot, player.stack)}"
    if rank >= 2:
        return f"raise {min(pot // 2, player.stack)}" if random.random() < 0.6 else "call"
    if to_call == 0:
        return "bet 30" if random.random() < 0.5 else "check"
    return "call"


def human_agent(player, opponent, board, pot, to_call) -> str:
    """Interactive agent for a human player."""
    print(f"\n  Your hole cards: {player.hole}")
    print(f"  Board: {board}")
    print(f"  Pot: {pot}  |  To call: {to_call}  |  Your stack: {player.stack}")
    if board:
        score, best5 = best_hand(player.hole + board)
        print(f"  Current best hand: {hand_name(score)} {best5}")
    options = "fold / call / check / bet N / raise N"
    while True:
        raw = input(f"  Action ({options}): ").strip().lower()
        parts = raw.split()
        if parts[0] in ("fold", "check", "call"):
            return raw
        if parts[0] in ("bet", "raise") and len(parts) == 2 and parts[1].isdigit():
            return raw
        print("  Invalid input, try again.")

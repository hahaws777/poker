"""
Texas Hold'em hand evaluator and 2-agent game engine.

Hand ranks (highest to lowest):
  8 Royal Flush, 7 Straight Flush, 6 Four of a Kind, 5 Full House,
  4 Flush, 3 Straight, 2 Three of a Kind, 1 Two Pair, 0 One Pair, -1 High Card
"""

from itertools import combinations
from collections import Counter
import random


RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VAL = {r: i for i, r in enumerate(RANKS, 2)}   # '2'->2 ... 'A'->14

HAND_NAMES = {
    8: "Royal Flush",
    7: "Straight Flush",
    6: "Four of a Kind",
    5: "Full House",
    4: "Flush",
    3: "Straight",
    2: "Three of a Kind",
    1: "Two Pair",
    0: "One Pair",
    -1: "High Card",
}


# ---------------------------------------------------------------------------
# Card / Deck
# ---------------------------------------------------------------------------

class Card:
    __slots__ = ("rank", "suit")

    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit

    def __repr__(self):
        return f"{self.rank}{self.suit}"

    def val(self) -> int:
        return RANK_VAL[self.rank]


class Deck:
    def __init__(self):
        self.cards = [Card(r, s) for r in RANKS for s in SUITS]
        random.shuffle(self.cards)

    def deal(self, n: int = 1) -> list[Card]:
        if len(self.cards) < n:
            raise RuntimeError("Deck exhausted")
        drawn, self.cards = self.cards[:n], self.cards[n:]
        return drawn


# ---------------------------------------------------------------------------
# 5-card hand evaluator
# ---------------------------------------------------------------------------

def _eval5(hand: list[Card]) -> tuple:
    """
    Return a comparable tuple for exactly 5 cards.
    Higher tuple == stronger hand.
    Format: (rank_class, *tiebreak_vals)
    """
    vals = sorted((c.val() for c in hand), reverse=True)
    suits = [c.suit for c in hand]
    counts = Counter(vals)
    freq = sorted(counts.values(), reverse=True)   # e.g. [2,2,1] for two pair
    groups = sorted(counts.keys(), key=lambda v: (counts[v], v), reverse=True)

    is_flush = len(set(suits)) == 1
    is_straight = (len(set(vals)) == 5 and vals[0] - vals[4] == 4)
    # Wheel straight: A-2-3-4-5
    if set(vals) == {14, 2, 3, 4, 5}:
        is_straight = True
        vals = [5, 4, 3, 2, 1]   # treat A as low

    if is_straight and is_flush:
        if vals[0] == 14:
            return (8, 14)       # royal flush
        return (7, vals[0])      # straight flush

    if freq[0] == 4:
        return (6, *groups)      # four of a kind

    if freq[:2] == [3, 2]:
        return (5, *groups)      # full house

    if is_flush:
        return (4, *vals)        # flush

    if is_straight:
        return (3, vals[0])      # straight

    if freq[0] == 3:
        return (2, *groups)      # three of a kind

    if freq[:2] == [2, 2]:
        return (1, *groups)      # two pair

    if freq[0] == 2:
        return (0, *groups)      # one pair

    return (-1, *vals)           # high card


def best_hand(cards: list[Card]) -> tuple[tuple, list[Card]]:
    """
    Find the best 5-card hand from any number of cards (typically 7).
    Returns (score_tuple, best_5_cards).
    """
    best_score = None
    best_5 = None
    for combo in combinations(cards, 5):
        score = _eval5(list(combo))
        if best_score is None or score > best_score:
            best_score = score
            best_5 = list(combo)
    return best_score, best_5


def hand_name(score: tuple) -> str:
    return HAND_NAMES[score[0]]


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

class Player:
    def __init__(self, name: str, stack: int = 1000):
        self.name = name
        self.stack = stack
        self.hole: list[Card] = []
        self.bet_this_round = 0
        self.folded = False
        self.all_in = False

    def reset_round(self):
        self.hole = []
        self.folded = False
        self.all_in = False
        self.bet_this_round = 0

    def __repr__(self):
        return f"{self.name}(stack={self.stack})"


class GameState:
    def __init__(self, p1: Player, p2: Player, small_blind: int = 10):
        self.players = [p1, p2]
        self.small_blind = small_blind
        self.big_blind = small_blind * 2
        self.dealer = 0          # index of dealer / small-blind in heads-up
        self.hand_num = 0

    # --- helpers ------------------------------------------------------------

    def _other(self, idx: int) -> int:
        return 1 - idx

    def _active_players(self):
        return [p for p in self.players if not p.folded]

    # --- betting engine -----------------------------------------------------

    def _apply_bet(self, player: Player, amount: int) -> int:
        """Put `amount` into pot from player's stack; handle all-in. Returns actual amount."""
        actual = min(amount, player.stack)
        player.stack -= actual
        player.bet_this_round += actual
        if player.stack == 0:
            player.all_in = True
        return actual

    def _betting_round(self, deck: Deck, board: list[Card], pot: int,
                       first_to_act: int, current_bet: int,
                       agent_fn) -> tuple[int, bool]:
        """
        Run one betting street.  Returns (new_pot, someone_folded).
        agent_fn(player, opp, board, pot, to_call) -> action string
        """
        players = self.players
        n = len(players)
        acted = [False] * n
        idx = first_to_act

        # Reset per-street bet tracking
        for p in players:
            p.bet_this_round = 0

        # Re-seat current_bet as what was already posted (blinds case)
        # (caller passes current_bet=0 for post-flop)

        while True:
            active = self._active_players()
            if len(active) <= 1:
                break

            player = players[idx]
            if player.folded or player.all_in:
                idx = self._other(idx)
                continue

            # Check if we're done: everyone acted and bets are equal
            all_acted = all(
                acted[i] or players[i].folded or players[i].all_in
                for i in range(n)
            )
            bets_equal = all(
                players[i].bet_this_round == current_bet
                for i in range(n)
                if not players[i].folded and not players[i].all_in
            )
            if all_acted and bets_equal:
                break

            opp = players[self._other(idx)]
            to_call = current_bet - player.bet_this_round

            action = agent_fn(player, opp, board, pot, to_call)
            action = action.strip().lower()

            if action == "fold":
                player.folded = True
                acted[idx] = True
                print(f"  {player.name} folds.")
                idx = self._other(idx)
                return pot, True

            elif action in ("check", "call"):
                if to_call == 0:
                    print(f"  {player.name} checks.")
                else:
                    paid = self._apply_bet(player, to_call)
                    pot += paid
                    print(f"  {player.name} calls {paid}. Pot: {pot}")
                acted[idx] = True

            elif action.startswith("raise ") or action.startswith("bet "):
                parts = action.split()
                try:
                    amount = int(parts[1])
                except (IndexError, ValueError):
                    amount = self.big_blind

                # Must at least call first, then raise on top
                call_paid = self._apply_bet(player, to_call)
                raise_paid = self._apply_bet(player, amount)
                pot += call_paid + raise_paid
                current_bet = player.bet_this_round
                acted = [False] * n   # reopen action
                acted[idx] = True
                print(f"  {player.name} raises to {current_bet}. Pot: {pot}")

            else:
                # Default: check/call
                if to_call == 0:
                    print(f"  {player.name} checks (default).")
                else:
                    paid = self._apply_bet(player, to_call)
                    pot += paid
                    print(f"  {player.name} calls {paid} (default). Pot: {pot}")
                acted[idx] = True

            idx = self._other(idx)

        return pot, False

    # --- main hand ----------------------------------------------------------

    def play_hand(self, agent_fns: list) -> str | None:
        """
        Play one hand.  agent_fns[i](player, opp, board, pot, to_call) -> str
        Returns name of winner, or None on tie.
        """
        self.hand_num += 1
        deck = Deck()
        board: list[Card] = []
        pot = 0

        p0, p1 = self.players
        for p in self.players:
            p.reset_round()

        print(f"\n{'='*50}")
        print(f"Hand #{self.hand_num}  |  {p0.name}: {p0.stack}  |  {p1.name}: {p1.stack}")
        print(f"{'='*50}")

        # --- post blinds (heads-up: dealer = small blind) ---
        sb_idx = self.dealer
        bb_idx = self._other(sb_idx)
        sb, bb = self.players[sb_idx], self.players[bb_idx]

        sb_post = self._apply_bet(sb, self.small_blind)
        pot += sb_post
        bb_post = self._apply_bet(bb, self.big_blind)
        pot += bb_post
        print(f"Blinds: {sb.name} posts SB {sb_post}, {bb.name} posts BB {bb_post}. Pot: {pot}")

        # --- deal hole cards ---
        for p in self.players:
            p.hole = deck.deal(2)
        print(f"{p0.name} hole: {p0.hole}   |   {p1.name} hole: [hidden]")

        # --- pre-flop betting (SB acts first in heads-up) ---
        print("\n-- Pre-Flop --")
        current_bet = self.big_blind
        pot, folded = self._betting_round(
            deck, board, pot,
            first_to_act=sb_idx,
            current_bet=current_bet,
            agent_fn=lambda pl, op, b, p2, tc: agent_fns[self.players.index(pl)](pl, op, b, p2, tc)
        )
        if folded:
            winner = self._active_players()[0]
            winner.stack += pot
            print(f"\n>> {winner.name} wins {pot} (opponent folded)")
            self.dealer = self._other(self.dealer)
            return winner.name

        # --- flop ---
        board += deck.deal(3)
        print(f"\n-- Flop: {board} --")
        pot, folded = self._betting_round(
            deck, board, pot,
            first_to_act=bb_idx,
            current_bet=0,
            agent_fn=lambda pl, op, b, p2, tc: agent_fns[self.players.index(pl)](pl, op, b, p2, tc)
        )
        if folded:
            winner = self._active_players()[0]
            winner.stack += pot
            print(f"\n>> {winner.name} wins {pot} (opponent folded)")
            self.dealer = self._other(self.dealer)
            return winner.name

        # --- turn ---
        board += deck.deal(1)
        print(f"\n-- Turn: {board} --")
        pot, folded = self._betting_round(
            deck, board, pot,
            first_to_act=bb_idx,
            current_bet=0,
            agent_fn=lambda pl, op, b, p2, tc: agent_fns[self.players.index(pl)](pl, op, b, p2, tc)
        )
        if folded:
            winner = self._active_players()[0]
            winner.stack += pot
            print(f"\n>> {winner.name} wins {pot} (opponent folded)")
            self.dealer = self._other(self.dealer)
            return winner.name

        # --- river ---
        board += deck.deal(1)
        print(f"\n-- River: {board} --")
        pot, folded = self._betting_round(
            deck, board, pot,
            first_to_act=bb_idx,
            current_bet=0,
            agent_fn=lambda pl, op, b, p2, tc: agent_fns[self.players.index(pl)](pl, op, b, p2, tc)
        )
        if folded:
            winner = self._active_players()[0]
            winner.stack += pot
            print(f"\n>> {winner.name} wins {pot} (opponent folded)")
            self.dealer = self._other(self.dealer)
            return winner.name

        # --- showdown ---
        print("\n-- Showdown --")
        results = []
        for p in self._active_players():
            score, best5 = best_hand(p.hole + board)
            print(f"  {p.name}: {p.hole} -> {hand_name(score)} {best5}")
            results.append((score, p))

        results.sort(key=lambda x: x[0], reverse=True)
        if results[0][0] == results[1][0]:
            # Tie: split pot
            half = pot // 2
            for _, p in results:
                p.stack += half
            if pot % 2:
                results[0][1].stack += 1   # odd chip to best seat
            print(f"\n>> Tie — pot split ({half} each)")
            self.dealer = self._other(self.dealer)
            return None
        else:
            winner = results[0][1]
            winner.stack += pot
            print(f"\n>> {winner.name} wins {pot} with {hand_name(results[0][0])}")
            self.dealer = self._other(self.dealer)
            return winner.name

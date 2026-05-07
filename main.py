"""
Run a heads-up Texas Hold'em match between two agents.

Usage examples:
    python main.py                         # tight vs aggressive, 10 hands
    python main.py --hands 50              # 50-hand match
    python main.py --a1 human --a2 tight   # human vs tight agent
    python main.py --a1 gto --a2 tight     # GTO vs tight (requires trained strategy)
    python main.py --a1 random --a2 call_station --hands 100

Available agents: random, call_station, tight, aggressive, human, gto
  gto requires:  python rl/train.py   (run once to build strategy.pkl)
"""

import argparse
from evaluator import Player, GameState
from agents import random_agent, call_station, tight_agent, aggressive_agent, human_agent

def _gto_agent(player, opponent, board, pot, to_call):
    from rl.gto_agent import gto_agent_0
    return gto_agent_0(player, opponent, board, pot, to_call)

AGENT_MAP = {
    "random":       random_agent,
    "call_station": call_station,
    "tight":        tight_agent,
    "aggressive":   aggressive_agent,
    "human":        human_agent,
    "gto":          _gto_agent,
}


def run_match(a1_name: str, a2_name: str, num_hands: int, stack: int = 1000,
              small_blind: int = 10):
    p1 = Player(a1_name.capitalize(), stack)
    p2 = Player(a2_name.capitalize(), stack)
    game = GameState(p1, p2, small_blind)

    fn1 = AGENT_MAP[a1_name]
    fn2 = AGENT_MAP[a2_name]
    agent_fns = [fn1, fn2]

    wins = {p1.name: 0, p2.name: 0, "Tie": 0}

    for _ in range(num_hands):
        if p1.stack <= 0 or p2.stack <= 0:
            print("\n*** A player has busted out! ***")
            break
        result = game.play_hand(agent_fns)
        if result is None:
            wins["Tie"] += 1
        else:
            wins[result] += 1

    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")
    print(f"  {p1.name}: {p1.stack} chips  ({wins[p1.name]} wins)")
    print(f"  {p2.name}: {p2.stack} chips  ({wins[p2.name]} wins)")
    print(f"  Ties: {wins['Tie']}")
    if p1.stack > p2.stack:
        print(f"\n  {p1.name} wins the match!")
    elif p2.stack > p1.stack:
        print(f"\n  {p2.name} wins the match!")
    else:
        print("\n  Match ends in a draw!")


def main():
    parser = argparse.ArgumentParser(description="Heads-up Texas Hold'em")
    parser.add_argument("--a1", default="tight", choices=AGENT_MAP.keys())
    parser.add_argument("--a2", default="aggressive", choices=AGENT_MAP.keys())
    parser.add_argument("--hands", type=int, default=10)
    parser.add_argument("--stack", type=int, default=1000)
    parser.add_argument("--blind", type=int, default=10)
    args = parser.parse_args()

    print(f"Heads-Up Texas Hold'em: {args.a1} vs {args.a2}")
    print(f"Hands: {args.hands}  |  Starting stack: {args.stack}  |  SB: {args.blind}")
    run_match(args.a1, args.a2, args.hands, args.stack, args.blind)


if __name__ == "__main__":
    main()

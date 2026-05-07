"""
Train the CFR solver and save the resulting strategy to disk.

Usage:
    python rl/train.py                   # 100k iterations (~30s)
    python rl/train.py --iters 500000    # higher quality strategy
    python rl/train.py --iters 1000000 --log 100000
"""

import sys, os, argparse, pickle, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.cfr import CFRTrainer, GState, hand_bucket, FOLD, CALL, HALF, POT
from evaluator import Card, RANKS, SUITS

SAVE_PATH = os.path.join(os.path.dirname(__file__), "strategy.pkl")


def exploitability_proxy(trainer: CFRTrainer, n_hands: int = 1000) -> float:
    """
    Simulate n_hands using the avg strategy for both players.
    Returns mean absolute utility per hand — a proxy for how far from
    a zero-sum equilibrium we are (true GTO → both players near 0 EV).
    """
    total = [0.0, 0.0]
    for _ in range(n_hands):
        deck = [Card(r, s) for r in RANKS for s in SUITS]
        random.shuffle(deck)
        holes     = [deck[:2], deck[2:4]]
        community = deck[4:9]
        bkts      = CFRTrainer._precompute_buckets(holes, community)

        state = GState()
        while not state.done:
            cur   = state.to_act
            bkt   = bkts[cur][state.street]
            key   = state.info_key(cur, bkt)
            valid = state.valid()
            strat = trainer.strategy_for(key, valid)

            r, acc, act = random.random(), 0.0, valid[0]
            for a in valid:
                acc += strat[a]
                if r < acc:
                    act = a
                    break
            state.apply(act)

        for p in range(2):
            total[p] += state.utility(p, holes, community)

    n = n_hands
    return abs(total[0] / n), abs(total[1] / n)


def print_sample_strategy(trainer: CFRTrainer):
    """Print a few key info-set strategies for inspection."""
    print("\n  Sample strategies (avg policy — closer to GTO with more iters):")
    samples = [
        ("Pre-flop SB premium (bucket 4)", "0|0|4|"),
        ("Pre-flop SB trash   (bucket 0)", "0|0|0|"),
        ("Pre-flop BB premium (bucket 4)", "0|1|4|c"),   # after SB calls
        ("Flop     SB strong  (bucket 4)", "1|0|4|"),
        ("Flop     SB weak    (bucket 0)", "1|0|0|"),
        ("River    SB nuts    (bucket 4)", "3|0|4|"),
    ]
    action_labels = ["fold", "call/chk", "bet-half", "bet-pot"]
    valid_all = [CALL, HALF, POT]   # no fold when first to act
    valid_bet = [FOLD, CALL, HALF, POT]

    for label, key in samples:
        facing_bet = key.endswith("|c") or key.endswith("|h") or key.endswith("|p")
        valid = valid_bet if facing_bet else valid_all
        strat = trainer.strategy_for(key, valid)
        parts = "  ".join(f"{action_labels[a]}:{strat[a]:.2f}" for a in valid)
        print(f"    {label:<40} → {parts}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=100_000)
    ap.add_argument("--log",   type=int, default=25_000)
    args = ap.parse_args()

    print(f"CFR Training  |  iterations: {args.iters:,}  |  log every: {args.log:,}\n")
    trainer = CFRTrainer()

    t0 = time.time()
    trainer.train(args.iters, log_every=args.log)
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.1f}s  ({args.iters / elapsed:,.0f} iter/s)")
    print(f"Info sets learned: {len(trainer.nodes):,}")

    u0, u1 = exploitability_proxy(trainer)
    print(f"Avg |utility| proxy — p0: {u0:.4f} BB  p1: {u1:.4f} BB")
    print("  (lower = closer to Nash equilibrium)")

    print_sample_strategy(trainer)

    with open(SAVE_PATH, "wb") as f:
        pickle.dump(trainer.nodes, f)
    print(f"\nStrategy saved → {SAVE_PATH}")


if __name__ == "__main__":
    main()

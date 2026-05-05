# Texas Hold'em Evaluator

Heads-up (2-player) Texas Hold'em engine with hand evaluation and pluggable agents.

## Files

| File | Purpose |
|---|---|
| `evaluator.py` | Card/deck primitives, 5-card hand evaluator, game engine |
| `agents.py` | Built-in agent strategies |
| `main.py` | CLI match runner |

## Usage

```bash
python3 main.py                                        # tight vs aggressive, 10 hands
python3 main.py --hands 50                             # longer match
python3 main.py --a1 human --a2 tight                 # play yourself
python3 main.py --a1 random --a2 call_station --hands 100
python3 main.py --a1 aggressive --a2 tight --stack 2000 --blind 25
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--a1` | `tight` | Agent for player 1 |
| `--a2` | `aggressive` | Agent for player 2 |
| `--hands` | `10` | Number of hands to play |
| `--stack` | `1000` | Starting chips per player |
| `--blind` | `10` | Small blind size |

### Available agents

| Name | Behavior |
|---|---|
| `random` | Random legal action each turn |
| `call_station` | Always check/call, never folds or raises |
| `tight` | Folds weak pre-flop hands; bets strong made hands |
| `aggressive` | Bets and raises frequently; folds only under heavy pressure |
| `human` | Interactive CLI — you type the action |

## Writing a custom agent

An agent is any callable with this signature:

```python
def my_agent(player, opponent, board, pot, to_call) -> str:
    ...
```

| Parameter | Type | Description |
|---|---|---|
| `player` | `Player` | Your player object (`.hole`, `.stack`, `.bet_this_round`) |
| `opponent` | `Player` | Opponent object (hole cards are hidden by convention) |
| `board` | `list[Card]` | Community cards dealt so far (0–5 cards) |
| `pot` | `int` | Current pot size |
| `to_call` | `int` | Chips needed to call; 0 means you can check |

**Return one of:**
- `"fold"`
- `"check"` or `"call"`
- `"bet N"` or `"raise N"` — where N is the raise amount on top of the call

### Example: always shove with pocket aces

```python
def aces_agent(player, opponent, board, pot, to_call):
    vals = [c.val() for c in player.hole]
    if vals.count(14) == 2:          # pocket aces
        return f"raise {player.stack}"
    return "call" if to_call > 0 else "check"
```

Wire it up directly:

```python
from evaluator import Player, GameState
from agents import tight_agent

p1 = Player("AcesBot", stack=1000)
p2 = Player("Tight", stack=1000)
game = GameState(p1, p2, small_blind=10)

for _ in range(20):
    game.play_hand([aces_agent, tight_agent])
```

## Hand rankings

| Rank | Name |
|---|---|
| 8 | Royal Flush |
| 7 | Straight Flush |
| 6 | Four of a Kind |
| 5 | Full House |
| 4 | Flush |
| 3 | Straight |
| 2 | Three of a Kind |
| 1 | Two Pair |
| 0 | One Pair |
| -1 | High Card |

Hands are compared as tuples — ties are broken by kicker values automatically.

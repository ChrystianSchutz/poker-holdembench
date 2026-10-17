"""Cards, deterministic deals and equity estimation."""

from __future__ import annotations

import itertools
import random
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache

from phevaluator import evaluate_cards

RANKS = "23456789TJQKA"
SUITS = "shdc"
DECK: tuple[str, ...] = tuple(r + s for r in RANKS for s in SUITS)
SUIT_SYMBOLS = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}


@dataclass(frozen=True, slots=True)
class Deal:
    """Hole cards for both pokerkit player indices and the full five-card runout."""

    seed: int
    holes: tuple[tuple[str, str], tuple[str, str]]
    board: tuple[str, str, str, str, str]

    @classmethod
    def from_seed(cls, seed: int) -> Deal:
        """Deal nine distinct cards from an independent seeded stream."""
        c = random.Random(seed).sample(DECK, 9)
        return cls(
            seed=seed, holes=((c[0], c[1]), (c[2], c[3])), board=(c[4], c[5], c[6], c[7], c[8])
        )


def hand_class(hole: tuple[str, str] | list[str]) -> str:
    """Canonical 169-class name, e.g. ``AKs``, ``T9o`` or ``77``."""
    a, b = sorted(hole, key=lambda c: RANKS.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ("s" if a[1] == b[1] else "o")


def permute_suits(cards: list[str], mapping: dict[str, str]) -> list[str]:
    """Apply a suit relabelling; strategically the spot is unchanged."""
    return [c[0] + mapping[c[1]] for c in cards]


def _rank(cards: list[str]) -> int:
    return int(evaluate_cards(*cards))  # lower is stronger


def equity_vs_hand(
    hole: list[str], opp: list[str], board: list[str], samples: int = 20000, seed: int = 0
) -> float:
    """Showdown equity (win + tie/2) against a known hand.

    Exact enumeration from the flop on; Monte Carlo preflop.
    """
    used = set(hole) | set(opp) | set(board)
    rest = [c for c in DECK if c not in used]
    need = 5 - len(board)
    if need <= 2:
        runouts: Iterable[tuple[str, ...]] = itertools.combinations(rest, need)
    else:
        rng = random.Random(seed)
        runouts = [tuple(rng.sample(rest, need)) for _ in range(samples)]
    wins = ties = total = 0
    for extra in runouts:
        full = board + list(extra)
        a, b = _rank(hole + full), _rank(opp + full)
        wins += a < b
        ties += a == b
        total += 1
    return (wins + ties / 2) / total


def equity_vs_random(
    hole: list[str], board: list[str], samples: int = 2000, seed: int = 0
) -> float:
    """Monte Carlo equity against one uniformly random opponent hand."""
    if not board:
        return preflop_equity(hand_class(hole))
    used = set(hole) | set(board)
    rest = [c for c in DECK if c not in used]
    need = 5 - len(board)
    rng = random.Random(seed)
    wins = ties = 0.0
    for _ in range(samples):
        draw = rng.sample(rest, 2 + need)
        full = board + draw[2:]
        a, b = _rank(hole + full), _rank(draw[:2] + full)
        wins += a < b
        ties += a == b
    return (wins + ties / 2) / samples


@lru_cache(maxsize=169)
def preflop_equity(cls: str, samples: int = 6000) -> float:
    """Equity of a 169-class hand against a random hand (cached Monte Carlo)."""
    r1, r2 = cls[0], cls[1]
    hole = [r1 + "s", r2 + ("s" if cls.endswith("s") else "h")]
    used = set(hole)
    rest = [c for c in DECK if c not in used]
    rng = random.Random(zlib.crc32(cls.encode()))
    wins = ties = 0.0
    for _ in range(samples):
        draw = rng.sample(rest, 7)
        a, b = _rank(hole + draw[2:]), _rank(draw[:2] + draw[2:])
        wins += a < b
        ties += a == b
    return (wins + ties / 2) / samples


def pretty(cards: list[str] | tuple[str, ...]) -> str:
    """Human-readable card list, e.g. ``A♠ K♦``."""
    return " ".join(c[0].replace("T", "10") + SUIT_SYMBOLS[c[1]] for c in cards) if cards else "-"

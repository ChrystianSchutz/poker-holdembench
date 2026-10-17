"""Free reference bots: deterministic given the observation's seed, no API calls."""

from __future__ import annotations

import random
import zlib
from typing import Any, ClassVar

from holdembench.cards import equity_vs_random, preflop_equity
from holdembench.engine import ActionOption, Decision, Observation


class _Bot:
    kind: ClassVar[str] = "bot"

    def __init__(self, name: str, **params: Any) -> None:
        self.name = name
        self.params = params

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "params": self.params}

    async def act(self, obs: Observation) -> Decision:
        raise NotImplementedError

    def _rng(self, obs: Observation) -> random.Random:
        return random.Random(obs.rng_seed ^ zlib.crc32(self.name.encode()))


def _frac(o: ActionOption, obs: Observation) -> float:
    assert o.to is not None
    return (o.to - obs.opp_bet) / max(obs.pot + obs.to_call, 1)


def _aggressive(obs: Observation) -> list[ActionOption]:
    return [o for o in obs.options if o.kind in ("bet", "raise", "all_in")]


def _passive(obs: Observation) -> ActionOption:
    o = obs.option("check") or obs.option("call")
    assert o is not None
    return o


def _closest(obs: Observation, frac: float, *, allow_all_in: bool = False) -> ActionOption | None:
    sized = [o for o in _aggressive(obs) if allow_all_in or o.kind != "all_in"]
    if not sized:
        sized = _aggressive(obs)
    return min(sized, key=lambda o: abs(_frac(o, obs) - frac), default=None)


def _check_or_fold(obs: Observation) -> ActionOption:
    o = obs.option("check") or obs.option("fold")
    assert o is not None
    return o


class RandomBot(_Bot):
    """Uniform over the legal menu: the floor every model must clear."""

    kind = "random"

    async def act(self, obs: Observation) -> Decision:
        return Decision(self._rng(obs).choice(obs.labels))


class CallingStation(_Bot):
    """Never folds, never raises. Punishes bluffs, pays off value."""

    kind = "calling_station"

    async def act(self, obs: Observation) -> Decision:
        return Decision(_passive(obs).label)


class Maniac(_Bot):
    """Raises a pot-sized amount whenever allowed, otherwise calls."""

    kind = "maniac"

    async def act(self, obs: Observation) -> Decision:
        o = _closest(obs, 1.0) or _passive(obs)
        return Decision(o.label)


def _hand_equity(obs: Observation, samples: int) -> float:
    if not obs.board:
        return preflop_equity_of(obs.hole)
    return equity_vs_random(obs.hole, obs.board, samples=samples, seed=obs.rng_seed)


def preflop_equity_of(hole: list[str]) -> float:
    """Equity of a concrete hole-card pair against a random hand."""
    from holdembench.cards import hand_class

    return preflop_equity(hand_class(hole))


class Rock(_Bot):
    """Plays only premium hands, bets them, and folds everything else to pressure."""

    kind = "rock"

    async def act(self, obs: Observation) -> Decision:
        eq = _hand_equity(obs, 600)
        if eq >= 0.75 or (not obs.board and eq >= 0.62):
            o = _closest(obs, 0.75)
            if o is not None:
                return Decision(o.label, equity=eq)
        if eq >= 0.6:
            return Decision(_passive(obs).label, equity=eq)
        return Decision(_check_or_fold(obs).label, equity=eq)


class EquityTAG(_Bot):
    """Tight-aggressive heuristic: Monte Carlo equity, pot odds and a little bluffing.

    Equity is computed against a uniformly random hand and discounted for every
    aggressive opponent action this hand, a crude stand-in for range narrowing.
    """

    kind = "equity_tag"

    async def act(self, obs: Observation) -> Decision:
        rng = self._rng(obs)
        bluff = float(self.params.get("bluff", 0.12))
        raw = _hand_equity(obs, int(self.params.get("samples", 800)))
        pressure = sum(
            1 for e in obs.history if e.seat != obs.seat and e.kind in ("bet", "raise", "all_in")
        )
        eq = raw - 0.06 * pressure
        if eq >= 0.8 and obs.board and (o := _closest(obs, 1.0, allow_all_in=True)):
            return Decision(o.label, equity=raw)
        if eq >= 0.66 and (o := _closest(obs, 0.75)):
            return Decision(o.label, equity=raw)
        if obs.to_call > 0:
            pot_odds = obs.to_call / (obs.pot + obs.to_call)
            label = "call" if eq > pot_odds + 0.04 else "fold"
            return Decision(label, equity=raw)
        if (eq >= 0.55 or rng.random() < bluff) and (o := _closest(obs, 0.5)):
            return Decision(o.label, equity=raw)
        return Decision("check", equity=raw)


BOTS: dict[str, type[_Bot]] = {
    cls.kind: cls for cls in (RandomBot, CallingStation, Maniac, Rock, EquityTAG)
}

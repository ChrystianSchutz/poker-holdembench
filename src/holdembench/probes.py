"""Fixed probe spots: cheap, controlled comparisons of how agents decide.

Spots are captured from real hands between reference bots (so they are realistic
and cover every street). Each agent answers every spot three times:

* ``orig``  – the spot as captured;
* ``repeat`` – the identical prompt again (self-consistency);
* ``iso``   – the same spot with suits relabelled (strategically identical).

No HUD is shown, so every agent sees exactly the same text.
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import itertools
import random
from pathlib import Path
from typing import Any

import numpy as np
import orjson

from holdembench.agents.baselines import EquityTAG
from holdembench.cards import (
    SUITS,
    Deal,
    equity_vs_random,
    hand_class,
    permute_suits,
    preflop_equity,
)
from holdembench.engine import Agent, Decision, Observation, Rules, play_hand


class _Stop(Exception):
    pass


class _Capture:
    """Plays like ``inner`` until decision ``target``, then records the observation and stops."""

    kind = "capture"

    def __init__(self, name: str, inner: Agent, target: int, sink: list[Observation]) -> None:
        self.name = name
        self.inner = inner
        self.target = target
        self.sink = sink

    def describe(self) -> dict[str, Any]:
        return {}

    async def act(self, obs: Observation) -> Decision:
        if obs.decision_index == self.target:
            self.sink.append(copy.deepcopy(obs))
            raise _Stop
        return await self.inner.act(obs)


async def make_spots(n: int, seed: int = 4242, rules: Rules | None = None) -> list[Observation]:
    """Capture ``n`` distinct decision points, balanced across streets."""
    rules = rules or Rules()
    rng = random.Random(seed)
    per_street = {"preflop": n // 4, "flop": n // 4, "turn": n // 4, "river": n - 3 * (n // 4)}
    spots: list[Observation] = []
    tries = 0
    while any(v > 0 for v in per_street.values()) and tries < 50 * n:
        tries += 1
        deal = Deal.from_seed(rng.randrange(10**9))
        target = rng.randrange(0, 8)
        sink: list[Observation] = []
        bots = [EquityTAG("ref_a", bluff=0.25), EquityTAG("ref_b", bluff=0.25)]
        players = [_Capture(b.name, b, target, sink) for b in bots]
        with contextlib.suppress(_Stop):
            await play_hand(players, deal, rules, match_id="probe", hand_index=0)
        if sink and per_street[sink[0].street] > 0:
            obs = sink[0]
            obs.history = list(obs.history)
            obs.hud = None
            per_street[obs.street] -= 1
            spots.append(obs)
    return spots


def iso_variant(obs: Observation, rng: random.Random) -> Observation:
    """Relabel suits with a random non-identity permutation."""
    perm = list(SUITS)
    while perm == list(SUITS):
        rng.shuffle(perm)
    mapping = dict(zip(SUITS, perm, strict=True))
    o = copy.deepcopy(obs)
    o.hole = permute_suits(o.hole, mapping)
    o.board = permute_suits(o.board, mapping)
    return o


def spot_record(obs: Observation) -> dict[str, Any]:
    """Serializable spot description with reference equity."""
    eq = (
        preflop_equity(hand_class(obs.hole))
        if not obs.board
        else equity_vs_random(obs.hole, obs.board, 3000)
    )
    return {
        "street": obs.street,
        "seat": obs.seat,
        "hole": obs.hole,
        "board": obs.board,
        "pot": obs.pot,
        "to_call": obs.to_call,
        "labels": obs.labels,
        "equity_vs_random": eq,
        "text": obs.render(),
    }


async def run_probes(
    agents: list[Agent], spots: list[Observation], out: Path, seed: int = 99
) -> list[dict[str, Any]]:
    """Ask every agent every spot (orig, repeat, iso); append answers to ``probes.jsonl``."""
    rng = random.Random(seed)
    isos = [iso_variant(s, rng) for s in spots]
    path = out / "probes.jsonl"
    done = set()
    if path.exists():
        for line in path.read_bytes().splitlines():
            r = orjson.loads(line)
            done.add((r["agent"], r["spot"], r["variant"]))
    lock = asyncio.Lock()
    rows: list[dict[str, Any]] = []

    async def ask(agent: Agent, k: int, variant: str, obs: Observation) -> None:
        if (agent.name, k, variant) in done:
            return
        o = copy.deepcopy(obs)
        o.match_id = f"probe-{k}-{variant}"
        d = await agent.act(o)
        row = {
            "agent": agent.name,
            "spot": k,
            "variant": variant,
            "label": d.label,
            "equity": d.equity,
            "probabilities": d.probabilities,
            "cost": d.meta.get("cost"),
            "error": d.meta.get("error"),
        }
        async with lock:
            rows.append(row)
            with path.open("ab") as fh:
                fh.write(orjson.dumps(row) + b"\n")

    tasks = []
    for agent in agents:
        for k, (s, iso) in enumerate(zip(spots, isos, strict=True)):
            tasks += [
                ask(agent, k, "orig", s),
                ask(agent, k, "repeat", s),
                ask(agent, k, "iso", iso),
            ]
    await asyncio.gather(*tasks)
    return rows


def bucket(label: str, spot: dict[str, Any]) -> str:
    """Coarse action class used for agreement: fold / check / call / small / big / all-in."""
    if label in ("fold", "check", "call", "all_in"):
        return label
    try:
        to = int(label.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return "invalid"
    frac = to / max(spot["pot"] + spot["to_call"], 1)
    return "bet_small" if frac < 0.6 else "bet_big"


def analyze_probes(out: Path) -> dict[str, Any]:
    """Self-consistency, suit-isomorphism consistency, agreement matrix, blunders."""
    spots = orjson.loads((out / "spots.json").read_bytes())
    answers: dict[tuple[str, int, str], str] = {}
    for line in (out / "probes.jsonl").read_bytes().splitlines():
        r = orjson.loads(line)
        answers[(r["agent"], r["spot"], r["variant"])] = r["label"]
    agents = sorted({a for a, _, _ in answers})
    n = len(spots)

    def b(agent: str, k: int, v: str) -> str | None:
        lab = answers.get((agent, k, v))
        return None if lab is None or lab == "__invalid__" else bucket(lab, spots[k])

    per_agent = {}
    for a in agents:
        rep = [
            b(a, k, "orig") == b(a, k, "repeat")
            for k in range(n)
            if b(a, k, "orig") and b(a, k, "repeat")
        ]
        iso = [
            b(a, k, "orig") == b(a, k, "iso")
            for k in range(n)
            if b(a, k, "orig") and b(a, k, "iso")
        ]
        exact_rep = [answers.get((a, k, "orig")) == answers.get((a, k, "repeat")) for k in range(n)]
        acts = [b(a, k, "orig") for k in range(n)]
        nut_folds = sum(
            1 for k in range(n) if acts[k] == "fold" and spots[k]["equity_vs_random"] >= 0.9
        )
        nut_spots = sum(
            1 for k in range(n) if spots[k]["to_call"] > 0 and spots[k]["equity_vs_random"] >= 0.9
        )
        trash_calls = sum(
            1
            for k in range(n)
            if acts[k] in ("call", "all_in")
            and spots[k]["street"] == "river"
            and spots[k]["equity_vs_random"] < 0.15
        )
        per_agent[a] = {
            "self_consistency": float(np.mean(rep)) if rep else None,
            "self_consistency_exact": float(np.mean(exact_rep)),
            "iso_consistency": float(np.mean(iso)) if iso else None,
            "invalid": sum(
                1
                for k in range(n)
                for v in ("orig", "repeat", "iso")
                if answers.get((a, k, v)) == "__invalid__"
            ),
            "fold_rate_facing_bet": float(
                np.mean(
                    [acts[k] == "fold" for k in range(n) if spots[k]["to_call"] > 0 and acts[k]]
                )
            ),
            "aggression_rate": float(
                np.mean([str(acts[k]).startswith(("bet", "all")) for k in range(n) if acts[k]])
            ),
            "strong_hand_folds": f"{nut_folds}/{nut_spots}",
            "weak_river_calls": trash_calls,
            "mix": {x: sum(1 for y in acts if y == x) for x in sorted({str(y) for y in acts})},
        }
    m = np.full((len(agents), len(agents)), np.nan)
    for (i, a), (j, c) in itertools.product(enumerate(agents), repeat=2):
        pairs = [(b(a, k, "orig"), b(c, k, "orig" if i != j else "repeat")) for k in range(n)]
        pairs = [(x, y) for x, y in pairs if x and y]
        if pairs:
            m[i, j] = float(np.mean([x == y for x, y in pairs]))
    consensus = []
    for k in range(n):
        acts = [b(a, k, "orig") for a in agents if b(a, k, "orig")]
        if acts:
            top = max(set(acts), key=acts.count)
            consensus.append(
                {
                    "spot": k,
                    "top": top,
                    "share": acts.count(top) / len(acts),
                    "street": spots[k]["street"],
                }
            )
    return {
        "agents": agents,
        "per_agent": per_agent,
        "agreement": m.tolist(),
        "consensus": consensus,
        "spots": n,
    }

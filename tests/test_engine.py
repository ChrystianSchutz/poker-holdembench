import asyncio
import random

import pytest

from holdembench.agents.baselines import BOTS, CallingStation, EquityTAG
from holdembench.cards import Deal, equity_vs_hand, hand_class
from holdembench.engine import Decision, Observation, Rules, play_hand


def run(coro):
    return asyncio.run(coro)


class Scripted:
    kind = "scripted"

    def __init__(self, name, labels):
        self.name = name
        self.labels = list(labels)
        self.seen: list[Observation] = []

    def describe(self):
        return {}

    async def act(self, obs):
        self.seen.append(obs)
        return Decision(self.labels.pop(0) if self.labels else "check")


def test_deal_is_deterministic_and_distinct():
    d1, d2 = Deal.from_seed(5), Deal.from_seed(5)
    assert d1 == d2
    cards = [*d1.holes[0], *d1.holes[1], *d1.board]
    assert len(set(cards)) == 9


def test_hand_class():
    assert hand_class(["As", "Ks"]) == "AKs"
    assert hand_class(["2d", "Ah"]) == "A2o"
    assert hand_class(["7c", "7d"]) == "77"


@pytest.mark.parametrize("variant", ["nl", "fl"])
def test_every_bot_pair_is_zero_sum(variant):
    rules = Rules(variant=variant)
    bots = [cls(k) for k, cls in BOTS.items()]
    for seed in range(6):
        deal = Deal.from_seed(seed)
        a, b = random.Random(seed).sample(bots, 2)
        rec = run(play_hand((a, b), deal, rules, match_id="t", hand_index=0))
        assert sum(rec["payoffs"]) == 0
        assert abs(sum(rec["allin_adjusted"])) < 1e-6
        assert rec["fallbacks"] == 0


def test_observation_hides_opponent_cards():
    deal = Deal.from_seed(1)
    a, b = Scripted("a", []), Scripted("b", [])
    run(play_hand((a, b), deal, Rules(), match_id="t", hand_index=0))
    for agent, idx in ((a, 0), (b, 1)):
        for obs in agent.seen:
            text = obs.render()
            for c in deal.holes[1 - idx]:
                assert c not in obs.hole
                assert c not in text or c in obs.board


def test_sb_acts_first_preflop_and_has_sized_raises():
    sb = Scripted("sb", ["fold"])
    bb = Scripted("bb", [])
    rec = run(play_hand((bb, sb), Deal.from_seed(2), Rules(), match_id="t", hand_index=0))
    first = sb.seen[0]
    assert first.seat == "SB" and first.to_call == 50 and first.pot == 150
    assert {"fold", "call", "all_in"} <= set(first.labels)
    assert any(lab.startswith("raise_") for lab in first.labels)
    assert rec["payoffs"] == [50, -50]


def test_invalid_label_falls_back_to_check_or_fold():
    sb = Scripted("sb", ["nonsense"])
    bb = Scripted("bb", [])
    rec = run(play_hand((bb, sb), Deal.from_seed(3), Rules(), match_id="t", hand_index=0))
    assert rec["actions"][0]["label"] == "fold" and rec["actions"][0]["fallback"]


def test_all_in_adjustment_uses_equity():
    deal = Deal.from_seed(11)
    sb = Scripted("sb", ["all_in"])
    bb = Scripted("bb", ["call"])
    rec = run(play_hand((bb, sb), deal, Rules(), match_id="t", hand_index=0))
    assert rec["allin"]["board_len"] == 0
    eq = equity_vs_hand(list(deal.holes[0]), list(deal.holes[1]), [], seed=deal.seed)
    assert rec["allin_adjusted"][0] == pytest.approx(eq * 20000 - 10000)


def test_no_allin_record_without_allin():
    rec = run(
        play_hand(
            (CallingStation("a"), CallingStation("b")),
            Deal.from_seed(4),
            Rules(),
            match_id="t",
            hand_index=0,
        )
    )
    assert rec["allin"] is None
    assert rec["allin_adjusted"] == [float(p) for p in rec["payoffs"]]


def test_equity_bot_is_deterministic():
    deal = Deal.from_seed(9)
    r1 = run(play_hand((EquityTAG("x"), EquityTAG("y")), deal, Rules(), match_id="t", hand_index=0))
    r2 = run(play_hand((EquityTAG("x"), EquityTAG("y")), deal, Rules(), match_id="t", hand_index=0))
    assert [a["label"] for a in r1["actions"]] == [a["label"] for a in r2["actions"]]

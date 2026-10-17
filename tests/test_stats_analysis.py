import asyncio

import pytest

from holdembench.agents.baselines import Maniac, Rock
from holdembench.analysis import bradley_terry, duplicate_outcomes, margin_rating
from holdembench.cards import Deal
from holdembench.engine import Rules, play_hand
from holdembench.providers import Budget, BudgetExceededError, parse_answer
from holdembench.stats import hand_flags, summarize


def _hands(n=20):
    out = []
    a, b = Maniac("maniac"), Rock("rock")
    for d in range(n):
        for o, players in enumerate(((a, b), (b, a))):
            h = asyncio.run(
                play_hand(players, Deal.from_seed(d), Rules(), match_id="m", hand_index=2 * d + o)
            )
            h["deal_index"], h["deal_key"] = d, str(d)
            out.append(h)
    return out


def test_maniac_and_rock_styles():
    hands = _hands()
    m = summarize(hand_flags(h, "maniac") for h in hands)
    r = summarize(hand_flags(h, "rock") for h in hands)
    assert m["pfr"] > 0.5 and r["vpip"] < 0.4
    assert m["af"] > r["af"] or r["af"] is None or m["af"] > 1


def test_duplicate_outcomes_pair_orientations():
    hands = _hands(10)
    rows = duplicate_outcomes(hands)
    assert len(rows) == 10
    blocks = duplicate_outcomes(hands, block=5)
    assert len(blocks) == 2
    assert sum(r["net_a"] for r in rows) == pytest.approx(sum(r["net_a"] for r in blocks))


def test_ratings_order_dominant_player_first():
    rows = [{"a": "x", "b": "y", "net_a": 100.0, "deal": str(i)} for i in range(20)]
    rows += [{"a": "y", "b": "z", "net_a": 100.0, "deal": str(i)} for i in range(20)]
    agents = ["x", "y", "z"]
    bt = bradley_terry(rows, agents)
    assert bt[0] > bt[1] > bt[2]
    mr = margin_rating(rows, agents)
    assert mr[0] - mr[1] == pytest.approx(500, rel=1e-6)
    assert mr.sum() == pytest.approx(0, abs=1e-6)


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ('{"action": "call", "equity": 40}', ("call", 0.4)),
        ('thinking... {"action":"raise_300","equity":55}\n', ("raise_300", 0.55)),
        ('{"action": "bet_300", "equity": 85}\n\n{"answer": "', ("bet_300", 0.85)),
        ("I will fold here.", ("fold", None)),
        ("", (None, None)),
        ('{"action": "teleport", "equity": 5}', (None, None)),
    ],
)
def test_parse_answer(content, expected):
    assert parse_answer(content, ["fold", "call", "raise_300", "bet_300"]) == expected


def test_budget_waits_then_refuses():
    async def scenario():
        b = Budget(0.10)
        await b.reserve(0.06)
        waiter = asyncio.create_task(b.reserve(0.06))
        await asyncio.sleep(0.01)
        assert not waiter.done()
        await b.settle(0.06, 0.01)
        await waiter
        await b.settle(0.06, 0.08)
        with pytest.raises(BudgetExceededError):
            await b.reserve(0.02)
        assert b.spent == pytest.approx(0.09)

    asyncio.run(scenario())


def test_valid_hands_voids_provider_failures_and_keeps_latest():
    from holdembench.runner import valid_hands

    ok = {"match_id": "m", "hand_index": 0, "actions": [{"m_error": "unparseable answer: ''"}]}
    bad = {"match_id": "m", "hand_index": 1, "actions": [{"m_error": "HTTP 429: rate limited"}]}
    replay = {"match_id": "m", "hand_index": 1, "actions": [{}]}
    assert valid_hands([ok, bad]) == [ok]
    assert valid_hands([ok, bad, replay]) == [ok, replay]


def test_design_hash_allows_extending_deals() -> None:
    from holdembench.runner import RunConfig, design_hash

    base = RunConfig.model_validate(
        {
            "name": "x",
            "deals": 20,
            "agents": [{"id": "r", "kind": "rock"}, {"id": "m", "kind": "maniac"}],
        }
    )
    assert design_hash(base) == design_hash(
        base.model_copy(update={"deals": 25, "budget_usd": 9.0})
    )
    assert design_hash(base) != design_hash(base.model_copy(update={"deal_seed": 1}))

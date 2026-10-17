"""Standard poker-tracker statistics computed from hand records.

Definitions follow common HUD conventions (PokerTracker / Hold'em Manager), adapted
to heads-up play:

* VPIP: voluntarily put chips in preflop (a call, bet or raise; the big blind's free
  check does not count).
* PFR: raised preflop.
* 3-bet: re-raised preflop after the opponent raised (the SB open counts as the first raise;
  the big blind posting is not a raise).
* C-bet: the last preflop raiser bets the flop when given the chance to bet first.
* AF: postflop (bets + raises) / calls.
* WTSD: reached showdown, as a share of hands that saw the flop. W$SD: won at showdown.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

AGGRESSIVE = ("bet", "raise", "all_in")


def player_index(record: dict[str, Any], player: str) -> int:
    """PokerKit index (0 = BB, 1 = SB) of ``player`` in this hand."""
    players: list[str] = record["players"]
    return players.index(player)


def hand_flags(record: dict[str, Any], player: str) -> dict[str, Any]:
    """Per-hand indicators for one player."""
    idx = player_index(record, player)
    seat = "BB" if idx == 0 else "SB"
    acts = record["actions"]
    pre = [a for a in acts if a["street"] == "preflop"]
    mine_pre = [a for a in pre if a["seat"] == seat]
    f: dict[str, Any] = {"seat": seat}
    f["vpip"] = any(a["kind"] in ("call", *AGGRESSIVE) for a in mine_pre)
    f["pfr"] = any(a["kind"] in AGGRESSIVE for a in mine_pre)
    f["limp"] = seat == "SB" and bool(mine_pre) and mine_pre[0]["kind"] == "call"
    f["folded_pre"] = any(a["kind"] == "fold" for a in mine_pre)

    raises_before = 0
    f["three_bet_opp"] = f["three_bet"] = False
    f["faced_three_bet"] = f["fold_to_three_bet"] = False
    for a in pre:
        if a["seat"] == seat:
            if raises_before == 1:
                f["three_bet_opp"] = True
                f["three_bet"] = f["three_bet"] or a["kind"] in AGGRESSIVE
            if raises_before >= 2 and not f["faced_three_bet"]:
                f["faced_three_bet"] = True
                f["fold_to_three_bet"] = a["kind"] == "fold"
        if a["kind"] in AGGRESSIVE:
            raises_before += 1

    last_raiser = next((a["seat"] for a in reversed(pre) if a["kind"] in AGGRESSIVE), None)
    flop = [a for a in acts if a["street"] == "flop"]
    f["saw_flop"] = len(record["board"]) >= 3 and not any(a["kind"] == "fold" for a in pre)
    f["cbet_opp"] = f["cbet"] = f["faced_cbet"] = f["fold_to_cbet"] = False
    if f["saw_flop"] and flop:
        first_bet = next((a for a in flop if a["kind"] in AGGRESSIVE), None)
        mine_flop = [a for a in flop if a["seat"] == seat]
        if last_raiser == seat and mine_flop:
            first_mine = mine_flop[0]
            if (
                first_bet is None
                or first_bet is first_mine
                or flop.index(first_mine) < flop.index(first_bet)
            ):
                f["cbet_opp"] = True
                f["cbet"] = first_mine["kind"] in AGGRESSIVE
        if (
            last_raiser not in (seat, None)
            and first_bet is not None
            and first_bet["seat"] == last_raiser
        ):
            after = [a for a in flop[flop.index(first_bet) + 1 :] if a["seat"] == seat]
            if after:
                f["faced_cbet"] = True
                f["fold_to_cbet"] = after[0]["kind"] == "fold"

    post = [a for a in acts if a["street"] != "preflop" and a["seat"] == seat]
    f["post_aggr"] = sum(a["kind"] in AGGRESSIVE for a in post)
    f["post_calls"] = sum(a["kind"] == "call" for a in post)
    f["post_checks"] = sum(a["kind"] == "check" for a in post)
    f["post_folds"] = sum(a["kind"] == "fold" for a in post)
    f["post_actions"] = len(post)
    faced = [a for a in acts if a["seat"] == seat and a["to_call"] > 0]
    f["faced_bets"] = len(faced)
    f["folds_to_bet"] = sum(a["kind"] == "fold" for a in faced)
    f["showdown"] = bool(record["showdown"])
    f["wtsd"] = f["saw_flop"] and f["showdown"]
    f["won_sd"] = f["showdown"] and record["payoffs"][idx] > 0
    f["allin"] = any(a["kind"] == "all_in" for a in acts if a["seat"] == seat)
    f["sizes"] = [
        (a["street"], (a["to"] - a["opp_bet"]) / max(a["pot_before"] + a["to_call"], 1))
        for a in acts
        if a["seat"] == seat and a["kind"] in AGGRESSIVE and a.get("to") is not None
    ]
    f["net"] = record["payoffs"][idx]
    f["net_adj"] = record["allin_adjusted"][idx]
    return f


def summarize(flags: Iterable[dict[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate per-hand flags into rates."""
    rows = list(flags)
    n = len(rows)

    def rate(num: str, den: str | None = None) -> float | None:
        d = [r for r in rows if den is None or r[den]]
        return sum(bool(r[num]) for r in d) / len(d) if d else None

    calls = sum(r["post_calls"] for r in rows)
    aggr = sum(r["post_aggr"] for r in rows)
    faced = sum(r["faced_bets"] for r in rows)
    return {
        "hands": n,
        "vpip": rate("vpip"),
        "pfr": rate("pfr"),
        "limp": rate("limp", None) if n else None,
        "three_bet": rate("three_bet", "three_bet_opp"),
        "fold_to_three_bet": rate("fold_to_three_bet", "faced_three_bet"),
        "cbet": rate("cbet", "cbet_opp"),
        "fold_to_cbet": rate("fold_to_cbet", "faced_cbet"),
        "af": aggr / calls if calls else (float(aggr) if aggr else None),
        "fold_to_bet": sum(r["folds_to_bet"] for r in rows) / faced if faced else None,
        "wtsd": rate("wtsd", "saw_flop"),
        "wsd": rate("won_sd", "showdown"),
        "allin_rate": rate("allin"),
    }


def hud_line(stats: dict[str, Any]) -> dict[str, str]:
    """Compact opponent summary shown to agents (percentages rounded)."""

    def pct(x: float | None) -> str:
        return "n/a" if x is None else f"{100 * x:.0f}%"

    af = stats.get("af")
    return {
        "hands observed": str(stats["hands"]),
        "VPIP": pct(stats["vpip"]),
        "PFR": pct(stats["pfr"]),
        "3-bet": pct(stats["three_bet"]),
        "fold to bet": pct(stats["fold_to_bet"]),
        "postflop aggression factor": "n/a" if af is None else f"{af:.1f}",
        "went to showdown after flop": pct(stats["wtsd"]),
    }

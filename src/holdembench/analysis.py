"""Results analysis: duplicate win rates, ratings, style, calibration, similarity.

Uncertainty always comes from a *deal-cluster bootstrap*: deal seeds are resampled
with replacement and every hand of a resampled seed (both seat orientations, every
pairing) moves together. This respects the fact that all matches share the cards.
"""

from __future__ import annotations

import itertools
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from scipy.optimize import minimize

from holdembench.cards import equity_vs_hand, equity_vs_random, hand_class, preflop_equity
from holdembench.stats import hand_flags, summarize

BB = 100
ELO = 400 / math.log(10)


def load_hands(run_dirs: list[Path]) -> list[dict[str, Any]]:
    """Hands from one or more run directories (match ids are prefixed when merging)."""
    from holdembench.runner import valid_hands

    hands: list[dict[str, Any]] = []
    for d in run_dirs:
        records = []
        for line in (d / "hands.jsonl").read_bytes().splitlines():
            if line.strip():
                try:
                    records.append(orjson.loads(line))
                except orjson.JSONDecodeError:
                    continue
        for h in valid_hands(records):
            if len(run_dirs) > 1:
                h["match_id"] = f"{d.name}:{h['match_id']}"
                h["deal_key"] = f"{d.name}:{h['deal_seed']}"
            else:
                h["deal_key"] = str(h["deal_seed"])
            hands.append(h)
    return hands


def duplicate_outcomes(
    hands: list[dict[str, Any]], *, adjusted: bool = False, block: int = 1
) -> list[dict[str, Any]]:
    """One row per (match, deal block) with both orientations complete: A's duplicate net.

    ``block`` > 1 sums consecutive deals; the block key is shared by every match because
    all matches use the same deal list, so the cluster bootstrap still moves cards together.
    """
    key = "allin_adjusted" if adjusted else "payoffs"
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for h in hands:
        groups[(h["match_id"], h["deal_key"])].append(h)
    if block > 1:
        merged: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for (match, _), hs in groups.items():
            if len(hs) == 2:
                prefix = (
                    hs[0]["deal_key"].rsplit(":", 1)[0] + ":" if ":" in hs[0]["deal_key"] else ""
                )
                merged[(match, f"{prefix}b{hs[0]['deal_index'] // block}")].extend(hs)
        groups = merged
    rows = []
    for (match, deal), hs in groups.items():
        if len(hs) < 2 or len(hs) % 2:
            continue
        a, b = sorted(hs[0]["players"])
        net_a = sum(h[key][h["players"].index(a)] for h in hs)
        rows.append(
            {
                "match": match,
                "deal": deal,
                "a": a,
                "b": b,
                "net_a": float(net_a),
                "errors": sum(h["errors"] + h["fallbacks"] for h in hs),
            }
        )
    return rows


def _cluster_resample(rows: list[dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    deals = sorted({r["deal"] for r in rows})
    by_deal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_deal[r["deal"]].append(r)
    out: list[dict[str, Any]] = []
    for d in rng.choices(deals, k=len(deals)):
        out.extend(by_deal[d])
    return out


def pairwise(
    rows: list[dict[str, Any]], reps: int = 1000, seed: int = 7
) -> dict[tuple[str, str], dict[str, float]]:
    """A's win rate against B in mbb/hand with a deal bootstrap CI."""
    res: dict[tuple[str, str], dict[str, float]] = {}
    by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        by_pair[(r["a"], r["b"])].append(r["net_a"])
    rng = np.random.default_rng(seed)
    for pair, nets in by_pair.items():
        x = np.array(nets) / 2 / BB * 1000
        boots = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
        res[pair] = {
            "mbb": float(x.mean()),
            "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)),
            "deals": len(x),
            "win_share": float(np.mean(x > 0) + 0.5 * np.mean(x == 0)),
        }
    return res


def _agents(rows: list[dict[str, Any]]) -> list[str]:
    return sorted({r["a"] for r in rows} | {r["b"] for r in rows})


def bradley_terry(
    rows: list[dict[str, Any]], agents: list[str], prior_sd: float = 2.0
) -> np.ndarray:
    """MAP Bradley-Terry strengths from duplicate-deal wins (ties count half)."""
    idx = {a: i for i, a in enumerate(agents)}
    ai = np.array([idx[r["a"]] for r in rows])
    bi = np.array([idx[r["b"]] for r in rows])
    y = np.array([1.0 if r["net_a"] > 0 else 0.0 if r["net_a"] < 0 else 0.5 for r in rows])

    def nll(theta: np.ndarray) -> tuple[float, np.ndarray]:
        d = theta[ai] - theta[bi]
        p = 1 / (1 + np.exp(-d))
        eps = 1e-12
        loss = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)) + np.sum(theta**2) / (
            2 * prior_sd**2
        )
        g = np.zeros_like(theta)
        np.add.at(g, ai, -(y - p))
        np.add.at(g, bi, (y - p))
        return float(loss), g + theta / prior_sd**2

    theta = minimize(nll, np.zeros(len(agents)), jac=True, method="L-BFGS-B").x
    out: np.ndarray = theta - theta.mean()
    return out


def margin_rating(rows: list[dict[str, Any]], agents: list[str]) -> np.ndarray:
    """Least-squares additive rating in mbb/hand: E[A vs B] = s_A - s_B, sum(s) = 0."""
    idx = {a: i for i, a in enumerate(agents)}
    X = np.zeros((len(rows) + 1, len(agents)))
    y = np.zeros(len(rows) + 1)
    for k, r in enumerate(rows):
        X[k, idx[r["a"]]] = 1
        X[k, idx[r["b"]]] = -1
        y[k] = r["net_a"] / 2 / BB * 1000
    X[-1, :] = 1.0  # sum-to-zero constraint as an extra equation
    return np.linalg.lstsq(X, y, rcond=None)[0]


def ratings(
    rows: list[dict[str, Any]], reps: int = 300, seed: int = 11
) -> dict[str, dict[str, float]]:
    """Elo-scale BT and margin ratings with deal-cluster bootstrap intervals."""
    agents = _agents(rows)
    bt = bradley_terry(rows, agents)
    mr = margin_rating(rows, agents)
    rng = random.Random(seed)
    bt_s, mr_s = [], []
    for _ in range(reps):
        sample = _cluster_resample(rows, rng)
        bt_s.append(bradley_terry(sample, agents))
        mr_s.append(margin_rating(sample, agents))
    bt_a, mr_a = np.array(bt_s), np.array(mr_s)
    out = {}
    for i, a in enumerate(agents):
        out[a] = {
            "elo": 1500 + ELO * bt[i],
            "elo_lo": 1500 + ELO * float(np.percentile(bt_a[:, i], 2.5)),
            "elo_hi": 1500 + ELO * float(np.percentile(bt_a[:, i], 97.5)),
            "mbb": float(mr[i]),
            "mbb_lo": float(np.percentile(mr_a[:, i], 2.5)),
            "mbb_hi": float(np.percentile(mr_a[:, i], 97.5)),
        }
    ranks = np.argsort(-bt_a, axis=1).argsort(axis=1) + 1
    for i, a in enumerate(agents):
        out[a]["rank_lo"] = int(np.percentile(ranks[:, i], 2.5))
        out[a]["rank_hi"] = int(np.percentile(ranks[:, i], 97.5))
    return out


def style(hands: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Tracker stats, action mix and bet sizing per agent."""
    flags: dict[str, list[dict[str, Any]]] = defaultdict(list)
    mix: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for h in hands:
        for p in set(h["players"]):
            flags[p].append(hand_flags(h, p))
        for a in h["actions"]:
            mix[a["player"]][a["street"]][a["kind"]] += 1
    out = {}
    for p, fl in flags.items():
        s: dict[str, Any] = dict(summarize(fl))
        sizes = [sz for f in fl for _, sz in f["sizes"]]
        post_sizes = [sz for f in fl for st, sz in f["sizes"] if st != "preflop"]
        s["median_size_postflop"] = float(np.median(post_sizes)) if post_sizes else None
        s["size_hist"] = (
            np.histogram(np.clip(sizes, 0, 3), bins=[0, 0.4, 0.6, 0.85, 1.2, 2, 3.01])[0].tolist()
            if sizes
            else []
        )
        s["action_mix"] = {st: dict(v) for st, v in mix[p].items()}
        out[p] = s
    return out


def decisions(hands: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Cost, latency, tokens, reliability per agent."""
    acc: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
    for h in hands:
        for a in h["actions"]:
            d = acc[a["player"]]
            d["fallback"].append(a["fallback"])
            if "m_cost" in a:
                d["cost"].append(a.get("m_cost") or 0.0)
                d["latency"].append(a.get("m_latency_s"))
                d["out_tok"].append(a.get("m_out_tok"))
                d["in_tok"].append(a.get("m_in_tok"))
                d["error"].append(bool(a.get("m_error")))
                d["mismatch"].append(bool(a.get("m_argmax_mismatch")))
                d["reasoning_tokens"].append(a.get("m_reasoning_tokens"))
    hands_per: dict[str, int] = defaultdict(int)
    for h in hands:
        for p in set(h["players"]):
            hands_per[p] += 1
    out = {}
    for p, d in acc.items():

        def med(k: str, d: dict[str, list[Any]] = d) -> float | None:
            v = [x for x in d[k] if isinstance(x, (int, float))]
            return float(np.median(v)) if v else None

        def p90(k: str, d: dict[str, list[Any]] = d) -> float | None:
            v = [x for x in d[k] if isinstance(x, (int, float))]
            return float(np.percentile(v, 90)) if v else None

        n = len(d["fallback"])
        cost = float(sum(d["cost"]))
        out[p] = {
            "decisions": n,
            "hands": hands_per[p],
            "decisions_per_hand": n / max(hands_per[p], 1),
            "fallback_rate": sum(d["fallback"]) / n if n else 0.0,
            "error_rate": (sum(d["error"]) / len(d["error"])) if d["error"] else 0.0,
            "argmax_mismatch_rate": (sum(d["mismatch"]) / len(d["mismatch"]))
            if d["mismatch"]
            else 0.0,
            "cost_usd": cost,
            "usd_per_1k_hands": 1000 * cost / max(hands_per[p], 1),
            "usd_per_decision": cost / n if n else 0.0,
            "latency_med_s": med("latency"),
            "latency_p90_s": p90("latency"),
            "out_tok_med": med("out_tok"),
            "in_tok_med": med("in_tok"),
            "reasoning_tok_med": med("reasoning_tokens"),
        }
    return out


def calibration(
    hands: list[dict[str, Any]], max_points: int = 4000, seed: int = 3
) -> dict[str, dict[str, Any]]:
    """Self-reported equity vs. equity against a random hand and vs. the actual hand."""
    pts: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    rng = random.Random(seed)
    items = [(h, a) for h in hands for a in h["actions"] if a.get("equity_est") is not None]
    rng.shuffle(items)
    counts: dict[str, int] = defaultdict(int)
    for h, a in items:
        p = a["player"]
        if counts[p] >= max_points // 4:
            continue
        counts[p] += 1
        i = h["players"].index(p)
        hole, opp = h["holes"][i], h["holes"][1 - i]
        board = h["full_board"][: a["board_len"]]
        vs_rand = (
            preflop_equity(hand_class(hole))
            if not board
            else equity_vs_random(hole, board, 300, seed=1)
        )
        vs_hand = equity_vs_hand(hole, opp, board, samples=400, seed=1)
        pts[p].append((float(a["equity_est"]), vs_rand, vs_hand))
    out = {}
    for p, v in pts.items():
        arr = np.array(v)
        out[p] = {
            "n": len(v),
            "mae_vs_random": float(np.mean(np.abs(arr[:, 0] - arr[:, 1]))),
            "bias_vs_random": float(np.mean(arr[:, 0] - arr[:, 1])),
            "corr_vs_random": float(np.corrcoef(arr[:, 0], arr[:, 1])[0, 1])
            if len(v) > 2
            else None,
            "brier_vs_hand": float(np.mean((arr[:, 0] - arr[:, 2]) ** 2)),
            "points": arr[:400].round(3).tolist(),
        }
    return out


PREMIUM = frozenset({"AA", "KK", "QQ", "AKs", "AKo"})


def blunders(hands: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Unambiguous mistakes, judged only from information the agent had.

    * premium fold: folding AA/KK/QQ/AK preflop;
    * strong river fold: folding the river with >= 90% equity against a random hand;
    * hopeless river call: calling a river bet with < 10% equity against a random hand.
    """
    acc: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for h in hands:
        for a in h["actions"]:
            p = a["player"]
            i = h["players"].index(p)
            hole = h["holes"][i]
            if a["street"] == "preflop" and a["to_call"] > 0:
                if hand_class(hole) in PREMIUM:
                    acc[p]["premium_spots"] += 1
                    acc[p]["premium_folds"] += a["kind"] == "fold"
            elif a["street"] == "river" and a["to_call"] > 0:
                eq = equity_vs_random(hole, h["full_board"], 400, seed=7)
                if eq >= 0.9:
                    acc[p]["strong_river_spots"] += 1
                    acc[p]["strong_river_folds"] += a["kind"] == "fold"
                if eq < 0.1:
                    acc[p]["hopeless_river_spots"] += 1
                    acc[p]["hopeless_river_calls"] += a["kind"] == "call"
    out = {}
    for p, d in acc.items():

        def r(num: str, den: str, d: dict[str, int] = d) -> float | None:
            return d[num] / d[den] if d[den] else None

        out[p] = {
            **dict(d),
            "premium_fold_rate": r("premium_folds", "premium_spots"),
            "strong_river_fold_rate": r("strong_river_folds", "strong_river_spots"),
            "hopeless_river_call_rate": r("hopeless_river_calls", "hopeless_river_spots"),
        }
    return out


def _bucket(a: dict[str, Any]) -> str:
    kind: str = a["kind"]
    if kind in ("fold", "check", "call"):
        return kind
    if a["kind"] == "all_in":
        return "all_in"
    frac = (a["to"] - a["opp_bet"]) / max(a["pot_before"] + a["to_call"], 1)
    return "raise_small" if frac < 0.6 else "raise_mid" if frac < 1.1 else "raise_big"


def first_action_agreement(hands: list[dict[str, Any]]) -> dict[str, Any]:
    """Agreement on the identical spot: the SB's first preflop decision of each deal.

    The cards, position and menu are identical across all matches; only the opponent
    HUD text differs. Agreement is measured on coarse action buckets.
    """
    spots: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for h in hands:
        first = h["actions"][0] if h["actions"] else None
        if first and first["seat"] == "SB":
            spots[first["player"]][f"{h['deal_key']}"].append(_bucket(first))
    agents = sorted(spots)
    mat = np.full((len(agents), len(agents)), np.nan)
    for i, j in itertools.product(range(len(agents)), repeat=2):
        common = set(spots[agents[i]]) & set(spots[agents[j]])
        agree: list[bool] = []
        for s in common:
            xs, ys = spots[agents[i]][s], spots[agents[j]][s]
            if i == j:
                agree.extend(x == y for x, y in itertools.combinations(xs, 2))
            else:
                agree.extend(x == y for x in xs for y in ys)
        if agree:
            mat[i, j] = float(np.mean(agree))
    return {"agents": agents, "matrix": mat.tolist()}


def power(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Median per-deal SD of a pairing (mbb/hand) and deals needed for a given CI half-width."""
    by: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        by[(r["a"], r["b"])].append(r["net_a"] / 2 / BB * 1000)
    sds = [float(np.std(v, ddof=1)) for v in by.values() if len(v) > 5]
    if not sds:
        return {}
    sd = float(np.median(sds))
    return {
        "median_deal_sd_mbb": sd,
        **{f"deals_for_ci_{h}mbb": float((1.96 * sd / h) ** 2) for h in (1000, 500, 250)},
    }


def _merge_ratings(
    deal_level: dict[str, dict[str, float]], block_level: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """mbb/hand from deal-level rows; Elo from block-level rows; deal-level Elo kept as ``elo_deal``."""
    out: dict[str, dict[str, float]] = {}
    for a, r in deal_level.items():
        b = block_level.get(a, {})
        out[a] = {
            **r,
            "elo_deal": r["elo"],
            **{k: b[k] for k in ("elo", "elo_lo", "elo_hi", "rank_lo", "rank_hi") if k in b},
        }
    return out


def analyze(run_dirs: list[Path], reps: int = 300, block: int = 10) -> dict[str, Any]:
    """Everything the report needs."""
    hands = load_hands(run_dirs)
    rows_raw = duplicate_outcomes(hands)
    rows = duplicate_outcomes(hands, adjusted=True)
    blocks = duplicate_outcomes(hands, adjusted=True, block=block)
    res: dict[str, Any] = {
        "hands": len(hands),
        "duplicate_deals": len(rows),
        "elo_block_deals": block,
        "primary_metric": "all-in adjusted chips",
        "ratings": _merge_ratings(ratings(rows, reps=reps), ratings(blocks, reps=reps))
        if rows
        else {},
        "ratings_raw": ratings(rows_raw, reps=max(reps // 3, 50)) if rows_raw else {},
        "power": power(rows),
        "pairwise": {f"{a}|{b}": v for (a, b), v in pairwise(rows).items()},
        "pairwise_raw": {f"{a}|{b}": v for (a, b), v in pairwise(rows_raw).items()},
        "style": style(hands),
        "decisions": decisions(hands),
        "calibration": calibration(hands),
        "agreement": first_action_agreement(hands),
        "blunders": blunders(hands),
    }
    return res

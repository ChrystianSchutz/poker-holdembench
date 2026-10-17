"""Compact data bundle for the interactive blog post (ratings, styles, probes, hand replays)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import orjson

from holdembench.analysis import PREMIUM, analyze, load_hands
from holdembench.cards import equity_vs_hand, hand_class
from holdembench.stats import hand_flags

SIZE_BINS = [0.0, 0.4, 0.6, 0.9, 1.2, 2.0, 4.0]
SIZE_LABELS = ["<0.4", "0.4-0.6", "0.6-0.9", "0.9-1.2", "1.2-2", "2-4", "4+"]


def _bin(frac: float) -> int:
    return max(i for i, lo in enumerate(SIZE_BINS) if frac >= lo)


def bet_sizes(hands: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Histogram of postflop bet/raise sizes (fraction of pot) per agent."""
    out: dict[str, list[int]] = defaultdict(lambda: [0] * len(SIZE_BINS))
    for h in hands:
        for p in h["players"]:
            for street, frac in hand_flags(h, p)["sizes"]:
                if street != "preflop":
                    out[p][_bin(frac)] += 1
    return dict(out)


def _replay(h: dict[str, Any], title: str, note: str, run: str) -> dict[str, Any]:
    return {
        "title": title,
        "note": note,
        "run": run,
        "players": h["players"],
        "seats": h["seats"],
        "holes": h["holes"],
        "board": h["full_board"],
        "showdown": h["showdown"],
        "payoffs": h["payoffs"],
        "actions": [
            {
                "p": a["player"],
                "street": a["street"],
                "label": a["label"],
                "kind": a["kind"],
                "added": a["added"],
                "pot": a["pot_before"],
                "to_call": a["to_call"],
                "board_len": a["board_len"],
                "equity": a.get("equity_est"),
            }
            for a in h["actions"]
        ],
    }


def _pot(h: dict[str, Any]) -> int:
    return sum(abs(x) for x in h["payoffs"])


def pick_hands(run_hands: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """A handful of telling hands, chosen by rule so the selection is reproducible."""
    picks: list[dict[str, Any]] = []
    flat = [(run, h) for run, hs in run_hands.items() for h in hs]

    def first(pred: Any, key: Any) -> tuple[str, dict[str, Any]] | None:
        cands = [(r, h) for r, h in flat if pred(h)]
        return max(cands, key=lambda rh: key(rh[1])) if cands else None

    def premium_fold(agent: str) -> Any:
        def pred(h: dict[str, Any]) -> bool:
            if agent not in h["players"]:
                return False
            i = h["players"].index(agent)
            acts = [a for a in h["actions"] if a["player"] == agent]
            return (
                hand_class(h["holes"][i]) in PREMIUM
                and bool(acts)
                and acts[-1]["kind"] == "fold"
                and acts[-1]["street"] == "preflop"
            )

        return pred

    def river_fold_winner(agent: str) -> Any:
        def pred(h: dict[str, Any]) -> bool:
            if agent not in h["players"]:
                return False
            i = h["players"].index(agent)
            last = h["actions"][-1]
            if last["player"] != agent or last["kind"] != "fold" or last["street"] != "river":
                return False
            return equity_vs_hand(h["holes"][i], h["holes"][1 - i], h["full_board"]) == 1.0

        return pred

    def overbet(agent: str) -> Any:
        def pred(h: dict[str, Any]) -> bool:
            return agent in h["players"] and any(
                street != "preflop" and frac >= 4 for street, frac in hand_flags(h, agent)["sizes"]
            )

        return pred

    rules = [
        (premium_fold("solar_decide"), _pot, "Solar folds a premium hand", "solar_decide"),
        (overbet("jev"), _pot, "Jev's signature overbet", "jev"),
        (
            river_fold_winner("pplx_decider"),
            _pot,
            "pplx folds the best hand on the river",
            "pplx_decider",
        ),
    ]
    for pred, key, title, who in rules:
        hit = first(pred, key)
        if hit:
            picks.append(_replay(hit[1], title, who, hit[0]))
    chat = {"luna", "muse_contrib", "deepseek_flash", "glm_flash", "mimo_flash"}
    decision = {"jev", "pplx_decider", "solar_decide", "mercury_decide"}

    def script_vs(model: str) -> Any:
        def pred(h: dict[str, Any]) -> bool:
            return set(h["players"]) == {"equity_tag", model} and bool(h["showdown"])

        return pred

    hit = first(script_vs("opus"), _pot)
    if hit:
        picks.append(_replay(hit[1], "The 30-line script vs Claude Opus", "opus", hit[0]))
    groups = [
        ("Bielik, the Polish local model", lambda ps: "bielik" in ps),
        ("Biggest pot between two cheap LLMs", lambda ps: ps <= chat),
        ("Cheap LLM against a decision model", lambda ps: bool(ps & chat) and bool(ps & decision)),
    ]
    for title, ok in groups:
        hit = first(lambda h, ok=ok: ok(set(h["players"])) and h["showdown"], _pot)
        if hit:
            picks.append(_replay(hit[1], title, "", hit[0]))
    return picks


def _action_class(label: str, pot: int) -> str:
    if label in ("fold", "check", "call", "all_in"):
        return label
    m = re.search(r"(\d+)$", label)
    if not m:
        return "invalid"
    frac = int(m.group(1)) / max(pot, 1)
    return "small" if frac < 0.6 else "medium" if frac < 1.1 else "big"


def probe_grid(probe_dir: Path) -> dict[str, Any]:
    spots = orjson.loads((probe_dir / "spots.json").read_bytes())
    res = orjson.loads((probe_dir / "probe_results.json").read_bytes())
    grid: dict[str, dict[int, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    for line in (probe_dir / "probes.jsonl").read_bytes().splitlines():
        r = orjson.loads(line)
        grid[r["agent"]][r["spot"]][r["variant"]] = _action_class(
            r["label"], spots[r["spot"]]["pot"]
        )
    return {
        "agents": res["agents"],
        "agreement": res["agreement"],
        "per_agent": {
            a: {k: v for k, v in r.items() if k in ("self_consistency", "iso_consistency", "mix")}
            for a, r in res["per_agent"].items()
        },
        "spots": [
            {k: s[k] for k in ("street", "seat", "hole", "board", "pot", "to_call")} for s in spots
        ],
        "answers": {
            a: [[g[i].get(v, "") for v in ("orig", "repeat", "iso")] for i in range(len(spots))]
            for a, g in grid.items()
        },
    }


def spend(results_root: Path) -> dict[str, Any]:
    by_model: defaultdict[str, float] = defaultdict(float)
    calls: Counter[str] = Counter()
    for f in results_root.glob("*/calls.jsonl"):
        for line in f.read_bytes().splitlines():
            c = orjson.loads(line)
            by_model[c["model"]] += c.get("cost") or 0.0
            calls[c["model"]] += 1
    return {
        "total_usd": round(sum(by_model.values()), 4),
        "by_model": {
            m: {"usd": round(v, 4), "calls": calls[m]}
            for m, v in sorted(by_model.items(), key=lambda kv: -kv[1])
        },
    }


def build(
    runs: list[Path],
    probe_dir: Path | None,
    labels: dict[str, str],
    reps: int = 300,
    res: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Blog bundle; pass ``res`` to reuse an existing ``analyze`` result."""
    res = res if res is not None else analyze(runs, reps=reps)
    hands = load_hands(runs)
    run_hands = {d.name: load_hands([d]) for d in runs}
    agents = sorted(res["ratings"], key=lambda a: -res["ratings"][a]["mbb"])
    tiers = {}
    for d in runs:
        cfg = orjson.loads((d / "manifest.json").read_bytes())["config"]
        tiers.update({s["id"]: s.get("tier", "bot") for s in cfg["agents"]})
    return {
        "hands": res["hands"],
        "deals": res["duplicate_deals"],
        "runs": {d.name: len(v) for d, v in zip(runs, run_hands.values(), strict=True)},
        "agents": [
            {
                "id": a,
                "label": labels.get(a, a),
                "tier": tiers.get(a, "bot"),
                **{k: round(v, 1) for k, v in res["ratings"][a].items() if isinstance(v, float)},
                "rank_lo": res["ratings"][a]["rank_lo"],
                "rank_hi": res["ratings"][a]["rank_hi"],
                "usd_per_1k_hands": round(res["decisions"][a]["usd_per_1k_hands"], 4),
                "usd_per_decision": res["decisions"][a]["usd_per_decision"],
                "latency_s": res["decisions"][a]["latency_med_s"],
                "out_tok": res["decisions"][a]["out_tok_med"],
                "error_rate": res["decisions"][a]["error_rate"],
                "style": {k: v for k, v in res["style"][a].items() if isinstance(v, int | float)},
                "blunders": res["blunders"].get(a),
                "calibration": {
                    k: v for k, v in res["calibration"].get(a, {}).items() if k != "points"
                },
            }
            for a in agents
        ],
        "pairwise": res["pairwise"],
        "power": res["power"],
        "bet_sizes": {"bins": SIZE_LABELS, "counts": bet_sizes(hands)},
        "replays": pick_hands(run_hands),
        "probes": probe_grid(probe_dir) if probe_dir else None,
        "spend": spend(runs[0].parent),
    }

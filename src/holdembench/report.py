"""Markdown report, figures and a JSON bundle for the blog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import orjson

TIER_COLORS = {
    "bot": "#8a8f98",
    "decision": "#d9822b",
    "cheap": "#2b7bd9",
    "sota": "#a23bd9",
    "": "#555555",
}


def _tiers(run_dirs: list[Path]) -> dict[str, str]:
    tiers: dict[str, str] = {}
    for d in run_dirs:
        m = orjson.loads((d / "manifest.json").read_bytes())
        for a in m["config"]["agents"]:
            tiers[a["id"]] = a.get("tier") or (
                "bot" if a["kind"] not in ("chat", "decision") else a["kind"]
            )
    return tiers


def _pct(x: float | None) -> str:
    return "–" if x is None else f"{100 * x:.0f}%"


def _num(x: float | None, fmt: str = "{:.2f}") -> str:
    return "–" if x is None else fmt.format(x)


def write_report(res: dict[str, Any], run_dirs: list[Path], out: Path, title: str) -> Path:
    """Render ``report.md``, PNG figures and ``results.json`` into ``out``."""
    out.mkdir(parents=True, exist_ok=True)
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    tiers = _tiers(run_dirs)
    res["tiers"] = tiers
    rt = res["ratings"]
    order = sorted(rt, key=lambda a: -rt[a]["mbb"])
    dec = res["decisions"]
    sty = res["style"]

    lines = [
        f"# {title}",
        "",
        f"{res['hands']} hands, {res['duplicate_deals']} complete duplicate deal pairs. "
        "Intervals are 95% deal-cluster bootstrap intervals.",
        "",
        "## Leaderboard",
        "",
        "| # | Agent | Tier | mbb/hand (all-in adj.) | 95% CI | raw mbb/hand | Elo | 95% CI | Rank CI | Elo on single deals | $ / 1k hands |",
        "|---:|---|---|---:|---|---:|---:|---|---|---:|---:|",
    ]
    adj = res.get("ratings_raw", {})
    for k, a in enumerate(order, 1):
        r = rt[a]
        cost = dec.get(a, {}).get("usd_per_1k_hands")
        lines.append(
            f"| {k} | {a} | {tiers.get(a, '')} | {r['mbb']:+.0f} | {r['mbb_lo']:+.0f} to {r['mbb_hi']:+.0f} | "
            f"{adj.get(a, {}).get('mbb', float('nan')):+.0f} | {r['elo']:.0f} | {r['elo_lo']:.0f}–{r['elo_hi']:.0f} | "
            f"{r['rank_lo']}–{r['rank_hi']} | {r['elo_deal']:.0f} | {_num(cost, '${:.3f}')} |"
        )
    pw = res.get("power") or {}
    if pw:
        lines += [
            "",
            f"Per-deal SD of a pairing's duplicate result: {pw['median_deal_sd_mbb']:.0f} mbb/hand; "
            f"a ±1,000 mbb/hand (±1 BB) 95% CI for one pairing needs "
            f"~{pw['deals_for_ci_1000mbb']:.0f} deals, ±500 needs ~{pw['deals_for_ci_500mbb']:.0f}.",
        ]
    lines += [
        "",
        "mbb/hand (primary): all-in-adjusted additive least-squares rating in milli-big-blinds per hand, "
        "fitted to every pairing's duplicate result. Elo: Bradley–Terry on wins of "
        f"{res.get('elo_block_deals', 1)}-deal duplicate blocks (both seat orientations), mean 1500. "
        "'Elo on single deals' counts every duplicate deal as a game; it rewards winning many small "
        "pots and is shown to illustrate why it is not used.",
        "",
    ]

    lines += [
        "## Playing style",
        "",
        "| Agent | VPIP | PFR | 3-bet | C-bet | Fold to c-bet | Fold to bet | AF | WTSD | W$SD | All-in | Median postflop size (x pot) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in order:
        s = sty.get(a, {})
        lines.append(
            f"| {a} | {_pct(s.get('vpip'))} | {_pct(s.get('pfr'))} | {_pct(s.get('three_bet'))} | "
            f"{_pct(s.get('cbet'))} | {_pct(s.get('fold_to_cbet'))} | {_pct(s.get('fold_to_bet'))} | "
            f"{_num(s.get('af'))} | {_pct(s.get('wtsd'))} | {_pct(s.get('wsd'))} | {_pct(s.get('allin_rate'))} | "
            f"{_num(s.get('median_size_postflop'))} |"
        )
    model_agents = [
        a
        for a in order
        if a in dec and dec[a].get("usd_per_decision") is not None and tiers.get(a) != "bot"
    ]
    if model_agents:
        lines += [
            "",
            "## Cost, latency, reliability",
            "",
            "| Agent | Decisions | $ total | $ / decision | median latency s | p90 latency s | median output tokens | errors | fallbacks | argmax mismatch |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for a in model_agents:
            d = dec[a]
            lines.append(
                f"| {a} | {d['decisions']} | ${d['cost_usd']:.4f} | ${d['usd_per_decision']:.6f} | "
                f"{_num(d['latency_med_s'], '{:.1f}')} | {_num(d['latency_p90_s'], '{:.1f}')} | "
                f"{_num(d['out_tok_med'], '{:.0f}')} | {_pct(d['error_rate'])} | {_pct(d['fallback_rate'])} | "
                f"{_pct(d['argmax_mismatch_rate'])} |"
            )
    cal = res["calibration"]
    if cal:
        lines += [
            "",
            "## Equity self-assessment",
            "",
            "Every model decision also reports the agent's estimated showdown equity. "
            "Reference: Monte Carlo equity against a uniformly random hand (what the agent could compute) "
            "and against the opponent's actual hand (hindsight).",
            "",
            "| Agent | n | MAE vs random-hand equity | bias | correlation | Brier vs actual hand |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for a in sorted(cal, key=lambda x: cal[x]["mae_vs_random"]):
            c = cal[a]
            lines.append(
                f"| {a} | {c['n']} | {c['mae_vs_random']:.3f} | {c['bias_vs_random']:+.3f} | "
                f"{_num(c['corr_vs_random'])} | {c['brier_vs_hand']:.3f} |"
            )

    bl = res.get("blunders", {})
    if bl:
        lines += [
            "",
            "## Clear mistakes",
            "",
            "Judged only from the agent's own cards and the board (equity against a random hand).",
            "",
            "| Agent | Folded AA/KK/QQ/AK preflop | Folded river with >=90% equity | Called river with <10% equity |",
            "|---|---:|---:|---:|",
        ]

        def frac(d: dict[str, Any], num: str, den: str) -> str:
            return f"{d.get(num, 0)}/{d.get(den, 0)}" + (
                f" ({d[num] / d[den]:.0%})" if d.get(den) else ""
            )

        for a in order:
            d = bl.get(a, {})
            lines.append(
                f"| {a} | {frac(d, 'premium_folds', 'premium_spots')} | "
                f"{frac(d, 'strong_river_folds', 'strong_river_spots')} | "
                f"{frac(d, 'hopeless_river_calls', 'hopeless_river_spots')} |"
            )
    lines += ["", "## Figures", ""]
    for name, caption in _figures(res, order, tiers, fig_dir):
        lines.append(f"![{caption}](figures/{name})")
        lines.append(f"*{caption}*")
        lines.append("")
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "results.json").write_bytes(
        orjson.dumps(res, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY)
    )
    return out / "report.md"


def _color(a: str, tiers: dict[str, str]) -> str:
    return TIER_COLORS.get(tiers.get(a, ""), "#555555")


def _figures(
    res: dict[str, Any], order: list[str], tiers: dict[str, str], fig_dir: Path
) -> list[tuple[str, str]]:
    figs: list[tuple[str, str]] = []
    rt = res["ratings"]

    fig, ax = plt.subplots(figsize=(7, 0.42 * len(order) + 1.2))
    for y, a in enumerate(reversed(order)):
        r = rt[a]
        ax.plot([r["elo_lo"], r["elo_hi"]], [y, y], color=_color(a, tiers), lw=2.5, alpha=0.6)
        ax.plot(r["elo"], y, "o", color=_color(a, tiers))
    ax.set_yticks(range(len(order)), list(reversed(order)))
    ax.axvline(1500, color="#bbb", lw=0.8, ls="--")
    ax.set_xlabel("Elo (Bradley–Terry on duplicate deals), 95% CI")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "elo.png", dpi=160)
    plt.close(fig)
    figs.append(("elo.png", "Elo-scale ratings with deal-cluster bootstrap intervals"))

    n = len(order)
    mat = np.full((n, n), np.nan)
    for key, v in res["pairwise"].items():
        a, b = key.split("|")
        if a in order and b in order:
            i, j = order.index(a), order.index(b)
            mat[i, j], mat[j, i] = v["mbb"], -v["mbb"]
    lim = np.nanpercentile(np.abs(mat), 95) if np.isfinite(mat).any() else 1
    fig, ax = plt.subplots(figsize=(0.6 * n + 2.5, 0.55 * n + 1.8))
    im = ax.imshow(mat, cmap="RdBu", vmin=-lim, vmax=lim)
    ax.set_xticks(range(n), order, rotation=60, ha="right")
    ax.set_yticks(range(n), order)
    for i in range(n):
        for j in range(n):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="row vs column, mbb/hand")
    fig.tight_layout()
    fig.savefig(fig_dir / "pairwise.png", dpi=160)
    plt.close(fig)
    figs.append(("pairwise.png", "Head-to-head duplicate win rate (row agent's perspective)"))

    sty = res["style"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for a in order:
        s = sty.get(a, {})
        if s.get("vpip") is None:
            continue
        size = 40 + 60 * min(s.get("af") or 0, 6)
        ax.scatter(
            100 * s["vpip"], 100 * (s.get("pfr") or 0), s=size, color=_color(a, tiers), alpha=0.75
        )
        ax.annotate(
            a,
            (100 * s["vpip"], 100 * (s.get("pfr") or 0)),
            fontsize=7,
            xytext=(4, 3),
            textcoords="offset points",
        )
    ax.plot([0, 100], [0, 100], color="#ccc", lw=0.8)
    ax.set_xlabel("VPIP %")
    ax.set_ylabel("PFR %")
    ax.set_title("Preflop style (bubble size = postflop aggression factor)", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "style.png", dpi=160)
    plt.close(fig)
    figs.append(("style.png", "Loose/tight vs passive/aggressive"))

    dec = res["decisions"]
    pts = [
        (a, dec[a]["usd_per_1k_hands"], rt[a]["elo"])
        for a in order
        if a in dec and dec[a]["cost_usd"] > 0
    ]
    if pts:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for a, c, e in pts:
            ax.scatter(c, e, color=_color(a, tiers), s=50)
            ax.annotate(a, (c, e), fontsize=7, xytext=(4, 3), textcoords="offset points")
        ax.set_xscale("log")
        ax.set_xlabel("USD per 1,000 hands (log scale)")
        ax.set_ylabel("Elo")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "cost_vs_elo.png", dpi=160)
        plt.close(fig)
        figs.append(("cost_vs_elo.png", "What a rating point costs"))

    kinds = ["fold", "check", "call", "bet", "raise", "all_in"]
    colors = ["#c44", "#999", "#5a9", "#48c", "#36a", "#a3c"]
    for street in ("preflop", "flop"):
        fig, ax = plt.subplots(figsize=(7, 0.38 * len(order) + 1.2))
        for y, a in enumerate(reversed(order)):
            mix = sty.get(a, {}).get("action_mix", {}).get(street, {})
            tot = sum(mix.values()) or 1
            left = 0.0
            for k, c in zip(kinds, colors, strict=True):
                w = mix.get(k, 0) / tot
                ax.barh(y, w, left=left, color=c, label=k if y == 0 else None)
                left += w
        ax.set_yticks(range(len(order)), list(reversed(order)))
        ax.set_xlim(0, 1)
        ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
        fig.tight_layout()
        fig.savefig(fig_dir / f"mix_{street}.png", dpi=160)
        plt.close(fig)
        figs.append((f"mix_{street}.png", f"Action mix, {street}"))

    agr = res["agreement"]
    if agr["agents"]:
        ag = [a for a in order if a in agr["agents"]]
        idx = [agr["agents"].index(a) for a in ag]
        m = np.array(agr["matrix"])[np.ix_(idx, idx)]
        fig, ax = plt.subplots(figsize=(0.6 * len(ag) + 2.5, 0.55 * len(ag) + 1.8))
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(ag)), ag, rotation=60, ha="right")
        ax.set_yticks(range(len(ag)), ag)
        for i in range(len(ag)):
            for j in range(len(ag)):
                if np.isfinite(m[i, j]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
        fig.colorbar(im, ax=ax, label="agreement on identical first decision")
        fig.tight_layout()
        fig.savefig(fig_dir / "agreement.png", dpi=160)
        plt.close(fig)
        figs.append(
            (
                "agreement.png",
                "Do agents choose the same action in the same spot? (diagonal = self-consistency)",
            )
        )

    cal = res["calibration"]
    if cal:
        fig, ax = plt.subplots(figsize=(6, 5))
        for a, c in cal.items():
            arr = np.array(c["points"])
            if len(arr) < 5:
                continue
            bins = np.linspace(0, 1, 11)
            which = np.digitize(arr[:, 1], bins) - 1
            xs, ys = [], []
            for b in range(10):
                sel = which == b
                if sel.sum() >= 3:
                    xs.append(arr[sel, 1].mean())
                    ys.append(arr[sel, 0].mean())
            ax.plot(xs, ys, "-o", ms=3, color=_color(a, tiers), label=a, alpha=0.8)
        ax.plot([0, 1], [0, 1], color="#ccc", lw=0.8)
        ax.set_xlabel("equity vs random hand (Monte Carlo)")
        ax.set_ylabel("self-reported equity")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(fig_dir / "calibration.png", dpi=160)
        plt.close(fig)
        figs.append(("calibration.png", "Equity self-assessment vs computed equity"))
    return figs


def write_probe_report(res: dict[str, Any], out: Path, tiers: dict[str, str] | None = None) -> Path:
    """Consistency table, agreement heatmap and a behavioural dendrogram for probe results."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    tiers = tiers or {}
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    agents = res["agents"]
    per = res["per_agent"]
    lines = [
        "# Probe results",
        "",
        f"{res['spots']} fixed spots (balanced across streets), each asked as the original, an identical "
        "repeat, and a suit-relabelled copy. Actions are compared in coarse classes "
        "(fold / check / call / small bet / big bet / all-in).",
        "",
        "| Agent | Same class on identical prompt | Same exact label | Same class after suit relabelling | "
        "Fold when facing a bet | Bet/raise share | All-in | Invalid |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for a in sorted(agents, key=lambda x: -(per[x]["iso_consistency"] or 0)):
        r = per[a]
        lines.append(
            f"| {a} | {_pct(r['self_consistency'])} | {_pct(r['self_consistency_exact'])} | "
            f"{_pct(r['iso_consistency'])} | {_pct(r['fold_rate_facing_bet'])} | "
            f"{_pct(r['aggression_rate'])} | {r['mix'].get('all_in', 0)} | {r['invalid']} |"
        )
    m = np.array(res["agreement"], dtype=float)
    n = len(agents)
    fig, ax = plt.subplots(figsize=(0.6 * n + 2.5, 0.55 * n + 1.8))
    im = ax.imshow(m, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n), agents, rotation=60, ha="right")
    ax.set_yticks(range(n), agents)
    for i in range(n):
        for j in range(n):
            if np.isfinite(m[i, j]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=6, color="w")
    fig.colorbar(im, ax=ax, label="share of spots with the same action class")
    fig.tight_layout()
    fig.savefig(fig_dir / "probe_agreement.png", dpi=160)
    plt.close(fig)
    lines += [
        "",
        "![agreement](figures/probe_agreement.png)",
        "*Pairwise agreement on the original spots; the diagonal is agreement with its own repeat.*",
        "",
    ]
    if n >= 3:
        d = 1 - np.nan_to_num(m, nan=0.0)
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0)
        z = linkage(squareform(d, checks=False), method="average")
        fig, ax = plt.subplots(figsize=(7, 0.35 * n + 1.5))
        dendrogram(z, labels=agents, orientation="right", ax=ax, color_threshold=0)
        for lbl in ax.get_ymajorticklabels():
            lbl.set_color(_color(lbl.get_text(), tiers))
        ax.set_xlabel("1 − agreement (average linkage)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(fig_dir / "probe_dendrogram.png", dpi=160)
        plt.close(fig)
        lines += [
            "![dendrogram](figures/probe_dendrogram.png)",
            "*Who plays like whom: hierarchical clustering of probe decisions.*",
            "",
        ]
    cons = res["consensus"]
    if cons:
        lines += [
            "## Consensus by street",
            "",
            "| Street | Mean share choosing the plurality action |",
            "|---|---:|",
        ]
        for st in ("preflop", "flop", "turn", "river"):
            xs = [c["share"] for c in cons if c["street"] == st]
            if xs:
                lines.append(f"| {st} | {np.mean(xs):.0%} |")
    path = out / "probe_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path

"""Command line: ``holdembench run|report|probe|export|blogdata|release|estimate|credit``."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Annotated

import orjson
import typer
from rich.console import Console

from holdembench.runner import RunConfig, load_env, run

app = typer.Typer(add_completion=False, help="Duplicate heads-up hold'em benchmark.")
console = Console(emoji=False, highlight=False)


@app.command("run")
def run_cmd(
    config: Path,
    out: Annotated[Path | None, typer.Option("-o", help="output directory")] = None,
    dry_run: Annotated[bool, typer.Option(help="bots only, no API calls")] = False,
) -> None:
    """Run or resume an experiment."""
    load_env()
    cfg = RunConfig.load(config)
    out = out or Path("results") / cfg.name
    last = [0.0]

    def progress(state: dict[str, object]) -> None:
        now = time.monotonic()
        if now - last[0] > 5 or state["done"] == state["total"]:
            last[0] = now
            b = state["budget"]
            console.print(
                f"[dim]{time.strftime('%H:%M:%S')}[/] {state['done']}/{state['total']} hands  "
                f"spent ${b.spent:.4f} / ${b.cap:.2f}"  # type: ignore[attr-defined]
            )

    summary = asyncio.run(run(cfg, out, progress=progress, dry_run=dry_run))
    console.print(summary)


@app.command()
def report(
    runs: list[Path],
    out: Annotated[Path | None, typer.Option("-o")] = None,
    title: Annotated[str, typer.Option()] = "HoldemBench results",
    reps: Annotated[int, typer.Option(help="bootstrap replicates")] = 300,
) -> None:
    """Analyse one or more run directories (merged) and write a report."""
    from holdembench.analysis import analyze
    from holdembench.report import write_report

    res = analyze(runs, reps=reps)
    path = write_report(res, runs, out or runs[0] / "report", title)
    console.print(f"wrote {path}")


@app.command()
def estimate(
    config: Path,
    usd_per_decision: Annotated[list[str], typer.Option(help="agent=usd overrides")] = [],  # noqa: B006
    decisions_per_hand: float = 2.2,
) -> None:
    """Rough cost forecast from per-decision prices (measure them in a smoke run)."""
    cfg = RunConfig.load(config)
    price = {k: float(v) for k, v in (x.split("=") for x in usd_per_decision)}
    hands: dict[str, int] = {}
    for a, b in cfg.pairs():
        for x in (a, b):
            hands[x] = hands.get(x, 0) + 2 * cfg.deals
    total = 0.0
    for spec in cfg.paid_agents():
        p = price.get(spec.id, spec.max_cost_per_call / 4)
        c = hands.get(spec.id, 0) * decisions_per_hand * p
        total += c
        console.print(f"{spec.id:>16}: {hands.get(spec.id, 0):6d} hands  ~${c:.3f}")
    console.print(f"total ~${total:.3f}  (budget cap ${cfg.budget_usd})")


@app.command()
def credit() -> None:
    """Show OpenRouter key limit and usage (no charge)."""
    import os

    import httpx

    load_env()
    r = httpx.get(
        "https://openrouter.ai/api/v1/key",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    )
    data = r.json()["data"]
    console.print({k: data.get(k) for k in ("limit", "limit_remaining", "usage")})


@app.command()
def spend(run_dir: Path) -> None:
    """Sum journaled API charges per model for a run directory."""
    totals: dict[str, list[float]] = {}
    for line in (run_dir / "calls.jsonl").read_bytes().splitlines():
        c = orjson.loads(line)
        t = totals.setdefault(c["model"], [0, 0.0, 0])
        t[0] += 1
        t[1] += c["cost"] or 0.0
        t[2] += c["cost"] is None
    for m, (n, usd, unk) in sorted(totals.items()):
        console.print(f"{m:>40}: {int(n):5d} calls  ${usd:.5f}  unknown={int(unk)}")


@app.command()
def probe(
    config: Path,
    out: Annotated[Path, typer.Option("-o")],
    spots: Annotated[int, typer.Option(help="number of probe spots")] = 40,
    only: Annotated[list[str], typer.Option(help="agent ids to include (default: all)")] = [],  # noqa: B006
) -> None:
    """Ask every agent the same fixed spots (orig / repeat / suit-isomorphic) and analyse."""
    from holdembench.probes import analyze_probes, make_spots, run_probes, spot_record
    from holdembench.providers import Budget, Journal, OpenRouter
    from holdembench.runner import build_agent, prior_spend

    load_env()
    cfg = RunConfig.load(config)
    out.mkdir(parents=True, exist_ok=True)

    async def main() -> None:
        spot_path = out / "spots.json"
        obs = await make_spots(spots)
        records = [spot_record(o) for o in obs]
        if spot_path.exists() and len(orjson.loads(spot_path.read_bytes())) != len(records):
            msg = "existing spots.json has a different spot count"
            raise typer.BadParameter(msg)
        spot_path.write_bytes(orjson.dumps(records, option=orjson.OPT_INDENT_2))
        budget = Budget(cfg.budget_usd, prior_spend(out))
        router = OpenRouter(budget, Journal(out / "calls.jsonl")) if cfg.paid_agents() else None
        specs = [s for s in cfg.agents if not only or s.id in only]
        agents = [build_agent(s, router, cfg.ask_equity) for s in specs]
        try:
            await run_probes(agents, obs, out)
        finally:
            if router:
                await router.aclose()
        console.print(f"spent ${budget.spent:.4f}")

    asyncio.run(main())
    res = analyze_probes(out)
    (out / "probe_results.json").write_bytes(orjson.dumps(res, option=orjson.OPT_INDENT_2))
    from holdembench.report import write_probe_report

    tiers = {s.id: s.tier for s in cfg.agents}
    console.print(f"wrote {write_probe_report(res, out, tiers)}")
    for a, r in res["per_agent"].items():
        console.print(
            f"{a:>16}  self {r['self_consistency']}  iso {r['iso_consistency']}  "
            f"invalid {r['invalid']}  strong-folds {r['strong_hand_folds']}  mix {r['mix']}"
        )


@app.command()
def export(
    runs: list[Path],
    out: Annotated[Path, typer.Option("-o")] = Path("hf"),
) -> None:
    """Flatten runs into a Hugging Face-ready dataset (one row per decision)."""
    rows = _export(runs, out)
    console.print(f"wrote {rows} decisions to {out}")


def _export(runs: list[Path], out: Path) -> int:
    from holdembench.analysis import load_hands

    out.mkdir(parents=True, exist_ok=True)
    rows = 0
    with (out / "decisions.jsonl").open("wb") as dec, (out / "hands.jsonl").open("wb") as hf:
        for run_dir in runs:
            for h in load_hands([run_dir]):
                hf.write(orjson.dumps({"run": run_dir.name, **h}) + b"\n")
                for a in h["actions"]:
                    i = h["players"].index(a["player"])
                    dec.write(
                        orjson.dumps(
                            {
                                "run": run_dir.name,
                                "match_id": h["match_id"],
                                "hand_index": h["hand_index"],
                                "deal_seed": h["deal_seed"],
                                "decision": a["i"],
                                "agent": a["player"],
                                "opponent": h["players"][1 - i],
                                "seat": a["seat"],
                                "street": a["street"],
                                "hole": h["holes"][i],
                                "board": h["full_board"][: a["board_len"]],
                                "pot": a["pot_before"],
                                "to_call": a["to_call"],
                                "stack": a["stack_before"],
                                "options": a["options"],
                                "action": a["label"],
                                "fallback": a["fallback"],
                                "equity_estimate": a["equity_est"],
                                "probabilities": a["probabilities"],
                                "model": a.get("m_model"),
                                "latency_s": a.get("m_latency_s"),
                                "cost_usd": a.get("m_cost"),
                                "output_tokens": a.get("m_out_tok"),
                                "hand_result_chips": h["payoffs"][i],
                            }
                        )
                        + b"\n"
                    )
                    rows += 1
    return rows


LABELS = {
    "equity_tag": "equity_tag (30-line script)",
    "rock": "Rock bot",
    "maniac": "Maniac bot",
    "calling_station": "Calling station bot",
    "random": "Random bot",
    "jev": "Jev 1.13",
    "pplx_decider": "pplx-decider 27B",
    "solar_decide": "Solar Decide",
    "mercury_decide": "Mercury Decide",
    "luna": "GPT-6 Luna",
    "muse_contrib": "Muse Spark Contributor",
    "deepseek_flash": "DeepSeek V4.1 Flash",
    "glm_flash": "GLM 5.3 Flash",
    "mimo_flash": "MiMo V2.6 Flash",
    "sol": "GPT-6.1 Sol",
    "opus": "Claude Opus 5.5",
    "muse_spark": "Muse Spark 1.3",
    "bielik": "Bielik 11B v3 (local)",
}


@app.command()
def blogdata(
    runs: list[Path],
    out: Annotated[Path, typer.Option("-o")],
    probes: Annotated[Path | None, typer.Option(help="probe run directory")] = None,
    reps: Annotated[int, typer.Option(help="bootstrap replicates")] = 300,
) -> None:
    """Write the data bundle for the interactive blog post (`window.HOLDEM = {...}`)."""
    from holdembench.blogdata import build

    data = build(runs, probes, LABELS, reps=reps)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"window.HOLDEM = " + orjson.dumps(data, default=float) + b";\n")
    console.print(f"wrote {out} ({out.stat().st_size // 1024} KiB, {data['hands']} hands)")


def _leaderboard(data: dict[str, object]) -> list[dict[str, object]]:
    """Leaderboard rows for the dataset, taken from the same bundle the blog shows."""
    keys = (
        "tier",
        "mbb",
        "mbb_lo",
        "mbb_hi",
        "elo",
        "elo_lo",
        "elo_hi",
        "elo_deal",
        "rank_lo",
        "rank_hi",
    )
    rows = []
    for a in data["agents"]:  # type: ignore[attr-defined]
        rows.append(
            {
                "agent": a["id"],
                "label": a["label"],
                **{k: a[k] for k in keys if k in a},
                "usd_per_1k_hands": a["usd_per_1k_hands"],
                "latency_med_s": a["latency_s"],
                "error_rate": a["error_rate"],
            }
        )
    return rows


@app.command()
def release(
    runs: list[Path],
    probes: Annotated[Path, typer.Option(help="probe run directory (probes.jsonl, spots.json)")],
    out: Annotated[Path, typer.Option("-o", help="Hugging Face dataset folder")] = Path(
        "results/hf"
    ),
    extra: Annotated[
        list[Path] | None, typer.Option(help="runs exported to the dataset but not rated")
    ] = None,
    blog: Annotated[Path | None, typer.Option(help="also write the blog summary JSON here")] = None,
    report_dir: Annotated[Path | None, typer.Option("--report", help="report + figures")] = None,
    reps: Annotated[int, typer.Option(help="bootstrap replicates")] = 300,
) -> None:
    """Everything that is published, from one analysis, so the numbers cannot drift.

    Writes the dataset (decisions, hands, probes, spots, leaderboard, card) to ``out``, and
    optionally the blog summary and the Markdown report with figures.
    """
    import shutil

    from holdembench.analysis import analyze
    from holdembench.blogdata import build
    from holdembench.report import write_report

    res = analyze(runs, reps=reps)
    data = build(runs, probes, LABELS, reps=reps, res=res)
    out.mkdir(parents=True, exist_ok=True)
    rows = _export([*(extra or []), *runs], out)
    shutil.copyfile(probes / "probes.jsonl", out / "probes.jsonl")
    shutil.copyfile(probes / "spots.json", out / "spots.json")
    (out / "leaderboard.json").write_bytes(
        orjson.dumps(_leaderboard(data), default=float, option=orjson.OPT_INDENT_2)
    )
    card = Path(__file__).resolve().parents[2] / "docs" / "hf_dataset_card.md"
    shutil.copyfile(card, out / "README.md")
    console.print(f"dataset: {out} ({rows} decisions, {data['hands']} rated hands)")
    if blog:
        blog.parent.mkdir(parents=True, exist_ok=True)
        blog.write_bytes(orjson.dumps(data, default=float, option=orjson.OPT_INDENT_2))
        console.print(f"blog summary: {blog}")
    if report_dir:
        console.print(f"report: {write_report(res, runs, report_dir, 'HoldemBench results')}")
    console.print(f"total spend: ${data['spend']['total_usd']:.2f}")

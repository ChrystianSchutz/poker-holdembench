# HoldemBench

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-poker--holdembench-yellow)](https://huggingface.co/datasets/chrystians/poker-holdembench)
[![Write-up](https://img.shields.io/badge/write--up-interactive-blue)](https://chrystianschutz.com/blog/ai-poker-holdembench/)
[![Technical report](https://img.shields.io/badge/report-PDF-lightgrey)](docs/holdembench-paper.pdf)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Duplicate heads-up hold'em for decision models, cheap and frontier LLMs, local models and classic poker bots.**

HoldemBench measures how well AI agents play heads-up no-limit Texas hold'em, and *how*
they play, for cents per model. It compares five families through one identical action
interface:

| Family | Examples | Price per decision |
|---|---|---|
| Typed **decision models** (OpenRouter `/alpha/decisions`) | Jev 1.13, Solar Decide, Mercury Decide, pplx-decider-27B | ~$0.00005 |
| Cheap reasoning **chat LLMs** | GPT-6 Luna, Muse Spark Contributor, DeepSeek V4.1 Flash, GLM 5.3 Flash, MiMo V2.6 Flash | $0.0002–0.002 |
| **Frontier LLMs** | GPT-6.1 Sol, Claude Opus 5.5 | $0.003–0.007 |
| **Local** open-weights (any OpenAI-compatible server) | Bielik 11B v3 via llama.cpp | free |
| Reference **bots** | random, calling station, maniac, rock, Monte Carlo equity TAG | free |

## Results (7,729 hands, $16.44 of API calls)

All-in-adjusted win rate in big blinds per hand against the whole field, 95% bootstrap interval.

| # | Player | Tier | BB/hand | 95% CI | $ / 1k hands |
|---:|---|---|---:|---|---:|
| 1 | DeepSeek V4.1 Flash | cheap LLM | +5.4 | +3.4 to +8.0 | $6.45 |
| 2 | Muse Spark Contributor | cheap LLM | +4.3 | +2.9 to +6.0 | $1.59 |
| 3 | GPT-6.1 Sol | frontier | +3.4 | +2.3 to +4.6 | $8.68 |
| 4 | GPT-6 Luna | cheap LLM | +3.0 | +1.8 to +4.2 | $0.71 |
| 5 | Claude Opus 5.5 | frontier | +2.7 | +1.6 to +3.7 | $21.97 |
| 6 | GLM 5.3 Flash | cheap LLM | +2.5 | +0.6 to +4.2 | $0.60 |
| 7 | MiMo V2.6 Flash | cheap LLM | +2.4 | +1.5 to +3.5 | $0.97 |
| 8 | equity_tag (~30-line script) | bot | +2.4 | +1.5 to +3.5 | free |
| 9 | Jev 1.13 | decision | +2.0 | +0.5 to +3.1 | $0.16 |
| 10 | rock | bot | −0.1 | −1.1 to +0.8 | free |
| 11 | pplx-decider 27B | decision | −0.8 | −1.9 to +0.1 | $0.17 |
| 12 | maniac | bot | −1.8 | −3.9 to +0.0 | free |
| 13 | calling station | bot | −2.8 | −4.4 to −1.2 | free |
| 14 | Mercury Decide | decision | −3.2 | −6.3 to −0.2 | free |
| 15 | Solar Decide | decision | −4.0 | −5.6 to −2.5 | $0.15 |
| 16 | random | bot | −7.0 | −9.0 to −5.0 | free |
| 17 | Bielik 11B v3 (local) | local | −8.4 | −13.9 to −2.7 | free |

Interactive write-up: https://chrystianschutz.com/blog/ai-poker-holdembench/ ·
technical report: [`docs/paper.md`](docs/paper.md) ([PDF](docs/holdembench-paper.pdf)) ·
dataset: [chrystians/poker-holdembench](https://huggingface.co/datasets/chrystians/poker-holdembench) ([card](docs/hf_dataset_card.md)) ·
lab notebook: [`docs/blog.md`](docs/blog.md).

## Why this design

* **Duplicate poker.** Every deal is played twice with seats swapped, and every pairing
  plays the same deals, so card luck cancels instead of dominating a few hundred hands.
* **One action menu for everyone.** Fold, check/call, five pot-fraction sizes, all-in.
  A decision model and a chat model choose from exactly the same options.
* **Rules by [PokerKit](https://github.com/uoftcprg/pokerkit).** No hand-rolled betting
  or showdown logic.
* **Honest statistics.** Win rate (mbb/hand) from an additive model over all pairings,
  Bradley–Terry Elo on 10-deal duplicate blocks, a deal-cluster bootstrap for every
  interval, and an all-in equity adjustment.
* **Behaviour, not just profit.** VPIP, PFR, 3-bet, c-bet, aggression factor, WTSD,
  bet sizing, action mix by street, equity self-assessment vs. Monte Carlo truth,
  and *probes*: identical-prompt consistency and suit-relabelling invariance.
* **Cost safety.** Every request reserves its worst-case cost against a hard budget
  before it is sent, settles to the provider-reported charge, and is journaled. A
  circuit breaker stops a run when a model keeps failing, instead of letting
  fallbacks masquerade as play.

## Quick start

```bash
uv sync
uv run holdembench run configs/bots.yaml          # free, deterministic, ~3 min
uv run holdembench report results/bots            # report.md + figures + results.json
```

With an OpenRouter key in `.env` (`OPENROUTER_API_KEY=...`):

```bash
uv run holdembench credit                          # key limit and usage, no charge
uv run holdembench run configs/smoke.yaml          # every model, 4 hands each, ~$0.03
uv run holdembench probe configs/roster-cheap.yaml -o results/probes --spots 40
uv run holdembench run configs/run1a-decision-league.yaml
uv run holdembench report results/run1a-decision-league results/run1b-chat-league -o results/run1
```

Runs resume from where they stopped; a different configuration in the same output
directory is refused.

## Reproducing the published numbers

Raw run directories (`results/`) are not in git; the flattened data is on [Hugging Face](https://huggingface.co/datasets/chrystians/poker-holdembench). With the
run directories in place, one command rebuilds everything that is published from a single
analysis, so the dataset leaderboard, the report and the blog cannot disagree:

```bash
uv run holdembench release results/run1a-decision-league results/run1a-mercury   results/run1b-chat-league results/run1c-bielik results/run2-frontier results/run2b-confirm   --probes results/probes-01 --extra results/bots -o results/hf --report results/release-report
uv run python scripts/paper_pdf.py      # docs/paper.md -> docs/holdembench-paper.pdf
```

## Layout

```
src/holdembench/
  cards.py        deterministic deals, Monte Carlo / exact equity (phevaluator)
  engine.py       PokerKit hand runner, action menu, observation text, hand records
  agents/         reference bots; chat and decision-model agents
  providers.py    OpenRouter chat + Decisions clients, budget guard, call journal
  runner.py       configs, duplicate scheduling, resumable concurrent execution
  stats.py        tracker statistics (VPIP, PFR, 3-bet, c-bet, AF, WTSD, ...)
  analysis.py     ratings, bootstrap, style, calibration, agreement
  probes.py       fixed-spot consistency and invariance probes
  report.py       Markdown report and figures
configs/          frozen experiment definitions
docs/             technical report (Markdown + PDF), dataset card, lab notebook
scripts/          paper PDF rendering
```

## Development

```bash
uv run pytest -q
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy            # strict
```

## Citation

```bibtex
@techreport{schutz2026holdembench,
  title  = {HoldemBench: Duplicate Heads-Up Hold'em as a Cheap, Variance-Controlled Benchmark
            for Decision Models, Chat LLMs, Local Models and Scripts},
  author = {Schutz, Chrystian},
  year   = {2026},
  month  = oct,
  url    = {https://chrystianschutz.com/blog/ai-poker-holdembench/}
}
```

## License

MIT

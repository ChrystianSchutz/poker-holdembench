---
license: mit
task_categories: [reinforcement-learning, text-generation]
tags: [poker, texas-holdem, llm-evaluation, benchmark, decision-making, game-theory, openrouter]
pretty_name: HoldemBench
size_categories: [10K<n<100K]
configs:
  - config_name: decisions
    data_files: decisions.jsonl
  - config_name: hands
    data_files: hands.jsonl
  - config_name: probes
    data_files: probes.jsonl
---

# HoldemBench: 17 AIs, 7,700 hands of heads-up no-limit hold'em

Every hand and every decision from HoldemBench, a benchmark that asks which kind of AI makes
better sequential decisions under uncertainty: OpenRouter **decision models** (Decisions API),
**cheap reasoning LLMs**, **frontier LLMs**, a **local open-weights model**, and **plain scripts**.

All games are heads-up no-limit Texas hold'em, blinds 50/100, 100 big blinds per player, stacks
reset every hand. Every deal is played **twice with the seats swapped** (duplicate poker), and
all-in hands before the river are scored by their exact equity, so most card luck cancels out.

## Players

| Tier | Players |
|---|---|
| Decision models (Decisions API) | Jev 1.13, pplx-decider 27B, Solar Decide, Mercury Decide |
| Cheap reasoning LLMs | GPT-6 Luna, Muse Spark 1.3 Contributor, DeepSeek V4.1 Flash, GLM 5.3 Flash, MiMo V2.6 Flash |
| Frontier LLMs | GPT-6.1 Sol, Claude Opus 5.5 (default reasoning) |
| Local open-weights | Bielik 11B v3.0 Instruct (Q8_0 GGUF, llama.cpp) |
| Scripts | equity_tag (Monte Carlo equity, ~30 lines), rock, maniac, calling station, random |

## Files

| File | Rows | Content |
|---|---:|---|
| `decisions.jsonl` | ~52,000 | One row per decision: private cards, board, pot, stack, legal menu, chosen action, the player's own win-probability estimate, the full option distribution (decision models), model id, latency, cost in USD, output tokens, the hand's final chip result. |
| `hands.jsonl` | ~9,700 | Full hand records: both players' cards, the five board cards, every action with its metadata, payoffs, all-in-adjusted payoffs, showdown flag. Includes the bot-only calibration league. |
| `probes.jsonl` | ~2,000 | 40 frozen spots asked to every player three times: original, repeated, and with suits renamed (strategically identical). |
| `spots.json` | 40 | The frozen probe spots with their full prompt text. |
| `leaderboard.json` | 17 | Final ratings (BB/hand, all-in adjusted, with 95% bootstrap intervals), Elo on 10-deal blocks, cost and latency per player. |

### `decisions.jsonl` columns

| Column | Type | Meaning |
|---|---|---|
| `run`, `match_id`, `hand_index`, `deal_seed` | str / int | League, pairing, hand within the match, deal seed (both orientations of a deal share it) |
| `decision` | int | Index of the action within the hand |
| `agent`, `opponent`, `seat` | str | Acting player, opponent, `SB` or `BB` |
| `street`, `hole`, `board`, `pot`, `to_call`, `stack` | | Situation the player saw (chips; big blind = 100) |
| `options` | list[str] | Legal menu, e.g. `fold`, `call`, `raise_400`, `all_in` (numbers are chip totals) |
| `action` | str | Chosen option |
| `fallback` | bool | True when the model's answer was unusable and check/fold was played instead |
| `equity_estimate` | float or null | The player's own stated chance to win (0–1) |
| `probabilities` | dict or null | Decision models only: probability of every option |
| `model`, `latency_s`, `cost_usd`, `output_tokens` | | Model id, wall time, provider-reported cost and output tokens of the call |
| `hand_result_chips` | int | The player's chip result for the whole hand (not all-in adjusted) |

`hands.jsonl` has one row per hand: `players`, `seats`, `holes`, `full_board`, `actions` (each with
`pot_before`, `to_call`, `added`, `equity_est`, `probabilities`, `fallback`), `payoffs`,
`allin_adjusted` (all-in hands scored by equity, the benchmark's primary result), `showdown`,
`deal_index` and `orientation`. `leaderboard.json` is written by the same analysis as the
technical report and the write-up, so its numbers match them exactly.

`run` names the league: `run1a-decision-league` (decision models and bots), `run1a-mercury`,
`run1b-chat-league` (cheap LLMs), `run1c-bielik` (local model), `run2-frontier` and
`run2b-confirm` (frontier models), `bots` (bot-only calibration).

## Headline results

- Cheap reasoning LLMs (DeepSeek V4.1 Flash +5.4, Muse Spark Contributor +4.3, Luna +3.0 BB/hand)
  played as well as GPT-6.1 Sol (+3.4) and Claude Opus 5.5 (+2.7), at 1/10 to 1/30 of the price.
- The ~30-line `equity_tag` script beat every decision model it played enough hands against
  (Jev by 2.7 BB/hand over 100 duplicate deals) and drew with Opus over 70 deals.
- Decision models are deterministic but play exploitable poker (Jev's median postflop bet is 6x pot;
  Solar goes all-in in 56% of hands and folded aces).
- Bielik finished last (−8.4 BB/hand): it never folded to a bet.

## Notes

- Hidden reasoning is not included; prompts can be rebuilt exactly from the state with
  `holdembench.engine.Observation.render`.
- Hands voided by provider failures (rate limits, timeouts, HTTP errors) were replayed and are
  excluded; model failures (truncated or unparseable answers) count against the model and appear
  as `fallback: true`.
- Model outputs are subject to each provider's terms of use.

Code, methodology and technical report: https://github.com/ChrystianSchutz/poker-holdembench ·
Write-up: https://chrystianschutz.com/blog/ai-poker-holdembench/

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

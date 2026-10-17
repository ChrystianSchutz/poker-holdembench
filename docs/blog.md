# HoldemBench: can a "decision model" play poker? (working draft)

> Status: **lab notebook / draft.** Numbers below are labelled by phase. Anything
> marked *anecdote* is a single observation, not a result.

## The question

Most LLM poker experiments pit a few chat models against each other for a few hundred
hands and report who ended up with the most chips. Heads-up no-limit hold'em has an
enormous variance, so those leaderboards are mostly measuring the deck.

This project asks three narrower questions:

1. **Decision models vs. chat LLMs vs. classic bots.** OpenRouter now serves
   *decision models* (`typesafe/jev-1.13`, `upstage/solar-decide`,
   `inception/mercury-decide`, `perplexity/pplx-decider-v1-27b`) through a typed
   `/alpha/decisions` endpoint: you send a state and a finite menu of options, they
   return a choice plus a probability for every option. They are 10-100x cheaper per
   decision than a reasoning chat model. Are they any good at poker?
2. **Cheap vs. expensive.** Can a large pool of cheap games build a rating scale that
   a small number of expensive frontier-model games can be placed on?
3. **Do models play alike?** Same leaks, same bet sizes, same reaction to the same
   spot? Or are their "styles" as different as human regulars'?

## Design in one paragraph

Heads-up no-limit Texas hold'em, blinds 50/100, 100 big blinds, stacks reset every
hand. Rules are enforced by [PokerKit](https://github.com/uoftcprg/pokerkit), not by
hand-written code. Every deal is played **twice with the seats swapped** (duplicate
poker), and every pairing uses the **same list of deals**, so card luck cancels
between the two orientations and between opponents. Every agent, whether LLM,
decision model or bot, chooses from the same finite menu of actions (fold, check/call,
a few pot-fraction sizes, all-in), so differences come from choices, not from
interface. All-in hands are additionally scored by all-in equity (the standard
"all-in EV" luck adjustment).

## Field notes

### Day 0: the Decisions API works, and it is extremely cheap

The first probe sent one identical spot to all four decision models: we hold
**A♠ K♦ on the button**, facing the big blind, with options fold / call / raise to
2.5 BB / raise to 3 BB / all-in. A second question asked each model for its
showdown equity in 10% buckets.

| Model | Choice | P(choice) | Equity bucket | Latency | Cost |
|---|---|---:|---|---:|---:|
| Jev 1.13 | raise to 3 BB | 0.70 | 50-60% | 1.2 s | $0.000031 |
| Solar Decide | **fold** | 0.44 (tied with call) | 40-50% | 15.9 s | $0.000053 |
| Mercury Decide (free) | raise to 2.5 BB | 0.68 | 70-80% | 0.5 s | $0 |
| pplx-decider-v1-27b | raise to 3 BB | 0.42 | 60-70% | 0.4 s | $0.000024 |

*Anecdote, n = 1:*
- **Solar folded A-K offsuit on the button.** Folding AKo heads-up is one of the
  clearest mistakes in poker; it is a top-10% hand. Solar's distribution was an exact
  tie between fold and call (0.436 / 0.436), and it broke the tie toward folding.
  Whether this is a systematic leak or just a coin flip is something the probe set
  (same spot, repeated and with suits permuted) is designed to answer.
- **Equity estimates disagree a lot.** AKo against a random hand has about 65%
  equity. The buckets chosen for this spot ranged from 40-50% (Solar) to 70-80%
  (Mercury). Every decision in the benchmark records this estimate, so calibration
  becomes a measurable property rather than a vibe.
- **The models return full probability distributions,** not just a choice. That makes
  it possible to measure how *confident* each model is, and later to compare argmax
  play against sampling from the model's own mixed strategy.
- **Price.** Four decisions with two questions each cost about **$0.0001 in total**.
  For comparison, a reasoning chat model like GPT-6 Luna averaged about $0.0004 per
  decision in the author's earlier Figgie benchmark, while GLM 5.3 Flash averaged
  about $0.004, roughly **100x the price of a Jev decision**.

### Prior art from the Figgie benchmark (what not to repeat)

The same author's earlier project, FiggieBench, ran into lessons this project
absorbs from day one:

- Reasoning chat models are **slow** (median 40-120 s per call for Muse, MiMo and
  GLM in that harness) and their cost is dominated by hidden reasoning tokens.
- Free or promotional routes fail in new ways: HTTP 200 responses with no choices,
  gateway timeouts with unknown billing. Every call here is journaled with its cost or
  marked *unknown*, and a hard budget guard reserves money before each request.
- Jev occasionally returns probabilities rounded to two decimals, where the "chosen"
  option is not exactly the maximum. This harness logs it rather than discarding the
  decision.

### Day 0, later: "Elo" can crown a maniac

The first free league (five scripted bots, 2,000 duplicate hands) produced a ranking
that was obviously wrong when Elo was computed the textbook way, with every duplicate
deal counted as one game won or lost:

| Bot | mbb/hand | Elo, one deal = one game | Elo, 10-deal blocks |
|---|---:|---:|---:|
| equity_tag (Monte Carlo equity + pot odds) | **+4,283** | 1,503 (3rd) | **1,763 (1st)** |
| maniac (pot-raises everything) | −147 | **1,592 (1st)** | 1,462 |
| random | −2,425 | 1,560 (2nd) | 1,462 |

The maniac "wins" most deals because most hands end with someone folding to a raise;
it loses the few large pots that decide the money. Poker scores are not binary, so the
benchmark's primary metric is **chip win rate (mbb/hand)**, and its Elo is fitted on
blocks of ten duplicate deals, where the size of a win matters. The single-deal Elo is
kept in the report as a cautionary column.

### Bug of the day: the all-in adjustment fired on every street

PokerKit has a short phase between betting rounds where nobody is "to act" while the
next card is not dealt yet. The first version of the all-in detector treated that
moment as "both players are all-in" and replaced real results with equity-based
expectations on almost every hand. Symptom: equity_tag's adjusted win rate was 7x
smaller than its real one. Fix: require that a stack is actually empty. A regression
test now pins it down.

### Reasoning knobs do not mean the same thing across providers

The route smoke test (9 models, 36 hands, $0.031) surfaced two failure modes, both
from *thinking too much*, not too little:

- **GLM 5.3 Flash**, at its default effort (`max` on this route), twice produced 8,192
  reasoning tokens and no answer: $0.004 per call, ~80x a Jev decision, for a
  fallback fold. With `effort: medium` it **stopped reasoning altogether** (15-90
  output tokens). With `effort: high` it used 40-600 tokens and answered every time.
  Same label, completely different behavior.
- **DeepSeek V4.1 Flash** truncated 36% of decisions at 8,192 tokens. With a 32,768
  ceiling it always answered, but its reasoning length ranged from **496 to 31,322
  tokens** per decision. The longest one was spent deciding to *check* with ~15%
  equity.

Treatment adopted: each provider's default reasoning, hidden from the response,
with a 32,768-token ceiling; GLM gets `high` because its default never terminates.
The same choice was made in the earlier Figgie benchmark.

### Probes: are decision models consistent?

A probe set of 40 realistic spots (10 per street, captured from bot-vs-bot hands) is
asked three times per model: the original, an identical repeat, and the same spot with
**suits relabelled**, which changes nothing strategically.

| Model | Same answer on identical prompt | Same answer after suit relabelling | All-in in 40 spots |
|---|---:|---:|---:|
| Jev 1.13 | 100% | 87.5% | 2 |
| pplx-decider 27B | 100% | 92.5% | 1 |
| Mercury Decide | 97.5% | 82.5% | 1 |
| Solar Decide | **57.5%** | **50%** | **19** |
| equity_tag (bot) | 100% | 100% | 0 |

- **Solar shoves almost half the time** and is close to a coin flip with itself.
  The AK-fold from the first probe was not an isolated quirk; Solar's poker policy
  is high-variance noise.
- Jev and pplx are deterministic, yet **suit relabelling alone flips 1 decision in
  8 to 13**. A player who reasons about poker should be exactly invariant to it.
  This is a cheap, assumption-free test of "understanding vs. pattern matching".

### Infrastructure is part of the result: rate limits that look like folds

Two provider-side limits hit the first real league, and both would have silently
corrupted a naive benchmark:

- **The free tier has a daily cap.** Mercury Decide is free, but OpenRouter allows
  1,000 free-model requests per day on this account (`free-models-per-day-high-balance`).
  Smoke tests, probes and 300 league hands used it up. The circuit breaker stopped the
  Mercury league after ten consecutive refusals; the league resumes after the daily reset.
- **pplx-decider was rate-limited by its upstream** (HTTP 429, "Request rate limit
  exceeded") on about **14% of hands** at a concurrency of 8. Because successes were
  interleaved with failures, the consecutive-failure breaker never tripped. Each refused
  call fell back to check/fold, so pplx would have appeared to be a timid player that
  folds too often, and it would have been ranked for a strategy it never chose.

The fix separates two kinds of failure. **Provider failures** (HTTP errors, timeouts,
rate limits) void the hand: analysis ignores it and the runner replays it on resume.
**Model failures** (truncated reasoning, unparseable or illegal answers) stay in the
data and count against the model, because they are part of what that model does.
429s now get up to six slower retries; refusals are not billed.

### Run 1A: the decision-model league (5,600 hands, $0.83)

Three decision models and five bots, every pair, 100 duplicate deals per pair
(Mercury Decide plays a separate, rate-limited league). Primary metric: all-in-adjusted
mbb/hand with 95% deal-cluster bootstrap intervals.

| # | Agent | mbb/hand | 95% CI | Elo (10-deal blocks) |
|---:|---|---:|---|---:|
| 1 | equity_tag (bot) | +4,010 | +2,517 to +5,583 | 1,766 |
| 2 | **Jev 1.13** | **+3,709** | +2,268 to +5,225 | 1,617 |
| 3 | rock (bot) | +1,600 | +987 to +2,427 | 1,680 |
| 4 | pplx-decider 27B | +584 | −137 to +1,667 | 1,510 |
| 5 | maniac (bot) | −733 | −2,425 to +645 | 1,437 |
| 6 | calling_station (bot) | −1,330 | −2,676 to −380 | 1,366 |
| 7 | **Solar Decide** | **−2,401** | −3,520 to −1,128 | 1,405 |
| 8 | random (bot) | −5,438 | −7,093 to −3,815 | 1,220 |

What stands out:

- **Jev is nearly as profitable as a Monte Carlo equity bot,** at $0.15 per 1,000
  hands. But it is not playing "good poker" in the textbook sense. It almost never
  raises preflop (PFR 3%, versus a sensible 40-80% heads-up), is passive postflop
  (aggression factor 0.22), and when it does bet after the flop it is usually all-in
  (median bet 6x the pot). Its profit comes from punishing the bots' bluffs and
  over-aggression: +7.4 to +9.1 BB/hand against maniac, calling station, Solar and
  random, but −2.7 BB/hand against equity_tag.
- **pplx-decider is a tighter, weaker cousin of Jev:** the same passive pattern, plus a
  real leak. It folded the river with ≥90% equity in 7 of 29 such spots (24%).
- **Solar Decide loses to the calling station.** It goes all-in in 56% of hands,
  3-bets 76% of the time when it can, and **folded AA/KK/QQ/AK in 4 of 13 spots**.
  Every red flag from the first probe and the consistency test showed up in play.
- **Every decision model is pessimistic about its own hand.** Asked for showdown
  equity, Jev, pplx and Solar underestimate it by 16-20 percentage points on average
  (bias −0.16 to −0.20 against Monte Carlo equity vs. a random hand). Jev's and pplx's
  estimates still correlate with the truth (r ≈ 0.6), Solar's barely (r = 0.38).

### Who plays like whom

Clustering the probe answers (40 spots, every agent) by how often two agents pick the
same action class gives a surprisingly clean picture:

- **Jev, pplx-decider and the calling-station bot form one cluster.** Jev picks the
  same action class as a bot that *only ever checks or calls* in about 82% of spots.
- **The chat LLMs form their own family.** Luna, DeepSeek and Muse are each other's
  nearest neighbours, and MiMo joins them. They bet or raise in 28-45% of probe spots,
  against 8% for Jev and pplx.
- **GLM 5.3 Flash pairs with Mercury Decide,** away from the other LLMs: the most
  bet-happy of the "reasonable" agents.
- **Solar sits alone,** closest to the random and maniac bots.

Self-consistency is a model-family trait too. Decision models are (except Solar)
deterministic, while the reasoning LLMs give the same coarse answer to an identical
prompt only 55-82% of the time (GLM 55%, MiMo 65%, Muse 70%, Luna 80%, DeepSeek 82%).
Their sampled reasoning acts like a mixed strategy, whether intended or not.

### How many hands does it take? (viability)

Duplicate dealing removes card luck *between* the two seat orientations, but not the
luck of how two specific opponents' hands interact. Measured on the bot league and the
decision-model league:

| Estimator | Per-deal SD of a pairing | Deals for a ±1 BB/hand 95% CI | for ±0.5 BB/hand |
|---|---:|---:|---:|
| Raw chips | 15-19 BB/hand | ~940-1,430 | ~3,800-5,700 |
| All-in adjusted | 9-10 BB/hand | ~340-420 | ~1,370-1,680 |

The all-in adjustment (replace the result of an all-in before the river by its exact
equity) is unbiased and cuts the required sample by ~2.7x, so it is the primary metric.
The consequence for budgets is stark:

- **Decision models:** ~1,000 deals per pairing costs about 2,000 hands x ~2 decisions
  x $0.00005 = **$0.20**. Tight rankings are affordable.
- **Cheap chat LLMs:** the same precision costs $1-$7 per pairing (DeepSeek at the top
  end), so a $5 budget only separates *large* differences. The design therefore pools
  every model's games across many opponents through an additive rating, instead of
  reading individual head-to-heads.
- **Frontier models:** at ~$0.01-$0.02 per decision, a single pairing at ±1 BB/hand
  would cost $50-$100. Phase 2 can only place them on the scale coarsely, through
  bridge opponents whose ratings are already pinned down cheaply.

### Run 1B: the cheap-LLM league (interim, ~775 of 800 hands, ~$2.4)

Round robin of the five cheap chat LLMs plus every one of them against the two anchors
shared with Run 1A (equity_tag and Jev), 20 duplicate deals per pair (extended to 25 to
reach exactly 1,000 hands). Merged with Run 1A, all-in adjusted, BB/hand:

| Player | BB/hand | 95% CI | $ / 1k hands | median s / decision |
|---|---:|---|---:|---:|
| DeepSeek V4.1 Flash | +6.0 | +3.6 to +8.9 | $6.47 | 104 |
| Muse Spark Contributor | +4.7 | +2.4 to +7.8 | $1.63 | 36 |
| GPT-6 Luna | +3.6 | +2.2 to +5.1 | $0.71 | 4 |
| MiMo V2.6 Flash | +2.6 | +1.6 to +3.7 | $1.00 | 8 |
| GLM 5.3 Flash | +2.2 | -0.1 to +4.5 | $0.65 | 6 |
| equity_tag (script) | +2.0 | +0.7 to +3.4 | free | - |
| Jev 1.13 | +1.6 | +0.3 to +3.0 | $0.16 | 0.3 |

Findings worth a sentence each:

- **Cheap reasoning LLMs beat the decision models.** Head-to-head against Jev: Muse
  +3.7 BB/hand (CI +0.5 to +8.9), Luna +1.7 (+0.1 to +4.0), MiMo +2.3 (-0.2 to +5.0),
  DeepSeek +6.4 (-1.5 to +15.3, only 20 deals), GLM ~0.
- **The 30-line script beat Jev head-to-head**: equity_tag vs Jev +2.7 BB/hand
  (CI +1.0 to +4.6) over 100 deals. It is roughly level with the cheap LLMs (all within noise).
- **Style**: LLMs play loose-aggressive (VPIP 68-85%, PFR 33-61%), size bets 0.5-1x pot,
  c-bet 44-83%. Decision models are passive preflop (Jev PFR 3%, pplx 3%) and either
  overbet (Jev, median 6x pot) or shove (Solar).
- **Calibration**: asked for their win probability, decision models are 15-18 points too
  pessimistic (bias vs a Monte Carlo of their hand against a random hand); the chat LLMs
  5-10 points; the scripts under 1. Correlation with the true equity: LLMs 0.71-0.82,
  Jev 0.61, pplx 0.62, Solar 0.39.
- **Blunders**: pplx folded the river with >=90% equity in 7 of 30 spots; Jev 2 of 20 and
  called hopeless rivers 2 of 7; no cheap LLM folded a strong river (small samples: 2-5 spots each).
- **Wall-clock time**: DeepSeek thinks ~100 s per decision on average and up to ~5 min on hard spots, so one hand against it takes 3-8 min; its last hands kept the league running long after every other pair had finished.
- **Reliability**: DeepSeek is the only model with real errors (3% of decisions hit the
  32k output cap). Everyone else: 0% errors in the league.
- **Note for the write-up:** GLM's point estimate ties the script; do not write "all five
  LLMs beat every bot" without checking the final numbers.

### Frontier price check (before Run 2)

Four probe spots per model, default reasoning, $0.07 in total:

| Model | $ / decision | latency | reasoning tokens |
|---|---:|---:|---|
| GPT-6 Sol | ~$0.0037 | 2-9 s | 46-607 |
| Claude Opus 5.5 | ~$0.0063 | 3-10 s | 0-427 (none on easy spots) |
| Muse Spark 1.3 (full) | ~$0.0086 | 11-75 s | 317-4,352 |

Opus spending zero reasoning tokens on easy spots is itself a finding: "default thinking"
is adaptive. Muse Spark (full) was dropped from Run 2: most expensive and ~10x slower.

### Run 2 design (authorised 2026-10-04, running)

GPT-6.1 Sol and Claude Opus 5.5 each play equity_tag, Jev, pplx, Luna and Muse Contributor,
plus Sol vs Opus: 11 pairs x 20 duplicate deals = 440 hands, cap $14 (forecast $8-11).
With a per-deal SD of ~9.6 BB, 120 deals per frontier model gives roughly +/-1.7 BB/hand
on its rating, enough to separate it from the decision models and the script if the gaps
look like Run 1's (2-6 BB). Bridges are already rated, so one joint fit places them.

## The interactive post (for whoever writes the final version)

- Lives in `C:\BookmarkHandler\blogChrystianSchutzOstatAstro`:
  `src/content/blog/ai-poker-holdembench.md` (draft: true), styles `src/styles/holdem.css`,
  figures `public/holdem/holdem.js`, data `public/holdem/data.js`.
- Regenerate the data with
  `holdembench blogdata results/run1a-decision-league results/run1a-mercury results/run1b-chat-league results/run2-frontier --probes results/probes-01 -o <blog>/public/holdem/data.js`.
- Figures (placeholders `<div class="hb" data-hb="NAME">`): `replay` (hero: animated
  table replaying rule-selected hands), `counters`, `leaderboard` (BB/hand or Elo with CIs),
  `sizes` (postflop bet-size small multiples), `stylemap` (VPIP vs PFR, tracker view),
  `consistency` (40 probe spots x asked/again/suits swapped), `similarity` (agreement
  matrix), `h2h` (pairwise matrix), `cost` (BB/hand vs $/1k hands, log), `spend`.
- Replayed hands are chosen by rule in `blogdata.pick_hands` (largest pot matching:
  Solar premium fold, Jev overbet >= 4 pots, pplx river fold of the winning hand, biggest
  frontier pot, biggest LLM-vs-LLM pot, biggest LLM-vs-decision pot), so they are reproducible
  and not hand-picked.
- Story beats that test well: Solar folds AA to a random bot while saying "I think I win
  48%"; Jev bets 6x pot; a 30-line script beats the best decision model; cheap reasoning
  LLMs win for under $1 per 1,000 hands; Elo crowned the maniac; rate limits looked like folds.
- Screenshots of every figure (desktop + phone) are produced with Playwright
  (`channel="chrome"`), never with the user's browser.

## Costs so far

| Phase | Calls | Spend |
|---|---:|---:|
| Decisions API probe (4 models x 1 spot) | 4 | ~$0.0001 |
| Route smoke tests (9 models, plus GLM/DeepSeek reasoning variants) | ~150 | ~$0.05 |
| Decision-model probes (4 models x 40 spots x 3) | 480 | ~$0.02 |
| Chat-model probes (5 models x 40 spots x 3) | 600 | ~$0.2 |
| Run 1A decision league (5,600 hands) | ~11,000 | $0.83 |
| Run 1A Mercury (free model, daily cap) | ~600 | $0 |
| Run 1B cheap-LLM league (1,000 hands) | ~3,800 | ~$2.4 (to be finalised) |
| Frontier price check (3 models x 4 spots) | 12 | $0.07 |
| Run 2 frontier league (440 hands) | - | running, cap $14 |

Measured price per decision (median latency in brackets): pplx-decider $0.00006 (2.5 s
under pacing), Jev $0.00005 (0.3 s), Solar $0.0001 (13 s), Mercury Decide free (22 s
under pacing), GLM-high $0.0002 (6 s), Luna $0.0002 (4 s), MiMo $0.0003 (8 s), Muse
Contributor $0.0005 (36 s), DeepSeek $0.002 (104 s), Sol ~$0.004, Opus ~$0.006.

## Open questions / TODO for the write-up

- Final numbers after Run 1B (1,000 hands), Mercury completion and Run 2.
- Does Solar's fold-vs-call tie reflect a calibrated mixed strategy or noise?
- Does a frontier model beat the 30-line script, and by how much per dollar?

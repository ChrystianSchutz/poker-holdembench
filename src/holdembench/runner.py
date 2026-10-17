"""Tournament configuration, scheduling and execution.

A *match* is one pairing played over the run's shared list of deals. Each deal is
played twice with the seats swapped (duplicate poker); every match uses the same
deal seeds, so all agents face identical cards. Hands inside a match run in order
(the opponent HUD depends on earlier hands); matches run concurrently.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import os
import platform
import time
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import orjson
import yaml
from pydantic import BaseModel, Field

from holdembench.agents.baselines import BOTS
from holdembench.agents.models import ChatAgent, DecisionAgent
from holdembench.cards import Deal
from holdembench.engine import Agent, Rules, play_hand, provider_failed
from holdembench.providers import (
    Budget,
    BudgetExceededError,
    ChatClient,
    DecisionClient,
    Journal,
    OpenRouter,
)
from holdembench.stats import hand_flags, hud_line, summarize


class AgentSpec(BaseModel):
    """One competitor."""

    id: str
    kind: str
    tier: str = ""
    model: str | None = None
    provider: dict[str, Any] | None = None
    reasoning: dict[str, Any] | None = None
    max_tokens: int = 8192
    structured: Literal["json_schema", "json_object", "none"] = "json_schema"
    temperature: float | None = None
    concurrency: int = 8
    min_interval_s: float = 0.0
    timeout_s: float = 300.0
    retries: int = 2
    max_cost_per_call: float = 0.02
    # OpenAI-compatible server for a locally hosted model, e.g. http://localhost:3000 (ThreadShelf).
    base_url: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class Schedule(BaseModel):
    """Which pairs play. ``round_robin`` plays every pair; ``pairs`` lists them."""

    mode: Literal["round_robin", "pairs", "focus"] = "round_robin"
    pairs: list[tuple[str, str]] = Field(default_factory=list)
    focus: list[str] = Field(default_factory=list)


class RunConfig(BaseModel):
    """A complete, frozen experiment description."""

    name: str
    variant: Literal["nl", "fl"] = "nl"
    deals: int = 50
    deal_seed: int = 1_000
    budget_usd: float = 0.5
    hud: bool = True
    ask_equity: bool = True
    match_concurrency: int = 32
    agents: list[AgentSpec]
    schedule: Schedule = Field(default_factory=Schedule)

    @classmethod
    def load(cls, path: Path) -> RunConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))

    def pairs(self) -> list[tuple[str, str]]:
        ids = [a.id for a in self.agents]
        match self.schedule.mode:
            case "pairs":
                return [tuple(p) for p in self.schedule.pairs]  # type: ignore[misc]
            case "focus":
                focus = set(self.schedule.focus)
                return [p for p in itertools.combinations(ids, 2) if focus & set(p)]
            case _:
                return list(itertools.combinations(ids, 2))

    def paid_agents(self) -> list[AgentSpec]:
        return [a for a in self.agents if a.kind in ("chat", "decision")]


def build_agent(spec: AgentSpec, router: OpenRouter | None, ask_equity: bool) -> Agent:
    """Instantiate one competitor."""
    if spec.kind in BOTS:
        return BOTS[spec.kind](spec.id, **spec.params)
    if router is None or spec.model is None:
        msg = f"agent {spec.id} needs a model and an API router"
        raise ValueError(msg)
    extra: dict[str, Any] = {"provider": spec.provider} if spec.provider else {}
    common: dict[str, Any] = {
        "max_cost_per_call": spec.max_cost_per_call,
        "concurrency": spec.concurrency,
        "min_interval_s": spec.min_interval_s,
        "timeout_s": spec.timeout_s,
        "retries": spec.retries,
        "extra": extra,
        "base_url": spec.base_url,
    }
    if spec.kind == "chat":
        client = ChatClient(
            router,
            spec.model,
            structured=spec.structured,
            max_tokens=spec.max_tokens,
            reasoning=spec.reasoning,
            temperature=spec.temperature,
            **common,
        )
        return ChatAgent(spec.id, client)
    if spec.kind == "decision":
        return DecisionAgent(
            spec.id, DecisionClient(router, spec.model, **common), ask_equity=ask_equity
        )
    msg = f"unknown agent kind {spec.kind!r}"
    raise ValueError(msg)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    for line in path.read_bytes().splitlines():
        if line.strip():
            try:
                out.append(orjson.loads(line))
            except orjson.JSONDecodeError:
                continue  # a torn final line from an interrupted run
    return out


def valid_hands(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Latest record per (match, hand), dropping hands voided by provider failures."""
    latest: dict[tuple[str, int], dict[str, Any]] = {}
    for h in records:
        latest[(h["match_id"], h["hand_index"])] = h
    return [h for h in latest.values() if not any(provider_failed(a) for a in h["actions"])]


def prior_spend(out: Path) -> float:
    """Spend already journaled in this run directory (unknown charges at their reserve)."""
    total = 0.0
    for c in _read_jsonl(out / "calls.jsonl"):
        total += c["cost"] if c.get("cost") is not None else c.get("reserve", 0.02)
    return total


def design_hash(config: RunConfig) -> str:
    """Hash of the experimental design, used to decide whether a run directory can resume.

    Operational knobs (budget cap, pacing, concurrency) are excluded so a stopped run can be
    resumed with a higher cap or slower pacing. ``deals`` is excluded too: deal ``d`` always
    uses seed ``deal_seed + d``, so a longer run only appends deals to a shorter one.
    """
    operational = {"concurrency", "min_interval_s", "timeout_s", "retries", "max_cost_per_call"}
    design = config.model_dump_json(
        exclude={
            "budget_usd": True,
            "match_concurrency": True,
            "deals": True,
            "agents": dict.fromkeys(range(len(config.agents)), operational),
        }
    ).encode()
    return hashlib.sha256(design).hexdigest()


async def run(
    config: RunConfig,
    out: Path,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute (or resume) a run and return a summary."""
    out.mkdir(parents=True, exist_ok=True)  # noqa: ASYNC240 - one-time setup before any I/O-bound work
    cfg_bytes = config.model_dump_json(indent=2).encode()
    cfg_hash = design_hash(config)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = orjson.loads(manifest_path.read_bytes())
        if design_hash(RunConfig.model_validate(old["config"])) != cfg_hash:
            msg = f"{out} holds a different config; use a new output directory"
            raise RuntimeError(msg)
    hands_path = out / "hands.jsonl"
    existing = valid_hands(_read_jsonl(hands_path))
    done = {(h["match_id"], h["hand_index"]) for h in existing}

    router: OpenRouter | None = None
    budget = Budget(config.budget_usd, prior_spend(out))
    if config.paid_agents() and not dry_run:
        router = OpenRouter(budget, Journal(out / "calls.jsonl"))
        credit = await router.credit()
        remaining = credit.get("limit_remaining")
        if remaining is not None and remaining < budget.cap - budget.spent:
            budget.cap = budget.spent + float(remaining)
    agents = {
        s.id: build_agent(s, router, config.ask_equity)
        for s in config.agents
        if not dry_run or s.kind in BOTS
    }
    rules = Rules(variant=config.variant)
    manifest = {
        "config_sha256": hashlib.sha256(cfg_bytes).hexdigest(),
        "design_sha256": cfg_hash,
        "config": orjson.loads(cfg_bytes),
        "agents": {k: a.describe() for k, a in agents.items()},
        "started": time.time(),
        "python": platform.python_version(),
        "pokerkit": version("pokerkit"),
        "holdembench": version("holdembench"),
    }
    manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    pairs = [p for p in config.pairs() if p[0] in agents and p[1] in agents]
    total = len(pairs) * config.deals * 2
    state: dict[str, Any] = {"done": len(done), "total": total, "stopped": None, "budget": budget}
    write_lock = asyncio.Lock()
    sem = asyncio.Semaphore(config.match_concurrency)
    by_match: dict[str, list[dict[str, Any]]] = {}
    for h in existing:
        by_match.setdefault(h["match_id"], []).append(h)

    async def play_match(a_id: str, b_id: str) -> None:
        match_id = f"{a_id}__vs__{b_id}"
        history = sorted(by_match.get(match_id, []), key=lambda h: h["hand_index"])
        async with sem:
            for d in range(config.deals):
                deal = Deal.from_seed(config.deal_seed + d)
                for orient, (p0, p1) in enumerate(((a_id, b_id), (b_id, a_id))):
                    hand_index = 2 * d + orient
                    if (match_id, hand_index) in done:
                        continue
                    if state["stopped"]:
                        return
                    huds: list[dict[str, Any] | None] = [None, None]
                    if config.hud:
                        for i in range(2):
                            opp = p1 if i == 0 else p0
                            flags = [hand_flags(h, opp) for h in history]
                            huds[i] = hud_line(summarize(flags)) if flags else None
                    try:
                        record = await play_hand(
                            (agents[p0], agents[p1]),
                            deal,
                            rules,
                            match_id=match_id,
                            hand_index=hand_index,
                            huds=huds,
                        )
                    except BudgetExceededError as exc:
                        state["stopped"] = str(exc)
                        return
                    record["deal_index"] = d
                    record["orientation"] = orient
                    history.append(record)
                    async with write_lock:
                        with hands_path.open("ab") as fh:
                            fh.write(orjson.dumps(record) + b"\n")
                        state["done"] += 1
                        if progress:
                            progress(state)

    await asyncio.gather(*(play_match(a, b) for a, b in pairs))
    if router:
        await router.aclose()
    summary = {
        "hands": state["done"],
        "planned": total,
        "stopped": state["stopped"],
        "spent_usd": round(budget.spent, 6),
        "unknown_cost_calls": budget.unknown_calls,
    }
    manifest["finished"] = time.time()
    manifest["summary"] = summary
    manifest_path.write_bytes(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))
    return summary


def load_env() -> None:
    """Load ``.env`` without overriding exported variables."""
    from dotenv import load_dotenv

    load_dotenv(Path.cwd() / ".env", override=False)
    os.environ.setdefault("PYTHONUTF8", "1")

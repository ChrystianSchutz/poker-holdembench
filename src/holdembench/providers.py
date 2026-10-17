"""OpenRouter clients for chat LLMs and typed decision models, with a hard budget.

Every request reserves a worst-case cost before it is sent and settles to the
provider-reported ``usage.cost`` afterwards. A failed request whose billing is
unknown keeps its reservation as spend, so the budget can only be over-counted.
Every call, successful or not, is appended to a JSONL journal.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx
import orjson

OPENROUTER = "https://openrouter.ai/api"


class BudgetExceededError(RuntimeError):
    """Raised before a request that could push spend over the cap."""


class ProviderDownError(BudgetExceededError):
    """Raised after repeated consecutive failures of one model (circuit breaker).

    Subclasses the budget stop so the runner halts cleanly instead of silently
    replacing a broken model's decisions with fallbacks.
    """


class Budget:
    """Process-wide spend guard. ``spent`` includes reservations for unknown charges."""

    def __init__(self, cap_usd: float, spent_usd: float = 0.0) -> None:
        self.cap = cap_usd
        self.spent = spent_usd
        self.reserved = 0.0
        self.unknown_calls = 0
        self._cond = asyncio.Condition()

    async def reserve(self, amount: float) -> None:
        """Wait until the worst case fits beside in-flight calls; fail once spend leaves no room."""
        async with self._cond:
            while self.spent + self.reserved + amount > self.cap:
                if self.spent + amount > self.cap or self.reserved == 0:
                    msg = f"budget cap ${self.cap:.4f} reached (spent ${self.spent:.4f})"
                    raise BudgetExceededError(msg)
                await self._cond.wait()
            self.reserved += amount

    async def settle(self, reserved: float, actual: float | None) -> None:
        async with self._cond:
            self.reserved -= reserved
            if actual is None:
                self.unknown_calls += 1
                self.spent += reserved
            else:
                self.spent += actual
            self._cond.notify_all()


class Journal:
    """Append-only JSONL log of every API call."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, record: dict[str, Any]) -> None:
        line = orjson.dumps(record, default=str) + b"\n"
        async with self._lock:
            with self.path.open("ab") as fh:
                fh.write(line)


class OpenRouter:
    """Shared HTTP session, pacing and per-model concurrency."""

    def __init__(self, budget: Budget, journal: Journal, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            msg = "OPENROUTER_API_KEY is not set"
            raise RuntimeError(msg)
        self.budget = budget
        self.journal = journal
        self.http = httpx.AsyncClient(
            base_url=OPENROUTER,
            headers={
                "Authorization": f"Bearer {key}",
                "HTTP-Referer": "https://github.com/ChrystianSchutz/holdembench",
                "X-Title": "HoldemBench",
            },
            timeout=httpx.Timeout(600, connect=30),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
        )

    async def aclose(self) -> None:
        await self.http.aclose()

    async def credit(self) -> dict[str, Any]:
        """Key limit/usage, free of charge."""
        r = await self.http.get("/v1/key")
        r.raise_for_status()
        data: dict[str, Any] = r.json()["data"]
        return data


class ModelClient:
    """Base for one model endpoint: concurrency, pacing, retries, budget, journal."""

    path = ""

    def __init__(
        self,
        router: OpenRouter,
        model: str,
        *,
        max_cost_per_call: float,
        concurrency: int = 8,
        min_interval_s: float = 0.0,
        timeout_s: float = 300.0,
        retries: int = 2,
        extra: dict[str, Any] | None = None,
        base_url: str | None = None,
    ) -> None:
        self.router = router
        self.model = model
        # A local OpenAI-compatible server (llama.cpp, LM Studio, ThreadShelf): no key, no charge.
        self.local = base_url is not None
        self.http = (
            httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=httpx.Timeout(600, connect=30))
            if base_url
            else router.http
        )
        self.max_cost_per_call = max_cost_per_call
        self.sem = asyncio.Semaphore(concurrency)
        self.min_interval_s = min_interval_s
        self.timeout_s = timeout_s
        self.retries = retries
        self.extra = extra or {}
        self._pace = asyncio.Lock()
        self._last = 0.0
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5

    async def _send(
        self, payload: dict[str, Any], tag: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """POST with retries. Returns (body or None, call metadata)."""
        budget = self.router.budget
        if self.consecutive_failures >= self.max_consecutive_failures:
            msg = f"{self.model}: {self.consecutive_failures} consecutive failed calls, stopping"
            raise ProviderDownError(msg)
        meta: dict[str, Any] = {"model": self.model, "attempts": 0, "cost": 0.0}
        started = time.perf_counter()
        async with self.sem:
            for attempt in range(self.retries + 5):
                meta["attempts"] = attempt + 1
                await budget.reserve(self.max_cost_per_call)
                async with self._pace:
                    wait = self.min_interval_s - (time.monotonic() - self._last)
                    if wait > 0:
                        await asyncio.sleep(wait)
                    self._last = time.monotonic()
                status: int | None = None
                body: dict[str, Any] | None = None
                error: str | None = None
                try:
                    r = await self.http.post(self.path, json=payload, timeout=self.timeout_s)
                    status = r.status_code
                    try:
                        body = r.json()
                    except ValueError:
                        error = f"non-JSON body (HTTP {status}): {r.text[:200]}"
                    if body is not None and (status >= 400 or "error" in body):
                        error = f"HTTP {status}: {json.dumps(body.get('error', body))[:300]}"
                except (httpx.HTTPError, TimeoutError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
                cost = 0.0 if self.local else _cost(body)
                # Explicit 4xx refusals are not billed; anything else without a cost is unknown.
                if cost is None and status is not None and 400 <= status < 500:
                    cost = 0.0
                await budget.settle(self.max_cost_per_call, cost)
                meta["cost"] += cost if cost is not None else self.max_cost_per_call
                meta["cost_unknown"] = meta.get("cost_unknown", False) or cost is None
                await self.router.journal.write(
                    {
                        "at": time.time(),
                        "model": self.model,
                        "endpoint": self.path,
                        "attempt": attempt + 1,
                        "status": status,
                        "error": error,
                        "cost": cost,
                        "reserve": self.max_cost_per_call,
                        "usage": (body or {}).get("usage"),
                        "id": (body or {}).get("id"),
                        "provider": (body or {}).get("provider"),
                        **tag,
                        "response": _strip(body),
                    }
                )
                if error is None and body is not None:
                    self.consecutive_failures = 0
                    meta["latency_s"] = round(time.perf_counter() - started, 3)
                    meta["provider"] = body.get("provider")
                    meta["usage"] = body.get("usage")
                    return body, meta
                meta["error"] = error
                retryable = status is None or status in (408, 409, 425, 429) or status >= 500
                if not retryable or attempt >= self.retries + (4 if status == 429 else 0):
                    break
                # Rate limits get extra, slower retries: a 429 refusal is not billed.
                cap = 60 if status == 429 else 20
                await asyncio.sleep(min(2**attempt * 2, cap))
        meta["latency_s"] = round(time.perf_counter() - started, 3)
        meta["provider_error"] = True
        self.consecutive_failures += 1
        return None, meta


def _cost(body: dict[str, Any] | None) -> float | None:
    if not body:
        return None
    usage = body.get("usage") or {}
    cost = usage.get("cost")
    return float(cost) if isinstance(cost, (int, float)) else None


def _strip(body: dict[str, Any] | None) -> Any:
    """Keep the journal readable: drop reasoning blobs, keep the answer."""
    if not body:
        return body
    out = {k: v for k, v in body.items() if k not in ("usage",)}
    for ch in out.get("choices", []) or []:
        msg = ch.get("message") or {}
        for k in ("reasoning", "reasoning_details"):
            if k in msg:
                msg[k] = f"<{len(str(msg[k]))} chars omitted>"
    return out


class ChatClient(ModelClient):
    """``/v1/chat/completions`` with a JSON-schema answer ``{action, equity}``."""

    path = "/v1/chat/completions"

    def __init__(
        self,
        router: OpenRouter,
        model: str,
        *,
        structured: str = "json_schema",
        max_tokens: int = 8192,
        reasoning: dict[str, Any] | None = None,
        temperature: float | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(router, model, **kw)
        self.structured = structured
        self.max_tokens = max_tokens
        self.reasoning = reasoning if reasoning is not None else {"enabled": True, "exclude": True}
        self.temperature = temperature

    async def decide(
        self, system: str, user: str, labels: list[str], tag: dict[str, Any]
    ) -> tuple[str | None, float | None, dict[str, Any]]:
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": labels},
                "equity": {"type": "integer", "minimum": 0, "maximum": 100},
            },
            "required": ["action", "equity"],
            "additionalProperties": False,
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self.max_tokens,
            "usage": {"include": True},
            **self.extra,
        }
        if self.reasoning:
            payload["reasoning"] = self.reasoning
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.structured == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "poker_action", "strict": True, "schema": schema},
            }
        elif self.structured == "json_object":
            payload["response_format"] = {"type": "json_object"}
        body, meta = await self._send(payload, tag)
        if body is None:
            return None, None, meta
        try:
            choice = body["choices"][0]
            content = choice["message"].get("content") or ""
            meta["finish_reason"] = choice.get("finish_reason")
        except (KeyError, IndexError, TypeError):
            meta["error"] = "response without choices"
            return None, None, meta
        meta["reasoning_tokens"] = (
            (body.get("usage") or {}).get("completion_tokens_details") or {}
        ).get("reasoning_tokens")
        label, equity = parse_answer(content, labels)
        if label is None:
            meta["error"] = f"unparseable answer: {content[:200]!r}"
        return label, equity, meta


def parse_answer(content: str, labels: list[str]) -> tuple[str | None, float | None]:
    """Extract ``{action, equity}`` from a possibly chatty reply."""
    for match in reversed(list(re.finditer(r"\{[^{}]*\}", content, re.DOTALL))):
        try:
            obj = json.loads(match.group(0))
        except ValueError:
            continue
        action = str(obj.get("action", "")).strip()
        if action in labels:
            eq = obj.get("equity")
            equity = float(eq) / 100 if isinstance(eq, (int, float)) and 0 <= eq <= 100 else None
            return action, equity
    hits = [lab for lab in labels if re.search(rf"\b{re.escape(lab)}\b", content)]
    return (hits[-1], None) if len(set(hits)) == 1 else (None, None)


EQUITY_BUCKETS = {f"p{i * 10:02d}_{i * 10 + 10}": f"{i * 10}-{i * 10 + 10}%" for i in range(10)}


class DecisionClient(ModelClient):
    """``/alpha/decisions``: a choice over the action menu plus an equity bucket."""

    path = "/alpha/decisions"

    async def decide(
        self,
        rules: str,
        situation: str,
        options: dict[str, str],
        tag: dict[str, Any],
        *,
        ask_equity: bool = True,
    ) -> tuple[str | None, float | None, dict[str, Any], dict[str, float] | None]:
        questions: dict[str, Any] = {
            "action": {
                "type": "choice",
                "instructions": (
                    "Choose exactly one legal poker action that maximizes your expected chip "
                    "winnings in this hand."
                ),
                "criteria": options,
            }
        }
        if ask_equity:
            questions["equity"] = {
                "type": "choice",
                "instructions": (
                    "Estimate the probability that your hand would win at showdown against "
                    "the opponent's likely holdings, given the action so far."
                ),
                "criteria": EQUITY_BUCKETS,
            }
        payload = {
            "model": self.model,
            "state": {"rules": rules, "situation": situation},
            "questions": questions,
            **self.extra,
        }
        body, meta = await self._send(payload, tag)
        if body is None:
            return None, None, meta, None
        try:
            ans = body["answers"]["action"]
            choice = ans["choice"]
            probs = {k: float(v) for k, v in ans["probabilities"].items()}
            meta["confidence"] = ans.get("confidence")
        except (KeyError, TypeError, ValueError):
            meta["error"] = "decision response without an action answer"
            return None, None, meta, None
        if choice not in options:
            meta["error"] = f"unknown option {choice!r}"
            return None, None, meta, probs
        if probs and probs.get(choice, 0) + 0.011 < max(probs.values()):
            meta["argmax_mismatch"] = True
        equity = None
        eq_ans = body["answers"].get("equity") if ask_equity else None
        if isinstance(eq_ans, dict) and isinstance(eq_ans.get("probabilities"), dict):
            ep = eq_ans["probabilities"]
            total = sum(float(v) for v in ep.values()) or 1.0
            equity = (
                sum(
                    float(v) * (int(k[1:3]) + 5) / 100 for k, v in ep.items() if k in EQUITY_BUCKETS
                )
                / total
            )
        return choice, equity, meta, probs

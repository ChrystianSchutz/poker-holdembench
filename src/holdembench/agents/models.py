"""Model-backed agents. Both families see the same rules text and situation text."""

from __future__ import annotations

from typing import Any

from holdembench.engine import Decision, Observation
from holdembench.providers import ChatClient, DecisionClient

SYSTEM_PROMPT = """You are an expert poker player in a heads-up match. Your goal is to win as many chips as possible over many hands.

{rules}

Each turn you receive the current situation and a menu of legal actions. Choose exactly one action label from the menu.
Reply with JSON only, no other text:
{{"action": "<one label from the legal actions>", "equity": <integer 0-100: your estimated probability of winning at showdown against the opponent's likely hand range>}}"""

INVALID = "__invalid__"


class ChatAgent:
    """A chat LLM answering with ``{action, equity}`` JSON."""

    kind = "chat"

    def __init__(self, name: str, client: ChatClient) -> None:
        self.name = name
        self.client = client

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.client.model,
            "max_tokens": self.client.max_tokens,
            "reasoning": self.client.reasoning,
            "structured": self.client.structured,
            "extra": self.client.extra,
        }

    async def act(self, obs: Observation) -> Decision:
        system = SYSTEM_PROMPT.format(rules=obs.rules.describe())
        tag = {
            "agent": self.name,
            "match": obs.match_id,
            "hand": obs.hand_index,
            "decision": obs.decision_index,
        }
        label, equity, meta = await self.client.decide(system, obs.render(), obs.labels, tag)
        return Decision(label or INVALID, equity=equity, meta=_slim(meta))


class DecisionAgent:
    """A typed decision model choosing from the same menu via ``/alpha/decisions``."""

    kind = "decision"

    def __init__(self, name: str, client: DecisionClient, *, ask_equity: bool = True) -> None:
        self.name = name
        self.client = client
        self.ask_equity = ask_equity

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.client.model,
            "ask_equity": self.ask_equity,
            "extra": self.client.extra,
        }

    async def act(self, obs: Observation) -> Decision:
        tag = {
            "agent": self.name,
            "match": obs.match_id,
            "hand": obs.hand_index,
            "decision": obs.decision_index,
        }
        options = {o.label: o.description for o in obs.options}
        rules = obs.rules.describe() + " Your goal is to win as many chips as possible."
        label, equity, meta, probs = await self.client.decide(
            rules, obs.render(), options, tag, ask_equity=self.ask_equity
        )
        return Decision(label or INVALID, equity=equity, probabilities=probs, meta=_slim(meta))


def _slim(meta: dict[str, Any]) -> dict[str, Any]:
    usage = meta.pop("usage", None) or {}
    meta["in_tok"] = usage.get("prompt_tokens", usage.get("input_tokens"))
    meta["out_tok"] = usage.get("completion_tokens", usage.get("output_tokens"))
    return meta

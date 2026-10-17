"""Heads-up hold'em hand runner on top of PokerKit.

PokerKit enforces the rules (blinds, betting order, raise sizes, side pots, showdown).
This module only adds three things: deterministic dealing from a :class:`Deal`, a
finite *action menu* shared by every agent, and a structured hand record.

PokerKit heads-up convention: player index 0 is the big blind, player index 1 is the
small blind / button, who acts first preflop and last on every later street.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol

from pokerkit import Automation, FixedLimitTexasHoldem, Mode, NoLimitTexasHoldem, State

from holdembench.cards import Deal, equity_vs_hand

STREETS = ("preflop", "flop", "turn", "river")
SEATS = ("BB", "SB")  # indexed by pokerkit player index
_AUTOMATIONS = (
    Automation.ANTE_POSTING,
    Automation.BET_COLLECTION,
    Automation.BLIND_OR_STRADDLE_POSTING,
    Automation.CARD_BURNING,
    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
    Automation.HAND_KILLING,
    Automation.CHIPS_PUSHING,
    Automation.CHIPS_PULLING,
    Automation.RUNOUT_COUNT_SELECTION,
)


@dataclass(frozen=True, slots=True)
class Rules:
    """Table parameters. Stacks are reset to ``stack`` before every hand."""

    variant: Literal["nl", "fl"] = "nl"
    small_blind: int = 50
    big_blind: int = 100
    stack: int = 10_000
    bet_fractions: tuple[float, ...] = (0.33, 0.5, 0.75, 1.0, 1.5)

    def describe(self) -> str:
        """One-line rules summary used in prompts."""
        name = "no-limit" if self.variant == "nl" else "fixed-limit"
        return (
            f"Heads-up {name} Texas hold'em cash game. Blinds {self.small_blind}/{self.big_blind}. "
            f"Both players start every hand with {self.stack} chips "
            f"({self.stack // self.big_blind} big blinds); stacks reset after each hand."
        )


@dataclass(frozen=True, slots=True)
class ActionOption:
    """One entry of the legal action menu."""

    label: str
    kind: Literal["fold", "check", "call", "bet", "raise", "all_in"]
    to: int | None
    description: str


@dataclass(frozen=True, slots=True)
class Event:
    """A public betting action."""

    street: str
    seat: str
    kind: str
    to: int | None
    added: int


@dataclass(slots=True)
class Observation:
    """Everything the acting player may know. Never contains the opponent's cards."""

    match_id: str
    hand_index: int
    decision_index: int
    rng_seed: int
    rules: Rules
    seat: str
    hole: list[str]
    board: list[str]
    street: str
    pot: int
    stack: int
    opp_stack: int
    my_bet: int
    opp_bet: int
    to_call: int
    history: list[Event]
    options: list[ActionOption]
    hud: dict[str, Any] | None = None

    @property
    def labels(self) -> list[str]:
        """Option labels in menu order."""
        return [o.label for o in self.options]

    def option(self, label: str) -> ActionOption | None:
        """Look up an option by label."""
        return next((o for o in self.options if o.label == label), None)

    def render(self) -> str:
        """Plain-text situation shared verbatim by every model-based agent."""
        bb = self.rules.big_blind
        pos = (
            "small blind / button (you act first preflop, last after the flop)"
            if self.seat == "SB"
            else "big blind (you act last preflop, first after the flop)"
        )
        lines = [
            f"Hand {self.hand_index + 1} of this match. Your position: {pos}.",
            f"Your hole cards: {' '.join(self.hole)}",
            f"Board: {' '.join(self.board) if self.board else '(none yet)'}  [street: {self.street}]",
            f"Pot (all chips committed so far, including current bets): {self.pot} "
            f"({self.pot / bb:.1f} BB)",
            f"Your remaining stack: {self.stack}. Opponent remaining stack: {self.opp_stack}.",
            f"Current street bets: you {self.my_bet}, opponent {self.opp_bet}. "
            f"Amount to call: {self.to_call}.",
            "Betting so far:",
        ]
        by_street: dict[str, list[str]] = {}
        for e in self.history:
            who = "you" if e.seat == self.seat else "opponent"
            verb = {
                "fold": "fold",
                "check": "check",
                "call": "call",
                "bet": f"bet {e.to}",
                "raise": f"raise to {e.to}",
                "all_in": f"go all-in (to {e.to})",
            }[e.kind]
            by_street.setdefault(e.street, []).append(f"{who} ({e.seat}) {verb}")
        if not by_street:
            lines.append("  preflop: blinds posted, no action yet")
        for street, acts in by_street.items():
            lines.append(f"  {street}: " + ", ".join(acts))
        if self.hud:
            lines.append(
                "Opponent tendencies in this match so far: "
                + ", ".join(f"{k} {v}" for k, v in self.hud.items())
            )
        lines.append("Legal actions:")
        lines.extend(f"  {o.label}: {o.description}" for o in self.options)
        return "\n".join(lines)


@dataclass(slots=True)
class Decision:
    """An agent's answer. ``equity`` is the agent's own showdown-equity estimate (0-1)."""

    label: str
    equity: float | None = None
    probabilities: dict[str, float] | None = None
    meta: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    """Anything that can choose from the action menu."""

    @property
    def name(self) -> str:
        """Unique agent id."""
        ...

    @property
    def kind(self) -> str:
        """Agent family (bot kind, ``chat`` or ``decision``)."""
        ...

    async def act(self, obs: Observation) -> Decision:
        """Choose one option label."""
        ...

    def describe(self) -> dict[str, Any]:
        """Manifest entry."""
        ...


def new_state(rules: Rules) -> State:
    """A fresh heads-up hand with blinds posted and no cards dealt."""
    stacks = (rules.stack, rules.stack)
    blinds = (rules.small_blind, rules.big_blind)
    if rules.variant == "nl":
        return NoLimitTexasHoldem.create_state(
            _AUTOMATIONS, True, 0, blinds, rules.big_blind, stacks, 2, mode=Mode.CASH_GAME
        )
    return FixedLimitTexasHoldem.create_state(
        _AUTOMATIONS,
        True,
        0,
        blinds,
        rules.big_blind,
        2 * rules.big_blind,
        stacks,
        2,
        mode=Mode.CASH_GAME,
    )


def legal_options(state: State, rules: Rules) -> list[ActionOption]:
    """The finite menu for the current actor: fold, check/call, a few sizes, all-in."""
    i = state.actor_index
    assert i is not None
    to_call = state.checking_or_calling_amount or 0
    opp_bet = max(state.bets)
    pot = sum(rules.stack - s for s in state.stacks)
    bb = rules.big_blind
    options: list[ActionOption] = []
    if to_call > 0:
        options.append(ActionOption("fold", "fold", None, "Fold and give up the pot"))
        all_in_call = to_call >= state.stacks[i]
        options.append(
            ActionOption(
                "call",
                "call",
                None,
                f"Call {to_call}" + (" (this puts you all-in)" if all_in_call else ""),
            )
        )
    else:
        options.append(ActionOption("check", "check", None, "Check"))
    if state.can_complete_bet_or_raise_to():
        lo = state.min_completion_betting_or_raising_to_amount
        hi = state.max_completion_betting_or_raising_to_amount
        assert lo is not None and hi is not None
        verb = "raise" if opp_bet > 0 else "bet"
        if rules.variant == "fl":
            options.append(_sized(verb, lo, opp_bet, pot, to_call, bb, all_in=False))
            return options
        targets: list[int] = []
        for frac in rules.bet_fractions:
            target = round((opp_bet + frac * (pot + to_call)) / 10) * 10
            target = max(target, lo)
            if target < hi and target not in targets:
                targets.append(target)
        options.extend(_sized(verb, t, opp_bet, pot, to_call, bb, all_in=False) for t in targets)
        options.append(_sized(verb, hi, opp_bet, pot, to_call, bb, all_in=True))
    return options


def _sized(
    verb: str, to: int, opp_bet: int, pot: int, to_call: int, bb: int, *, all_in: bool
) -> ActionOption:
    frac = (to - opp_bet) / (pot + to_call) if pot + to_call else 0.0
    head = "Go all-in" if all_in else ("Bet" if verb == "bet" else "Raise to")
    desc = f"{head} {to} ({to / bb:.1f} BB total this street; {frac:.2f}x pot)"
    if all_in:
        return ActionOption("all_in", "all_in", to, desc)
    return ActionOption(f"{verb}_{to}", "bet" if verb == "bet" else "raise", to, desc)


def _flat(cards: Sequence[Sequence[Any]]) -> int:
    return sum(len(c) for c in cards)


async def play_hand(
    players: Sequence[Agent],
    deal: Deal,
    rules: Rules,
    *,
    match_id: str,
    hand_index: int,
    huds: Sequence[dict[str, Any] | None] = (None, None),
) -> dict[str, Any]:
    """Play one hand. ``players[i]`` sits at pokerkit index ``i`` (0 = BB, 1 = SB)."""
    with warnings.catch_warnings():
        # PokerKit warns when dealt cards are not drawn from its own shuffled deck.
        warnings.simplefilter("ignore", UserWarning)
        state = new_state(rules)
        state.deal_hole("".join(deal.holes[0]))
        state.deal_hole("".join(deal.holes[1]))
        history: list[Event] = []
        actions: list[dict[str, Any]] = []
        allin: dict[str, Any] | None = None
        board: list[str] = []
        decision_index = 0
        while state.status:
            if state.actor_index is not None:
                i = state.actor_index
                options = legal_options(state, rules)
                street = STREETS[state.street_index or 0]
                obs = Observation(
                    match_id=match_id,
                    hand_index=hand_index,
                    decision_index=decision_index,
                    rng_seed=(deal.seed * 1_000_003 + hand_index * 7919 + decision_index)
                    & 0x7FFFFFFF,
                    rules=rules,
                    seat=SEATS[i],
                    hole=list(deal.holes[i]),
                    board=list(board),
                    street=street,
                    pot=sum(rules.stack - s for s in state.stacks),
                    stack=state.stacks[i],
                    opp_stack=state.stacks[1 - i],
                    my_bet=state.bets[i],
                    opp_bet=state.bets[1 - i],
                    to_call=state.checking_or_calling_amount or 0,
                    history=list(history),
                    options=options,
                    hud=huds[i],
                )
                decision = await players[i].act(obs)
                chosen = obs.option(decision.label)
                fallback = chosen is None
                if chosen is None:
                    chosen = obs.option("check") or obs.option("fold")
                    assert chosen is not None
                before = state.stacks[i]
                if chosen.kind == "fold":
                    state.fold()
                elif chosen.kind in ("check", "call"):
                    state.check_or_call()
                else:
                    state.complete_bet_or_raise_to(chosen.to)
                added = before - state.stacks[i]
                history.append(Event(street, SEATS[i], chosen.kind, chosen.to, added))
                actions.append(
                    {
                        "i": decision_index,
                        "street": street,
                        "seat": SEATS[i],
                        "player": players[i].name,
                        "label": chosen.label,
                        "kind": chosen.kind,
                        "to": chosen.to,
                        "added": added,
                        "pot_before": obs.pot,
                        "to_call": obs.to_call,
                        "my_bet": obs.my_bet,
                        "opp_bet": obs.opp_bet,
                        "stack_before": obs.stack,
                        "n_options": len(options),
                        "options": obs.labels,
                        "board_len": len(board),
                        "raw_label": decision.label,
                        "fallback": fallback,
                        "equity_est": decision.equity,
                        "probabilities": decision.probabilities,
                        **{f"m_{k}": v for k, v in decision.meta.items()},
                    }
                )
                decision_index += 1
                if (
                    allin is None
                    and state.status
                    and state.actor_index is None
                    and all(state.statuses)
                    and 0 in state.stacks
                    and len(board) < 5
                ):
                    allin = {
                        "board_len": len(board),
                        "contrib": [rules.stack - s for s in state.stacks],
                    }
            elif state.can_deal_board():
                n = len(board)
                cards = list(deal.board[n : n + (3 if n == 0 else 1)])
                state.deal_board("".join(cards))
                board.extend(cards)
            else:  # pragma: no cover - automations cover every other phase
                msg = f"unexpected pokerkit phase in {match_id}#{hand_index}"
                raise RuntimeError(msg)

    payoffs = [int(p) for p in state.payoffs]
    folded = any(a["kind"] == "fold" for a in actions)
    adjusted = [float(p) for p in payoffs]
    if allin is not None:
        known = list(deal.board[: allin["board_len"]])
        eq0 = equity_vs_hand(list(deal.holes[0]), list(deal.holes[1]), known, seed=deal.seed)
        pot = sum(allin["contrib"])
        adjusted = [eq0 * pot - allin["contrib"][0], (1 - eq0) * pot - allin["contrib"][1]]
        allin["equity"] = [eq0, 1 - eq0]
    return {
        "match_id": match_id,
        "hand_index": hand_index,
        "deal_seed": deal.seed,
        "players": [p.name for p in players],
        "seats": {SEATS[0]: players[0].name, SEATS[1]: players[1].name},
        "holes": [list(h) for h in deal.holes],
        "board": board,
        "full_board": list(deal.board),
        "actions": actions,
        "payoffs": payoffs,
        "allin_adjusted": adjusted,
        "allin": allin,
        "showdown": not folded,
        "fallbacks": sum(a["fallback"] for a in actions),
        "errors": sum(1 for a in actions if a.get("m_error")),
        "provider_errors": sum(1 for a in actions if provider_failed(a)),
    }


def provider_failed(action: dict[str, Any]) -> bool:
    """True when a decision failed for transport/provider reasons, not because of the model.

    Such hands are void: the runner replays them on resume and analysis ignores them.
    Unparseable, truncated or illegal answers are the model's own failures and count.
    """
    if action.get("m_provider_error"):
        return True
    err = str(action.get("m_error") or "")
    return err.startswith(("HTTP ", "non-JSON")) or "Error:" in err or "Timeout" in err


def event_dict(e: Event) -> dict[str, Any]:
    """Serialise an event."""
    return asdict(e)

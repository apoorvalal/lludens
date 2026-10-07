"""Small simultaneous economic games for typed, single-turn decision models."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

from ..actions import ActionSpace
from ..environments import Phase, PhasedGame, RoundState


@dataclass(frozen=True)
class EconomicGame:
    name: str
    actions: tuple[str, ...]
    descriptions: Mapping[str, str]
    rules: str
    cooperative_action: str

    def payoff(self, a: str, b: str) -> tuple[float, float]:
        if a not in self.actions or b not in self.actions:
            raise ValueError("Action outside this game's action space")
        if self.name == "prisoners_dilemma":
            return {("C", "C"): (3, 3), ("C", "D"): (0, 5),
                    ("D", "C"): (5, 0), ("D", "D"): (1, 1)}[a, b]
        if self.name == "stag_hunt":
            return {("Stag", "Stag"): (4, 4), ("Stag", "Hare"): (0, 3),
                    ("Hare", "Stag"): (3, 0), ("Hare", "Hare"): (3, 3)}[a, b]
        if self.name == "public_goods":
            x, y = int(a), int(b)
            share = 0.8 * (x + y)
            return 10 - x + share, 10 - y + share
        raise ValueError("Unknown economic game")

    def contribution(self, action: str) -> float:
        if action not in self.actions:
            raise ValueError("Invalid action")
        return int(action) / 10 if self.name == "public_goods" else float(action == self.cooperative_action)


GAMES = {
    "prisoners_dilemma": EconomicGame(
        "prisoners_dilemma", ("C", "D"),
        {"C": "Choose Cooperate", "D": "Choose Defect"},
        "Repeated Prisoner's Dilemma. Both choose C or D simultaneously. "
        "Payoffs (your action, other action -> your points, other points): "
        "C,C -> 3,3; C,D -> 0,5; D,C -> 5,0; D,D -> 1,1.", "C"),
    "stag_hunt": EconomicGame(
        "stag_hunt", ("Stag", "Hare"),
        {"Stag": "Choose Stag", "Hare": "Choose Hare"},
        "Repeated Stag Hunt. Both choose Stag or Hare simultaneously. "
        "Payoffs (your action, other action -> your points, other points): "
        "Stag,Stag -> 4,4; Stag,Hare -> 0,3; Hare,Stag -> 3,0; Hare,Hare -> 3,3.", "Stag"),
    "public_goods": EconomicGame(
        "public_goods", tuple(str(i) for i in range(0, 11, 2)),
        {str(i): f"Contribute {i} of your 10 tokens to the joint pool; keep {10-i} tokens" for i in range(0, 11, 2)},
        "Repeated TWO-PLAYER public goods game. Each player receives a fresh endowment of 10 tokens each round. "
        "Each simultaneously contributes 0, 2, 4, 6, 8, or 10 tokens. "
        "The joint contribution is multiplied by 1.6 and divided equally between the two players. "
        "Your points = 10 - your contribution + 0.8*(your contribution + other contribution). "
        "Other points = 10 - other contribution + 0.8*(your contribution + other contribution). "
        "For example, both contribute 10 -> 16 points each; both contribute 0 -> 10 each; "
        "you contribute 0 and other contributes 10 -> you get 18, other gets 8. "
        "Unused tokens do not carry into the next endowment; points accumulate.", "10"),
}


INSTRUCTIONS = (
    "Choose your action for the current round to maximize YOUR OWN expected total points "
    "over the entire match, including future rounds. You are a player, not an observer "
    "classifying the history. Select one of the allowed actions."
)


class RepeatedDecisionGame(PhasedGame):
    """Uses PhasedGame's staged simultaneous moves; all history is explicit.

    Neither opponent identity nor current-round actions/probabilities are in prompts.
    No action fallback, corrective retries, free-form chat, or covert state is used.
    """

    def __init__(self, game: str, agents: Mapping[int, object], n_rounds: int = 20):
        if set(agents) != {1, 2} or type(n_rounds) is not int or n_rounds < 1:
            raise ValueError("Exactly two players and a positive integer horizon are required")
        self.spec = GAMES[game]
        phase = Phase("move", "action", action_space=ActionSpace(self.spec.actions), invalid_retries=0)
        super().__init__(agents, n_rounds, (phase,))

    def prompt_for(self, phase: Phase, player: int, state: RoundState) -> str:
        other = 3 - player
        history = [
            {"round": h["round"], "your_action": h["actions"][str(player)],
             "other_action": h["actions"][str(other)], "your_points": h["payoffs"][str(player)],
             "other_points": h["payoffs"][str(other)]} for h in self.history
        ]
        total = sum(h["your_points"] for h in history)
        return (
            f"{self.spec.rules}\nYou are player {player} in a two-player match. "
            f"This is round {state.round_number} of exactly {self.n_rounds}. "
            "Both players know the rules and the fixed horizon, and face the same opponent throughout. "
            "Both choose without seeing the other's current move. After each round both actions and payoffs "
            "are revealed. There is no communication. You do not know the opponent's strategy or identity. "
            "No side payments, other rewards, or later matches affect your score. "
            "Your objective is your own total points, not beating the other's score or maximizing the joint score.\n"
            f"Your points so far: {total}.\nComplete public history from your perspective: "
            + json.dumps(history, separators=(",", ":"))
        )

    def resolve_round(self, state: RoundState) -> dict:
        a, b = state.value("move", 1), state.value("move", 2)
        u, v = self.spec.payoff(a, b)
        return {"round": state.round_number, "actions": {"1": a, "2": b},
                "payoffs": {"1": u, "2": v}}

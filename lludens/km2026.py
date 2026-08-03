from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
from pathlib import Path
from typing import Iterable, Literal, Mapping

from .actions import BINARY_ACTION_SPACE, parse_binary_action
from .agent import AgentRequest, PromptParameter, TotreLLM
from .environments import Phase, PhasedGame, RoundState


Action = Literal["Cooperate", "Defect"]
Treatment = Literal["sim", "seq", "chat"]


@dataclass(frozen=True)
class PayoffMatrix:
    mutual_cooperate: int = 32
    sucker: int = 12
    temptation: int = 50
    mutual_defect: int = 25

    def payoff(self, player1_action: str, player2_action: str) -> tuple[int, int]:
        player1_action = parse_binary_action(player1_action)
        player2_action = parse_binary_action(player2_action)
        if player1_action == "Cooperate" and player2_action == "Cooperate":
            return self.mutual_cooperate, self.mutual_cooperate
        if player1_action == "Cooperate" and player2_action == "Defect":
            return self.sucker, self.temptation
        if player1_action == "Defect" and player2_action == "Cooperate":
            return self.temptation, self.sucker
        return self.mutual_defect, self.mutual_defect

    def describe(self) -> str:
        return (
            "Payoff matrix, with your payoff listed first: "
            f"Cooperate/Cooperate = {self.mutual_cooperate}, {self.mutual_cooperate}; "
            f"Cooperate/Defect = {self.sucker}, {self.temptation}; "
            f"Defect/Cooperate = {self.temptation}, {self.sucker}; "
            f"Defect/Defect = {self.mutual_defect}, {self.mutual_defect}."
        )


def classify_chat_message(message: str | None) -> str:
    text = (message or "").lower()
    cooperative = [
        r"\bcooperate\b",
        r"\bcooperation\b",
        r"\bcollaborat",
        r"\bwork together\b",
        r"\bteam up\b",
        r"\bmutual\b",
        r"\bboth choose c\b",
        r"\bchoose c\b",
        r"\btrust\b",
    ]
    defective = [
        r"\bdefect\b",
        r"\bchoose d\b",
        r"\bdon'?t cooperate\b",
        r"\bnot cooperate\b",
        r"\bbetray\b",
        r"\bcompete\b",
    ]
    if any(re.search(pattern, text) for pattern in cooperative):
        return "cooperative"
    if any(re.search(pattern, text) for pattern in defective):
        return "defective"
    return "neutral"


def public_history(history: list[dict], max_rounds: int = 12) -> str:
    if not history:
        return "No previous rounds."
    parts = []
    for row in history[-max_rounds:]:
        if row["treatment"] == "chat":
            parts.append(
                "Round {round}: messages: player 1={m1!r}; player 2={m2!r}. "
                "Actions: player 1 {a1}, player 2 {a2}. Payoffs: {p1}, {p2}.".format(
                    round=row["round"],
                    m1=row["player1_message"],
                    m2=row["player2_message"],
                    a1=row["player1_action"],
                    a2=row["player2_action"],
                    p1=row["payoff1"],
                    p2=row["payoff2"],
                )
            )
        else:
            parts.append(
                "Round {round}: player 1 {a1}, player 2 {a2}. Payoffs: {p1}, {p2}.".format(
                    round=row["round"],
                    a1=row["player1_action"],
                    a2=row["player2_action"],
                    p1=row["payoff1"],
                    p2=row["payoff2"],
                )
            )
    prefix = f"Only the last {max_rounds} rounds are shown. " if len(history) > max_rounds else ""
    return prefix + "\n".join(parts)


class LLMPolicy(TotreLLM):
    """Repeated-PD configuration of the general `TotreLLM` agent."""

    def __init__(
        self,
        model_id: str,
        *,
        temperature: float | None = None,
        max_tokens: int = 64,
        label: str | None = None,
        private_type: str | None = None,
        private_gamma: float | None = None,
    ):
        parameters: dict[str, PromptParameter] = {}
        if private_type is not None:
            parameters["type"] = PromptParameter(
                "type",
                private_type,
                "Your private experimental type is {value}.",
            )
        if private_gamma is not None:
            parameters["gamma"] = PromptParameter(
                "gamma",
                private_gamma,
                (
                    "Your private gamma is {value}. Add gamma to your own stage utility whenever "
                    "you choose Cooperate. Your opponent does not observe your gamma, and you do "
                    "not observe theirs. Keep this value fixed throughout the match."
                ),
            )
        options: dict[str, float | int] = {"max_tokens": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        super().__init__(
            model_id,
            system=(
                "You are a subject in an economics experiment. Your objective is to maximize "
                "your own utility over the whole match. Other players are independent subjects "
                "with the same objective. Do not explain an action unless asked for a message."
            ),
            options=options,
            label=label,
            private_parameters=parameters,
        )

    @property
    def private_gamma(self) -> float | None:
        parameter = self.private_parameters.get("gamma")
        return None if parameter is None else float(parameter.value)


class ScriptedPolicy:
    def __init__(self, strategy: str, seed: int | None = None, label: str | None = None):
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.label = label or f"scripted/{strategy}"
        self.private_parameters: dict[str, PromptParameter] = {}

    def observe(self, observation) -> None:
        return None

    def communicate(self, request: AgentRequest) -> str:
        action = self._action(request)
        if action == "Cooperate":
            return "I propose that we both cooperate."
        return "I am not committing to cooperation."

    def respond(self, request: AgentRequest) -> Action:
        return self._action(request)

    def _action(self, request: AgentRequest) -> Action:
        metadata = request.metadata or {}
        history = metadata.get("history", [])
        state = metadata.get("state")
        strategy = self.strategy.lower().replace("-", "_")
        if strategy in {"always_cooperate", "cooperate"}:
            return "Cooperate"
        if strategy in {"always_defect", "defect"}:
            return "Defect"
        if strategy == "random":
            return "Cooperate" if self.rng.random() < 0.5 else "Defect"
        opponent = 2 if request.player == 1 else 1
        if strategy == "grim":
            if any(row[f"player{opponent}_action"] == "Defect" for row in history):
                return "Defect"
            return "Cooperate"
        if strategy == "tit_for_tat":
            if request.phase == "action" and state is not None:
                current = state.value("action", opponent)
                if current is not None:
                    return current
            if not history:
                return "Cooperate"
            return history[-1][f"player{opponent}_action"]
        raise ValueError(f"Unknown scripted strategy: {self.strategy}")


class RepeatedPDGame(PhasedGame):
    """PD payoff and prompts expressed through reusable environment phases."""

    def __init__(
        self,
        treatment: Treatment,
        horizon: int,
        player1,
        player2,
        *,
        payoff_matrix: PayoffMatrix | None = None,
        seed: int = 0,
        match_id: str | None = None,
    ):
        if treatment not in {"sim", "seq", "chat"}:
            raise ValueError(f"Unknown treatment: {treatment}")
        if treatment == "chat":
            phases = (
                Phase("message", "communication", simultaneous=True),
                Phase("action", "action", simultaneous=True, action_space=BINARY_ACTION_SPACE),
            )
        elif treatment == "seq":
            phases = (
                Phase("action", "action", simultaneous=False, action_space=BINARY_ACTION_SPACE),
            )
        else:
            phases = (
                Phase("action", "action", simultaneous=True, action_space=BINARY_ACTION_SPACE),
            )
        super().__init__({1: player1, 2: player2}, horizon, phases)
        self.treatment = treatment
        self.payoff_matrix = payoff_matrix or PayoffMatrix()
        self.seed = seed
        self.match_id = match_id or f"{treatment}-h{horizon}-s{seed}"

    def request_metadata(self, phase: Phase, player: int, state: RoundState) -> Mapping:
        return {
            "environment": self,
            "history": self.history,
            "phase": phase,
            "state": state,
        }

    def prompt_for(self, phase: Phase, player: int, state: RoundState) -> str:
        if self.treatment == "seq":
            role = "first mover" if player == 1 else "second mover"
            treatment_note = "Actions this round are sequential."
        else:
            role = "simultaneous mover"
            treatment_note = (
                "A simultaneous message phase occurs before simultaneous actions."
                if self.treatment == "chat"
                else "Actions this round are simultaneous."
            )
        current = []
        if self.treatment == "seq" and player == 2:
            current.append(f"The first mover already chose {state.value('action', 1)}.")
        if phase.name == "action" and self.treatment == "chat":
            current.append(
                "This round's messages are: "
                f"player 1={state.value('message', 1)!r}; "
                f"player 2={state.value('message', 2)!r}."
            )
        base = (
            f"You are player {player} ({role}). This is round {state.round_number} of {self.n_rounds}. "
            f"{treatment_note}\n{self.payoff_matrix.describe()}\n"
            + ("\n".join(current) + "\n" if current else "")
            + "Completed-round outcomes are in your conversation history."
        )
        if phase.kind == "communication":
            return base + "\n\nWrite one message to the other player, under 25 words."
        return base + "\n\nChoose one action. Respond with exactly: Cooperate or Defect."

    def resolve_round(self, state: RoundState) -> dict:
        action1 = state.value("action", 1)
        action2 = state.value("action", 2)
        action1_response = state.response("action", 1)
        action2_response = state.response("action", 2)
        message1 = state.value("message", 1)
        message2 = state.value("message", 2)
        payoff1, payoff2 = self.payoff_matrix.payoff(action1, action2)
        message1_label = classify_chat_message(message1)
        message2_label = classify_chat_message(message2)
        return {
            "match_id": self.match_id,
            "seed": self.seed,
            "treatment": self.treatment,
            "horizon": self.n_rounds,
            "round": state.round_number,
            "player1_model": getattr(self.agents[1], "label", type(self.agents[1]).__name__),
            "player2_model": getattr(self.agents[2], "label", type(self.agents[2]).__name__),
            "player1_gamma": private_parameter_value(self.agents[1], "gamma"),
            "player2_gamma": private_parameter_value(self.agents[2], "gamma"),
            "player1_message": message1,
            "player2_message": message2,
            "player1_message_label": message1_label,
            "player2_message_label": message2_label,
            "both_cooperative_messages": self.treatment == "chat"
            and message1_label == "cooperative"
            and message2_label == "cooperative",
            "player1_raw_action": action1_response.raw,
            "player2_raw_action": action2_response.raw,
            "player1_action": action1,
            "player2_action": action2,
            "player1_invalid_action": action1_response.invalid,
            "player2_invalid_action": action2_response.invalid,
            "payoff1": payoff1,
            "payoff2": payoff2,
            "player1_cooperated": action1 == "Cooperate",
            "player2_cooperated": action2 == "Cooperate",
            "mutual_cooperation": action1 == "Cooperate" and action2 == "Cooperate",
        }

    def observation_for(self, player: int, record: Mapping) -> str:
        if self.treatment == "chat":
            message_text = (
                f" Messages: player 1={record['player1_message']!r}; "
                f"player 2={record['player2_message']!r}."
            )
        else:
            message_text = ""
        return (
            f"Round {record['round']} completed.{message_text} "
            f"Player 1 chose {record['player1_action']}; player 2 chose {record['player2_action']}. "
            f"Payoffs were {record['payoff1']} and {record['payoff2']}."
        )


def private_parameter_value(agent, name: str):
    parameter = getattr(agent, "private_parameters", {}).get(name)
    return None if parameter is None else parameter.value


def run_match(
    treatment: Treatment,
    horizon: int,
    player1,
    player2,
    *,
    payoff_matrix: PayoffMatrix | None = None,
    seed: int = 0,
    match_id: str | None = None,
) -> list[dict]:
    game = RepeatedPDGame(
        treatment,
        horizon,
        player1,
        player2,
        payoff_matrix=payoff_matrix,
        seed=seed,
        match_id=match_id,
    )
    return game.run_game()


def run_llm_sweep(
    *,
    model1: str,
    model2: str,
    horizons: Iterable[int] = (10, 20, 50),
    treatments: Iterable[Treatment] = ("sim", "seq", "chat"),
    seed: int = 20260802,
    temperature: float | None = None,
) -> list[dict]:
    rows: list[dict] = []
    for horizon in horizons:
        for treatment in treatments:
            player1 = LLMPolicy(model1, temperature=temperature)
            player2 = LLMPolicy(model2, temperature=temperature)
            match_id = f"{treatment}-h{horizon}-{model1.split('/')[-1]}-{model2.split('/')[-1]}-s{seed}"
            rows.extend(
                run_match(
                    treatment,
                    horizon,
                    player1,
                    player2,
                    seed=seed,
                    match_id=match_id,
                )
            )
    return rows


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as file:
        return [json.loads(line) for line in file if line.strip()]


def cooperation_summary(rows: Iterable[dict]):
    import pandas as pd

    data = pd.DataFrame(rows)
    summary = (
        data.groupby(["treatment", "horizon"], as_index=False)
        .agg(
            rounds=("round", "count"),
            player1_cooperation=("player1_cooperated", "mean"),
            player2_cooperation=("player2_cooperated", "mean"),
            mutual_cooperation=("mutual_cooperation", "mean"),
            player1_payoff=("payoff1", "sum"),
            player2_payoff=("payoff2", "sum"),
        )
        .sort_values(["horizon", "treatment"])
    )
    for column in ["player1_cooperation", "player2_cooperation", "mutual_cooperation"]:
        summary[column] = summary[column].round(3)
    return summary


def seq_second_mover_summary(rows: Iterable[dict]):
    import pandas as pd

    data = pd.DataFrame(rows)
    sequential = data[data["treatment"] == "seq"].copy()
    if sequential.empty:
        return pd.DataFrame()
    return (
        sequential.groupby(["horizon", "player1_action"], as_index=False)
        .agg(
            rounds=("round", "count"),
            second_mover_cooperation=("player2_cooperated", "mean"),
        )
        .sort_values(["horizon", "player1_action"])
    )


def chat_summary(rows: Iterable[dict]):
    import pandas as pd

    data = pd.DataFrame(rows)
    chat = data[data["treatment"] == "chat"].copy()
    if chat.empty:
        return pd.DataFrame()
    both = chat[chat["both_cooperative_messages"]]
    follow = float(both["mutual_cooperation"].mean()) if len(both) else float("nan")
    return pd.DataFrame(
        [
            {
                "rounds": len(chat),
                "player1_cooperative_messages": (chat["player1_message_label"] == "cooperative").mean(),
                "player2_cooperative_messages": (chat["player2_message_label"] == "cooperative").mean(),
                "both_cooperative_messages": chat["both_cooperative_messages"].mean(),
                "follow_through_after_mutual_coop_messages": follow,
            }
        ]
    ).round(3)

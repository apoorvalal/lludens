from __future__ import annotations

from dataclasses import dataclass
import json
import random
import re
from pathlib import Path
from typing import Iterable, Literal

import llm

from .actions import parse_binary_action


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


@dataclass(frozen=True)
class Observation:
    treatment: Treatment
    horizon: int
    round_number: int
    player: int
    role: str
    payoff_matrix: PayoffMatrix
    history: list[dict]
    first_mover_action: str | None = None
    player1_message: str | None = None
    player2_message: str | None = None

    @property
    def opponent(self) -> int:
        return 2 if self.player == 1 else 1


def parse_or_defect(raw_response: str) -> tuple[Action, bool]:
    try:
        return parse_binary_action(raw_response), False
    except ValueError:
        return "Defect", True


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


def public_history(history: list[dict], player: int | None = None, max_rounds: int = 12) -> str:
    if not history:
        return "No previous rounds."
    rows = history[-max_rounds:]
    parts = []
    for row in rows:
        if row.get("treatment") == "chat":
            parts.append(
                "Round {round}: messages: player 1={m1!r}; player 2={m2!r}. "
                "Actions: player 1 {a1}, player 2 {a2}. Payoffs: {p1}, {p2}.".format(
                    round=row["round"],
                    m1=row.get("player1_message", ""),
                    m2=row.get("player2_message", ""),
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
    if len(history) > max_rounds:
        prefix = f"Only the last {max_rounds} rounds are shown. "
    else:
        prefix = ""
    return prefix + "\n".join(parts)


class ScriptedPolicy:
    def __init__(self, strategy: str, seed: int | None = None, label: str | None = None):
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.label = label or f"scripted/{strategy}"

    def message(self, observation: Observation) -> str:
        action = self.act(observation)
        if action == "Cooperate":
            return "I propose that we both cooperate."
        return "I am not committing to cooperation."

    def act(self, observation: Observation) -> Action:
        strategy = self.strategy.lower().replace("-", "_")
        if strategy in {"always_cooperate", "cooperate"}:
            return "Cooperate"
        if strategy in {"always_defect", "defect"}:
            return "Defect"
        if strategy == "random":
            return "Cooperate" if self.rng.random() < 0.5 else "Defect"
        if strategy == "grim":
            opponent_key = f"player{observation.opponent}_action"
            if any(row[opponent_key] == "Defect" for row in observation.history):
                return "Defect"
            return "Cooperate"
        if strategy == "tit_for_tat":
            if not observation.history:
                return "Cooperate"
            return observation.history[-1][f"player{observation.opponent}_action"]
        raise ValueError(f"Unknown scripted strategy: {self.strategy}")


class LLMPolicy:
    def __init__(
        self,
        model_id: str,
        *,
        temperature: float | None = 0.1,
        label: str | None = None,
        private_type: str | None = None,
        private_gamma: float | None = None,
    ):
        self.model_id = model_id
        self.model = llm.get_model(model_id)
        self.options = {} if temperature is None else {"temperature": temperature}
        self.label = label or model_id
        self.private_type = private_type
        self.private_gamma = private_gamma

    def _system(self) -> str:
        type_note = ""
        if self.private_type:
            type_note = f" Private note: {self.private_type}"
        gamma_note = ""
        if self.private_gamma is not None:
            gamma_note = (
                f" Your private gamma is {self.private_gamma:g}. "
                "Add gamma to your own stage utility whenever you choose Cooperate. "
                "Your opponent does not observe your gamma, and you do not observe theirs. "
                "Keep this private information fixed throughout the match."
            )
        return (
            "You are a subject in an economics experiment. "
            "Your objective is to maximize your own point total over the whole match. "
            "Other players are independent subjects with the same objective. "
            "Do not explain your answer unless asked for a chat message."
            + type_note
            + gamma_note
        )

    def _base_prompt(self, observation: Observation) -> str:
        treatment_note = {
            "sim": "Actions this round are simultaneous.",
            "seq": "Actions this round are sequential.",
            "chat": "A chat message phase occurs before simultaneous actions.",
        }[observation.treatment]
        current = ""
        if observation.first_mover_action is not None:
            current += f"\nThe first mover already chose: {observation.first_mover_action}."
        if observation.player1_message is not None or observation.player2_message is not None:
            current += (
                f"\nThis round's messages are: player 1: {observation.player1_message!r}; "
                f"player 2: {observation.player2_message!r}."
            )
        return (
            f"You are player {observation.player} ({observation.role}). "
            f"This is round {observation.round_number} of {observation.horizon}. "
            f"{treatment_note}\n"
            f"{observation.payoff_matrix.describe()}\n"
            f"{current}\n"
            f"Public history:\n{public_history(observation.history, observation.player)}"
        )

    def message(self, observation: Observation) -> str:
        prompt = (
            self._base_prompt(observation)
            + "\n\nWrite one short free-form message to the other player before choosing actions. "
            "Keep it under 25 words."
        )
        return self.model.prompt(prompt, system=self._system(), **self.options).text().strip()

    def act(self, observation: Observation) -> str:
        prompt = (
            self._base_prompt(observation)
            + "\n\nChoose your action for this round. "
            "Respond with exactly one word: Cooperate or Defect."
        )
        return self.model.prompt(prompt, system=self._system(), **self.options).text().strip()


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
    if treatment not in {"sim", "seq", "chat"}:
        raise ValueError(f"Unknown treatment: {treatment}")
    payoff_matrix = payoff_matrix or PayoffMatrix()
    match_id = match_id or f"{treatment}-h{horizon}-s{seed}"
    history: list[dict] = []

    for round_number in range(1, horizon + 1):
        message1 = message2 = None
        if treatment == "chat":
            obs1_msg = Observation(treatment, horizon, round_number, 1, "simultaneous mover", payoff_matrix, history)
            obs2_msg = Observation(treatment, horizon, round_number, 2, "simultaneous mover", payoff_matrix, history)
            message1 = player1.message(obs1_msg)
            message2 = player2.message(obs2_msg)

        if treatment == "seq":
            obs1 = Observation(treatment, horizon, round_number, 1, "first mover", payoff_matrix, history)
            raw1 = player1.act(obs1)
            action1, invalid1 = parse_or_defect(raw1)
            obs2 = Observation(
                treatment,
                horizon,
                round_number,
                2,
                "second mover",
                payoff_matrix,
                history,
                first_mover_action=action1,
            )
            raw2 = player2.act(obs2)
            action2, invalid2 = parse_or_defect(raw2)
        else:
            role = "simultaneous mover"
            obs1 = Observation(
                treatment,
                horizon,
                round_number,
                1,
                role,
                payoff_matrix,
                history,
                player1_message=message1,
                player2_message=message2,
            )
            obs2 = Observation(
                treatment,
                horizon,
                round_number,
                2,
                role,
                payoff_matrix,
                history,
                player1_message=message1,
                player2_message=message2,
            )
            raw1 = player1.act(obs1)
            raw2 = player2.act(obs2)
            action1, invalid1 = parse_or_defect(raw1)
            action2, invalid2 = parse_or_defect(raw2)

        payoff1, payoff2 = payoff_matrix.payoff(action1, action2)
        message1_label = classify_chat_message(message1)
        message2_label = classify_chat_message(message2)
        row = {
            "match_id": match_id,
            "seed": seed,
            "treatment": treatment,
            "horizon": horizon,
            "round": round_number,
            "player1_model": getattr(player1, "label", type(player1).__name__),
            "player2_model": getattr(player2, "label", type(player2).__name__),
            "player1_gamma": getattr(player1, "private_gamma", None),
            "player2_gamma": getattr(player2, "private_gamma", None),
            "player1_message": message1,
            "player2_message": message2,
            "player1_message_label": message1_label,
            "player2_message_label": message2_label,
            "both_cooperative_messages": treatment == "chat"
            and message1_label == "cooperative"
            and message2_label == "cooperative",
            "player1_raw_action": raw1,
            "player2_raw_action": raw2,
            "player1_action": action1,
            "player2_action": action2,
            "player1_invalid_action": invalid1,
            "player2_invalid_action": invalid2,
            "payoff1": payoff1,
            "payoff2": payoff2,
            "player1_cooperated": action1 == "Cooperate",
            "player2_cooperated": action2 == "Cooperate",
            "mutual_cooperation": action1 == "Cooperate" and action2 == "Cooperate",
        }
        history.append(row)

    return history


def run_llm_sweep(
    *,
    model1: str,
    model2: str,
    horizons: Iterable[int] = (10, 20, 50),
    treatments: Iterable[Treatment] = ("sim", "seq", "chat"),
    seed: int = 20260802,
    temperature: float = 0.1,
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
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def cooperation_summary(rows: Iterable[dict]):
    import pandas as pd

    df = pd.DataFrame(rows)
    summary = (
        df.groupby(["treatment", "horizon"], as_index=False)
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
    for col in ["player1_cooperation", "player2_cooperation", "mutual_cooperation"]:
        summary[col] = summary[col].round(3)
    return summary


def seq_second_mover_summary(rows: Iterable[dict]):
    import pandas as pd

    df = pd.DataFrame(rows)
    seq = df[df["treatment"] == "seq"].copy()
    if seq.empty:
        return pd.DataFrame()
    return (
        seq.groupby(["horizon", "player1_action"], as_index=False)
        .agg(
            rounds=("round", "count"),
            second_mover_cooperation=("player2_cooperated", "mean"),
        )
        .sort_values(["horizon", "player1_action"])
    )


def chat_summary(rows: Iterable[dict]):
    import pandas as pd

    df = pd.DataFrame(rows)
    chat = df[df["treatment"] == "chat"].copy()
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

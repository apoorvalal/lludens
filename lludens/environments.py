from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Literal, Mapping

from .actions import ActionSpace
from .agent import AgentRequest


PhaseKind = Literal["action", "communication", "response"]


@dataclass(frozen=True)
class Phase:
    """One stage of a round, with simultaneous or ordered responses."""

    name: str
    kind: PhaseKind
    actors: tuple[int, ...] = (1, 2)
    simultaneous: bool = True
    action_space: ActionSpace | None = None
    invalid_retries: int = 2


@dataclass(frozen=True)
class PhaseResponse:
    raw: str
    value: Any
    invalid: bool = False


@dataclass
class RoundState:
    round_number: int
    responses: dict[str, dict[int, PhaseResponse]] = field(default_factory=dict)

    def set_response(self, phase: str, actor: int, response: PhaseResponse) -> None:
        self.responses.setdefault(phase, {})[actor] = response

    def response(self, phase: str, actor: int) -> PhaseResponse | None:
        return self.responses.get(phase, {}).get(actor)

    def value(self, phase: str, actor: int, default: Any = None) -> Any:
        response = self.response(phase, actor)
        return default if response is None else response.value


class PhasedGame:
    """Generic repeated-game runner composed from ordered response phases."""

    def __init__(self, agents: Mapping[int, Any], n_rounds: int, phases: tuple[Phase, ...]):
        if not agents:
            raise ValueError("PhasedGame requires at least one agent.")
        self.agents = dict(agents)
        self.n_rounds = n_rounds
        self.phases = phases
        self.history: list[dict[str, Any]] = []

    def prompt_for(self, phase: Phase, player: int, state: RoundState) -> str:
        raise NotImplementedError()

    def observation_for(self, player: int, record: Mapping[str, Any]) -> str | Mapping[str, Any]:
        return record

    def resolve_round(self, state: RoundState) -> dict[str, Any]:
        raise NotImplementedError()

    def system_context_for(self, phase: Phase, player: int, state: RoundState) -> str | None:
        return None

    def request_metadata(self, phase: Phase, player: int, state: RoundState) -> Mapping[str, Any]:
        return {
            "environment": self,
            "history": self.history,
            "phase": phase,
            "state": state,
        }

    def _respond(self, player: int, phase: Phase, state: RoundState) -> PhaseResponse:
        agent = self.agents[player]
        request = AgentRequest(
            prompt=self.prompt_for(phase, player, state),
            phase=phase.name,
            round_number=state.round_number,
            player=player,
            system_context=self.system_context_for(phase, player, state),
            metadata=self.request_metadata(phase, player, state),
        )
        for attempt in range(phase.invalid_retries + 1):
            if phase.kind == "communication" and hasattr(agent, "communicate"):
                raw = agent.communicate(request)
            elif hasattr(agent, "respond"):
                raw = agent.respond(request)
            else:
                raw = agent.interact(request.prompt)
            if phase.action_space is None:
                return PhaseResponse(raw=raw, value=raw)
            parsed = phase.action_space.resolve(raw)
            if not parsed.invalid or attempt == phase.invalid_retries:
                return PhaseResponse(raw=parsed.raw, value=parsed.value, invalid=parsed.invalid)
        raise AssertionError("unreachable")

    def play_round(self, round_number: int) -> dict[str, Any]:
        state = RoundState(round_number)
        for phase in self.phases:
            staged: dict[int, PhaseResponse] = {}
            for player in phase.actors:
                response = self._respond(player, phase, state)
                if phase.simultaneous:
                    staged[player] = response
                else:
                    state.set_response(phase.name, player, response)
            for player, response in staged.items():
                state.set_response(phase.name, player, response)
        record = self.resolve_round(state)
        self.history.append(record)
        for player, agent in self.agents.items():
            observation = self.observation_for(player, record)
            if hasattr(agent, "observe"):
                agent.observe(observation)
            elif hasattr(agent, "update_history"):
                if isinstance(observation, str):
                    agent.update_history(observation)
                else:
                    agent.update_history(json.dumps(observation, sort_keys=True))
        return record

    def run_game(self) -> list[dict[str, Any]]:
        for round_number in range(1, self.n_rounds + 1):
            self.play_round(round_number)
        return self.history


class Game:
    """Backward-compatible two-player simultaneous game interface."""

    def __init__(self, agent1, agent2, n_rounds):
        self.agent1 = agent1
        self.agent2 = agent2
        self.n_rounds = n_rounds
        self.history = []
        self.total_payoffs = [0, 0]

    def get_payoff(self, move1, move2):
        raise NotImplementedError()

    def get_system_prompt(self):
        raise NotImplementedError()

    def get_valid_moves(self):
        raise NotImplementedError()

    def play_round(self, round_num):
        prompt = (
            "What is your move for this round? Respond with one of: "
            + ", ".join(self.get_valid_moves())
            + "."
        )
        move1 = self.agent1.interact(prompt)
        move2 = self.agent2.interact(prompt)
        payoff1, payoff2 = self.get_payoff(move1, move2)
        self.total_payoffs[0] += payoff1
        self.total_payoffs[1] += payoff2
        summary = (
            f"Round {round_num}: Agent 1 chose {move1}, Agent 2 chose {move2}. "
            f"Payoffs: Agent 1 = {payoff1}, Agent 2 = {payoff2}."
        )
        self.history.append(summary)
        self.agent1.update_history("You are agent 1. " + summary)
        self.agent2.update_history("You are agent 2. " + summary)
        return move1, move2, payoff1, payoff2, summary

    def run_game(self):
        for round_num in range(1, self.n_rounds + 1):
            print(f"\n--- Round {round_num} ---")
            *_, summary = self.play_round(round_num)
            print(summary)
        print("\nFinal Total Payoffs:")
        print("Agent 1:", self.total_payoffs[0])
        print("Agent 2:", self.total_payoffs[1])

    def introspect(self, agent_num, question):
        if agent_num == 1:
            return self.agent1.interact(question)
        if agent_num == 2:
            return self.agent2.interact(question)
        return "Invalid agent number."

    def get_history(self):
        return self.history

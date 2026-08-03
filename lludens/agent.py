from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping

import llm


@dataclass(frozen=True)
class PromptParameter:
    """A private parameter rendered into an agent's system prompt."""

    name: str
    value: Any
    instruction: str = "{name} = {value}"

    def render(self) -> str:
        return self.instruction.format(name=self.name, value=self.value)


@dataclass(frozen=True)
class AgentRequest:
    """A phase-specific request sent from an environment to an agent."""

    prompt: str
    phase: str
    round_number: int
    player: int
    observation: str | None = None
    system_context: str | None = None
    metadata: Mapping[str, Any] | None = None


class TotreLLM:
    """General LLM agent with private system parameters and explicit observations."""

    def __init__(
        self,
        model_id: str,
        system: str | None = None,
        options: dict | None = None,
        *,
        label: str | None = None,
        private_parameters: Mapping[str, PromptParameter | Any] | None = None,
    ):
        self.model_id = model_id
        self.label = label or model_id
        self.model = llm.get_model(model_id)
        self.conversation = self.model.conversation()
        self.options = {"temperature": 0.1} if options is None else dict(options)
        self.system = system or ""
        self.private_parameters: dict[str, PromptParameter] = {}
        self.pending_observations: list[str] = []
        self.transcript: list[dict[str, Any]] = []
        for name, parameter in (private_parameters or {}).items():
            if isinstance(parameter, PromptParameter):
                self.private_parameters[name] = parameter
            else:
                self.private_parameters[name] = PromptParameter(name, parameter)

    def set_private_parameter(
        self,
        name: str,
        value: Any,
        instruction: str = "{name} = {value}",
    ) -> None:
        self.private_parameters[name] = PromptParameter(name, value, instruction)

    def render_system(self, extra: str | None = None) -> str:
        sections = [self.system.strip()]
        if self.private_parameters:
            rendered = "\n".join(
                f"- {parameter.render()}" for parameter in self.private_parameters.values()
            )
            sections.append(
                "Private parameters for this agent only. Do not reveal them unless the experiment asks you to:\n"
                + rendered
            )
        if extra:
            sections.append(extra.strip())
        return "\n\n".join(section for section in sections if section)

    def observe(self, observation: str | Mapping[str, Any]) -> None:
        if isinstance(observation, str):
            text = observation
        else:
            text = json.dumps(observation, sort_keys=True)
        self.pending_observations.append(text)
        self.transcript.append({"type": "observation", "content": text})

    def interact(
        self,
        prompt: str,
        attachments: list | None = None,
        *,
        context: str | None = None,
        system_context: str | None = None,
    ) -> str:
        observations = [*self.pending_observations]
        self.pending_observations.clear()
        if context:
            observations.append(context)
        effective_prompt = prompt
        if observations:
            effective_prompt = "New observations:\n" + "\n".join(observations) + "\n\n" + prompt
        response = self.conversation.prompt(
            effective_prompt,
            attachments=attachments,
            system=self.render_system(system_context),
            **self.options,
        )
        text = response.text().strip()
        self.transcript.append(
            {
                "type": "interaction",
                "prompt": effective_prompt,
                "response": text,
            }
        )
        return text

    def respond(self, request: AgentRequest) -> str:
        return self.interact(
            request.prompt,
            context=request.observation,
            system_context=request.system_context,
        )

    def communicate(self, request: AgentRequest) -> str:
        return self.respond(request)

    def get_history(self):
        return self.conversation.responses

    def get_transcript(self) -> list[dict[str, Any]]:
        return list(self.transcript)

    def update_history(self, summary: str) -> None:
        self.observe(summary)

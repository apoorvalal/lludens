from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Generic, TypeVar


Action = TypeVar("Action", bound=str)


@dataclass(frozen=True)
class ParsedAction(Generic[Action]):
    raw: str
    value: Action
    invalid: bool = False


@dataclass(frozen=True)
class ActionSpace(Generic[Action]):
    """A reusable discrete action space with LLM-response normalization."""

    choices: tuple[Action, ...]
    aliases: dict[str, Action] = field(default_factory=dict)
    fallback: Action | None = None

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("ActionSpace requires at least one choice.")
        if self.fallback is not None and self.fallback not in self.choices:
            raise ValueError("ActionSpace fallback must be one of its choices.")

    def parse(self, text: str) -> Action:
        normalized = str(text).strip().lower()
        lookup = {choice.lower(): choice for choice in self.choices}
        lookup.update({alias.lower(): value for alias, value in self.aliases.items()})
        if normalized in lookup:
            return lookup[normalized]

        final_match = re.search(
            r"(?:final\s+)?(?:answer|action|choice)\s*(?:is|:)?\s*([A-Za-z_-]+)",
            normalized,
        )
        if final_match and final_match.group(1) in lookup:
            return lookup[final_match.group(1)]

        tokens = re.findall(r"[A-Za-z_-]+", normalized)
        mentioned = [lookup[token] for token in tokens if token in lookup]
        if mentioned and len(set(mentioned)) == 1:
            return mentioned[0]
        if mentioned:
            return mentioned[0]
        raise ValueError(f"Could not parse action from response: {text!r}")

    def resolve(self, text: str) -> ParsedAction[Action]:
        try:
            return ParsedAction(raw=text, value=self.parse(text))
        except ValueError:
            if self.fallback is None:
                raise
            return ParsedAction(raw=text, value=self.fallback, invalid=True)

    def instruction(self) -> str:
        return ", ".join(self.choices)


BINARY_ACTION_SPACE = ActionSpace(
    choices=("Cooperate", "Defect"),
    aliases={
        "c": "Cooperate",
        "cooperates": "Cooperate",
        "cooperated": "Cooperate",
        "cooperating": "Cooperate",
        "collaborate": "Cooperate",
        "collaborates": "Cooperate",
        "d": "Defect",
        "defects": "Defect",
        "defected": "Defect",
        "defecting": "Defect",
    },
    fallback="Defect",
)


def parse_binary_action(text: str) -> str:
    """Normalize an LLM response to `Cooperate` or `Defect`."""
    return BINARY_ACTION_SPACE.parse(text)

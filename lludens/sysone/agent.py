"""Game agents backed by typed constrained choices."""
from __future__ import annotations

import random
import math
from typing import Callable, Mapping

from ..agent import AgentRequest
from .choices import MODELS, evaluate_choice, normalize_choice
from .games import INSTRUCTIONS


class DecisionAgent:
    """Stateless API calls, explicit history, independently reproducible order/sampling.

    ``sample`` is an experimenter-imposed stochastic policy, not a claim that
    classification probabilities are strategic mixed-strategy probabilities.
    """

    def __init__(self, provider: str, descriptions: Mapping[str, str | None], *, seed: int,
                 policy: str = "argmax", call: Callable | None = None,
                 instructions: str = INSTRUCTIONS):
        if provider not in MODELS or policy not in {"argmax", "sample"}:
            raise ValueError("Unsupported provider or action policy")
        if len(descriptions) < 2 or not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("At least two options and nonempty instructions are required")
        self.provider, self.descriptions = provider, dict(descriptions)
        self.seed, self.policy, self.instructions = seed, policy, instructions
        self.call = call or evaluate_choice
        self.last_decision = None

    def respond(self, request: AgentRequest) -> str:
        self.last_decision = None
        labels = list(self.descriptions)
        random.Random(self.seed + 7919 * request.round_number).shuffle(labels)
        options = {k: self.descriptions[k] for k in labels}
        result = self.call(self.provider, request.prompt, options, self.instructions)
        # Revalidate even replayed records or injected transports.
        action, probs = normalize_choice(result["answer"], options)
        if result["choice"] != action:
            raise ValueError("Result choice disagrees with the validated raw answer")
        # Earlier journal writers normalized in a different floating-point order.
        # Keep their last-bit representation, but reject substantive disagreement.
        recorded_probs = result.get("probabilities", probs)
        if set(recorded_probs) != set(probs) or any(
            isinstance(recorded_probs[k], bool) or not isinstance(recorded_probs[k], (int, float))
            or not math.isclose(recorded_probs[k], probs[k], rel_tol=0, abs_tol=1e-12)
            for k in probs
        ):
            raise ValueError("Result probabilities disagree with the validated raw answer")
        draw = None
        if self.policy == "sample":
            draw = random.Random(self.seed + 104729 * request.round_number).random()
            cumulative = 0.0
            # Canonical order separates the sampling seed from the options order.
            for label in self.descriptions:
                cumulative += probs[label]
                action = label
                if draw < cumulative:
                    break
        self.last_decision = {**result, "probabilities": recorded_probs, "action": action,
                              "policy": self.policy, "sample_uniform": draw,
                              "options": options, "prompt": request.prompt,
                              "instructions": self.instructions}
        return action

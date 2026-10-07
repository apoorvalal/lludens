"""Typed decision agent using Simon Willison's two LLM plugins.

The plugin owns API transport/key handling. This adapter validates the full
distribution and never translates an API error into a game action.
"""
from __future__ import annotations

import json
import math
import random
import re
import threading
import time
from typing import Callable

import llm

from .agent import AgentRequest
from .decision_games import INSTRUCTIONS


MODELS = {"decisions": "openai-decisions/gpt-6-luna", "jev": "typesafe/jev-latest"}
_MODEL_LOCK = threading.Lock()


def get_decision_model(provider):
    # LLM 0.36 lazily registers plugins; its first registration is not thread safe.
    with _MODEL_LOCK:
        return llm.get_model(MODELS[provider])


def normalize_choice(answer: dict, options: dict) -> tuple[str, dict[str, float]]:
    if answer.get("type") != "choice":
        raise ValueError("Expected a choice, not a refusal or another answer type")
    raw = answer.get("probabilities")
    if isinstance(raw, list):
        pairs = [(r["value"], r["probability"]) for r in raw]
        if len({x[0] for x in pairs}) != len(pairs):
            raise ValueError("Duplicate probabilities")
        probs = dict(pairs)
    elif isinstance(raw, dict):
        probs = dict(raw)
    else:
        raise ValueError("Missing probability distribution")
    if set(probs) != set(options):
        raise ValueError("Probability labels do not match legal actions")
    if any(isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
           or not 0 <= p <= 1 for p in probs.values()):
        raise ValueError("Invalid probability")
    mass = sum(probs.values())
    if not math.isclose(mass, 1, abs_tol=1e-4):
        # Jev 1.13 returns probabilities rounded to two decimals. Six options
        # can therefore sum to 0.99/1.01. Accept only errors consistent with
        # that quantization; preserve raw values in `answer` and normalize
        # the distribution used by the explicitly sampled policy.
        rounded = all(math.isclose(p * 100, round(p * 100), abs_tol=1e-8) for p in probs.values())
        if not rounded or abs(mass - 1) > 0.005 * len(probs) + 1e-9:
            raise ValueError("Probabilities do not sum to one within rounding precision")
    probs = {label: p / mass for label, p in probs.items()}
    choice = answer.get("choice")
    if choice not in probs or probs[choice] < max(probs.values()) - 1e-6:
        raise ValueError("Choice is not a legal highest-probability action")
    return choice, probs


def evaluate_choice(provider: str, prompt: str, options: dict, instructions: str = INSTRUCTIONS) -> dict:
    """Make one decision with bounded retries for explicit rate-limit responses only."""
    started = time.perf_counter()
    options_key = "choices" if provider == "decisions" else "criteria"
    for attempt in range(3):
        try:
            response = get_decision_model(provider).prompt(
                prompt, system=instructions, answer_type="choice", **{options_key: options})
            answer = json.loads(response.text())
            choice, probs = normalize_choice(answer, options)
            raw = response.json()
            return {"provider": provider, "model_id": MODELS[provider], "model": raw["model"],
                    "choice": choice, "probabilities": probs, "answer": answer,
                    "usage": raw["usage"], "latency_seconds": time.perf_counter() - started,
                    "attempts": attempt + 1}
        except llm.ModelError as exc:
            # Do not log raw transport exceptions/headers. Do not replay ambiguous timeouts.
            match = re.search(r"HTTP (\d{3})", str(exc))
            status = int(match.group(1)) if match else None
            if status in {429, 529} and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"{provider}: API/transport failure (HTTP {status}); no action substituted") from None


class DecisionAgent:
    """Stateless API calls, explicit history, independently reproducible order/sampling.

    ``sample`` is an experimenter-imposed stochastic policy, not a claim that
    classification probabilities are strategic mixed-strategy probabilities.
    """

    def __init__(self, provider: str, descriptions: dict, *, seed: int, policy: str = "argmax",
                 call: Callable | None = None):
        if provider not in MODELS or policy not in {"argmax", "sample"}:
            raise ValueError("Unsupported provider or action policy")
        self.provider, self.descriptions, self.seed, self.policy = provider, descriptions, seed, policy
        self.call = call or evaluate_choice
        self.last_decision = None

    def respond(self, request: AgentRequest) -> str:
        labels = list(self.descriptions)
        random.Random(self.seed + 7919 * request.round_number).shuffle(labels)
        options = {k: self.descriptions[k] for k in labels}
        result = self.call(self.provider, request.prompt, options, INSTRUCTIONS)
        # Revalidate even replayed records or injected transports.
        _, probs = normalize_choice(result["answer"], options)
        action = result["choice"]
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
        self.last_decision = {**result, "action": action, "policy": self.policy, "sample_uniform": draw,
                              "options": options, "prompt": request.prompt, "instructions": INSTRUCTIONS}
        return action

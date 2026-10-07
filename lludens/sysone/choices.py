"""Typed decision agent using Simon Willison's two LLM plugins.

The plugin owns API transport/key handling. This adapter validates the full
distribution and never translates an API error into a game action.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from typing import Mapping

import llm

from .games import INSTRUCTIONS


MODELS = {"decisions": "openai-decisions/gpt-6-luna", "jev": "typesafe/jev-latest"}
_MODEL_LOCK = threading.Lock()


def get_decision_model(provider: str):
    """Resolve an installed LLM plugin, without sending a request."""
    if provider not in MODELS:
        raise ValueError(f"Unsupported decision provider: {provider}")
    # LLM 0.36 lazily registers plugins; its first registration is not thread safe.
    with _MODEL_LOCK:
        return llm.get_model(MODELS[provider])


def choice_kwargs(provider: str, prompt: str, options: Mapping[str, str | None],
                  instructions: str = INSTRUCTIONS) -> dict:
    """Return the exact keyword arguments passed to ``llm.Model.prompt``.

    Option insertion order is preserved. Both plugins map ``system`` to the
    choice question's instructions, not to a conversational system message.
    """
    if provider not in MODELS:
        raise ValueError(f"Unsupported decision provider: {provider}")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("A nonempty state prompt is required")
    if not isinstance(instructions, str) or not instructions.strip():
        raise ValueError("Nonempty choice instructions are required")
    if not isinstance(options, Mapping) or len(options) < 2 or any(
        not isinstance(k, str) or not k.strip()
        or (v is not None and (not isinstance(v, str) or not v.strip()))
        for k, v in options.items()
    ):
        raise ValueError("At least two named options with text/null descriptions are required")
    key = "choices" if provider == "decisions" else "criteria"
    return {"prompt": prompt, "system": instructions, "answer_type": "choice",
            key: dict(options)}


def preview_choice(provider: str, prompt: str, options: Mapping[str, str | None],
                   instructions: str = INSTRUCTIONS) -> dict:
    """Build the plugin's request body offline, without credentials or inference.

    This uses the pinned plugins' ``build_payload`` method. It reconstructs the
    body from the LLM arguments; it is not a captured HTTP transaction.
    """
    kwargs = choice_kwargs(provider, prompt, options, instructions)
    model = get_decision_model(provider)
    response = model.prompt(**kwargs)  # Lazy: do not call text()/json().
    return {"provider": provider, "model_id": MODELS[provider], "llm_kwargs": kwargs,
            "api_payload": model.build_payload(response.prompt)}


def normalize_choice(answer: dict, options: dict) -> tuple[str, dict[str, float]]:
    """Validate a typed answer, retaining only bounded rounding adjustments.

    The strict game policy requires the provider's choice to be a probability
    maximum (ties allowed). Raw answers are never changed in place.
    """
    if not options or not isinstance(answer, dict) or answer.get("type") != "choice":
        raise ValueError("Expected a choice, not a refusal or another answer type")
    raw = answer.get("probabilities")
    if isinstance(raw, list):
        if any(not isinstance(r, dict) or "value" not in r or "probability" not in r
               or not isinstance(r["value"], str) for r in raw):
            raise ValueError("Malformed probability entries")
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
    if mass <= 0:
        raise ValueError("Probability mass must be positive")
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
    if not isinstance(choice, str) or choice not in probs or probs[choice] < max(probs.values()) - 1e-6:
        raise ValueError("Choice is not a legal highest-probability action")
    return choice, probs


def evaluate_choice(provider: str, prompt: str, options: dict, instructions: str = INSTRUCTIONS) -> dict:
    """Make one decision with bounded retries for explicit rate-limit responses only."""
    started = time.perf_counter()
    kwargs = choice_kwargs(provider, prompt, options, instructions)
    for attempt in range(3):
        try:
            response = get_decision_model(provider).prompt(**kwargs)
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

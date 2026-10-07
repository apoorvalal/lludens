"""System One constrained choices and repeated economic games.

Transport is supplied by the optional ``lludens[sysone]`` LLM plugins.
Importing this module or replaying a journal never calls a provider.
"""
from .agent import DecisionAgent
from .choices import (
    MODELS,
    choice_kwargs,
    evaluate_choice,
    get_decision_model,
    normalize_choice,
    preview_choice,
)
from .games import EconomicGame, GAMES, INSTRUCTIONS, RepeatedDecisionGame
from .journal import DecisionJournal, read_jsonl
from .runner import run_match

__all__ = [
    "DecisionAgent", "DecisionJournal", "EconomicGame", "GAMES", "INSTRUCTIONS",
    "MODELS", "RepeatedDecisionGame", "choice_kwargs", "evaluate_choice",
    "get_decision_model", "normalize_choice", "preview_choice", "read_jsonl",
    "run_match",
]

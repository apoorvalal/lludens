"""Compatibility imports; new code should use :mod:`lludens.sysone`."""
from .sysone.games import EconomicGame, GAMES, INSTRUCTIONS, RepeatedDecisionGame

__all__ = ["EconomicGame", "GAMES", "INSTRUCTIONS", "RepeatedDecisionGame"]

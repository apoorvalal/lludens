"""Checkpointed two-player matches; independent of any experiment directory."""
from pathlib import Path
import json

from .agent import DecisionAgent
from .games import GAMES, RepeatedDecisionGame
from .journal import DecisionJournal, read_jsonl


def run_match(spec: dict, journal: DecisionJournal, rounds_path: str | Path) -> dict:
    """Resume a match specification against a journal and a round checkpoint.

    Required fields: ``id``, ``game``, ``players`` (two provider names), ``seed``,
    ``policy``, and ``rounds``. The journal owns successful calls; the checkpoint
    only contains completed simultaneous rounds. See ``docs/sysone.md``.
    """
    rounds_path = Path(rounds_path)
    if len(spec["players"]) != 2:
        raise ValueError("Exactly two providers are required")
    agents = {
        player: DecisionAgent(spec["players"][player-1], dict(GAMES[spec["game"]].descriptions),
                              seed=spec["seed"] + player, policy=spec["policy"],
                              call=journal.caller(spec["id"], player)) for player in (1, 2)
    }
    game = RepeatedDecisionGame(spec["game"], agents, spec["rounds"])
    existing = read_jsonl(rounds_path)
    if len(existing) > spec["rounds"] or [r["round"] for r in existing] != list(range(1, len(existing)+1)):
        raise ValueError("Noncontiguous round checkpoint")
    for row in existing:
        a, b = row["actions"]["1"], row["actions"]["2"]
        if list(GAMES[spec["game"]].payoff(a,b)) != [row["payoffs"]["1"], row["payoffs"]["2"]]:
            raise ValueError("Payoff mismatch in checkpoint")
    game.history = existing
    if len(existing) < spec["rounds"]:
        rounds_path.parent.mkdir(parents=True, exist_ok=True)
    for r in range(len(existing)+1, spec["rounds"]+1):
        record = game.play_round(r)
        # The engine publishes history only after both actions were selected.
        record["decisions"] = {
            str(p): {k: v for k, v in agents[p].last_decision.items()
                     if k not in {"prompt", "instructions", "options", "answer"}}
            for p in (1, 2)
        }
        with rounds_path.open("a") as stream:
            stream.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
    return {"id": spec["id"], "rounds": len(game.history),
            "scores": {str(p): sum(r["payoffs"][str(p)] for r in game.history) for p in (1, 2)}}

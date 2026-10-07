"""Direct-action payoff accounting, independent of the live model adapters.

Exact rational payoffs avoid confusing floating-point summation with attainment.
The Pareto frontier is computed over feasible cumulative payoff vectors, not by
equating efficiency with maximum joint payoff or with an individual score cap.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import isclose

TITLES = {"prisoners_dilemma": "Prisoner’s Dilemma", "stag_hunt": "Stag Hunt",
          "public_goods": "Two-player public goods"}
NAMES = {"decisions": "GPT-6 Luna", "jev": "Jev 1.13.0"}


def stage_payoffs(game):
    """Independent statement of the three games' economic primitives."""
    if game == "prisoners_dilemma":
        values = {("C", "C"): (3, 3), ("C", "D"): (0, 5),
                  ("D", "C"): (5, 0), ("D", "D"): (1, 1)}
    elif game == "stag_hunt":
        values = {("Stag", "Stag"): (4, 4), ("Stag", "Hare"): (0, 3),
                  ("Hare", "Stag"): (3, 0), ("Hare", "Hare"): (3, 3)}
    elif game == "public_goods":
        values = {(str(a), str(b)): (10 - a + F(4, 5) * (a + b),
                                     10 - b + F(4, 5) * (a + b))
                  for a, b in product(range(0, 11, 2), repeat=2)}
    else:
        raise ValueError(f"Unknown game: {game}")
    return {actions: tuple(map(F, payoffs)) for actions, payoffs in values.items()}


def nondominated(points):
    """Two-dimensional Pareto maxima, with weak gains and one strict gain."""
    frontier = []
    highest_second = None
    for point in sorted(set(points), reverse=True):
        if highest_second is None or point[1] > highest_second:
            frontier.append(point)
            highest_second = point[1]
    return frozenset(frontier)


@lru_cache(maxsize=None)
def pareto_frontier(game, rounds):
    if type(rounds) is not int or rounds < 1:
        raise ValueError("A positive integer horizon is required")
    # Dominated partial totals can be discarded: the same continuations remain
    # feasible after every history, and points add without discounting.
    stage = nondominated(stage_payoffs(game).values())
    frontier = frozenset({(F(0), F(0))})
    for _ in range(rounds):
        frontier = nondominated((u + a, v + b) for u, v in frontier for a, b in stage)
    return frontier


def benchmark(game, rounds):
    outcomes = stage_payoffs(game)
    best = max(outcomes.values(), key=sum)
    maximizers = [actions for actions, payoff in outcomes.items() if sum(payoff) == sum(best)]
    assert len(maximizers) == 1 and best[0] == best[1]
    return {"rounds": rounds, "actions": list(maximizers[0]),
            "individual": [float(x * rounds) for x in best],
            "joint": float(sum(best) * rounds), "stage_joint": float(sum(best))}


def evaluate_match(match):
    if match["policy"] != "argmax":
        raise ValueError("Direct-action accounting does not include randomized policies")
    game, horizon = match["game"], match["rounds"]
    history = match["history"]
    if len(history) != horizon or [h["round"] for h in history] != list(range(1, horizon + 1)):
        raise ValueError("A complete, consecutively numbered match is required")
    outcomes, ref = stage_payoffs(game), benchmark(game, horizon)
    totals = [F(0), F(0)]
    cooperative_rounds = 0
    profiles = Counter()
    for record in history:
        actions = tuple(record["actions"][str(p)] for p in (1, 2))
        payoffs = outcomes[actions]
        for i, payoff in enumerate(payoffs):
            if not isclose(float(payoff), record["payoffs"][str(i + 1)], abs_tol=1e-9, rel_tol=0):
                raise ValueError("Saved payoff disagrees with the economic primitives")
            totals[i] += payoff
        cooperative_rounds += list(actions) == ref["actions"]
        # Head-to-head action profiles are always Luna first, regardless of seat.
        ordered = actions[::-1] if match["kind"] == "h2h" and match["players"][0] == "jev" else actions
        profiles[" / ".join(ordered)] += 1
    joint = sum(totals)
    efficient = tuple(totals) in pareto_frontier(game, horizon)
    joint_maximum = joint == F(str(ref["joint"]))
    assert not joint_maximum or efficient
    row = {key: match[key] for key in ("id", "game", "kind", "pair", "swap", "players", "rounds")}
    row.update({"seat1_payoff": float(totals[0]), "seat2_payoff": float(totals[1]),
                "joint_payoff": float(joint), "joint_maximum": ref["joint"],
                "joint_share": float(joint / F(str(ref["joint"]))),
                "joint_shortfall": float(F(str(ref["joint"])) - joint),
                "joint_maximum_attained": joint_maximum, "pareto_efficient": efficient,
                "joint_optimal_rounds": cooperative_rounds, "action_profiles": dict(sorted(profiles.items()))})
    if match["kind"] == "h2h":
        assert sorted(match["players"]) == ["decisions", "jev"]
        row["luna_payoff"] = float(totals[match["players"].index("decisions")])
        row["jev_payoff"] = float(totals[match["players"].index("jev")])
    return row


def aggregate(rows):
    def mean(key):
        return float(sum(F(str(row[key])) for row in rows) / len(rows))

    result = {"matches": len(rows), "rounds_observed": sum(r["rounds"] for r in rows),
              "mean_seat1_payoff": mean("seat1_payoff"), "mean_seat2_payoff": mean("seat2_payoff"),
              "mean_joint_payoff": mean("joint_payoff"), "joint_maximum": rows[0]["joint_maximum"],
              "mean_joint_shortfall": mean("joint_shortfall"),
              "joint_maximum_matches": sum(r["joint_maximum_attained"] for r in rows),
              "pareto_efficient_matches": sum(r["pareto_efficient"] for r in rows),
              "joint_optimal_rounds": sum(r["joint_optimal_rounds"] for r in rows)}
    result["mean_joint_share"] = result["mean_joint_payoff"] / result["joint_maximum"]
    profiles = Counter()
    for row in rows:
        profiles.update(row["action_profiles"])
    result["action_profiles"] = dict(sorted(profiles.items()))
    if rows[0]["kind"] == "h2h":
        result.update({"seat_pairs": len({r["pair"] for r in rows}),
                       "mean_luna_payoff": mean("luna_payoff"), "mean_jev_payoff": mean("jev_payoff"),
                       "luna_wins": sum(r["luna_payoff"] > r["jev_payoff"] for r in rows),
                       "jev_wins": sum(r["luna_payoff"] < r["jev_payoff"] for r in rows),
                       "ties": sum(r["luna_payoff"] == r["jev_payoff"] for r in rows)})
    return result


def summarize_direct(replay):
    rows = [evaluate_match(m) for m in replay if m["policy"] == "argmax"]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate match IDs")
    horizons = {r["rounds"] for r in rows}
    if len(horizons) != 1:
        raise ValueError("Report summaries require a common horizon")
    horizon = horizons.pop()
    head = []; self_play = []
    for game in TITLES:
        group = [r for r in rows if r["game"] == game and r["kind"] == "h2h"]
        head.append({"game": game, **aggregate(group)})
        for provider in NAMES:
            group = [r for r in rows if r["game"] == game and r["kind"] == "self"
                     and r["players"] == [provider, provider]]
            self_play.append({"game": game, "provider": provider, **aggregate(group)})
    return {"action_rule": "Use each fresh API response's selected action; no added randomization.",
            "horizon": horizon, "matches": len(rows), "rounds": sum(r["rounds"] for r in rows),
            "decisions": 2 * sum(r["rounds"] for r in rows),
            "benchmarks": {game: benchmark(game, horizon) for game in TITLES},
            "h2h": head, "self_play": self_play, "match_results": rows}

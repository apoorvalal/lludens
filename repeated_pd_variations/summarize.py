from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize repeated PD round-robin output.")
    parser.add_argument("path", nargs="?", type=Path, default=Path("data/repeated_pd_variations.jsonl"))
    args = parser.parse_args()

    data = pd.read_json(args.path, lines=True)
    summary = (
        data.groupby(
            ["player1_model", "player2_model", "treatment", "horizon"],
            as_index=False,
        )
        .agg(
            matches=("match_id", "nunique"),
            rounds=("round", "count"),
            player1_cooperation=("player1_cooperated", "mean"),
            player2_cooperation=("player2_cooperated", "mean"),
            mutual_cooperation=("mutual_cooperation", "mean"),
            player1_payoff=("payoff1", "mean"),
            player2_payoff=("payoff2", "mean"),
        )
        .sort_values(["treatment", "horizon", "player1_model", "player2_model"])
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Iterable

from lludens.km2026 import LLMPolicy, RepeatedPDGame, Treatment


DEFAULT_HORIZONS = (10, 20, 50)
DEFAULT_TREATMENTS: tuple[Treatment, ...] = ("sim", "seq", "chat")


@dataclass(frozen=True)
class ModelSpec:
    label: str
    model_id: str


@dataclass(frozen=True)
class MatchSpec:
    match_id: str
    player1: ModelSpec
    player2: ModelSpec
    treatment: Treatment
    horizon: int
    replicate: int
    seed: int
    player1_gamma: float
    player2_gamma: float


def load_models(path: str | Path) -> list[ModelSpec]:
    with Path(path).open() as file:
        rows = json.load(file)
    models = [ModelSpec(**row) for row in rows]
    if not models:
        raise ValueError("The model configuration is empty.")
    labels = [model.label for model in models]
    if len(labels) != len(set(labels)):
        raise ValueError("Model labels must be unique.")
    return models


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(payload.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def build_plan(
    models: Iterable[ModelSpec],
    *,
    treatments: Iterable[Treatment] = DEFAULT_TREATMENTS,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    replicates: int = 1,
    gamma_values: Iterable[float] = (0.0,),
    base_seed: int = 20260803,
) -> list[MatchSpec]:
    models = list(models)
    treatments = tuple(treatments)
    horizons = tuple(horizons)
    gamma_values = tuple(gamma_values)
    if replicates < 1:
        raise ValueError("replicates must be at least one.")
    if not gamma_values:
        raise ValueError("gamma_values must contain at least one value.")

    plan: list[MatchSpec] = []
    for player1 in models:
        for player2 in models:
            for treatment in treatments:
                for horizon in horizons:
                    for replicate in range(1, replicates + 1):
                        seed = stable_seed(
                            base_seed,
                            player1.label,
                            player2.label,
                            treatment,
                            horizon,
                            replicate,
                        )
                        gamma_rng = random.Random(seed)
                        player1_gamma = gamma_rng.choice(gamma_values)
                        player2_gamma = gamma_rng.choice(gamma_values)
                        match_id = "-".join(
                            [
                                slug(player1.label),
                                "vs",
                                slug(player2.label),
                                treatment,
                                f"h{horizon}",
                                f"r{replicate}",
                                f"s{seed}",
                            ]
                        )
                        plan.append(
                            MatchSpec(
                                match_id=match_id,
                                player1=player1,
                                player2=player2,
                                treatment=treatment,
                                horizon=horizon,
                                replicate=replicate,
                                seed=seed,
                                player1_gamma=player1_gamma,
                                player2_gamma=player2_gamma,
                            )
                        )
    return plan


def load_progress(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    progress: dict[str, list[dict]] = {}
    with path.open() as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                progress.setdefault(row["match_id"], []).append(row)
    for rows in progress.values():
        rows.sort(key=lambda row: row["round"])
    return progress


def append_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


def write_plan(path: Path, plan: Iterable[MatchSpec]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as file:
        json.dump(
            [
                {
                    "match_id": spec.match_id,
                    "player1_label": spec.player1.label,
                    "player1_model": spec.player1.model_id,
                    "player2_label": spec.player2.label,
                    "player2_model": spec.player2.model_id,
                    "treatment": spec.treatment,
                    "horizon": spec.horizon,
                    "replicate": spec.replicate,
                    "seed": spec.seed,
                    "player1_gamma": spec.player1_gamma,
                    "player2_gamma": spec.player2_gamma,
                }
                for spec in plan
            ],
            file,
            indent=2,
        )
        file.write("\n")


def run_plan(
    plan: Iterable[MatchSpec],
    output_path: Path,
    temperature: float | None,
    max_tokens: int,
) -> None:
    progress = load_progress(output_path)
    for index, spec in enumerate(plan, start=1):
        prior_rows = progress.get(spec.match_id, [])
        if len(prior_rows) >= spec.horizon:
            print(f"[{index}] skip completed {spec.match_id}", flush=True)
            continue
        start_round = len(prior_rows) + 1
        print(
            f"[{index}] run {spec.match_id} from round {start_round}",
            flush=True,
        )
        player1 = LLMPolicy(
            spec.player1.model_id,
            label=spec.player1.label,
            private_gamma=spec.player1_gamma,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        player2 = LLMPolicy(
            spec.player2.model_id,
            label=spec.player2.label,
            private_gamma=spec.player2_gamma,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        game = RepeatedPDGame(
            spec.treatment,
            spec.horizon,
            player1,
            player2,
            seed=spec.seed,
            match_id=spec.match_id,
        )
        game.history.extend(prior_rows)
        for prior_row in prior_rows:
            player1.observe(game.observation_for(1, prior_row))
            player2.observe(game.observation_for(2, prior_row))
        for round_number in range(start_round, spec.horizon + 1):
            row = game.play_round(round_number)
            row["replicate"] = spec.replicate
            append_rows(output_path, [row])


def parse_args() -> argparse.Namespace:
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Run ordered round-robin repeated PD experiments.")
    parser.add_argument("--models", default=folder / "models.json", type=Path)
    parser.add_argument("--treatments", nargs="+", choices=DEFAULT_TREATMENTS, default=DEFAULT_TREATMENTS)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--gamma-values", nargs="+", type=float, default=[0.0])
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/repeated_pd_variations.jsonl"))
    parser.add_argument("--plan-output", type=Path, default=Path("data/repeated_pd_variations_plan.json"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = load_models(args.models)
    plan = build_plan(
        models,
        treatments=args.treatments,
        horizons=args.horizons,
        replicates=args.replicates,
        gamma_values=args.gamma_values,
        base_seed=args.seed,
    )
    if args.shard_count < 1:
        raise ValueError("shard-count must be at least one.")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be between zero and shard-count minus one.")
    plan = [
        spec
        for index, spec in enumerate(plan)
        if index % args.shard_count == args.shard_index
    ]
    write_plan(args.plan_output, plan)
    total_rounds = sum(spec.horizon for spec in plan)
    print(
        f"Planned {len(plan)} matches and {total_rounds} game rounds "
        f"across {len(models)} models.",
        flush=True,
    )
    if not args.dry_run:
        run_plan(plan, args.output, args.temperature, args.max_tokens)


if __name__ == "__main__":
    main()

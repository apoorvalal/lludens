"""Resumable Decisions-vs-Jev pilot. Run from the repository root.

python experiments/decisions_jev_games_2026_10_06/run.py --workers 8
All live calls must run in the host environment with its existing API connections.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from lludens.sysone import DecisionJournal as Journal, GAMES, run_match

HERE = Path(__file__).resolve().parent


def make_plan():
    matches = []
    for policy, pairs in [("argmax", 8), ("sample", 4)]:
        for seed in range(pairs):
            for game in GAMES:
                for swap in (0, 1):
                    players = ["decisions", "jev"] if swap == 0 else ["jev", "decisions"]
                    matches.append({"id": f"{game}-{policy}-{seed:02d}-swap{swap}", "game": game,
                                    "policy": policy, "seed": 2026100600 + seed * 100,
                                    "pair": seed, "swap": swap, "players": players,
                                    "rounds": 20, "kind": "h2h"})
    for seed in range(2):
        for game in GAMES:
            for provider in ("decisions", "jev"):
                matches.append({"id": f"{game}-self-{provider}-{seed:02d}", "game": game,
                                "policy": "argmax", "seed": 2026100600 + seed * 100,
                                "pair": seed, "swap": 0, "players": [provider, provider],
                                "rounds": 20, "kind": "self"})
    return {"schema": 1, "matches": matches, "action_rule": "API choice (argmax); sample is separate",
            "payoffs": {k: {"rules": v.rules, "actions": v.actions} for k, v in GAMES.items()},
            "option_order": "Seeded shuffle separately for each player/round; seat swap keeps seat seeds",
            "disclosure": "No opponent identity, current action, or probability is sent to the other player"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    (data / "matches").mkdir(exist_ok=True)
    plan = make_plan()
    # JSON converts tuples to lists; compare the canonical serialized form.
    plan = json.loads(json.dumps(plan))
    path = data / "plan.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Plan differs from saved run; use a new experiment directory")
    path.write_text(json.dumps(plan, indent=2) + "\n")
    matches = plan["matches"][:args.limit]
    if args.dry_run:
        print(json.dumps({"matches": len(matches), "rounds": sum(m["rounds"] for m in matches),
                          "maximum_new_successful_calls": 2 * sum(m["rounds"] for m in matches)}))
        return
    journal = Journal(data / "calls.jsonl")
    start = time.perf_counter()
    metadata = {"started_utc": datetime.now(timezone.utc).isoformat(), "workers": args.workers,
                "python": sys.version, "packages": {k: importlib.metadata.version(k)
                for k in ("llm", "llm-openai-decisions", "llm-typesafe", "httpx2")}}
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_match, spec, journal, data / "matches" / f"{spec['id']}.jsonl"): spec
                   for spec in matches}
        for future in as_completed(futures):
            try:
                print(json.dumps({"completed": future.result()}), flush=True)
            except Exception as exc:
                # API adapter errors are sanitized, never include raw headers or credentials.
                spec = futures[future]
                failure = {"match": spec["id"], "type": type(exc).__name__, "message": str(exc)[:200]}
                failures.append(failure)
                print(json.dumps({"failed": failure}), flush=True)
    metadata.update(elapsed_seconds=time.perf_counter()-start, failed_matches=failures,
                    saved_successful_calls=len(journal.saved), matches_requested=len(matches))
    (data / ("run_pilot.json" if args.limit else "run.json")).write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

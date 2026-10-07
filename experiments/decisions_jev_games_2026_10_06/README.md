# Decisions API and Jev in repeated economic games

Date: 6 October 2026 (Pacific). Original experiment branch: `experiment/decisions-jev-games-2026-10-06`.
Reusable-module refactor: `feature/sysone-constrained-games`.

## Scope and findings

The existing local `lludens` checkout is the canonical repository. This experiment
uses the reusable `lludens.sysone` package without changing existing agents or
game environments. The earlier `decision_agent` and `decision_games` modules
remain compatibility shims. See [the module guide](../../docs/sysone.md). Pre-existing README, notebook, data, paper,
and planning edits are not part of this experiment.

The models are **OpenAI Decisions / GPT-6 Luna** and **TypeSafe Jev 1.13.0**.
Both are called through Simon Willison's `llm` plugins, with identical instructions
and role-adjusted state text. The inspected plugin commits are pinned in
`requirements.txt`. OpenAI's plugin was installed from its repository at version
0.1; the announcement's 0.1a0 package was not available from the package index.

The main comparison uses each API's selected action (argmax). Mean points/round:

| Game | GPT-6 Luna | Jev |
| --- | ---: | ---: |
| Prisoner's Dilemma | 0.9625 | 1.1969 |
| Stag Hunt | 2.8781 | 3.0000 |
| Two-player public goods | 9.9788 | 10.0850 |

Each row summarizes 16 twenty-round matches (eight seat-swapped pairs). Jev's
small advantage does **not** imply a universal ranking: the separate sampled
policies reverse the payoff ordering in Prisoner's Dilemma and public goods.
Classification probabilities are not automatically strategic mixed strategies.
Both models score 20/20 on one-shot, known-opponent best-response checks.

## Protocol

- 48 primary head-to-head matches: 3 games × 8 seeds × 2 seats.
- 24 sampled-policy matches: 3 games × 4 seeds × 2 seats.
- 12 self-play matches: 3 games × 2 models × 2 seeds.
- All matches: 20 simultaneous rounds, fixed known horizon, complete public
  action/payoff history, same opponent throughout, maximize own total points.
- No communication, opponent model identity, cross-match memory, payoff noise,
  private types, or hidden chain-of-thought. Only the application retains history.
- Public goods has **two players**, fresh endowment 10, multiplier 1.6, equal
  sharing, contributions {0,2,4,6,8,10}; it is not a four-player treatment.
- Action descriptions are permuted independently by seat and round. Seat swaps
  preserve each seat's schedule; the payoff explanation's order is fixed.
- 166 probes: 40 known-opponent one-shot choices (both option orders), and 126
  fixed-history choices (21 states × 2 models × forward/reverse/repeat).
- Errors and refusals never become fallback actions. API distributions are
  validated. Jev's two-decimal probabilities can sum to 0.99/1.01; the adapter
  preserves raw values and normalizes only errors consistent with rounding.

The plans were frozen before the corresponding calls. The probes' histories were
specified before inspecting their results. API output drives the play; no moves
or probabilities were hand-written. The pilot is conditional on one framing,
one payoff specification per game, and one finite horizon.

## Reproduction

From the repository (or replication bundle) root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  -r experiments/decisions_jev_games_2026_10_06/requirements.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python experiments/decisions_jev_games_2026_10_06/analyze.py
```

Analysis is offline. `analyze.py` regenerates all 3,360 exact LLM requests and
replays all 1,680 rounds against the exact recorded responses, including sampling
seeds, then regenerates tables, figures, and the interactive report. A missing
request fails rather than making an API call.

To also execute the notebook and refresh the complete replication download:

```bash
.venv/bin/python experiments/decisions_jev_games_2026_10_06/build_report.py
```

The report begins with both models' actual opening prompts, LLM keyword arguments,
and recorded typed answers. HTTP request bodies are reconstructed by the pinned
plugins, not presented as captured traffic. All report building is offline.

The executed notebook is `notebooks/decisions_jev_games.ipynb`. It uses the same
offline audit and saved data. Use the full replication bundle to execute a
downloaded notebook; the single notebook file does not contain raw journals.

For live runs, use the plugins' existing API connections in the host execution
environment. Here those are supplied by the OpenClaw Gateway's protected route;
no credentials belong in experiment files or command arguments.

```bash
.venv/bin/python experiments/decisions_jev_games_2026_10_06/run.py --dry-run
.venv/bin/python experiments/decisions_jev_games_2026_10_06/run.py --workers 8
.venv/bin/python experiments/decisions_jev_games_2026_10_06/probes.py
```

The same command resumes saved histories and reuses exact successful calls.
Complete matches are skipped. Use a new experiment directory and seeds for a new
sample; do not overwrite this run's frozen plans.

## Artifacts

- `data/plan.json`: immutable match design, economic rules, and seeds.
- `data/matches/*.jsonl`: append-only round results.
- `data/calls.jsonl.gz`: exact requests, raw typed answers, normalized distributions,
  usage, resolved model IDs, timestamps, and client latency; no credentials.
- `data/probes_plan.json`, `probes.json`, `probe_calls.jsonl.gz`: controlled inputs/results.
- `data/summary.json`: aggregate outcomes, diagnostics, and usage.
- `data/audit.json`: offline full replay result.
- `data/run.json`: final successful resume metadata.
- `data/initial_registration_failures.json`, `quantization_failures.json`: earlier
  interrupted passes, retained for provenance rather than silently discarded.
- `report/opening_examples.json`: actual first-round LLM arguments and typed
  answers, plus request bodies reconstructed offline by the pinned plugins.
- `report/index.html`: standalone report with interactive replay of all matches.
- `report/*.png`: standard Matplotlib figures, suitable for export.

## Execution limitations

An expired local proxy certificate was renewed with the same key/identity and
unchanged TLS/host verification. LLM 0.36's lazy plugin discovery is serialized
to avoid concurrent initialization failures. The initial strict probability-mass
validator interrupted 11 public-goods matches and one diagnostic; accepted calls
and earlier game rounds were preserved, and the incomplete suffixes resumed.
Those 12 rejected responses were not journaled by the initial adapter and were
reissued, rather than interpreted as a game move. A separate one-call inspection
identified Jev's probability rounding. All final planned cases completed.

Reported input costs (~$0.181 combined) use the 3,526 retained game/probe responses;
they exclude two smoke responses, the 12 rejected responses, and one inspection
response. Client latency includes connection setup and concurrent scheduling.
The recorded provider versions are `gpt-6-luna` and `jev-1.13.0`; aliases may change
in future live reruns. Seeded wrappers do not guarantee deterministic API services.

## Sources

- [Announcement](https://simonwillison.net/2026/Oct/6/llm-openai-decisions/)
- [OpenAI Decisions API](https://developers.openai.com/api/docs/guides/decisions)
- [Decisions plugin](https://github.com/simonw/llm-openai-decisions/tree/4827324f5b55a5a4e8626b81518d48706d790a08)
- [Jev plugin](https://github.com/simonw/llm-typesafe/tree/225932cfde461ee6e38ae2b612040c50c086db8c)
- [TypeSafe API](https://docs.typesafe.ai/api)
- [Jev models and pricing](https://docs.typesafe.ai/models)

# Decisions API and Jev in repeated economic games

Date: 6 October 2026 (Pacific). Original experiment branch: `experiment/decisions-jev-games-2026-10-06`.
Reusable-module refactor: `feature/sysone-constrained-games`.

## Scope and findings

The revised memo compares **direct API-selected actions only**. Each player is
queried afresh in every round with the complete realized action/payoff history.
No returned probability vector is used to randomize these actions. The models
are **GPT-6 Luna / OpenAI Decisions** and **TypeSafe Jev 1.13.0**, called through
Simon Willison's pinned `llm` plugins with identical instructions and role-adjusted
state text. The existing `lludens.sysone` module and compatibility imports are
unchanged; see [the module guide](../../docs/sysone.md).

Mean **cumulative points per twenty-round head-to-head match**, with 16 matches
(eight seat-swapped pairs) per game:

| Game | Luna | Jev | Joint | Joint maximum | Mean share | Matches attaining maximum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prisoner's Dilemma | 19.2500 | 23.9375 | 43.1875 | 120 | 35.99% | 0/16 |
| Stag Hunt | 57.5625 | 60.0000 | 117.5625 | 160 | 73.48% | 0/16 |
| Two-player public goods | 199.5750 | 201.7000 | 401.2750 | 640 | 62.70% | 0/16 |

Jev earns more on average in each game. This individual advantage is separate
from collective payoff attainment. Every head-to-head cumulative vector is
strictly Pareto-dominated by the corresponding symmetric cooperative benchmark:
(60,60), (80,80), or (320,320). No head-to-head round attains its joint maximum.
There is no claim of a general ranking of the two models.

The memo also reports 12 self-play matches separately (two per model per game).
One Luna Stag Hunt match reaches (80,80), the only efficient cumulative outcome
among all 60 direct-action matches. Two runs per setting cannot establish an
equilibrium-selection rate.

**Efficiency accounting.** Individual cumulative payoffs are sums of the 20
round payoffs; joint payoff is their sum. The share of the joint maximum is not
an attainment frequency. `direct_results.py` computes the full cumulative Pareto
frontier using exact rational arithmetic and distinguishes efficient asymmetric
vectors from the symmetric joint maximum. For example, perpetual (D,C) in PD
yields the efficient vector (100,0), but not the maximum joint payoff 120.

## Games and information

- PD: CC=(3,3), CD=(0,5), DC=(5,0), DD=(1,1).
- Stag Hunt: SS=(4,4), SH=(0,3), HS=(3,0), HH=(3,3).
- Public goods: **two players**, fresh endowment 10, contributions {0,2,4,6,8,10},
  multiplier 1.6 and equal sharing. Own payoff is `10-c_i+0.8*(c_i+c_j)`;
  joint payoff is `20+0.6*(c_i+c_j)`. Unspent tokens do not carry forward.
- All matches have 20 simultaneous rounds, a fixed known horizon and the same
  opponent throughout. The objective is own expected total points, including
  future rounds—not the lead over the opponent or the joint score.
- Before round t: rules, legal actions, own player number, current round/horizon,
  **every earlier action pair and both payoffs**, and own cumulative points.
  Other cumulative points are inferable from the history. No current-round move
  is revealed until both actions have been selected.
- Explicit per-request history, no persistent conversation, cross-match memory,
  communication, opponent brand/strategy, side payments or private reasoning trace.
- **40 fresh decisions per match**: 20 per player. The direct panel contains
  60 matches, 1,200 rounds and 2,400 retained decisions. Primary head-to-head
  accounts for 48 matches, 960 rounds and 1,920 decisions.
- Options are shuffled by seat and round; swaps preserve each seat's schedule.
  The payoff explanation is fixed. Seeds do not ensure service determinism.
- No API error is converted to a fallback action.

The joint benchmark is not necessarily an equilibrium. Under standard common
knowledge of rationality, finite-horizon backward induction selects defection
in PD and zero contribution in public goods. Low cooperation is not by itself
evidence of misunderstanding. Hare/Hare is an equilibrium in Stag Hunt, although
Stag/Stag Pareto-dominates it. These results describe one framing, payoff
specification and known horizon; they do not identify the models' strategies.

## Preserved archive and exclusions

The original archive is unchanged: 84 matches, 1,680 rounds, 3,360 game decisions,
plus 166 diagnostics. In addition to the 60 direct-action matches, it contains
24 matches with experimenter-imposed action randomization. Those runs change
the agent's policy; they are **not** a second estimate of the direct-action
model comparison and do not enter the revised memo, notebook tables or replay.
Diagnostics also remain in the archive rather than the memo's results.

The plans were frozen before their corresponding calls. The revision does not
remove observed poor outcomes or rerun models. `data/summary.json` preserves the
original all-panel accounting; `data/direct_summary.json` supplies the revised
memo and is copied to `report/summary.json` for download. Every original match
continues to be checked by the offline audit.

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
replays all 1,680 archived rounds against the exact recorded responses, including
sampling seeds. It then selects the 60 direct-action matches for the memo tables,
figure and interactive replay. A missing
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

[Annotation copy on Quomodoc](https://lalten.org/quomodoc/docs/lludens-repeated-games).
Generate its self-contained HTML with `python export_quomodoc.py` from this
experiment directory. The figure is embedded; script-driven replay controls are
replaced with a link to the private interactive report. No Quomodoc scripts or
security settings are changed. Publishing the exported HTML is a separate step.

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
- `data/summary.json`: original all-panel outcomes, diagnostics, and usage (archive).
- `data/direct_summary.json`: revised direct-only cumulative payoffs and efficiency.
- `report/direct_matches.csv`: individual/joint scores and attainment for all 60 direct matches.
- `data/audit.json`: offline full replay result.
- `data/run.json`: final successful resume metadata.
- `data/initial_registration_failures.json`, `quantization_failures.json`: earlier
  interrupted passes, retained for provenance rather than silently discarded.
- `report/opening_examples.json`: actual first-round LLM arguments and typed
  answers, plus request bodies reconstructed offline by the pinned plugins.
- `report/index.html`: standalone memo with interactive replay of the 60 direct-action matches.
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

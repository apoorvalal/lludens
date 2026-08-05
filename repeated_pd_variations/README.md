# Repeated Prisoner's Dilemma variations

This folder runs a five-model ordered round robin through the repeated-game treatments in Kartal and Mueller (2026). The default model panel is:

- OpenAI GPT-5.6 Sol;
- Anthropic Claude Sonnet 5;
- Google Gemini 3.6 Flash;
- DeepSeek V4 Pro;
- xAI Grok 4.5.

All models are accessed through the `llm-openrouter` plugin. The model identifiers live in `models.json`, so the panel can be changed without editing the runner.

## Library API

The experiment is implemented through the reusable `lludens` API rather than a standalone game loop:

- `PromptParameter` represents a parameter rendered only in an agent's system prompt. `TotreLLM` accepts any number of these private parameters, so gamma is an experiment configuration rather than a special case in the base agent.
- `AgentRequest` is the environment-to-agent contract. It carries the current phase, player, round, prompt, optional observation, and environment metadata.
- `TotreLLM.respond()` handles ordinary action or response phases. `TotreLLM.communicate()` is the side-communication hook; its default behavior delegates to the same model, but custom agents can implement a distinct communication policy.
- `TotreLLM.observe()` adds public environment feedback to the next interaction without making a wasteful model call. The legacy `update_history()` method now delegates to `observe()`.
- `ActionSpace` owns valid actions, aliases, parsing, invalid-response flags, and fallbacks. The PD uses the shared `BINARY_ACTION_SPACE`.
- `Phase` declares who acts, whether responses are simultaneous or ordered, whether the phase is communication or action, and which action space validates responses.
- `PhasedGame` executes phases, enforces simultaneous information sets by staging responses, reveals ordered responses immediately, maintains structured history, and sends public observations back to agents.
- `RepeatedPDGame` supplies only the PD-specific phase list, prompts, payoff function, public observation, and round record. The simultaneous, sequential, and chat treatments are compositions of the same phase engine.

The old two-player `Game` API and `TotreLLM.interact()` remain available for existing notebooks. New experiments should prefer `PhasedGame`, `AgentRequest`, and `ActionSpace`.

## Problem setup

Two players repeatedly choose `Cooperate` or `Defect`. The stage-game payoffs are:

| Player 1 | Player 2 | Player 1 payoff | Player 2 payoff |
|---|---|---:|---:|
| Cooperate | Cooperate | 32 | 32 |
| Cooperate | Defect | 12 | 50 |
| Defect | Cooperate | 50 | 12 |
| Defect | Defect | 25 | 25 |

The fixed horizons are 10, 20, and 50 rounds. Each model occupies both seats against every model, including itself. Seat order is retained because it is strategically meaningful in the sequential treatment. With five models, three treatments, and three horizons, one replicate contains `5^2 * 3 * 3 = 225` matches and 6,000 game-round records.

## Completed baseline panel

The current five-model baseline fixes both private gamma values at zero and contains 225 ordered matches and 6,000 unique match-round records with no invalid actions. The analysis-ready combined file is:

```text
data/repeated_pd_variations_five_models_20260804.jsonl
```

The first four families were run as a 144-match panel. Grok was then added with `--involving-model grok-4.5`, producing exactly the 81 new cells that contain Grok: four incoming cross-family pairings, four outgoing cross-family pairings, and Grok self-play, each crossed with three treatments and three horizons. The four Grok shard outputs were concatenated with the original baseline only after row-count, match-count, uniqueness, and invalid-action checks passed.

Aggregate mutual cooperation in the completed panel is 26.2% in SIM, 36.3% in SEQ, and 85.8% in CHAT. Grok is strongly communication-dependent: its action-level cooperation rate is 6.5% in SIM, 17.6% in SEQ, and 90.5% in CHAT; Grok self-play mutual cooperation is 0% in SIM, 0% in SEQ, and 96.2% in CHAT. The full descriptive analysis is in `notebooks/in_silico_varieties_repeated_games.qmd`.

## Treatments

### Simultaneous

Both players receive the public history and choose an action without observing the opponent's current action. The two responses are collected before payoffs and the public history are updated.

### Sequential

Player 1 moves first. Player 2 observes Player 1's current action before choosing. The ordered round robin runs every distinct model pairing in both seat assignments, while self-play appears once per model.

### Chat

Each round has a simultaneous cheap-talk phase followed by simultaneous actions. Each player writes one message without seeing the opponent's current message. Both messages are then shown to both players before they choose actions. Messages, coded message labels, actions, and payoffs are stored in the round-level JSONL output.

## Private gamma

Each player receives a private, match-stable `gamma` through its system prompt. The instruction says that choosing `Cooperate` adds `gamma` to that player's stage utility. A player observes its own gamma but not its opponent's. Gamma affects the model's decision problem through the private prompt; reported monetary payoffs remain the common payoff matrix above so behavior and experimental payments stay separately interpretable.

`--gamma-values` specifies the support of the private-type distribution. The runner draws each player's gamma independently and deterministically from that support using the match seed. The default is `0`, which reproduces the untyped baseline. For example, `--gamma-values 0 10 20` spans low, intermediate, and high cooperation preferences; values above 18 eliminate the one-shot temptation to defect when the opponent cooperates under this payoff matrix.

The assigned private types are recorded as `player1_gamma` and `player2_gamma` in both the frozen plan and every round record. They are never included in the public history or the opponent's prompt.

## Running the experiment

Configure the OpenRouter key for `llm` and inspect the full plan without making model calls:

```bash
uv run llm keys set openrouter
uv run python -m repeated_pd_variations.run_experiments --dry-run
```

Run the untyped baseline:

```bash
uv run python -m repeated_pd_variations.run_experiments
```

Run three private gamma types with two independent replicates:

```bash
uv run python -m repeated_pd_variations.run_experiments \
  --gamma-values 0 10 20 \
  --replicates 2
```

The default outputs are:

- `data/repeated_pd_variations_plan.json`: immutable match plan, seeds, seats, and private types;
- `data/repeated_pd_variations.jsonl`: append-only round records.

The runner checkpoints after every round. Re-running the same command reconstructs public match history, resumes each partial match at its next missing round, and skips complete match IDs. Use a new output path or seed for a genuinely new batch.

Paid sweeps default to a 256-token response cap with model reasoning disabled; the game asks for one action or a message of at most 25 words, so hidden reasoning only adds cost and can consume the entire response allowance. Provider-specific overrides live in `models.json`: Gemini and Grok require small reasoning allowances, while DeepSeek uses a larger low-effort allowance to ensure it emits a final action. If a provider returns an invalid or truncated action, resume with `--rerun-invalid`. The runner truncates every affected match immediately before its first invalid round and regenerates that round and its dependent suffix. Pass `--reasoning` only for experiments that intentionally study reasoning-enabled behavior across models without a per-model override.

Large sweeps can run as independent shards. Each shard must use its own plan and JSONL paths:

```bash
uv run python -m repeated_pd_variations.run_experiments \
  --shard-count 4 --shard-index 0 \
  --output data/repeated_pd_variations_shard0.jsonl \
  --plan-output data/repeated_pd_variations_plan_shard0.json
```

Run shard indices `0` through `3` to cover the full plan exactly once, then concatenate their JSONL files for analysis.

To add one model's row and column without rerunning an existing panel, configure the new model and filter the plan to matches involving its label before sharding:

```bash
uv run python -m repeated_pd_variations.run_experiments \
  --involving-model grok-4.5 \
  --shard-count 4 --shard-index 0 \
  --output data/repeated_pd_variations_grok_shard0.jsonl \
  --plan-output data/repeated_pd_variations_grok_plan_shard0.json
```

Summarize completed results with:

```bash
uv run python -m repeated_pd_variations.summarize
```

## Changelog

### 2026-08-04 — Grok row-and-column expansion

- Added Grok 4.5 to the default OpenRouter model panel with its mandatory reasoning setting and a 64-token reasoning cap.
- Ran the 81-match incremental Grok row/column expansion in four deterministic shards: 2,160 rounds with zero invalid actions.
- Merged the expansion with the four-family baseline into a validated five-model panel containing 225 matches and 6,000 rounds.
- Updated the paper's aggregate results, pairwise matrices, model-level figure, interpretation, and reproducibility note for the five-model panel.

### 2026-08-03 — Core API refactor

- Replaced the bespoke repeated-PD orchestration loop with `PhasedGame` and reusable `Phase` objects.
- Generalized `TotreLLM` to accept private system-prompt parameters and phase-specific requests.
- Added an explicit side-communication hook and public-observation memory that does not trigger extra model calls.
- Generalized action validation through `ActionSpace` while preserving `parse_binary_action()`.
- Kept the legacy `Game` and `TotreLLM.interact()` interfaces for notebook compatibility.
- Added deterministic plan sharding for parallel paid sweeps.
- Changed paid-run checkpointing from match-level to round-level, including partial-match resume from reconstructed public history.
- Disabled hidden reasoning and raised the response cap to prevent empty action responses in paid sweeps.
- Added `--rerun-invalid` to regenerate an invalid round and every later round conditioned on it.
- Added per-model option overrides so provider constraints such as mandatory Gemini reasoning remain explicit and reproducible.
- Added phase-level invalid-action retries so transient empty provider responses are regenerated before a round is recorded.
- Made invalid-action retries corrective rather than identical by explicitly repeating the valid action set after a failed response.
- Added `--involving-model` for incremental row-and-column expansions of an existing round robin.

### 2026-08-03 — Initial scaffold

- Added the four-model ordered round robin, resumable JSONL output, frozen plans, three treatments, three horizons, private gamma assignments, and summary command.

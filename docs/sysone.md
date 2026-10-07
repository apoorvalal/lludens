# System One choices and games

`lludens.sysone` puts constrained action selection behind the same interface for
OpenAI Decisions and TypeSafe Jev. The application supplies state, instructions,
and legal options; the model returns a typed choice and a distribution. It does
not generate a strategy explanation or retain conversational memory.

## Installation

```bash
uv sync --extra sysone --extra test
```

The `sysone` extra installs the two Simon Willison `llm` plugins at the commits
used in the October 2026 pilot. Existing `lludens` agents do not need this extra.
Use the plugins' existing LLM credential configuration or a host-managed API
connection for live calls. No credentials belong in notebooks or call journals.
Reading saved data, importing `sysone`, and offline replay require no API access.

## A constrained choice outside a game

```python
from lludens.sysone import evaluate_choice, preview_choice

options = {"billing": "Payments and invoices", "technical": "Product failures"}
state = "I was charged twice for my order."
question = "Which team should handle this request?"

# Offline: exact LLM arguments and the request body built by the installed plugin.
preview = preview_choice("jev", state, options, question)
print(preview["api_payload"])

# Live: requires a configured provider connection.
result = evaluate_choice("jev", state, options, question)
print(result["choice"], result["probabilities"])
```

Use provider `"decisions"` for `openai-decisions/gpt-6-luna`, or `"jev"` for
`typesafe/jev-latest`. Jev's alias can resolve to a newer model on future calls;
the result records both the requested model ID and returned model name.
Supply explicit task instructions for non-game choices. For compatibility with
the pilot, the default instructions still describe maximizing own match points.

`choice_kwargs` constructs the keyword arguments passed to `llm.Model.prompt`.
Both plugins put `system` into the question's `instructions`; `prompt` becomes
OpenAI `input` or TypeSafe `state`. Options become `choices` or `criteria`.
`preview_choice` calls the plugin's payload builder on a lazy response object;
it does **not** evaluate that response or send an HTTP request. Its body is a
reconstruction, not a captured HTTP transaction. These previews contain state
text, so only publish them when that text is suitable for sharing.

## Play a match

```python
from lludens.sysone import DecisionAgent, GAMES, RepeatedDecisionGame

options = GAMES["stag_hunt"].descriptions
agents = {
    1: DecisionAgent("decisions", options, seed=41),
    2: DecisionAgent("jev", options, seed=42),
}
game = RepeatedDecisionGame("stag_hunt", agents, n_rounds=20)
history = game.run_game()  # 40 live decisions
```

Available game names are `prisoners_dilemma`, `stag_hunt`, and `public_goods`.
The last is a **two-player** game with fresh endowment 10, multiplier 1.6, and
contributions in `{0, 2, 4, 6, 8, 10}`. `EconomicGame.payoff` defines incentives
in code; prompts describe exactly those rules.

The existing `PhasedGame` engine stages both actions before publishing either.
Calls may be executed sequentially, but the information structure is simultaneous:
player 2's prompt cannot see player 1's current action. Each prompt includes the
full public history from that player's perspective, not the opponent's provider
identity or its probabilities. Model parameters are not updated during a match.

`DecisionAgent` independently seeds action-order shuffling and optional sampling.
The historical default `policy="argmax"` means **use the API-selected action**:
ties are not broken again in Python. This strict adapter rejects a choice that
is not a reported probability maximum. `policy="sample"` instead samples the
normalized distribution. That is an experimenter-imposed policy, not evidence
that the model intended a game-theoretic mixed strategy. A custom `instructions`
argument can replace the game objective for a different application.

## Resume without repeating successful decisions

```python
from pathlib import Path
from lludens.sysone import DecisionJournal, run_match

output = Path("my-sysone-run")
spec = {
    "id": "stag-hunt-example", "game": "stag_hunt",
    "players": ["decisions", "jev"], "rounds": 20,
    "seed": 40, "policy": "argmax",
}
journal = DecisionJournal(output / "calls.jsonl")
scores = run_match(spec, journal, output / "rounds.jsonl")
```

The journal records exact prompts, instructions, option order, raw typed answers,
normalized distributions, usage, model identity, timestamps, and latency. Only
successful decisions are cached. If the second player fails, the first player's
already-recorded choice is reused on resume, before the round is published.
Transport errors and refusals never become fallback game actions. Only explicit
HTTP 429/529 responses are retried (at most three attempts), not ambiguous timeouts.

Request hashes include **option insertion order**. Changing a prompt or the order
of options is a different request. Context/player IDs distinguish otherwise
identical calls in different matches. Keep one immutable specification per round
checkpoint: `run_match` checks continuity and payoffs, but a checkpoint alone is
not proof that the same model or seed produced it. Use a new output directory
when changing the design. A `DecisionJournal` is thread-safe for appends in one
process; it is not a multi-process queue or a single-flight request scheduler.

The round file permits fast resume, which skips already-completed rounds. A full
audit instead starts an empty game and asks an **offline** journal for every move:

```python
from pathlib import Path
from lludens.sysone import DecisionAgent, DecisionJournal, GAMES, RepeatedDecisionGame

data = Path("experiments/decisions_jev_games_2026_10_06/data")
journal = DecisionJournal(data / "calls.jsonl", offline=True)
match_id = "prisoners_dilemma-argmax-00-swap0"
agents = {
    p: DecisionAgent(provider, GAMES["prisoners_dilemma"].descriptions,
                     seed=2026100600 + p, call=journal.caller(match_id, p))
    for p, provider in ((1, "decisions"), (2, "jev"))
}
replayed = RepeatedDecisionGame("prisoners_dilemma", agents, 20).run_game()
assert len(replayed) == 20
```

The reader automatically opens `calls.jsonl.gz` if the uncompressed journal is
absent. Offline mode never expands the archive or invokes a transport. Missing
exact requests fail rather than silently calling an API. The complete pilot's
`analyze.py` verifies all 3,360 saved requests and 1,680 rounds this way.

## Validation and layout

`normalize_choice` checks labels, finite probabilities, positive total mass, and
the selected action. Jev 1.13.0 rounds probabilities to two decimals; only mass
errors bounded by that precision are renormalized. Original values remain in
`result["answer"]`. Provider confidence is retained there too; it is not treated
as a separate probability that a strategy is correct.

| Module | Responsibility |
| --- | --- |
| `sysone.choices` | LLM adapters, payload previews, response validation |
| `sysone.agent` | Seeded option order and action-extraction policy |
| `sysone.games` | Payoffs, prompt construction, simultaneous game environment |
| `sysone.journal` | Append-only call records and exact offline replay |
| `sysone.runner` | Resumable match execution with round checkpoints |

The old `lludens.decision_agent` and `lludens.decision_games` import paths remain
compatibility shims. New code should import from `lludens.sysone`.
Experiment design, analysis, plots, and report building stay in
`experiments/decisions_jev_games_2026_10_06/`, not inside the reusable package.

```bash
uv run --extra sysone --extra test pytest tests -q
uv run --extra sysone python experiments/decisions_jev_games_2026_10_06/analyze.py
```

All tests and saved-data analysis are offline. The example API calls above are
illustrative live usage, not steps performed by the test suite.

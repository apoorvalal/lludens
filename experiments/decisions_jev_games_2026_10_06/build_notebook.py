"""Generate the analysis notebook; execution is offline and never calls an API."""
from pathlib import Path
import nbformat as nbf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
n=nbf.v4.new_notebook()
n.metadata.kernelspec={'name':'python3','display_name':'Python 3','language':'python'}
n.cells=[
nbf.v4.new_markdown_cell('''# Decision models in repeated economic games

GPT-6 Luna (OpenAI Decisions API) and Jev 1.13.0 (TypeSafe), using Simon Willison's
`llm-openai-decisions` and `llm-typesafe` plugins. Data collected 6 October 2026 Pacific.

The primary comparison uses the APIs' selected actions. Sampling from their
distributions is a **separate experimenter-imposed policy**, not a claim that
classification probabilities are strategic mixing probabilities. Each agent
maximizes its own total payoff in a known twenty-round match with public history.

The panel contains 48 direct-choice head-to-head matches, 24 sampled head-to-head
matches, and 12 self-play matches. Seats are swapped within seeds; option order
varies by seat and round. All 1,680 rounds are replayed below from saved calls.
This notebook makes no network requests. Run from the complete repository or
downloaded replication bundle, not the notebook file alone.

[API documentation](https://developers.openai.com/api/docs/guides/decisions) ·
[Jev documentation](https://docs.typesafe.ai/api) ·
[Plugin announcement](https://simonwillison.net/2026/Oct/6/llm-openai-decisions/)'''),
nbf.v4.new_code_cell('''from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import display, Image
root = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p/'lludens/decision_games.py').exists())
experiment = root/'experiments/decisions_jev_games_2026_10_06'
sys.path.insert(0, str(root)); sys.path.insert(0, str(experiment))
from analyze import audit_and_load, summarize, TITLES
rounds, matches, replay, journal = audit_and_load()
summary, probes = summarize(rounds, matches, journal)
json.loads((experiment/'data/audit.json').read_text())'''),
nbf.v4.new_markdown_cell('''## Economic primitives

- Prisoner's Dilemma: CC=(3,3), CD=(0,5), DC=(5,0), DD=(1,1).
- Stag Hunt: SS=(4,4), SH=(0,3), HS=(3,0), HH=(3,3).
- Two-player public goods: fresh endowment 10; contributions {0,2,4,6,8,10};
  pool multiplier 1.6 and equal sharing. Payoff $10-c_i+0.8(c_i+c_j)$.

Defection and zero contribution are one-shot best responses in the first and
third games. Stag Hunt has two pure equilibria; Stag is optimal against a belief
of more than 0.75 on the other choosing Stag. A fixed finite horizon provides no
standard complete-information backward-induction support for cooperation in PD
or public goods. Low cooperation is not, on its own, a model error.'''),
nbf.v4.new_markdown_cell('''## Head-to-head behavior and payoffs

Means are computed across matches, not across independent rounds. Bootstrap
intervals resample entire seat-swapped seed pairs (eight for the primary panel,
four for sampling). They are small-sample, design-conditional pilot summaries.'''),
nbf.v4.new_code_cell('''head = pd.DataFrame(summary['h2h'])
display(head[['game','policy','matches','luna_points','jev_points',
              'decisions_contribution','jev_contribution','luna_minus_jev',
              'paired_bootstrap_low','paired_bootstrap_high']].round(4))
display(Image(filename=str(experiment/'report/payoffs.png')))'''),
nbf.v4.new_code_cell('''display(Image(filename=str(experiment/'report/cooperation.png')))
display(head[['game','policy','luna_wins','ties','jev_wins','joint_optimum_fraction']].round(4))'''),
nbf.v4.new_markdown_cell('''Jev's selected actions earn slightly more in all three primary panels, while
Luna occasionally attempts cooperation or Stag. Sampling changes the interaction
and reverses the PD and public-goods mean ranking. This is not evidence for a
general model leaderboard, and a payoff advantage need not increase group welfare.'''),
nbf.v4.new_markdown_cell('''## Known-opponent incentives and controlled histories

Each model sees every fixed opposing action in a one-round setting, under both
option orders. Separately, 21 fixed histories are queried forward, reversed, and
identically repeated. The histories were specified before inspecting probe results.'''),
nbf.v4.new_code_cell('''display(pd.DataFrame(summary['probes']).T)
rows = [{'game':p['game'],'history':p['case'],'model':p['provider'],
         'variant':p['variant'],'choice':p['result']['choice']}
        for p in probes if p['kind']=='history']
display(pd.DataFrame(rows).pivot(index=['game','history','model'],columns='variant',values='choice'))'''),
nbf.v4.new_markdown_cell('''Both models get 20/20 known-opponent best responses right. They condition
on supplied history: PD cooperation follows sustained cooperation, but not
sustained defection, and both defect in the final round after cooperation.
Luna's opening Stag Hunt choice flips when option order is reversed. Jev's
public-goods choice following sustained contribution changes on an identical
repeat. Neither diagnostic establishes universal order robustness or determinism.'''),
nbf.v4.new_code_cell('''display(pd.DataFrame(summary['self_play']).round(4))
display(pd.DataFrame(summary['usage']).T)'''),
nbf.v4.new_markdown_cell('''## Interpretation and replication limits

One Luna self-play Stag Hunt match coordinates on Stag for all twenty rounds;
the other does not. Jev self-play chooses Hare throughout both matches. Two
self-play matches per model/game are not enough to estimate equilibrium-selection
probabilities.

All 3,526 retained game and diagnostic responses are available in the journals,
including prompts, option ordering, returned answers, model IDs, usage, and timing.
Latency includes network/connection setup under concurrent load. The estimated
~18-cent total excludes setup calls and initially rejected validation responses.

Jev probabilities are rounded to two decimals. Raw values are retained and small
mass errors consistent with rounding are normalized for sampling. Twelve initial
validation rejections were not retained and were reissued; earlier successful
moves and histories were preserved. No failed call became a default action.

The complete report adds an interactive replay of every match. See the experiment
README for pinned package commits, live-run instructions, and execution provenance.
The scope is one prompt framing, one finite horizon, one payoff matrix per game,
no communication, and a two-player rather than larger-group public-goods game.''')]
nbf.write(n,ROOT/'notebooks/decisions_jev_games.ipynb')
print('Notebook generated')

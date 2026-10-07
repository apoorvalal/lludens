"""Economic checks for cumulative efficiency and the direct-only estimand."""
from fractions import Fraction as F
from itertools import product
import json
from pathlib import Path
import sys

import pytest

EXPERIMENT = Path(__file__).resolve().parents[1] / 'experiments/decisions_jev_games_2026_10_06'
sys.path.insert(0, str(EXPERIMENT))
from direct_results import evaluate_match, pareto_frontier, stage_payoffs, summarize_direct


def match(game, actions, policy='argmax'):
    return {'id':'fixture', 'game':game, 'kind':'h2h', 'pair':0, 'swap':0,
            'players':['decisions','jev'], 'rounds':len(actions), 'policy':policy,
            'history':[{'round':t, 'actions':dict(zip(('1','2'), pair)),
                        'payoffs':dict(zip(('1','2'), map(float, stage_payoffs(game)[pair])))}
                       for t,pair in enumerate(actions, 1)]}


@pytest.mark.parametrize('game', ['prisoners_dilemma','stag_hunt','public_goods'])
def test_frontier_matches_exhaustive_three_round_paths(game):
    # Enumerate every complete path without intermediate pruning, then compare
    # every feasible pair for domination. This is independent of the DP scan.
    points = {tuple(sum(p[i] for p in path) for i in (0,1))
              for path in product(stage_payoffs(game).values(), repeat=3)}
    reference = {p for p in points if not any(q != p and q[0] >= p[0] and q[1] >= p[1]
                                             for q in points)}
    assert pareto_frontier(game, 3) == reference


@pytest.mark.parametrize('game,pair,scores', [
    ('prisoners_dilemma', ('D','C'), (100,0)),
    ('public_goods', ('0','10'), (360,160)),
])
def test_asymmetric_efficiency_is_not_joint_maximum(game, pair, scores):
    result = evaluate_match(match(game, [pair]*20))
    assert (result['seat1_payoff'], result['seat2_payoff']) == scores
    assert result['pareto_efficient']
    assert not result['joint_maximum_attained']
    assert result['joint_share'] < 1


def test_averaging_efficient_pd_stages_can_yield_dominated_total():
    result = evaluate_match(match('prisoners_dilemma', [('D','C'),('C','D')]))
    assert result['joint_payoff'] == 10  # Both players could earn 6 instead of 5.
    assert not result['pareto_efficient']
    assert not result['joint_maximum_attained']


def test_public_goods_exact_attainment_and_rejecting_wrong_payoffs():
    result = evaluate_match(match('public_goods', [('10','10')]*20))
    assert result['joint_payoff'] == 640 and result['joint_share'] == 1
    assert result['joint_maximum_attained'] and result['pareto_efficient']
    bad = match('public_goods', [('2','0')]*20)
    bad['history'][0]['payoffs']['1'] += 1
    with pytest.raises(ValueError, match='economic primitives'):
        evaluate_match(bad)
    with pytest.raises(ValueError, match='randomized'):
        evaluate_match(match('prisoners_dilemma', [('C','C')], policy='sample'))


def test_saved_panel_selection_seat_mapping_and_payoff_denominators():
    plan = json.loads((EXPERIMENT/'data/plan.json').read_text())
    replay = [{**spec, 'history':[json.loads(line) for line in
              (EXPERIMENT/'data/matches'/f"{spec['id']}.jsonl").read_text().splitlines()]}
              for spec in plan['matches']]
    summary = summarize_direct(replay)
    assert (summary['matches'], summary['rounds'], summary['decisions']) == (60,1200,2400)
    expected = {'prisoners_dilemma':(F(77,4),F(383,16),120),
                'stag_hunt':(F(921,16),F(60),160),
                'public_goods':(F(7983,40),F(2017,10),640)}
    for row in summary['h2h']:
        luna, jev, maximum = expected[row['game']]
        assert (row['matches'], row['seat_pairs'], row['rounds_observed']) == (16,8,320)
        assert row['mean_luna_payoff'] == float(luna)
        assert row['mean_jev_payoff'] == float(jev)
        assert row['mean_joint_payoff'] == float(luna+jev)
        assert row['mean_joint_share'] == pytest.approx(float((luna+jev)/maximum))
        assert row['joint_maximum_matches'] == row['pareto_efficient_matches'] == 0
        assert row['joint_optimal_rounds'] == 0
    luna_stag = next(r for r in summary['self_play'] if r['game']=='stag_hunt' and r['provider']=='decisions')
    assert luna_stag['matches'] == 2
    assert (luna_stag['mean_seat1_payoff'], luna_stag['mean_seat2_payoff']) == (68.5,68.5)
    assert luna_stag['mean_joint_payoff'] == 137
    assert luna_stag['joint_maximum_matches'] == luna_stag['pareto_efficient_matches'] == 1
    assert sum(r['pareto_efficient'] for r in summary['match_results']) == 1

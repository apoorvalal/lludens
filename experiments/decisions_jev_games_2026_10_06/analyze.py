"""Offline audit, statistical summaries, figures, and self-contained HTML report."""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import html
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(HERE))
from lludens.sysone import DecisionJournal as Journal, read_jsonl as jsonl
from lludens.sysone import DecisionAgent, normalize_choice
from lludens.sysone import GAMES, RepeatedDecisionGame
from opening_examples import opening_examples, examples_html
from direct_results import summarize_direct

TITLES = {"prisoners_dilemma":"Prisoner’s Dilemma", "stag_hunt":"Stag Hunt", "public_goods":"Public goods (2 players)"}
NAMES = {"decisions":"GPT-6 Luna", "jev":"Jev 1.13.0"}
COLORS = {"decisions":"#1f658f", "jev":"#b45725"}


def audit_and_load():
    data=HERE/'data'
    plan=json.loads((data/'plan.json').read_text())
    journal=Journal(data/'calls.jsonl',offline=True)
    rows=[]; matches=[]; replay=[]
    for spec in plan['matches']:
        saved=jsonl(data/'matches'/f"{spec['id']}.jsonl")
        assert len(saved)==spec['rounds']
        agents={p:DecisionAgent(spec['players'][p-1],dict(GAMES[spec['game']].descriptions),
                    seed=spec['seed']+p, policy=spec['policy'],call=journal.caller(spec['id'],p)) for p in (1,2)}
        env=RepeatedDecisionGame(spec['game'],agents,spec['rounds'])
        for record in saved:
            actual=env.play_round(record['round'])
            assert actual['actions']==record['actions'] and actual['payoffs']==record['payoffs']
            for p in (1,2):
                result=agents[p].last_decision
                assert result['choice']==record['decisions'][str(p)]['choice']
                assert result['probabilities']==record['decisions'][str(p)]['probabilities']
                rows.append({'match':spec['id'], 'game':spec['game'],'kind':spec['kind'],'policy':spec['policy'],
                    'pair':spec['pair'],'swap':spec['swap'],'round':record['round'],'player':p,
                    'provider':spec['players'][p-1],'action':record['actions'][str(p)],
                    'payoff':record['payoffs'][str(p)],
                    'contribution':GAMES[spec['game']].contribution(record['actions'][str(p)])})
        records=[]
        for r in saved:
            records.append({'round':r['round'],'actions':r['actions'],'payoffs':r['payoffs'],
                            'probabilities':{p:v['probabilities'] for p,v in r['decisions'].items()}})
        replay.append({**spec,'history':records})
    assert len(journal.saved)==len(rows)==3360
    frame=pd.DataFrame(rows)
    match=frame.groupby(['game','kind','policy','pair','swap','match','provider'],as_index=False).agg(
        score=('payoff','sum'),points_per_round=('payoff','mean'),contribution=('contribution','mean'))
    audit={'matches_replayed':len(plan['matches']),'rounds_replayed':len(rows)//2,
           'exact_saved_requests':len(journal.saved),'payoffs_and_actions_match':True,
           'network_calls_during_audit':0}
    (HERE/'data'/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    return frame,match,replay,journal


def summarize(frame, match, journal):
    h2h=match[match.kind=='h2h']
    summary=[]
    for (game,policy),sub in h2h.groupby(['game','policy']):
        pivot=sub.pivot(index=['pair','swap'],columns='provider',values='points_per_round')
        d=pivot['decisions']-pivot['jev']; pair_diff=d.groupby('pair').mean().to_numpy()
        rng=np.random.default_rng(4827)
        boots=rng.choice(pair_diff,size=(10000,len(pair_diff)),replace=True).mean(axis=1)
        entry={'game':game,'policy':policy,'matches':len(pivot),'seat_pairs':len(pair_diff),
               'luna_points':float(pivot.decisions.mean()),'jev_points':float(pivot.jev.mean()),
               'luna_minus_jev':float(d.mean()),'paired_bootstrap_low':float(np.quantile(boots,.025)),
               'paired_bootstrap_high':float(np.quantile(boots,.975)),
               'luna_wins':int((d>1e-9).sum()),'jev_wins':int((d < -1e-9).sum()),
               'ties':int((abs(d)<=1e-9).sum())}
        for provider in NAMES:
            entry[provider+'_contribution']=float(sub[sub.provider==provider].contribution.mean())
        entry['joint_optimum_fraction']=(entry['luna_points']+entry['jev_points'])/{'prisoners_dilemma':6,'stag_hunt':8,'public_goods':32}[game]
        summary.append(entry)
    probes=json.loads((HERE/'data'/'probes.json').read_text())
    probe_summary={}
    for provider in NAMES:
        br=[p for p in probes if p['provider']==provider and p['kind']=='best_response']
        history=[p for p in probes if p['provider']==provider and p['kind']=='history']
        grouped=defaultdict(dict)
        for p in history: grouped[(p['game'],p['case'])][p['variant']]=p['result']
        probe_summary[provider]={'best_response_correct':sum(p['result']['choice'] in p['optimal'] for p in br),
            'best_response_total':len(br),'states':len(grouped),
            'order_flips':sum(g['forward']['choice']!=g['reverse']['choice'] for g in grouped.values()),
            'repeat_flips':sum(g['forward']['choice']!=g['repeat']['choice'] for g in grouped.values())}
    usage={}
    all_calls=list(journal.saved.values())+jsonl(HERE/'data'/'probe_calls.jsonl')
    for provider in NAMES:
        calls=[x for x in all_calls if x['provider']==provider]
        lats=[x['result']['latency_seconds'] for x in calls]
        inputs=sum(x['result']['usage']['input_tokens'] for x in calls)
        sums=[sum((dict((x['value'],x['probability']) for x in c['result']['answer']['probabilities'])
                   if isinstance(c['result']['answer']['probabilities'],list) else c['result']['answer']['probabilities']).values()) for c in calls]
        usage[provider]={'retained_calls':len(calls),'input_tokens':inputs,'estimated_usd':inputs*({'decisions':.1,'jev':.042}[provider])/1e6,
                         'median_latency_ms':float(np.median(lats)*1000),'p90_latency_ms':float(np.quantile(lats,.9)*1000),
                         'quantized_mass_adjustments':sum(abs(s-1)>1e-4 for s in sums),
                         'models':sorted({c['result']['model'] for c in calls})}
    output={'h2h':summary,'probes':probe_summary,'usage':usage,
            'self_play':match[match.kind=='self'].groupby(['game','provider'],as_index=False).agg(
                mean_points=('points_per_round','mean'),mean_contribution=('contribution','mean')).to_dict('records')}
    (HERE/'data'/'summary.json').write_text(json.dumps(output,indent=2)+'\n')
    return output,probes


def make_figures(frame, summary):
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':10,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), layout='constrained')
    for ax, game in zip(axes, GAMES):
        sub = frame[(frame.game == game) & (frame.kind == 'h2h') & (frame.policy == 'argmax')]
        joint = sub.groupby(['match','round']).payoff.sum().unstack('round').cumsum(axis=1).mean()
        rounds = np.arange(0, 21)
        ceiling = rounds * summary['benchmarks'][game]['stage_joint']
        realized = np.r_[0, joint.to_numpy()]
        ax.plot(rounds, ceiling, color='#687780', linestyle='--', label='Feasible joint maximum')
        ax.plot(rounds, realized, color='#1f658f', linewidth=2.2, label='Observed mean joint payoff')
        ax.fill_between(rounds, realized, ceiling, color='#e8eef2')
        ax.set(title=TITLES[game], xlabel='Completed rounds', ylabel='Cumulative joint points',
               xlim=(0,20), ylim=(0, ceiling[-1]*1.06), xticks=[0,5,10,15,20])
        ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=8, loc='upper left')
    fig.savefig(HERE/'report'/'payoffs.png', dpi=180)
    plt.close(fig)


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def make_report(summary, replay, examples):
    def fmt(x):
        return f"{x:.3f}".rstrip('0').rstrip('.')

    main_table = table(
        ['Game', 'Luna total', 'Jev total', 'Joint J', 'Maximum J*', 'J / J*',
         'Mean shortfall J* − J', 'Attain J*'],
        [[TITLES[r['game']], fmt(r['mean_luna_payoff']), fmt(r['mean_jev_payoff']),
          fmt(r['mean_joint_payoff']), fmt(r['joint_maximum']), f"{100*r['mean_joint_share']:.1f}%",
          fmt(r['mean_joint_shortfall']), f"{r['joint_maximum_matches']} / {r['matches']}"]
         for r in summary['h2h']])
    behavior_table = table(
        ['Game', 'Realized action pairs (Luna / Jev)', 'Luna wins / ties / Jev wins',
         'Joint-optimal rounds', 'Pareto-efficient matches'],
        [[TITLES[r['game']], '; '.join(f"{pair}: {n}" for pair,n in r['action_profiles'].items()),
          f"{r['luna_wins']} / {r['ties']} / {r['jev_wins']}",
          f"{r['joint_optimal_rounds']} / {r['rounds_observed']}",
          f"{r['pareto_efficient_matches']} / {r['matches']}"] for r in summary['h2h']])
    self_table = table(
        ['Game', 'Both players', 'Mean U₁', 'Mean U₂', 'Mean joint J', 'J / J*', 'Attain J*'],
        [[TITLES[r['game']], NAMES[r['provider']], fmt(r['mean_seat1_payoff']),
          fmt(r['mean_seat2_payoff']), fmt(r['mean_joint_payoff']), f"{100*r['mean_joint_share']:.1f}%",
          f"{r['joint_maximum_matches']} / {r['matches']}"] for r in summary['self_play']])
    individual_table = table(
        ['Game', 'Pair', 'Seat 1', 'Luna total', 'Jev total', 'Joint J', 'J / J*', 'Attain J*'],
        [[TITLES[r['game']], r['pair'], NAMES[r['players'][0]], fmt(r['luna_payoff']),
          fmt(r['jev_payoff']), fmt(r['joint_payoff']), f"{100*r['joint_share']:.1f}%",
          'Yes' if r['joint_maximum_attained'] else 'No']
         for r in summary['match_results'] if r['kind']=='h2h'])
    self_matches = table(
        ['Game', 'Both players', 'Run', 'Seat 1 U₁', 'Seat 2 U₂', 'Joint J', 'Attain J*'],
        [[TITLES[r['game']], NAMES[r['players'][0]], r['pair'], fmt(r['seat1_payoff']),
          fmt(r['seat2_payoff']), fmt(r['joint_payoff']), 'Yes' if r['joint_maximum_attained'] else 'No']
         for r in summary['match_results'] if r['kind']=='self'])
    template = (HERE/'report_template.html').read_text()
    tokens = {'MAIN_TABLE': main_table, 'BEHAVIOR_TABLE': behavior_table,
              'SELF_TABLE': self_table, 'MATCH_TABLE': individual_table, 'SELF_MATCHES': self_matches,
              'OPENING_EXAMPLES': examples_html(examples),
              'SHARED_INSTRUCTIONS': html.escape(examples[0]['llm_kwargs']['system']),
              'REPLAY_DATA': json.dumps([m for m in replay if m['policy']=='argmax'],
                                        separators=(',',':')).replace('</','<\\/'),
              'BENCHMARK_DATA': json.dumps(summary['benchmarks'], separators=(',',':'))}
    for key,value in tokens.items():
        template = template.replace('@@'+key+'@@', value)
    assert '@@' not in template
    (HERE/'report'/'index.html').write_text(template)


def main():
    (HERE/'report').mkdir(exist_ok=True)
    frame, match, replay, journal = audit_and_load()
    summarize(frame, match, journal)  # Preserve the complete original archive summary.
    summary = summarize_direct(replay)
    (HERE/'data'/'direct_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    make_figures(frame, summary)
    examples = opening_examples(journal)
    (HERE/'report'/'opening_examples.json').write_text(json.dumps(examples, indent=2, ensure_ascii=False)+'\n')
    make_report(summary, replay, examples)
    frame.to_csv(HERE/'data'/'rounds.csv', index=False)
    match.to_csv(HERE/'data'/'matches.csv', index=False)
    columns = ['id','game','kind','pair','swap','players','rounds','seat1_payoff','seat2_payoff',
               'luna_payoff','jev_payoff','joint_payoff','joint_maximum','joint_share','joint_shortfall',
               'joint_maximum_attained','pareto_efficient','joint_optimal_rounds']
    pd.DataFrame(summary['match_results'])[columns].to_csv(HERE/'report'/'direct_matches.csv', index=False)
    print(json.dumps(summary['h2h'], indent=2))


if __name__=='__main__': main()

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


def make_figures(frame,summary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,game in zip(axs,GAMES):
        sub=frame[(frame.game==game)&(frame.kind=='h2h')&(frame.policy=='argmax')]
        for provider in NAMES:
            y=sub[sub.provider==provider].groupby('round').contribution.mean()
            ax.plot(y.index,100*y,label=NAMES[provider],color=COLORS[provider],marker='o',markersize=3)
        ax.set(title=TITLES[game],xlabel='Round',ylim=(-2,102),xticks=[1,5,10,15,20]);ax.grid(alpha=.15)
        ax.set_ylabel('Mean % of endowment contributed' if game=='public_goods' else 'Cooperate / Stag choices (%)')
    axs[0].legend(frameon=False,fontsize=9)
    fig.savefig(HERE/'report'/'cooperation.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,game in zip(axs,GAMES):
        sub=[r for r in summary['h2h'] if r['game']==game]
        for i,policy in enumerate(['argmax','sample']):
            row=next(x for x in sub if x['policy']==policy)
            ax.bar(i-.17,row['luna_points'],width=.32,color=COLORS['decisions'],label='GPT-6 Luna' if i==0 else None)
            ax.bar(i+.17,row['jev_points'],width=.32,color=COLORS['jev'],label='Jev 1.13.0' if i==0 else None)
        ax.set(title=TITLES[game],ylabel='Mean points per round',xticks=[0,1],xticklabels=['API choice','Sampled policy'])
        ax.grid(axis='y',alpha=.15)
    axs[0].legend(frameon=False,fontsize=9)
    fig.savefig(HERE/'report'/'payoffs.png',dpi=180);plt.close(fig)


def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'


def make_report(summary,probes,replay,examples):
    rows=[]; sample=[]
    for r in summary['h2h']:
        row=[TITLES[r['game']],r['matches'],f"{r['luna_points']:.3f}",f"{r['jev_points']:.3f}",
             f"{r['decisions_contribution']*100:.1f}%",f"{r['jev_contribution']*100:.1f}%"]
        (rows if r['policy']=='argmax' else sample).append(row)
    main_table=table(['Game','Matches','Luna points/round','Jev points/round','Luna C / Stag / contribution','Jev C / Stag / contribution'],rows)
    sample_table=table(['Game','Matches','Luna points/round','Jev points/round','Luna C / Stag / contribution','Jev C / Stag / contribution'],sample)
    probes_table=table(['Model','Known-opponent best responses','Order reversals changed choice','Identical repeats changed choice'],[
        [NAMES[k],f"{v['best_response_correct']}/{v['best_response_total']}",f"{v['order_flips']}/{v['states']}",f"{v['repeat_flips']}/{v['states']}"] for k,v in summary['probes'].items()])
    usage_table=table(['Model','Retained calls','Input tokens','Estimated input cost','Median / p90 latency'],[
        [NAMES[k],v['retained_calls'],f"{v['input_tokens']:,}",f"${v['estimated_usd']:.4f}",f"{v['median_latency_ms']:.0f} / {v['p90_latency_ms']:.0f} ms"] for k,v in summary['usage'].items()])
    interval_table=table(['Game / policy','Luna − Jev points/round','Paired bootstrap 95% interval','Luna wins / ties / Jev wins'],[
        [TITLES[r['game']]+' / '+r['policy'],f"{r['luna_minus_jev']:.3f}",f"[{r['paired_bootstrap_low']:.3f}, {r['paired_bootstrap_high']:.3f}]",f"{r['luna_wins']} / {r['ties']} / {r['jev_wins']}"] for r in summary['h2h']])
    self_table=table(['Game','Model','Mean points/round','C / Stag / contribution'],[
        [TITLES[r['game']],NAMES[r['provider']],f"{r['mean_points']:.3f}",f"{100*r['mean_contribution']:.1f}%"] for r in summary['self_play']])
    case_rows=[]
    for game in GAMES:
        for case in ['empty','mutual_cooperation','mutual_safe_or_selfish','you_exploited','other_exploited','alternating','final_round_after_cooperation']:
            row=[TITLES[game],case.replace('_',' ')]
            for provider in NAMES:
                row.append(' / '.join(next(p['result']['choice'] for p in probes if p['game']==game and p['case']==case and p['provider']==provider and p['kind']=='history' and p['variant']==v) for v in ['forward','reverse','repeat']))
            case_rows.append(row)
    history_table=table(['Game','Fixed history','Luna: forward / reverse / repeat','Jev: forward / reverse / repeat'],case_rows)
    template=(HERE/'report_template.html').read_text()
    tokens={'MAIN_TABLE':main_table,'SAMPLE_TABLE':sample_table,'PROBES_TABLE':probes_table,'USAGE_TABLE':usage_table,
            'OPENING_EXAMPLES':examples_html(examples),
            'SHARED_INSTRUCTIONS':html.escape(examples[0]['llm_kwargs']['system']),
            'INTERVAL_TABLE':interval_table,'SELF_TABLE':self_table,'HISTORY_TABLE':history_table,
            'REPLAY_DATA':json.dumps(replay,separators=(',',':')).replace('</','<\\/'),
            'TOTAL_COST':f"{sum(x['estimated_usd'] for x in summary['usage'].values()):.3f}"}
    for key,value in tokens.items(): template=template.replace('@@'+key+'@@',value)
    assert '@@' not in template
    (HERE/'report'/'index.html').write_text(template)


def main():
    (HERE/'report').mkdir(exist_ok=True)
    frame,match,replay,journal=audit_and_load()
    summary,probes=summarize(frame,match,journal)
    make_figures(frame,summary)
    examples=opening_examples(journal)
    (HERE/'report'/'opening_examples.json').write_text(json.dumps(examples,indent=2,ensure_ascii=False)+'\n')
    make_report(summary,probes,replay,examples)
    frame.to_csv(HERE/'data'/'rounds.csv',index=False)
    match.to_csv(HERE/'data'/'matches.csv',index=False)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()

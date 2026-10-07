"""Controlled one-shot incentives and identical-history order/repeat probes."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import HERE, Journal
from lludens.decision_games import GAMES, RepeatedDecisionGame, INSTRUCTIONS
from lludens.environments import RoundState


def make_probes():
    probes = []
    for name, spec in GAMES.items():
        for other in spec.actions:
            best = max(spec.payoff(a, other)[0] for a in spec.actions)
            optimal = [a for a in spec.actions if spec.payoff(a,other)[0] == best]
            for provider in ("decisions", "jev"):
                for reverse in (False, True):
                    options = list(spec.descriptions.items())
                    if reverse: options.reverse()
                    probes.append({"id":f"best-response-{name}-{other}-{provider}-{reverse}",
                        "kind":"best_response", "game":name, "provider":provider,
                        "case":other, "variant":"reverse" if reverse else "forward",
                        "prompt": spec.rules + f"\nThere is exactly ONE round and no future interaction. "
                        f"The other player's action is fixed, known, and guaranteed to be {other}. "
                        "Your action cannot change the other player's action. You want only your own points.",
                        "instructions":"Choose the action that gives YOU the most points in this single round.",
                        "options":dict(options), "optimal":optimal})
        cooperative, selfish = spec.cooperative_action, spec.actions[1] if name != "public_goods" else "0"
        histories = {
            "empty": [],
            "mutual_cooperation": [(cooperative,cooperative)]*5,
            "mutual_safe_or_selfish": [(selfish,selfish)]*5,
            "you_exploited": [(cooperative,selfish)]*5,
            "other_exploited": [(selfish,cooperative)]*5,
            "alternating": [(cooperative,cooperative),(selfish,selfish)]*2+[(cooperative,cooperative)],
            "final_round_after_cooperation": [(cooperative,cooperative)]*19,
        }
        for case, history in histories.items():
            env = RepeatedDecisionGame(name,{1:object(),2:object()},20)
            for t,(a,b) in enumerate(history,1):
                u,v=spec.payoff(a,b)
                env.history.append({"round":t,"actions":{"1":a,"2":b},"payoffs":{"1":u,"2":v}})
            prompt=env.prompt_for(env.phases[0],1,RoundState(len(history)+1))
            for provider in ("decisions", "jev"):
                for variant in ("forward", "reverse", "repeat"):
                    options=list(spec.descriptions.items())
                    if variant=="reverse": options.reverse()
                    probes.append({"id":f"history-{name}-{case}-{provider}-{variant}",
                        "kind":"history", "game":name, "provider":provider, "case":case,
                        "variant":variant,"prompt":prompt,"instructions":INSTRUCTIONS,"options":dict(options)})
    return probes


def main():
    probes=make_probes()
    path=HERE/'data'/'probes_plan.json'
    if path.exists() and json.loads(path.read_text())!=probes:
        raise ValueError('Probe plan changed')
    path.write_text(json.dumps(probes,indent=2)+'\n')
    journal=Journal(HERE/'data'/'probe_calls.jsonl')
    def run(probe):
        result=journal.caller(probe['id'],1)(probe['provider'],probe['prompt'],probe['options'],probe['instructions'])
        return {**probe,'result':result}
    results=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(run,p) for p in probes]):
            results.append(future.result())
    results.sort(key=lambda x:x['id'])
    (HERE/'data'/'probes.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'probes':len(results),'best_response':sum(p['kind']=='best_response' for p in results)}))


if __name__=='__main__': main()

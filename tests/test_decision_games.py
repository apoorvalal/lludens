import json
from pathlib import Path
import random
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lludens.agent import AgentRequest
from lludens.sysone import DecisionAgent, normalize_choice
from lludens.sysone import GAMES, RepeatedDecisionGame


@pytest.mark.parametrize("game", list(GAMES))
def test_payoffs_are_symmetric(game):
    spec = GAMES[game]
    for a in spec.actions:
        for b in spec.actions:
            assert spec.payoff(a,b) == tuple(reversed(spec.payoff(b,a)))


def test_game_incentives_and_group_optima():
    pd = GAMES["prisoners_dilemma"]
    assert all(pd.payoff("D", b)[0] > pd.payoff("C", b)[0] for b in pd.actions)
    sh = GAMES["stag_hunt"]
    assert sh.payoff("Stag","Stag")[0] > sh.payoff("Hare","Stag")[0]
    assert sh.payoff("Stag","Hare")[0] < sh.payoff("Hare","Hare")[0]
    pg = GAMES["public_goods"]
    for b in pg.actions:
        assert all(pg.payoff("0",b)[0] > pg.payoff(a,b)[0] for a in pg.actions if a != "0")
    assert max(sum(pg.payoff(a,b)) for a in pg.actions for b in pg.actions) == 32


def test_simultaneous_hidden_moves_and_perspective():
    seen = []
    class Fake:
        def __init__(self, action): self.action = action
        def respond(self, request):
            assert request.metadata["state"].responses == {}
            seen.append(request)
            return self.action
    game = RepeatedDecisionGame("prisoners_dilemma", {1: Fake("C"), 2: Fake("D")}, 2)
    game.play_round(1)
    game.play_round(2)
    assert '"your_action":"C","other_action":"D","your_points":0' in seen[2].prompt
    assert '"your_action":"D","other_action":"C","your_points":5' in seen[3].prompt
    assert "gpt-6" not in seen[2].prompt and "jev" not in seen[2].prompt
    assert game.history[0]["payoffs"] == {"1": 0, "2": 5}


@pytest.mark.parametrize("list_format", [True, False])
def test_plugins_normalize_identically(list_format):
    p = {"C": .2, "D": .8}
    raw = [{"value": k, "probability": v} for k,v in p.items()] if list_format else p
    assert normalize_choice({"type":"choice", "choice":"D", "probabilities":raw}, p) == ("D",p)


@pytest.mark.parametrize("answer", [
    {"type":"refusal"},
    {"type":"choice","choice":"D","probabilities":{"C":.4,"D":.4}},
    {"type":"choice","choice":"C","probabilities":{"C":.1,"D":.9}},
    {"type":"choice","choice":"D","probabilities":{"C":float("nan"),"D":1}},
    {"type":"choice","choice":"D","probabilities":{"C":.2,"E":.8}},
])
def test_invalid_api_outputs_never_become_moves(answer):
    with pytest.raises(ValueError):
        normalize_choice(answer, {"C":"Cooperate","D":"Defect"})


def test_action_sampling_is_reproducible_and_distinct_from_argmax():
    def fake(provider,prompt,options,instructions):
        return {"choice":"D", "answer":{"type":"choice","choice":"D","probabilities":{"C":.49,"D":.51}}}
    requests = [AgentRequest("same state", "move", r, 1) for r in range(1,101)]
    agents = [DecisionAgent("jev", {"C":"Cooperate","D":"Defect"}, seed=7, policy=policy, call=fake)
              for policy in ("sample","sample","argmax")]
    paths = [[a.respond(req) for req in requests] for a in agents]
    assert paths[0] == paths[1]
    assert set(paths[0]) == {"C","D"}
    assert set(paths[2]) == {"D"}


def test_transport_failure_does_not_default_to_defection():
    def fail(*args): raise RuntimeError("API unavailable")
    agent = DecisionAgent("jev", dict(GAMES["prisoners_dilemma"].descriptions),seed=1,call=fail)
    with pytest.raises(RuntimeError, match="API unavailable"):
        agent.respond(AgentRequest("state", "move", 1, 1))
    assert agent.last_decision is None


def test_quantized_multiclass_probabilities_preserve_relative_mass():
    raw = {"0":.93,"2":0.,"4":0.,"6":.01,"8":0.,"10":.05}
    answer = {"type":"choice","choice":"0","probabilities":raw}
    action, p = normalize_choice(answer,raw)
    assert action == "0" and sum(p.values()) == pytest.approx(1)
    assert p["10"] == pytest.approx(.05/.99)
    assert sum(raw.values()) == pytest.approx(.99)  # raw evidence unchanged

"""Offline contract tests for public APIs, adapters, and resumable decisions."""
import gzip
import json
from pathlib import Path
import sys

import llm
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lludens.agent import AgentRequest
from lludens.sysone import (
    DecisionAgent, DecisionJournal, GAMES, RepeatedDecisionGame, choice_kwargs,
    evaluate_choice, normalize_choice, preview_choice, read_jsonl, run_match,
)
from lludens.sysone import choices


OPTIONS = {"C": "Cooperate", "D": "Defect"}


def result():
    return {"choice": "D", "probabilities": {"C": .2, "D": .8},
            "answer": {"type": "choice", "choice": "D",
                       "probabilities": {"C": .2, "D": .8}}}


def test_compatibility_imports_are_same_classes():
    from lludens.decision_agent import DecisionAgent as OldAgent
    from lludens.decision_games import RepeatedDecisionGame as OldGame
    assert OldAgent is DecisionAgent
    assert OldGame is RepeatedDecisionGame


@pytest.mark.parametrize("provider", ["decisions", "jev"])
def test_payload_preview_uses_plugin_without_evaluating_response(provider, monkeypatch):
    model = choices.get_decision_model(provider)
    def fail(*args, **kwargs):
        raise AssertionError("Payload previews must not invoke inference")
    monkeypatch.setattr(type(model), "execute", fail)
    ordered = {"D": "Defect", "C": "Cooperate"}
    preview = preview_choice(provider, "current state", ordered, "choose an action")
    assert preview["llm_kwargs"] == choice_kwargs(provider, "current state", ordered, "choose an action")
    body = preview["api_payload"]
    if provider == "decisions":
        assert body == {
            "model": "gpt-6-luna", "input": "current state", "questions": [{
                "name": "evaluation", "instructions": "choose an action", "type": "choice",
                "choices": [{"value": k, "description": v} for k, v in ordered.items()],
            }],
        }
    else:
        assert body == {
            "model": "jev-latest", "state": "current state", "questions": {
                "evaluation": {"type": "choice", "instructions": "choose an action", "criteria": ordered},
            },
        }


@pytest.mark.parametrize("provider,option_key", [("decisions", "choices"), ("jev", "criteria")])
def test_evaluate_choice_preserves_answer_and_validates_distribution(provider, option_key, monkeypatch):
    seen = []
    answer = {"type": "choice", "choice": "D", "probabilities": {"D": .80, "C": .19}}
    class Response:
        def text(self): return json.dumps(answer)
        def json(self): return {"model": "test-version", "usage": {"input_tokens": 1, "output_tokens": 0}}
    class Model:
        def prompt(self, **kwargs):
            seen.append(kwargs)
            return Response()
    monkeypatch.setattr(choices, "get_decision_model", lambda provider: Model())
    out = evaluate_choice(provider, "state", OPTIONS, "instruction")
    assert seen == [{"prompt": "state", "system": "instruction", "answer_type": "choice", option_key: OPTIONS}]
    assert out["answer"] == answer
    assert out["probabilities"]["D"] == pytest.approx(.8 / .99)
    assert out["model"] == "test-version" and out["attempts"] == 1


@pytest.mark.parametrize("status,attempts,delays", [(429, 3, [1, 2]), (529, 3, [1, 2]),
                                                  (401, 1, []), (402, 1, []), (500, 1, []),
                                                  (None, 1, [])])
def test_only_explicit_rate_limits_retry_and_errors_are_sanitized(status, attempts, delays, monkeypatch):
    seen, slept = [], []
    class Model:
        def prompt(self, **kwargs):
            seen.append(kwargs)
            # A marker, not a credential: raw transport messages must stay private.
            raise llm.ModelError(f"HTTP {status}: do-not-echo-transport-detail")
    monkeypatch.setattr(choices, "get_decision_model", lambda provider: Model())
    monkeypatch.setattr(choices.time, "sleep", slept.append)
    with pytest.raises(RuntimeError, match="no action substituted") as error:
        evaluate_choice("jev", "state", OPTIONS)
    assert "do-not-echo" not in str(error.value)
    assert len(seen) == attempts and slept == delays


@pytest.mark.parametrize("options", [{}, {"only": "One"}, {"": "Empty", "D": "Defect"}])
def test_invalid_requests_rejected_before_model_lookup(options, monkeypatch):
    def fail(provider): raise AssertionError("Unexpected model lookup")
    monkeypatch.setattr(choices, "get_decision_model", fail)
    with pytest.raises(ValueError): evaluate_choice("jev", "state", options)


@pytest.mark.parametrize("probabilities", [[{"value": "C"}], [], {"C": 0, "D": 0},
                                          {"C": True, "D": 0},
                                          [{"value": "D", "probability": .5}] * 2])
def test_malformed_distributions_fail_closed(probabilities):
    with pytest.raises(ValueError):
        normalize_choice({"type": "choice", "choice": "D", "probabilities": probabilities}, OPTIONS)


def test_agent_rejects_inconsistent_replay_record_and_clears_previous_decision():
    def call(*args):
        return values.pop(0)
    values = [result(), {**result(), "choice": "C"}]
    agent = DecisionAgent("jev", OPTIONS, seed=5, call=call)
    assert agent.respond(AgentRequest("state", "move", 1, 1)) == "D"
    with pytest.raises(ValueError, match="disagrees"):
        agent.respond(AgentRequest("state", "move", 2, 1))
    assert agent.last_decision is None


def test_agent_custom_instructions_and_options_are_snapshotted():
    seen = []
    def call(provider, prompt, options, instructions):
        seen.append((options, instructions))
        return result()
    options = dict(OPTIONS)
    agent = DecisionAgent("jev", options, seed=5, call=call, instructions="Maximize joint payoff.")
    options["E"] = "Not part of the original space"
    agent.respond(AgentRequest("state", "move", 1, 1))
    assert set(seen[0][0]) == set(OPTIONS)
    assert seen[0][1] == agent.last_decision["instructions"] == "Maximize joint payoff."


def test_agent_retains_archived_roundoff_but_rejects_different_probabilities():
    stored = {**result(), "probabilities": {"C": .20000000000000004, "D": .7999999999999999}}
    agent = DecisionAgent("jev", OPTIONS, seed=5, call=lambda *args: stored)
    agent.respond(AgentRequest("state", "move", 1, 1))
    assert agent.last_decision["probabilities"] == stored["probabilities"]
    stored["probabilities"] = {"C": .4, "D": .6}
    with pytest.raises(ValueError, match="probabilities disagree"):
        agent.respond(AgentRequest("state", "move", 2, 1))


def test_compressed_journal_replays_exact_order_without_writing_or_calling(tmp_path):
    path = tmp_path / "calls.jsonl"
    live = DecisionJournal(path, call=lambda *args: result())
    live.caller("match", 1)("jev", "state", OPTIONS, "instructions")
    archived = path.with_suffix(".jsonl.gz")
    archived.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
    path.unlink()
    def fail(*args): raise AssertionError("Offline replay called the network")
    replay = DecisionJournal(path, offline=True, call=fail)
    assert replay.caller("match", 1)("jev", "state", OPTIONS, "instructions") == result()
    assert not path.exists()
    with pytest.raises(ValueError, match="Missing exact saved request"):
        replay.caller("match", 1)("jev", "state", dict(reversed(OPTIONS.items())), "instructions")
    with pytest.raises(ValueError, match="Missing exact saved request"):
        replay.caller("match", 2)("jev", "state", OPTIONS, "instructions")


def test_duplicate_journal_keys_are_rejected(tmp_path):
    path = tmp_path / "calls.jsonl"
    path.write_text('{"key":"duplicate"}\n' * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        DecisionJournal(path, offline=True)


def test_failed_second_move_resumes_first_without_recalling_it(tmp_path):
    calls = []
    def transport(provider, *args):
        calls.append(provider)
        if len(calls) == 2:
            raise RuntimeError("second move failed")
        return result()
    journal_path, rounds = tmp_path / "calls.jsonl", tmp_path / "rounds.jsonl"
    spec = {"id": "resume", "players": ["decisions", "jev"], "game": "prisoners_dilemma",
            "seed": 5, "rounds": 2, "policy": "argmax"}
    journal = DecisionJournal(journal_path, call=transport)
    with pytest.raises(RuntimeError, match="second move failed"):
        run_match(spec, journal, rounds)
    assert len(read_jsonl(journal_path)) == 1 and read_jsonl(rounds) == []
    journal = DecisionJournal(journal_path, call=transport)
    assert run_match(spec, journal, rounds)["scores"] == {"1": 2, "2": 2}
    assert calls == ["decisions", "jev", "jev", "decisions", "jev"]
    assert len(read_jsonl(journal_path)) == 4 and len(read_jsonl(rounds)) == 2
    # Replay into a fresh checkpoint: every request must come from the journal.
    replay = DecisionJournal(journal_path, offline=True)
    replay_path = tmp_path / "offline-rounds.jsonl"
    run_match(spec, replay, replay_path)
    assert read_jsonl(replay_path) == read_jsonl(rounds)


def test_saved_opening_examples_match_journals():
    root = Path(__file__).resolve().parents[1]
    experiment = root / "experiments/decisions_jev_games_2026_10_06"
    sys.path.insert(0, str(experiment))
    from opening_examples import opening_examples
    journal = DecisionJournal(experiment / "data/calls.jsonl", offline=True)
    for example in opening_examples(journal):
        source = journal.saved[example["journal_key"]]
        assert example["answer"] == source["result"]["answer"]
        assert example["llm_kwargs"]["prompt"] == source["prompt"]
        assert "history from your perspective: []" in source["prompt"]
        assert example["round"] == 1 and example["answer"]["choice"] == "D"


def test_all_saved_diagnostic_probes_replay():
    root = Path(__file__).resolve().parents[1]
    experiment = root / "experiments/decisions_jev_games_2026_10_06"
    sys.path.insert(0, str(experiment))
    from probes import make_probes
    probes = make_probes()
    saved = json.loads((experiment / "data/probes.json").read_text())
    assert probes == json.loads((experiment / "data/probes_plan.json").read_text())
    journal = DecisionJournal(experiment / "data/probe_calls.jsonl", offline=True)
    for probe in probes:
        actual = journal.caller(probe["id"], 1)(probe["provider"], probe["prompt"],
                                               probe["options"], probe["instructions"])
        assert actual == next(s["result"] for s in saved if s["id"] == probe["id"])
    assert len(probes) == len(journal.saved) == 166

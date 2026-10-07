"""Document the two recorded opening decisions, without making new API calls."""
from __future__ import annotations

import html
import json

from lludens.sysone import DecisionAgent, GAMES, RepeatedDecisionGame, preview_choice

MATCH_ID = "prisoners_dilemma-argmax-00-swap0"


def opening_examples(journal):
    """Replay round one and reconstruct bodies through the pinned LLM plugins."""
    if not journal.offline:
        raise ValueError("Report examples require an offline journal")
    agents = {
        player: DecisionAgent(provider, GAMES["prisoners_dilemma"].descriptions,
                              seed=2026100600 + player, call=journal.caller(MATCH_ID, player))
        for player, provider in ((1, "decisions"), (2, "jev"))
    }
    game = RepeatedDecisionGame("prisoners_dilemma", agents, 20)
    game.play_round(1)
    examples = []
    for player, agent in agents.items():
        decision = agent.last_decision
        record = next(r for r in journal.saved.values()
                      if r["match_id"] == MATCH_ID and r["player"] == player
                      and r["prompt"] == decision["prompt"])
        preview = preview_choice(record["provider"], record["prompt"], record["options"],
                                 record["instructions"])
        examples.append({
            "match_id": MATCH_ID, "round": 1, "player": player,
            "journal_key": record["key"], "recorded_utc": record["utc"],
            **preview, "returned_model": record["result"]["model"],
            "answer": record["result"]["answer"], "usage": record["result"]["usage"],
            "latency_seconds": record["result"]["latency_seconds"],
            "request_provenance": "Body reconstructed by the pinned plugin from the exact saved LLM arguments; not an HTTP capture.",
            "response_provenance": "Exact typed answer retained from response.text(); the full HTTP response envelope was not archived.",
        })
    return examples


def examples_html(examples):
    def code(value):
        return "<pre><code>" + html.escape(value) + "</code></pre>"

    chunks = []
    for e in examples:
        name = "GPT-6 Luna / Decisions" if e["provider"] == "decisions" else "Jev / TypeSafe"
        kwargs = e["llm_kwargs"]
        # Python repr produces executable keyword arguments, including option order.
        invocation = "import llm\n\nresponse = llm.get_model(" + repr(e["model_id"]) + ").prompt(\n"
        invocation += "\n".join(f"    {k}={v!r}," for k, v in kwargs.items())
        invocation += "\n)\nanswer = response.text()  # Evaluating this would make a live API call."
        chunks.append(
            f'<article class="opening-example" data-provider="{e["provider"]}">'
            f'<h3>{name} — player {e["player"]}</h3>'
            '<p><strong>Initial state prompt</strong> (the exact text passed to LLM).</p>'
            + code(kwargs["prompt"])
            + '<details><summary>Exact llm Python invocation</summary>' + code(invocation) + '</details>'
            + '<details><summary>Request JSON built by the pinned plugin</summary>'
            + '<p>This body is reconstructed offline from the saved arguments, not an HTTP capture.</p>'
            + code(json.dumps(e["api_payload"], indent=2, ensure_ascii=False)) + '</details>'
            + '<p><strong>Recorded response payload</strong> — the unchanged typed answer returned by '
            '<code>response.text()</code>, not a generated explanation.</p>'
            + code(json.dumps(e["answer"], indent=2, ensure_ascii=False))
            + f'<p class="muted">Returned model: <code>{html.escape(e["returned_model"])}</code>. '
            + f'Input tokens: {e["usage"]["input_tokens"]}; client time: {1000 * e["latency_seconds"]:.0f} ms. '
            + f'Journal key: <code>{html.escape(e["journal_key"])}</code>.</p></article>'
        )
    return ''.join(chunks)

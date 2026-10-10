"""Quick checks for T.R.I.A.D. that run without Ollama or an API key.

The real AI models are swapped for a fake one that gives fixed answers,
so we can check the debate flow in about a second.

Run with:  python test_app.py
"""
import json
from pathlib import Path

from langchain_core.messages import AIMessage

import app


class FakeModel:
    """Pretends to be an AI model. Each agent always votes the way we tell it to."""

    def __init__(self, agent, vote):
        self.agent = agent
        self.vote = vote

    def invoke(self, messages):
        prompt = messages[-1].content
        if "VOTE:" in prompt:
            return AIMessage(content=f"VOTE: {self.vote}\nREASON: {self.agent} has its reasons.")
        if "Rewrite your proposal" in prompt:
            return AIMessage(content=f"{self.agent} revised proposal")
        if self.agent == "judge":
            return AIMessage(content="Judge's final answer")
        if self.agent == "ego":
            return AIMessage(content="<think>let me reason</think>ego proposal")
        return AIMessage(content=f"{self.agent} proposal")


def run_debate(votes):
    """Run a full debate with fake models. Returns (final state, list of steps run)."""
    app.get_model = lambda agent, mode, key, think=True: FakeModel(agent, votes.get(agent))
    inputs = {"query": "test question", "mode": app.LOCAL, "api_key": "secret-key", "iteration": 0}
    steps, state = [], {}
    for kind, chunk in app.app_graph.stream(inputs, stream_mode=["updates", "values"]):
        if kind == "updates":
            steps += list(chunk)
        else:
            state = chunk
    return state, steps


def test_parse_vote():
    assert app.parse_vote("VOTE: APPROVE\nREASON: Good plan.") == "APPROVE"
    assert app.parse_vote("I cannot APPROVE this.\nVOTE: REJECT") == "REJECT"   # the old bug
    assert app.parse_vote("vote: modify") == "MODIFY"                            # lower case
    assert app.parse_vote("I approve.") == "APPROVE"                             # format ignored
    assert app.parse_vote("Not sure.") == "MODIFY"                               # nothing found


def test_parse_reason():
    assert app.parse_reason("VOTE: REJECT\nREASON: Too risky.") == "Too risky."
    assert app.parse_reason("VOTE: REJECT") == ""


def test_majority_ends_after_one_round():
    state, steps = run_debate({"id": "APPROVE", "ego": "APPROVE", "superego": "REJECT"})
    assert state["iteration"] == 1
    assert "revision" not in steps
    assert steps[-1] == "consensus"
    assert state["history"][0]["reasons"]["superego"] == "superego has its reasons."


def test_no_majority_revises_then_stops_at_limit():
    state, steps = run_debate({"id": "MODIFY", "ego": "REJECT", "superego": "APPROVE"})
    assert state["iteration"] == app.MAX_ROUNDS
    assert steps.count("revision") == app.MAX_ROUNDS - 1
    after_revision = steps[steps.index("revision") + 1]
    assert after_revision == "cross_critique"     # revised proposals get criticized again
    assert state["proposals"]["id"] == "id revised proposal"
    assert len(state["history"]) == app.MAX_ROUNDS


def test_judge_writes_consensus_and_ego_thought_kept():
    state, _ = run_debate({"id": "APPROVE", "ego": "APPROVE", "superego": "APPROVE"})
    assert state["final_consensus"] == "Judge's final answer"
    assert state["ego_thought"] == "let me reason"


def test_saved_debate_has_no_api_key():
    state, _ = run_debate({"id": "APPROVE", "ego": "APPROVE", "superego": "APPROVE"})
    path = Path(app.save_debate(state))
    data = json.loads(path.read_text(encoding="utf-8"))
    path.unlink()
    assert "api_key" not in data
    assert "secret-key" not in json.dumps(data)
    assert data["final_consensus"] == "Judge's final answer"


def test_debate_log_shows_reasons():
    state, _ = run_debate({"id": "APPROVE", "ego": "APPROVE", "superego": "REJECT"})
    page = app.render_debate_log(state["history"])
    assert "Why REJECT: superego has its reasons." in page


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print("passed:", test.__name__)
    print(f"\nAll {len(tests)} tests passed.")

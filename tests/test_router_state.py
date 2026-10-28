"""A new turn in an existing chat thread must not inherit the previous turn's rewrite state."""
from langchain_core.messages import HumanMessage

import backend.rag_graph as rag_graph
from backend.models import RouterDecision


class _Chain:
    def invoke(self, _):
        return RouterDecision(route="paper_qa")


def test_router_clears_state_left_by_the_previous_turn(monkeypatch):
    monkeypatch.setattr(rag_graph, "router_chain", _Chain())
    state = {
        "messages": [HumanMessage(content="new question")],
        "session_id": "s",
        "doc_titles": ["Paper"],
        "query": "new question",
        # left behind by an earlier turn that was rewritten and premise-corrected
        "original_query": "old question",
        "premise_refuted": True,
        "premise_correction": "old correction",
    }
    update = rag_graph.router_node(state)
    assert update["route"] == "paper_qa"
    assert update["original_query"] is None
    assert update["premise_refuted"] is None
    assert update["premise_correction"] is None

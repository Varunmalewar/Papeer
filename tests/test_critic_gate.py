from backend.rag_graph import _lexical_grounding, _needs_critic

CONTEXT = (
    "The encoder is composed of a stack of N = 6 identical layers. Each layer has two "
    "sub-layers: multi-head self-attention and a position-wise feed-forward network. "
    "The feed-forward network uses a ReLU activation between two linear transformations."
)


def test_clean_short_grounded_answer_skips_critic():
    draft = "The encoder stacks identical layers with self-attention and feed-forward sub-layers."
    needs, reasons, grounding = _needs_critic(draft, CONTEXT)
    assert needs is False, reasons
    assert grounding >= 0.45


def test_numbers_inside_math_do_not_fire():
    draft = "The feed-forward network is defined as $FFN(x) = \\max(0, xW_1 + b_1)W_2 + b_2$ using ReLU."
    needs, reasons, _ = _needs_critic(draft, CONTEXT)
    assert needs is False, reasons


def test_page_numbers_in_citations_do_not_fire():
    draft = "The encoder stacks identical layers with self-attention [Paper, Page 3]."
    needs, reasons, _ = _needs_critic(draft, CONTEXT)
    assert needs is False, reasons


def test_numeric_claim_fires():
    draft = "The encoder stacks 6 identical layers."
    needs, reasons, _ = _needs_critic(draft, CONTEXT)
    assert needs is True
    assert "numbers" in reasons


def test_causal_language_fires():
    draft = "The encoder uses self-attention because it captures long-range dependencies."
    needs, reasons, _ = _needs_critic(draft, CONTEXT)
    assert needs is True
    assert "causal" in reasons


def test_long_answer_fires():
    draft = " ".join(["layers"] * 95)
    needs, reasons, _ = _needs_critic(draft, CONTEXT)
    assert needs is True
    assert "long" in reasons


def test_fabricated_citation_fires():
    draft = "The encoder stacks identical layers."
    needs, reasons, _ = _needs_critic(draft, CONTEXT, fabricated_citation=True)
    assert needs is True
    assert "fabricated_citation" in reasons


def test_low_grounding_fires():
    draft = "Quantum annealing optimizes protein folding pathways in synthetic biology."
    needs, reasons, _ = _needs_critic(draft, CONTEXT)
    assert needs is True
    assert "low_grounding" in reasons


def test_lexical_grounding_bounds():
    assert _lexical_grounding("", CONTEXT) == 1.0
    assert _lexical_grounding("encoder layers", CONTEXT) == 1.0
    assert _lexical_grounding("zzzz qqqq", CONTEXT) == 0.0

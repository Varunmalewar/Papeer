from backend.rag_graph import _strip_stray_refusal

GROUNDED = (
    "Self-attention lets each position attend to all other positions in the same "
    "sequence [Paper, Page 2]. The encoder stacks identical layers with "
    "multi-head attention and feed-forward sub-layers [Paper, Page 3]."
)


def test_trailing_refusal_removed():
    out = _strip_stray_refusal(GROUNDED + " The paper does not state this.")
    assert "does not state this" not in out
    assert out.startswith("Self-attention")
    assert "[Paper, Page 3]" in out


def test_refusal_on_its_own_line_removed():
    out = _strip_stray_refusal(GROUNDED + "\n\nThe paper does not state this.")
    assert "does not state this" not in out
    assert out.rstrip().endswith("sub-layers [Paper, Page 3].")


def test_pure_refusal_kept():
    answer = "The paper does not state this."
    assert _strip_stray_refusal(answer) == answer


def test_fallback_refusal_kept():
    answer = (
        "I wasn't able to find relevant information in the uploaded papers to "
        "answer your question. You may want to rephrase your question or upload "
        "additional papers."
    )
    assert _strip_stray_refusal(answer) == answer


def test_dont_know_refusal_kept():
    answer = "I don't know the answer."
    assert _strip_stray_refusal(answer) == answer


def test_mid_answer_refusal_removed_and_citation_preserved():
    answer = (
        "The paper uses scaled dot-product attention [Paper, Page 4]. The paper "
        "does not state this. It also reports multi-head attention [Paper, Page 5]."
    )
    out = _strip_stray_refusal(answer)
    assert "does not state this" not in out
    assert "[Paper, Page 4]" in out and "[Paper, Page 5]" in out


def test_short_answer_not_destroyed():
    answer = "It uses ReLU. The paper does not state this."
    assert _strip_stray_refusal(answer) == answer

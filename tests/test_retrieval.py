"""Hermetic unit tests for retrieval internals (no network, no LLM).

Covers: RRF fusion, scope filter (per-user isolation), BM25 scoring,
and the Groq-safe history trimmer.
"""
from qdrant_client.models import FieldCondition, Filter, MatchAny

from backend.rag_graph import _fit_history, _history_units
from backend.vector_store import _bm25_scores, _rrf_fuse, _scope_filter, _tokenize


# ── RRF fusion ───────────────────────────────────────────────────────────────

def test_rrf_doc_in_both_lists_ranks_first():
    fused = _rrf_fuse([["a", "b", "c"], ["b", "c"]])
    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]


def test_rrf_single_list_is_rank_order():
    fused = _rrf_fuse([["x", "y"]])
    assert fused["x"] > fused["y"]


# ── Scope filter: per-user isolation at the Qdrant layer ─────────────────────

def _must_keys(fltr):
    return [c.key for c in fltr.must]


def test_filter_always_scopes_by_session():
    f = _scope_filter(["sess-1"], None, None)
    assert "metadata.session_id" in _must_keys(f)
    assert any(
        c.key == "metadata.session_id"
        and isinstance(c.match, MatchAny)
        and c.match.any == ["sess-1"]
        for c in f.must
    )


def test_filter_adds_user_id_when_given():
    f = _scope_filter(["sess-1"], None, "lib-abc")
    assert "metadata.user_id" in _must_keys(f)


def test_filter_without_user_id_has_no_user_condition():
    f = _scope_filter(["sess-1"], None, None)
    assert "metadata.user_id" not in _must_keys(f)


def test_filter_titles_become_match_any():
    f = _scope_filter(["sess-1"], ["A", "B"], "lib-abc")
    title_conditions = [
        c for c in f.must
        if c.key == "metadata.title" and isinstance(c.match, MatchAny)
    ]
    assert len(title_conditions) == 1
    assert sorted(title_conditions[0].match.any) == ["A", "B"]


# ── BM25 ─────────────────────────────────────────────────────────────────────

def test_bm25_prefers_doc_with_query_terms():
    q = _tokenize("runtime improvement nodes")
    docs = [
        _tokenize("MOCAA* achieves 35-40% runtime improvement over classical A*"),
        _tokenize("The museum exhibits contemporary art and film"),
    ]
    scores = _bm25_scores(q, docs)
    assert scores[0] > scores[1]


def test_bm25_no_query_tokens_all_zero():
    scores = _bm25_scores([], [[_tokenize("anything")[0]]])
    assert all(s == 0 for s in scores)


# ── History trimmer (Groq-safe) ──────────────────────────────────────────────

def test_fit_history_short_history_unchanged():
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert _fit_history(msgs, 12000) == msgs


def test_fit_history_trims_oldest_first():
    big = [{"role": "user", "content": "x" * 900}, {"role": "assistant", "content": "y" * 900}]
    small = [{"role": "user", "content": "current question"}]
    out = _fit_history(big + small, 1000)
    # newest message always survives, oldest (the "xxxx" turn) is dropped
    assert out[-1] is small[0]
    assert out[0]["role"] == "system" and "omitted" in out[0]["content"]
    assert not any("xxxx" in (m.get("content") or "") for m in out)


def test_history_units_keep_tool_pair_together():
    call = {"role": "assistant", "tool_calls": [{"id": "t1"}]}
    tool = {"role": "tool", "content": "result"}
    units = _history_units([call, tool])
    assert len(units) == 1 and len(units[0]) == 2


def test_fit_history_never_splits_tool_pair():
    filler = [{"role": "user", "content": "f" * 800}]
    call = {"role": "assistant", "tool_calls": [{"id": "t1"}]}
    tool = {"role": "tool", "content": "r" * 50}
    out = _fit_history(filler + [call, tool], 300)
    roles = [m.get("role") for m in out]
    assert "tool_calls" in out[-2] or out[-2].get("tool_calls")
    assert roles[-1] == "tool"

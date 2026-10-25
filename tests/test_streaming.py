from langchain_core.documents import Document

from api.routers.chat import _sse
from backend.rag_graph import _buffer_stream
from backend.streaming import emit, reset_emitter, set_emitter, streaming_active


def _doc(title="Paper", page=0):
    return Document(page_content="x", metadata={"title": title, "page": page})


def _state(query="Which activation does the paper use?", docs=None, **extra):
    state = {"query": query, "retrieved_docs": docs or [_doc()]}
    state.update(extra)
    return state


def test_simple_single_doc_question_streams():
    assert _buffer_stream(_state()) is False


def test_premise_refuted_is_buffered():
    assert _buffer_stream(_state(premise_refuted=True)) is True


def test_multiple_documents_are_buffered():
    docs = [_doc("A"), _doc("B")]
    assert _buffer_stream(_state(docs=docs)) is True


def test_many_docs_are_buffered():
    docs = [_doc() for _ in range(7)]
    assert _buffer_stream(_state(docs=docs)) is True


def test_numeric_questions_are_buffered():
    assert _buffer_stream(_state(query="How many encoder layers does the paper use?")) is True


def test_why_questions_are_buffered():
    assert _buffer_stream(_state(query="Why does the paper use ReLU?")) is True


def test_original_query_wins_for_buffer_policy():
    state = _state(query="rewritten phrasing", original_query="How many heads?")
    assert _buffer_stream(state) is True


def test_emitter_roundtrip_and_noop():
    assert streaming_active() is False
    emit({"type": "delta", "text": "ignored"})  # no emitter -> no crash
    seen = []
    token = set_emitter(seen.append)
    try:
        assert streaming_active() is True
        emit({"type": "delta", "text": "hi"})
    finally:
        reset_emitter(token)
    assert seen == [{"type": "delta", "text": "hi"}]
    assert streaming_active() is False


def test_sse_frame_shape():
    frame = _sse({"type": "delta", "text": "hello"})
    assert frame.startswith("event: delta\n")
    assert 'data: {"type": "delta", "text": "hello"}' in frame
    assert frame.endswith("\n\n")

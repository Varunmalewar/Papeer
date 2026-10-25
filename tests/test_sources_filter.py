from api.deps import build_sources_list
from langchain_core.documents import Document


def _doc(title, page, content="x"):
    return Document(page_content=content, metadata={"title": title, "page": page})


def test_sources_only_cited_pages():
    docs = [_doc("T", 4), _doc("T", 1), _doc("T", 7)]
    answer = "mechanism [T, Page 5]."
    out = build_sources_list({"answer": answer, "retrieved_docs": docs})
    assert out == [{"title": "T", "page": 5}]


def test_sources_fallback_when_no_citations():
    docs = [_doc("T", 4), _doc("T", 4), _doc("W", None, "y")]
    out = build_sources_list({"answer": "no citations here", "retrieved_docs": docs})
    # deduped unfiltered
    assert {"title": "T", "page": 5} in out
    assert len([e for e in out if e.get("page") == 5]) == 1


def test_sources_refusal_after_changes_returns_empty():
    docs = [_doc("T", 0), _doc("T", 8)]
    assert build_sources_list({"answer": "I wasn't able to find relevant information in the uploaded papers. Try rephrasing.", "retrieved_docs": docs}) == []
    assert build_sources_list({"answer": "I don't know the answer.", "retrieved_docs": docs}) == []
    assert build_sources_list({"answer": "The paper does not state this.", "retrieved_docs": docs}) == []

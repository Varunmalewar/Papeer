import io

import fitz
import pytest
from langchain_core.documents import Document
from PIL import Image

from api.deps import build_sources_list
from backend.blob_store import (
    LocalBlobStore,
    delete_paper_files,
    figure_key,
    paper_id_for,
    pdf_key,
    validate_key,
)
from backend.figure_extractor import (
    FigureIngestContext,
    extract_figures,
    ingest_figure_docs,
)
from backend.rag_graph import (
    _FIGURE_CITATION_RE,
    _format_context_with_citations,
    _strip_bad_citations,
)


def _png(width: int, height: int, colour=(120, 80, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buf, format="PNG")
    return buf.getvalue()


def _make_pdf(path, with_caption=True, image_size=(420, 300)):
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    page.insert_image(fitz.Rect(60, 120, 60 + image_size[0], 120 + image_size[1]), stream=_png(*image_size))
    if with_caption:
        page.insert_text((60, 460), "Figure 1: Overview of the proposed pipeline.", fontsize=11)
    page.insert_text((60, 500), "The model is described in Section 3.", fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


def test_extract_figures_caption_and_page(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf")
    records = extract_figures(pdf)
    assert len(records) == 1
    rec = records[0]
    assert rec.number == 1
    assert "proposed pipeline" in rec.caption
    assert rec.page == 0
    assert rec.image_bytes


def test_decorative_small_image_is_skipped(tmp_path):
    pdf = _make_pdf(tmp_path / "small.pdf", image_size=(60, 40))
    assert extract_figures(pdf) == []


def test_no_caption_means_no_figure(tmp_path):
    pdf = _make_pdf(tmp_path / "plain.pdf", with_caption=False)
    assert extract_figures(pdf) == []


def test_ingest_figure_docs_persists_image_and_metadata(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf")
    store = LocalBlobStore(tmp_path / "figures")
    ctx = FigureIngestContext(user_id="user1", paper_id=paper_id_for("Paper"), store=store)
    docs = ingest_figure_docs(pdf, "Paper", ctx)
    assert len(docs) == 1
    md = docs[0].metadata
    assert md["type"] == "figure"
    assert md["figure_number"] == 1
    assert md["title"] == "Paper"
    assert md["page"] == 0
    assert store.exists(md["image_path"])
    assert store.get(md["image_path"]).startswith(b"\x89PNG")
    assert "Figure 1:" in docs[0].page_content
    assert md["parent_id"] == md["figure_id"]


def test_figure_context_header_and_strip(tmp_path):
    pdf = _make_pdf(tmp_path / "paper.pdf")
    store = LocalBlobStore(tmp_path / "figures")
    ctx = FigureIngestContext(user_id="user1", paper_id=paper_id_for("Paper"), store=store)
    docs = ingest_figure_docs(pdf, "Paper", ctx)
    assert _format_context_with_citations(docs).startswith("[Paper, Figure 1, Page 1]")
    answer = "See the diagram [Paper, Figure 1, Page 1] and [Paper, Figure 9, Page 1]."
    stripped = _strip_bad_citations(answer, docs)
    assert "[Paper, Figure 1, Page 1]" in stripped
    assert "Figure 9" not in stripped


def test_figure_citation_regex_ignores_plain_page_citation():
    text = "as shown [Paper, Page 3] and [Paper, Figure 2, Page 3]"
    assert [m.group("figure") for m in _FIGURE_CITATION_RE.finditer(text)] == ["2"]


def test_sources_include_figure_image_url():
    fig = Document(
        page_content="Figure 2: plot",
        metadata={
            "type": "figure",
            "figure_id": "pid-f2-abcd1234",
            "figure_number": 2,
            "title": "Paper",
            "page": 4,
        },
    )
    text = Document(page_content="body", metadata={"title": "Paper", "page": 4})
    values = {
        "answer": "Trend rises [Paper, Figure 2, Page 5].",
        "retrieved_docs": [text, fig],
        "session_id": "lib-1",
    }
    out = build_sources_list(values)
    assert out == [
        {"title": "Paper", "page": 5, "figure": 2, "image_url": "/api/figure/lib-1/pid-f2-abcd1234"}
    ]


def test_blob_store_rejects_traversal(tmp_path):
    store = LocalBlobStore(tmp_path / "blobs")
    with pytest.raises(ValueError):
        store.save("../escape.txt", b"x")
    with pytest.raises(ValueError):
        validate_key("/abs/path.png")


def test_delete_paper_files_cascades(tmp_path, monkeypatch):
    figs = LocalBlobStore(tmp_path / "figures")
    pdfs = LocalBlobStore(tmp_path / "pdfs")
    pid = paper_id_for("Paper")
    figs.save(figure_key("u1", pid, "f1", "png"), b"img")
    pdfs.save(pdf_key("u1", pid), b"%PDF")
    monkeypatch.setattr("backend.blob_store.get_figure_store", lambda: figs)
    monkeypatch.setattr("backend.blob_store.get_pdf_store", lambda: pdfs)
    assert delete_paper_files("u1", pid) == 2
    assert not figs.exists(figure_key("u1", pid, "f1", "png"))
    assert not pdfs.exists(pdf_key("u1", pid))

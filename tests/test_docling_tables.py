"""Hermetic tests for the optional Docling table pass (Docling itself is never run)."""
import fitz
from langchain_core.documents import Document

import backend.paper_loader as pl


def _pdf(path, page_texts):
    doc = fitz.open()
    for text in page_texts:
        doc.new_page().insert_text((72, 100), text)
    doc.save(str(path))
    doc.close()
    return str(path)


def _raw(n):
    return [Document(page_content=f"flat {i}", metadata={"page": i}) for i in range(n)]


def test_table_pages_finds_only_captioned_pages(tmp_path):
    path = _pdf(tmp_path / "p.pdf", ["Introduction text", "Table 1: Results", "Conclusion"])
    assert pl._table_pages(path) == [2]


def test_table_pass_replaces_only_table_pages(monkeypatch):
    monkeypatch.setattr(pl, "_docling_enabled", lambda: True)
    monkeypatch.setattr(pl, "_table_pages", lambda p: [2])
    monkeypatch.setattr(pl, "_docling_pages", lambda p, pages: {2: "| a | b |"})
    docs = _raw(3)
    pl._use_docling_tables("x.pdf", docs)
    assert [d.page_content for d in docs] == ["flat 0", "| a | b |", "flat 2"]


def test_switch_off_leaves_text_untouched(monkeypatch):
    monkeypatch.setenv("PAPEER_DOCLING", "0")
    assert not pl._docling_enabled()
    monkeypatch.setattr(pl, "_table_pages", lambda p: pytest_fail())
    docs = _raw(2)
    pl._use_docling_tables("x.pdf", docs)
    assert [d.page_content for d in docs] == ["flat 0", "flat 1"]


def test_failure_falls_back_to_plain_text(monkeypatch):
    def boom(path):
        raise RuntimeError("model download failed")

    monkeypatch.setattr(pl, "_docling_enabled", lambda: True)
    monkeypatch.setattr(pl, "_table_pages", boom)
    docs = _raw(2)
    pl._use_docling_tables("x.pdf", docs)
    assert [d.page_content for d in docs] == ["flat 0", "flat 1"]


def pytest_fail():
    raise AssertionError("table detection must not run when Docling is off")

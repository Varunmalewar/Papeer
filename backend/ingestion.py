"""Ingestion orchestration used by the HTTP API.

One place owns: stable paper ids, source-PDF persistence, figure extraction
context, and the per-paper file cascade. Text chunking and embeddings remain in
paper_loader / vector_store and are not modified here.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document

from backend.blob_store import (
    delete_paper_files,
    get_figure_store,
    get_pdf_store,
    paper_id_for,
    pdf_key,
)
from backend.paper_loader import load_arxiv, load_document, load_pdf

load_dotenv(override=True)


def _pdf_persistence_enabled() -> bool:
    return os.environ.get("PAPEER_PERSIST_PDFS", "1").strip() != "0"


def _figure_context(title: str, user_id: str):
    from backend.figure_extractor import (
        DescriptionCache,
        FigureIngestContext,
        descriptions_enabled,
        figures_enabled,
        vision_model_name,
    )

    if not figures_enabled():
        return None
    describe = descriptions_enabled()
    return FigureIngestContext(
        user_id=user_id,
        paper_id=paper_id_for(title),
        store=get_figure_store(),
        describe=describe,
        vision_model=vision_model_name() if describe else "",
        cache=DescriptionCache() if describe else None,
    )


def persist_pdf(data: bytes, title: str, user_id: str) -> str | None:
    if not _pdf_persistence_enabled() or not data:
        return None
    key = pdf_key(user_id, paper_id_for(title))
    get_pdf_store().save(key, data)
    return key


def ingest_pdf_bytes(
    data: bytes,
    filename: str,
    session_id: str,
    user_id: str,
    title: str | None = None,
) -> list[Document]:
    """Persist the PDF, extract text + figures, and index everything."""
    from backend.vector_store import add_paper

    resolved_title = title or Path(filename).stem
    persist_pdf(data, resolved_title, user_id)
    suffix = Path(filename).suffix.lower() or ".pdf"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        docs = load_pdf(tmp_path, title=resolved_title, figures_ctx=_figure_context(resolved_title, user_id))
        add_paper(docs, session_id, user_id=user_id)
        return docs
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def ingest_arxiv(query: str, session_id: str, user_id: str) -> list[Document]:
    from backend.vector_store import add_paper

    def _persist(pdf_bytes: bytes, title: str) -> None:
        persist_pdf(pdf_bytes, title, user_id)

    docs = load_arxiv(
        query,
        on_pdf=_persist,
        figures_ctx_factory=lambda resolved: _figure_context(resolved, user_id),
    )
    add_paper(docs, session_id, user_id=user_id)
    return docs


def ingest_webpage(url: str, session_id: str, user_id: str) -> list[Document]:
    from backend.vector_store import add_paper

    docs = load_document(url)
    add_paper(docs, session_id, user_id=user_id)
    return docs


def delete_paper_assets(title: str, user_id: str) -> int:
    """Cascade for delete-paper: figure images + persisted source PDF."""
    return delete_paper_files(user_id, paper_id_for(title))

import importlib.util
import os
import re
import tempfile
import threading
import urllib.request
from functools import lru_cache
from pathlib import Path

import feedparser
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.text_clean import clean_pages

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)


# ── Optional Docling pass for table pages ────────────────────────────────────
# PyMuPDF flattens tables to one cell per line, so columns are lost. Docling
# rebuilds them as markdown but costs ~9 s/page and ~1.5 GB RAM, so it only runs on
# pages PyMuPDF flags as table pages. Not installed or PAPEER_DOCLING=0 -> plain text.

_TABLE_CAPTION_RE = re.compile(r"^\s*Table\s+\d+\s*[:.]", re.MULTILINE | re.IGNORECASE)
_docling_lock = threading.Lock()  # one conversion at a time; the converter is not thread-safe


def _docling_enabled() -> bool:
    return (
        os.environ.get("PAPEER_DOCLING", "1").strip() != "0"
        and importlib.util.find_spec("docling") is not None
    )


def _table_pages(file_path: str) -> list[int]:
    """1-based pages with a 'Table N:' caption or a ruled table (PyMuPDF only, fast)."""
    import fitz

    pages = []
    with fitz.open(file_path) as pdf:
        for number, page in enumerate(pdf, 1):
            if _TABLE_CAPTION_RE.search(page.get_text()) or page.find_tables(strategy="lines").tables:
                pages.append(number)
    return pages


@lru_cache(maxsize=1)
def _docling_converter():
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.document_converter import DocumentConverter, PdfFormatOption

    opts = PdfPipelineOptions()
    opts.do_ocr = False  # research PDFs already carry text; OCR is the slow part
    opts.do_table_structure = True
    opts.table_structure_options.mode = TableFormerMode.FAST
    return DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})


def _docling_pages(file_path: str, pages: list[int]) -> dict[int, str]:
    out: dict[int, str] = {}
    for number in pages:
        try:
            with _docling_lock:
                result = _docling_converter().convert(file_path, page_range=(number, number))
            md = re.sub(r"<!--\s*image\s*-->", "", result.document.export_to_markdown()).strip()
            if md:
                out[number] = md
        except Exception as e:  # one bad page must not break ingestion
            print(f"[docling] page {number} skipped: {e}")
    return out


def _use_docling_tables(file_path: str, raw_docs: list[Document]) -> None:
    """Replace the text of table pages with Docling markdown, in place."""
    if not _docling_enabled():
        return
    try:
        by_page = _docling_pages(file_path, _table_pages(file_path))
    except Exception as e:  # never break text ingestion
        print(f"[docling] table pass skipped: {e}")
        return
    for doc in raw_docs:
        md = by_page.get(int(doc.metadata.get("page", 0)) + 1)
        if md:
            doc.page_content = md


def _chunk_docs(
    raw_docs: list[Document], title: str, md: bool = False
) -> list[Document]:
    """Split each raw doc into CHUNK_SIZE chunks. The page number comes from the
    raw doc, so citations keep working."""
    splitter = _md_splitter if md else _splitter
    out: list[Document] = []
    for doc in raw_docs:
        text = doc.page_content or ""
        if not text.strip():
            continue
        page = doc.metadata.get("page", 0)
        base = {k: v for k, v in doc.metadata.items() if k != "page"}
        for piece in splitter.split_text(text):
            out.append(Document(
                page_content=piece,
                metadata={**base, "title": title, "page": page},
            ))
    return out


def load_pdf(
    file_path: str,
    title: str | None = None,
    figures_ctx=None,
) -> list[Document]:
    raw_docs = PyMuPDFLoader(file_path).load()
    _use_docling_tables(file_path, raw_docs)
    resolved_title = title or Path(file_path).stem
    # P0: clean per-page Documents before chunking; pages are never concatenated
    # so every surviving chunk keeps its original page metadata.
    pages = [
        (int(doc.metadata.get("page", 0)), doc.page_content or "")
        for doc in raw_docs
    ]
    for doc, (_page, text) in zip(raw_docs, clean_pages(pages)):
        doc.page_content = text
    docs = _chunk_docs(raw_docs, resolved_title)
    if figures_ctx is not None:
        try:
            from backend.figure_extractor import figures_enabled, ingest_figure_docs

            if figures_enabled():
                docs += ingest_figure_docs(file_path, resolved_title, figures_ctx)
        except Exception as e:  # figures must never break text ingestion
            print(f"[figures] extraction skipped for '{resolved_title}': {e}")
    return docs


def load_text(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _chunk_docs(docs, Path(file_path).stem)


def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _chunk_docs(docs, Path(file_path).stem, md=True)


def load_webpage(url: str) -> list[Document]:
    docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (docs[0].metadata.get("title") or url) if docs else url
    return _chunk_docs(docs, title)


def _extract_arxiv_id(query: str) -> str | None:
    """Return bare ArXiv ID (no version suffix) if one appears in the query."""
    m = _ARXIV_ID_RE.search(query)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    return None


def _arxiv_api_lookup(arxiv_id: str) -> str:
    """Fetch paper title by ID from the ArXiv Atom API using feedparser."""
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    feed = feedparser.parse(url)
    if feed.entries:
        return feed.entries[0].get("title", arxiv_id).strip()
    return arxiv_id


def _arxiv_search(query: str) -> str:
    """Search ArXiv by title phrase and return the top result's bare paper ID."""
    phrase = query.strip('"')
    import urllib.parse
    search_query = urllib.parse.quote(f'ti:"{phrase}"')
    url = f"https://export.arxiv.org/api/query?search_query={search_query}&max_results=1&sortBy=relevance"
    feed = feedparser.parse(url)
    if not feed.entries:
        raise ValueError(f"No ArXiv paper found for: {query}")
    # Entry id is like http://arxiv.org/abs/1706.03762v5
    entry_id = feed.entries[0].get("id", "")
    m = _ARXIV_ID_RE.search(entry_id)
    if not m:
        raise ValueError(f"No ArXiv paper found for: {query}")
    return re.sub(r"v\d+$", "", m.group(1))


def _load_arxiv_by_id(
    arxiv_id: str,
    figures_ctx=None,
    on_pdf=None,
    figures_ctx_factory=None,
) -> list[Document]:
    """Download and chunk an ArXiv paper PDF by its bare ID.

    ``figures_ctx`` enables figure extraction; ``figures_ctx_factory(title)``
    builds that context once the real title is known; ``on_pdf(pdf_bytes,
    title)`` lets callers persist the source PDF.
    """
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    with urllib.request.urlopen(pdf_url, timeout=60) as resp:
        pdf_bytes = resp.read()
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        raw_docs = PyMuPDFLoader(tmp_path).load()
        if not raw_docs:
            raise ValueError(f"Could not load PDF for ArXiv ID: {arxiv_id}")
        title = (raw_docs[0].metadata.get("title") or "").strip() or _arxiv_api_lookup(arxiv_id)
        if on_pdf is not None:
            try:
                on_pdf(pdf_bytes, title)
            except Exception as e:
                print(f"[ingest] could not persist arXiv PDF '{title}': {e}")
        ctx = figures_ctx if figures_ctx is not None else (
            figures_ctx_factory(title) if figures_ctx_factory is not None else None
        )
        return load_pdf(tmp_path, title=title, figures_ctx=ctx)
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


def load_arxiv(query: str, figures_ctx=None, on_pdf=None, figures_ctx_factory=None) -> list[Document]:
    arxiv_id = _extract_arxiv_id(query) or _arxiv_search(query)
    return _load_arxiv_by_id(
        arxiv_id,
        figures_ctx=figures_ctx,
        on_pdf=on_pdf,
        figures_ctx_factory=figures_ctx_factory,
    )


def load_document(source: str) -> list[Document]:
    """Dispatch to the appropriate loader based on URL prefix or file extension."""
    if source.startswith(("http://", "https://")):
        return load_webpage(source)
    ext = Path(source).suffix.lower()
    if ext == ".pdf":
        return load_pdf(source)
    if ext == ".txt":
        return load_text(source)
    if ext in (".md", ".markdown"):
        return load_markdown(source)
    raise ValueError(f"Unsupported file type: {ext}")

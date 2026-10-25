import hashlib
import os
import re
import tempfile
import urllib.request
from pathlib import Path

import feedparser
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader, WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_ARXIV_ID_RE = re.compile(r"(\d{4}\.\d{4,5}(?:v\d+)?)")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)
_md_splitter = RecursiveCharacterTextSplitter.from_language(
    "markdown", chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
)

# ── Parent-document (small-to-big) chunking ───────────────────────────────────
# Children (small chunks) are embedded and searched; the section-level parent
# block each child belongs to is stored in metadata and expanded at retrieval
# time, so tables/abstracts never lose their surrounding context.

_PARENT_CHARS = int(os.environ.get("PAPEER_PARENT_CHARS", "2400"))
_TABLE_CHARS = int(os.environ.get("PAPEER_TABLE_CHARS", "6000"))


def _looks_like_table(para: str) -> bool:
    """Heuristic: several lines with ≥4 digit chars, tabs, or wide gaps —
    typical of an extracted results table."""
    lines = [l for l in para.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    hits = sum(
        1
        for l in lines
        if len(re.findall(r"\d", l)) >= 4 or l.count("\t") >= 2 or "   " in l
    )
    return hits >= max(2, len(lines) // 2)


def _parent_blocks(text: str) -> list[str]:
    """Split a raw document's text into ~_PARENT_CHARS parent blocks on
    paragraph boundaries. Very long single paragraphs (common in PDFs) are
    hard-sliced at the cap so a block never grows unbounded — EXCEPT
    table-like blocks, which get a bigger cap so a table stays one chunk."""
    paras = [p.strip("\n") for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    blocks: list[str] = []
    cur: list[str] = []
    size = 0
    for para in paras:
        cap = _TABLE_CHARS if _looks_like_table(para) else _PARENT_CHARS
        while len(para) > cap:
            blocks.append(para[:cap])
            para = para[cap:]
        if size + len(para) > _PARENT_CHARS and cur:
            blocks.append("\n\n".join(cur))
            cur, size = [], 0
        cur.append(para)
        size += len(para) + 2
    if cur:
        blocks.append("\n\n".join(cur))
    return blocks


def _small_to_big(
    raw_docs: list[Document], title: str, md: bool = False
) -> list[Document]:
    """Chunk each raw doc into child chunks and stamp every child with the
    parent block's text + id. The page number comes from the raw doc, so
    citations keep working."""
    splitter = _md_splitter if md else _splitter
    out: list[Document] = []
    for doc in raw_docs:
        text = doc.page_content or ""
        if not text.strip():
            continue
        page = doc.metadata.get("page", 0)
        base = {k: v for k, v in doc.metadata.items() if k != "page"}
        for bi, block in enumerate(_parent_blocks(text)):
            pid = hashlib.md5(f"{title}|{page}|{bi}".encode("utf-8")).hexdigest()
            for piece in splitter.split_text(block):
                out.append(Document(
                    page_content=piece,
                    metadata={
                        **base,
                        "title": title,
                        "page": page,
                        "parent_id": pid,
                        "parent_text": block,
                    },
                ))
    return out


def load_pdf(
    file_path: str,
    title: str | None = None,
    figures_ctx=None,
) -> list[Document]:
    raw_docs = PyMuPDFLoader(file_path).load()
    resolved_title = title or Path(file_path).stem
    docs = _small_to_big(raw_docs, resolved_title)
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
    return _small_to_big(docs, Path(file_path).stem)


def load_markdown(file_path: str) -> list[Document]:
    docs = TextLoader(file_path, encoding="utf-8").load()
    return _small_to_big(docs, Path(file_path).stem, md=True)


def load_webpage(url: str) -> list[Document]:
    docs = WebBaseLoader(url, requests_kwargs={"timeout": 30}).load()
    title = (docs[0].metadata.get("title") or url) if docs else url
    return _small_to_big(docs, title)


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

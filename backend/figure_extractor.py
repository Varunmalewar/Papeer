"""Figure extraction for PDF ingestion.

Stage 1 (always on): every meaningful figure becomes a chunk whose text is the
caption plus the sentences that reference it, with its image saved to blob
storage. Stage 2 (PAPEER_FIGURE_DESCRIPTIONS=1): a vision model appends a
one-paragraph visual description at ingestion time; results are cached by image
hash so re-ingesting never calls the model twice.
"""
from __future__ import annotations

import base64
import hashlib
import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_core.documents import Document

from backend.blob_store import BlobStore, figure_key

load_dotenv(override=True)

_CAPTION_RE = re.compile(r"^\s*(?:Figure|Fig\.?)\s+(\d+)\s*[:.]\s*(.+)$", re.IGNORECASE | re.DOTALL)
_SECTION_RE = re.compile(r"^\s*(?:\d+(?:\.\d+)*)\s+[A-Z][A-Za-z].{0,70}$")
_KNOWN_SECTIONS = (
    "abstract", "introduction", "related work", "background", "method", "methods",
    "methodology", "approach", "experiments", "experimental setup", "results",
    "discussion", "analysis", "conclusion", "conclusions", "appendix", "references",
)

_MIN_W = int(os.environ.get("PAPEER_FIG_MIN_W", "200"))
_MIN_H = int(os.environ.get("PAPEER_FIG_MIN_H", "120"))
_MIN_AREA_FRAC = float(os.environ.get("PAPEER_FIG_MIN_AREA_FRAC", "0.02"))
_REF_CHARS = int(os.environ.get("PAPEER_FIG_REF_CHARS", "600"))

_MIME = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "webp": "image/webp",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "tiff": "image/tiff",
}


def figures_enabled() -> bool:
    return os.environ.get("PAPEER_FIGURES", "1").strip() != "0"


def descriptions_enabled() -> bool:
    return os.environ.get("PAPEER_FIGURE_DESCRIPTIONS", "0").strip() == "1"


def vision_model_name() -> str:
    return os.environ.get("PAPEER_VISION_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"


@dataclass
class FigureRecord:
    number: int
    caption: str
    page: int  # 0-based, same convention as text chunks
    image_bytes: bytes
    ext: str
    section: str
    refs: str
    sha256: str


@dataclass
class FigureIngestContext:
    user_id: str
    paper_id: str
    store: BlobStore
    describe: bool = False
    vision_model: str = "gpt-4o-mini"
    cache: "DescriptionCache | None" = None


# ── geometry helpers ─────────────────────────────────────────────────────────

def _text_blocks(page) -> list[dict]:
    data = page.get_text("dict")
    return [b for b in data.get("blocks", []) if b.get("type") == 0]


def _block_text(block: dict) -> str:
    lines = []
    for line in block.get("lines", []):
        lines.append("".join(span.get("text", "") for span in line.get("spans", [])))
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _caption_candidates(page) -> list[tuple[fitz.Rect, int, str]]:
    out = []
    for block in _text_blocks(page):
        text = _block_text(block)
        m = _CAPTION_RE.match(text)
        if not m:
            continue
        caption = re.sub(r"\s+", " ", m.group(2)).strip()
        if len(caption) < 8:
            continue
        out.append((fitz.Rect(block["bbox"]), int(m.group(1)), caption))
    return out


def _caption_for_rect(rect: fitz.Rect, captions: list[tuple[fitz.Rect, int, str]]):
    """Nearest caption block below the image (small fallback window above)."""
    best = None
    for cap_rect, number, caption in captions:
        h_overlap = min(rect.x1, cap_rect.x1) - max(rect.x0, cap_rect.x0)
        if h_overlap <= 0:
            continue
        below = cap_rect.y0 - rect.y1
        above = rect.y0 - cap_rect.y1
        if -8 <= below <= 260:
            score = below
        elif -8 <= above <= 140:
            score = above + 40  # prefer captions below
        else:
            continue
        if best is None or score < best[0]:
            best = (score, number, caption)
    return (best[1], best[2]) if best else (None, None)


def _nearest_section(page, caption_rect: fitz.Rect) -> str:
    candidates = []
    for block in _text_blocks(page):
        rect = fitz.Rect(block["bbox"])
        if rect.y1 > caption_rect.y0 + 2:
            continue
        text = _block_text(block)
        if not text or len(text) > 90:
            continue
        low = text.lower().strip()
        if low in _KNOWN_SECTIONS or _SECTION_RE.match(text):
            candidates.append((rect.y0, text))
    if not candidates:
        return ""
    return max(candidates, key=lambda c: c[0])[1]


def _reference_sentences(doc: "fitz.Document", page_no: int, number: int) -> str:
    pattern = re.compile(rf"\bFig(?:ure)?\.?\s*{number}\b", re.IGNORECASE)
    found: list[str] = []
    for pno in (page_no - 1, page_no, page_no + 1):
        if not 0 <= pno < doc.page_count:
            continue
        text = re.sub(r"\s+", " ", doc[pno].get_text("text") or "")
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if pattern.search(sentence):
                found.append(sentence.strip())
        if found:
            break
    return " ".join(found)[: _REF_CHARS].strip()


# ── extraction ───────────────────────────────────────────────────────────────

def extract_figures(
    pdf_path: str | os.PathLike,
    min_w: int | None = None,
    min_h: int | None = None,
    min_area_frac: float | None = None,
) -> list[FigureRecord]:
    """Meaningful captioned figures only: size + area filter, hash dedupe,
    one record per figure number."""
    min_w = _MIN_W if min_w is None else min_w
    min_h = _MIN_H if min_h is None else min_h
    min_area_frac = _MIN_AREA_FRAC if min_area_frac is None else min_area_frac

    records: list[FigureRecord] = []
    seen_hashes: set[str] = set()
    seen_numbers: set[int] = set()

    with fitz.open(str(pdf_path)) as doc:
        for page_no, page in enumerate(doc):
            page_area = float(page.rect.width * page.rect.height) or 1.0
            captions = _caption_candidates(page)
            if not captions:
                continue
            for img in page.get_images(full=True):
                xref = img[0]
                width, height = int(img[2] or 0), int(img[3] or 0)
                if width < min_w or height < min_h:
                    continue  # decorative / icon-sized
                rects = page.get_image_rects(xref)
                if not rects:
                    continue
                rect = max(rects, key=lambda r: r.get_area())
                if float(rect.get_area()) < min_area_frac * page_area:
                    continue  # tiny placement on the page
                number, caption = _caption_for_rect(rect, captions)
                if number is None or number in seen_numbers:
                    continue
                info = doc.extract_image(xref)
                data = info.get("image") or b""
                if not data:
                    continue
                sha = hashlib.sha256(data).hexdigest()
                if sha in seen_hashes:
                    continue  # repeated logo / header image
                seen_hashes.add(sha)
                seen_numbers.add(number)
                records.append(FigureRecord(
                    number=number,
                    caption=caption,
                    page=page_no,
                    image_bytes=data,
                    ext=(info.get("ext") or "png").lower(),
                    section=_nearest_section(page, fitz.Rect(rect)),
                    refs=_reference_sentences(doc, page_no, number),
                    sha256=sha,
                ))
    return records


def figure_chunk_text(record: FigureRecord, description: str = "") -> str:
    parts = [f"Figure {record.number}: {record.caption}"]
    if record.section:
        parts.append(f"Section: {record.section}")
    if record.refs:
        parts.append(f"Referenced in text: {record.refs}")
    if description:
        parts.append(f"Visual description: {description}")
    return "\n\n".join(parts)


# ── stage 2: vision descriptions (cached by image hash) ─────────────────────

_DESCRIBE_PROMPT = (
    "Describe this figure from a research paper so that a text-only retrieval "
    "system can match questions to it. State the plot/diagram type, axes or "
    "labels, and the key trend or comparison shown. Use 2-3 sentences. Only "
    "describe what is visible; do not speculate about the paper's claims."
)


class DescriptionCache:
    """SQLite cache: one vision call per (image hash, model), ever."""

    def __init__(self, path: str | os.PathLike | None = None) -> None:
        self.path = str(path or os.environ.get("PAPEER_FIGURE_CACHE_DB", "./data/figure_descriptions.sqlite"))
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS figure_descriptions ("
                "sha256 TEXT NOT NULL, model TEXT NOT NULL, description TEXT NOT NULL, "
                "PRIMARY KEY (sha256, model))"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def get(self, sha256: str, model: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT description FROM figure_descriptions WHERE sha256=? AND model=?",
                (sha256, model),
            ).fetchone()
        return row[0] if row else None

    def put(self, sha256: str, model: str, description: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO figure_descriptions (sha256, model, description) VALUES (?,?,?)",
                (sha256, model, description),
            )


def describe_image(image_bytes: bytes, ext: str, caption: str, model: str) -> str:
    """One vision call (ingestion time only, never at query time)."""
    from langchain_openai import ChatOpenAI

    mime = _MIME.get((ext or "png").lower(), "image/png")
    b64 = base64.b64encode(image_bytes).decode("ascii")
    llm = ChatOpenAI(
        model=model,
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0,
        max_tokens=220,
    )
    message = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Caption: {caption}\n\n{_DESCRIBE_PROMPT}"},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        ],
    }
    content = llm.invoke([message]).content
    return re.sub(r"\s+", " ", str(content)).strip()


def ingest_figure_docs(
    pdf_path: str | os.PathLike,
    title: str,
    ctx: FigureIngestContext,
) -> list[Document]:
    """Extract figures, persist images, and return index-ready Documents."""
    docs: list[Document] = []
    for record in extract_figures(pdf_path):
        figure_id = f"{ctx.paper_id}-f{record.number}-{record.sha256[:8]}"
        key = figure_key(ctx.user_id, ctx.paper_id, figure_id, record.ext)
        try:
            ctx.store.save(key, record.image_bytes)
        except Exception as e:
            print(f"[figures] could not persist {figure_id}: {e}")
            continue
        description = ""
        if ctx.describe:
            model = ctx.vision_model or vision_model_name()
            cache = ctx.cache
            description = (cache.get(record.sha256, model) if cache else None) or ""
            if not description:
                try:
                    description = describe_image(record.image_bytes, record.ext, record.caption, model)
                    if cache and description:
                        cache.put(record.sha256, model, description)
                except Exception as e:
                    print(f"[figures] vision description failed for {figure_id}: {e}")
        text = figure_chunk_text(record, description)
        docs.append(Document(
            page_content=text,
            metadata={
                "type": "figure",
                "figure_id": figure_id,
                "figure_number": record.number,
                "paper_id": ctx.paper_id,
                "title": title,
                "page": record.page,
                "section": record.section,
                "caption": record.caption,
                "image_path": key,
                "image_ext": record.ext,
                "image_sha256": record.sha256,
                "parent_id": figure_id,
                "parent_text": text,
            },
        ))
    return docs


def figure_document_for_point(payload: dict) -> dict | None:
    """Normalise a Qdrant payload into the fields the API needs."""
    md = (payload or {}).get("metadata") or {}
    if md.get("type") != "figure" or not md.get("figure_id"):
        return None
    return {
        "figure_id": md.get("figure_id"),
        "figure_number": md.get("figure_number"),
        "paper_id": md.get("paper_id"),
        "title": md.get("title"),
        "page": md.get("page"),
        "image_path": md.get("image_path"),
        "image_ext": md.get("image_ext"),
        "caption": md.get("caption"),
        "user_id": md.get("user_id"),
    }


def media_type_for(ext: str) -> str:
    return _MIME.get((ext or "png").lower(), "application/octet-stream")

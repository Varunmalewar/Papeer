"""Page-preserving text cleaning for PDF ingestion.

The module is deliberately dependency-free (stdlib only) so both
``backend.paper_loader`` and ``evaluate.py`` can import it without a cycle.
``backend`` must never import from ``evaluate``.

Cleaning is expressed over a list of ``(page_number, page_text)`` tuples.  A
document-level line stream is used for layout decisions (references,
acknowledgements, headings), but deletions are always applied back to the page
that owns each line.  Pages are never concatenated, so ``_small_to_big()`` keeps
the original page metadata.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass

logger = logging.getLogger(__name__)

CTX_BOILERPLATE_MIN_PAGES = 3
_BOILERPLATE_MIN_LEN = 3
_BOILERPLATE_MAX_LEN = 120
_ACK_MAX_LINES = 40
_BAND_SIZE = 3
_PAGE_NUMBER_RE = re.compile(r"^\d{1,4}$")

_STOPWORDS = {
    "a", "an", "the", "of", "and", "for", "in", "on", "to", "with", "at",
    "by", "from", "as", "or", "is", "are",
}

_NUM_PREFIX = r"(?:(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s*)?"
_KNOWN_HEADING_RE = re.compile(
    r"^\s*" + _NUM_PREFIX +
    r"(?:abstract|introduction|related work|background|"
    r"method(?:s|ology)?|approach|experiments?|results?|evaluation|"
    r"discussion|conclusions?|limitations?|future work|references|"
    r"bibliography|appendix|appendices|supplementary(?:\s+materials?)?|"
    r"acknowledg(?:e)?ments?|funding|author contributions|"
    r"conflicts? of interest|data availability)\b",
    re.IGNORECASE,
)
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s+\S"
)
_REF_HEADING_RE = re.compile(
    r"^\s*" + _NUM_PREFIX +
    r"(?:references|bibliography)\s*:?\s*$",
    re.IGNORECASE,
)
_APPENDIX_HEADING_RE = re.compile(
    r"^\s*" + _NUM_PREFIX +
    r"(?:appendix|appendices|supplementary(?:\s+materials?)?)\b",
    re.IGNORECASE,
)
_ACK_HEADING_RE = re.compile(
    r"^\s*" + _NUM_PREFIX +
    r"acknowledg(?:e)?ments?\s*:?\s*$",
    re.IGNORECASE,
)
_LETTERED_HEADING_RE = re.compile(r"^\s*[A-Z](?:\.\d+)?\s+\S")


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


def _looks_like_table(para: str) -> bool:
    """Heuristic: several lines with >=4 digit chars, tabs, or wide gaps —
    typical of an extracted results table."""
    lines = [line for line in para.splitlines() if line.strip()]
    if len(lines) < 3:
        return False
    hits = sum(
        1
        for line in lines
        if len(re.findall(r"\d", line)) >= 4
        or line.count("\t") >= 2
        or "   " in line
    )
    return hits >= max(2, len(lines) // 2)


def _line_blocks(text: str) -> list[tuple[int, int, str]]:
    """Return ``(start_line, end_line, block_text)`` for blank-line blocks."""
    lines = (text or "").split("\n")
    blocks: list[tuple[int, int, str]] = []
    start: int | None = None
    buf: list[str] = []
    for idx, line in enumerate(lines):
        if line.strip():
            if start is None:
                start = idx
            buf.append(line)
        elif start is not None:
            blocks.append((start, idx, "\n".join(buf)))
            start, buf = None, []
    if start is not None:
        blocks.append((start, len(lines), "\n".join(buf)))
    return blocks


def _collect_lines(text: str, seen: set[str]) -> None:
    for line in (text or "").splitlines():
        norm = _norm_line(line)
        if (
            _BOILERPLATE_MIN_LEN <= len(norm) <= _BOILERPLATE_MAX_LEN
            and re.search(r"[a-z]{3}", norm)
        ):
            seen.add(norm)


def boilerplate_lines(
    page_texts: dict[int, list[str] | str],
    *,
    skip_table_blocks: bool = False,
    min_pages: int = CTX_BOILERPLATE_MIN_PAGES,
) -> set[str]:
    """Lines repeated on at least ``min_pages`` pages.

    ``page_texts`` mirrors the old evaluate helper: page/slot -> one or more
    text blocks.  With ``skip_table_blocks=True`` (ingestion path), lines inside
    a table-like block are not counted, so a repeated table header is kept.
    """
    counts: Counter[str] = Counter()
    for _page, values in page_texts.items():
        texts = [values] if isinstance(values, str) else list(values)
        seen: set[str] = set()
        if skip_table_blocks:
            for text in texts:
                for _start, _end, block in _line_blocks(text):
                    if _looks_like_table(block):
                        continue
                    _collect_lines(block, seen)
        else:
            for text in texts:
                _collect_lines(text, seen)
        for norm in seen:
            counts[norm] += 1
    return {norm for norm, count in counts.items() if count >= min_pages}


# Backwards-compatible private name used by evaluate/tools.
_boilerplate_lines = boilerplate_lines


@dataclass(frozen=True)
class BoilerplateHit:
    pages: int
    occurrences: int


def _page_band_indices(text: str) -> tuple[set[int], set[int]]:
    """Return line indexes in the first/last three non-empty lines."""
    lines = (text or "").split("\n")
    nonempty = [idx for idx, line in enumerate(lines) if line.strip()]
    return set(nonempty[:_BAND_SIZE]), set(nonempty[-_BAND_SIZE:])


def _is_boilerplate_candidate(norm: str) -> bool:
    if _PAGE_NUMBER_RE.fullmatch(norm):
        return True
    return (
        _BOILERPLATE_MIN_LEN <= len(norm) <= _BOILERPLATE_MAX_LEN
        and re.search(r"[a-z]{3}", norm) is not None
    )


def position_boilerplate_lines(
    page_texts: dict[int, list[str] | str],
    *,
    skip_table_blocks: bool = True,
    min_pages: int = CTX_BOILERPLATE_MIN_PAGES,
) -> dict[tuple[str, str], BoilerplateHit]:
    """Find repeated header/footer lines, separately for top and bottom bands.

    Only the first/last three non-empty lines on each page are candidates.  A
    normalized string is removed only when it appears in the same band on at
    least ``min_pages`` pages.  Body lines are never candidates, even if they
    repeat on every page.
    """
    page_counts: Counter[tuple[str, str]] = Counter()
    occurrences: Counter[tuple[str, str]] = Counter()

    for _page, values in page_texts.items():
        text = values if isinstance(values, str) else "\n".join(values)
        lines = (text or "").split("\n")
        top, bottom = _page_band_indices(text)

        protected: set[int] = set()
        if skip_table_blocks:
            for start, end, block in _line_blocks(text):
                if _looks_like_table(block):
                    protected.update(range(start, end))

        for band, indexes in (("top", top), ("bottom", bottom)):
            seen: set[str] = set()
            for idx in indexes:
                if idx in protected:
                    continue
                norm = _norm_line(lines[idx])
                if not _is_boilerplate_candidate(norm):
                    continue
                seen.add(norm)
                occurrences[(band, norm)] += 1
            for norm in seen:
                page_counts[(band, norm)] += 1

    return {
        key: BoilerplateHit(pages=count, occurrences=occurrences[key])
        for key, count in page_counts.items()
        if count >= min_pages
    }


def strip_boilerplate(text: str, boiler: set[str]) -> str:
    lines = [line for line in text.splitlines() if _norm_line(line) not in boiler]
    return "\n".join(lines).strip()


# Backwards-compatible private name used by evaluate/tools.
_strip_boilerplate = strip_boilerplate


def _first_alpha_word(text: str) -> str:
    for word in text.split():
        cleaned = word.strip("\"'()[]{}:;,.!?")
        if cleaned and cleaned[0].isalpha():
            return cleaned
    return ""


def is_heading(line: str) -> bool:
    """Conservative section-heading heuristic (used to end ack blocks)."""
    text = (line or "").strip()
    if not text or len(text) > 100 or not re.search(r"[A-Za-z]", text):
        return False
    if text[-1] in ".!?;,":
        return False
    if _KNOWN_HEADING_RE.match(text):
        return True
    if _NUMBERED_HEADING_RE.match(text):
        return len(text.split()) <= 12
    if text.isupper():
        return len(text.split()) <= 12
    words = text.split()
    if len(words) > 8:
        return False
    content = [w for w in words if _first_alpha_word(w).lower() not in _STOPWORDS]
    if not content:
        return False
    capitalized = sum(
        1 for w in content if _first_alpha_word(w)[:1].isupper()
    )
    return capitalized / len(content) >= 0.6


def _is_reference_heading(line: str) -> bool:
    return bool(_REF_HEADING_RE.match(line.strip())) and is_heading(line)


def _is_appendix_heading(line: str) -> bool:
    return bool(_APPENDIX_HEADING_RE.match(line.strip())) and is_heading(line)


def _is_ack_heading(line: str) -> bool:
    return bool(_ACK_HEADING_RE.match(line.strip())) and is_heading(line)


@dataclass(frozen=True)
class _Line:
    doc: int
    page: int
    idx: int
    text: str


def _tagged_lines(pages: list[tuple[int, str]]) -> list[_Line]:
    out: list[_Line] = []
    for doc, (page, text) in enumerate(pages):
        for idx, line in enumerate((text or "").split("\n")):
            out.append(_Line(doc, int(page), idx, line))
    return out


def _apply_drops(
    pages: list[tuple[int, str]], drops: set[tuple[int, int]]
) -> list[tuple[int, str]]:
    if not drops:
        return list(pages)
    out: list[tuple[int, str]] = []
    for doc, (page, text) in enumerate(pages):
        indexes = {idx for d, idx in drops if d == doc}
        if not indexes:
            out.append((page, text))
            continue
        lines = (text or "").split("\n")
        kept = [line for idx, line in enumerate(lines) if idx not in indexes]
        out.append((page, "\n".join(kept)))
    return out


def _reference_drops(lines: list[_Line]) -> set[tuple[int, int]]:
    if len(lines) < 4:
        return set()
    threshold = len(lines) / 2.0
    start = next(
        (
            pos
            for pos, line in enumerate(lines)
            if pos >= threshold and _is_reference_heading(line.text)
        ),
        None,
    )
    if start is None:
        return set()

    end = len(lines)
    for pos in range(start + 1, len(lines)):
        if _is_appendix_heading(lines[pos].text):
            end = pos
            break
    else:
        # Bare "A Proofs" appendices are a known limitation.  Warn instead of
        # guessing where the reference list ends.
        for line in lines[start + 1:]:
            if _LETTERED_HEADING_RE.match(line.text.strip()) and is_heading(line.text):
                logger.warning(
                    "references tail on page %s contains an unrecognized "
                    "lettered heading %r; not guessing an appendix boundary",
                    line.page,
                    line.text.strip()[:80],
                )
                break

    return {(line.doc, line.idx) for line in lines[start:end]}


def strip_reference_tail(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop a latter-half References/Bibliography section up to an appendix."""
    return _apply_drops(pages, _reference_drops(_tagged_lines(pages)))


def _ack_drops(lines: list[_Line]) -> set[tuple[int, int]]:
    drops: set[tuple[int, int]] = set()
    pos = 0
    while pos < len(lines):
        if not _is_ack_heading(lines[pos].text):
            pos += 1
            continue
        stop = min(len(lines), pos + _ACK_MAX_LINES)
        end = pos + 1
        while end < stop and not is_heading(lines[end].text):
            end += 1
        for line in lines[pos:end]:
            drops.add((line.doc, line.idx))
        pos = max(end, pos + 1)
    return drops


def strip_acknowledgments(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop an acknowledgements heading plus its block (max 40 lines)."""
    return _apply_drops(pages, _ack_drops(_tagged_lines(pages)))


def _table_line_ids(pages: list[tuple[int, str]]) -> set[tuple[int, int]]:
    protected: set[tuple[int, int]] = set()
    for doc, (_page, text) in enumerate(pages):
        for start, end, block in _line_blocks(text):
            if _looks_like_table(block):
                protected.update((doc, idx) for idx in range(start, end))
    return protected


def clean_pages(
    pages: list[tuple[int, str]],
    *,
    min_pages: int = CTX_BOILERPLATE_MIN_PAGES,
) -> list[tuple[int, str]]:
    """Clean PDF pages while preserving page boundaries and page numbers."""
    if not pages:
        return []

    protected = _table_line_ids(pages)
    boiler = position_boilerplate_lines(
        {doc: text for doc, (_page, text) in enumerate(pages)},
        skip_table_blocks=True,
        min_pages=min_pages,
    )

    drops: set[tuple[int, int]] = set()
    for doc, (_page, text) in enumerate(pages):
        lines = (text or "").split("\n")
        top, bottom = _page_band_indices(text)
        for band, indexes in (("top", top), ("bottom", bottom)):
            for idx in indexes:
                if (doc, idx) in protected:
                    continue
                if (band, _norm_line(lines[idx])) in boiler:
                    drops.add((doc, idx))

    lines = _tagged_lines(pages)
    drops |= _reference_drops(lines)
    drops |= _ack_drops(lines)

    return _apply_drops(pages, drops)

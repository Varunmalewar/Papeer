"""P0 ingestion-cleaning tests: synthetic rules + real eval-PDF comparison."""
from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter
from pathlib import Path

import pytest
from langchain_community.document_loaders import PyMuPDFLoader

from backend.paper_loader import _small_to_big, load_pdf
from backend.text_clean import (
    _looks_like_table,
    boilerplate_lines,
    clean_pages,
    is_heading,
    position_boilerplate_lines,
    strip_reference_tail,
)

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = ROOT / "documents"


def _pages(pages: list[str]) -> list[tuple[int, str]]:
    return [(idx, text) for idx, text in enumerate(pages)]


def _body(n: int = 8) -> list[str]:
    return [f"Body paragraph line {i}, with ordinary prose and a period." for i in range(1, n + 1)]


# ── References / appendix rules ──────────────────────────────────────────────

def test_references_then_appendix_keeps_appendix():
    text = "\n".join(
        _body(8)
        + [
            "References",
            "[1] A. Author. A paper. 2024.",
            "[2] B. Author. Another paper. 2023.",
            "Appendix A",
            "Appendix proof body.",
        ]
    )
    cleaned = clean_pages([(0, text)])
    out = cleaned[0][1]

    assert "References" not in out
    assert "[1] A. Author" not in out
    assert "Appendix A" in out
    assert "Appendix proof body." in out


def test_references_without_appendix_drops_to_eof():
    text = "\n".join(
        _body(8)
        + [
            "References",
            "[1] A. Author. A paper. 2024.",
            "[2] B. Author. Another paper. 2023.",
        ]
    )
    out = clean_pages([(0, text)])[0][1]

    assert "References" not in out
    assert "[1] A. Author" not in out
    assert "Body paragraph line 8" in out


def test_acknowledgments_before_appendix_keeps_appendix():
    text = "\n".join(
        _body(6)
        + [
            "Acknowledgments",
            "We thank the reviewers for their feedback.",
            "Appendix A",
            "Appendix proof body.",
        ]
    )
    out = clean_pages([(0, text)])[0][1]

    assert "Acknowledgments" not in out
    assert "We thank the reviewers" not in out
    assert "Appendix A" in out
    assert "Appendix proof body." in out


def test_acknowledgments_stop_at_subheading():
    text = "\n".join(
        _body(5)
        + [
            "Acknowledgments",
            "We thank the reviewers for their feedback.",
            "Funding",
            "This work was supported by grant 123.",
        ]
    )
    out = clean_pages([(0, text)])[0][1]

    assert "Acknowledgments" not in out
    assert "We thank the reviewers" not in out
    assert "Funding" in out
    assert "grant 123" in out


# ── Boilerplate / table rules ────────────────────────────────────────────────

def test_repeated_table_header_is_not_stripped():
    table = "\n".join(
        [
            "Table 1. Results",
            "Dataset Accuracy F1",
            "A 0.91 0.88",
            "B 0.92 0.89",
        ]
    )
    assert _looks_like_table(table)
    pages = _pages([table, table, table])
    cleaned = clean_pages(pages)

    assert all("Dataset Accuracy F1" in text for _page, text in cleaned)
    assert "Dataset Accuracy F1" not in boilerplate_lines(
        {page: text for page, text in pages}, skip_table_blocks=True
    )
    position_rule = position_boilerplate_lines(
        {page: text for page, text in pages}, skip_table_blocks=True
    )
    assert ("top", "dataset accuracy f1") not in position_rule


def test_repeated_body_label_is_kept():
    pages = []
    for _ in range(4):
        lines = _body(10)
        lines[4] = "HyDE"
        pages.append("\n".join(lines))
    cleaned = clean_pages(_pages(pages))

    assert all("HyDE" in text for _page, text in cleaned)
    rule = position_boilerplate_lines(
        {page: text for page, text in _pages(pages)}, skip_table_blocks=True
    )
    assert ("top", "hyde") not in rule
    assert ("bottom", "hyde") not in rule


def test_repeated_top_header_is_removed():
    header = "Proceedings of the 2024 Conference on Empirical Methods"
    pages = ["\n".join([header, *_body(6)]) for _ in range(4)]
    cleaned = clean_pages(_pages(pages))

    assert all(header not in text for _page, text in cleaned)
    rule = position_boilerplate_lines(
        {page: text for page, text in _pages(pages)}, skip_table_blocks=True
    )
    hit = rule[("top", header.lower())]
    assert hit.pages == 4
    assert hit.occurrences == 4


def test_repeated_bare_page_number_in_bottom_band_is_removed():
    pages = ["\n".join([*_body(6), "9"]) for _ in range(4)]
    cleaned = clean_pages(_pages(pages))

    assert all(
        "9" not in [line.strip() for line in text.splitlines()]
        for _page, text in cleaned
    )
    rule = position_boilerplate_lines(
        {page: text for page, text in _pages(pages)}, skip_table_blocks=True
    )
    assert rule[("bottom", "9")].pages == 4


def test_table_number_in_body_is_kept():
    pages = []
    for _ in range(4):
        lines = _body(10)
        lines[4] = "41.5"
        pages.append("\n".join(lines))
    cleaned = clean_pages(_pages(pages))

    assert all("41.5" in text for _page, text in cleaned)
    rule = position_boilerplate_lines(
        {page: text for page, text in _pages(pages)}, skip_table_blocks=True
    )
    assert ("top", "41.5") not in rule
    assert ("bottom", "41.5") not in rule


def test_body_paragraph_containing_references_is_untouched():
    text = "\n".join(
        _body(6)
        + ["The references are discussed in Section 3 of this document."]
    )
    out = clean_pages([(0, text)])[0][1]

    assert "The references are discussed in Section 3" in out
    assert out == text


def test_references_heading_in_first_half_is_not_tail():
    text = "\n".join(
        [
            "References",
            "This heading is part of the body introduction.",
            *_body(10),
        ]
    )
    out = strip_reference_tail([(0, text)])[0][1]

    assert out == text


def test_section_headings_and_title_are_not_dropped():
    text = "\n".join(
        [
            "“MOCAA* Enhanced: Multi-Objective Context-Aware A* Pathfinding",
            "with Integration of Multiple Heuristics”",
            "",
            "1.2 Objective",
            "The objective is to find a low-cost path.",
            "",
            "3. Module Description",
            "This section describes the modules.",
        ]
    )
    out = clean_pages([(0, text)])[0][1]

    assert "MOCAA* Enhanced" in out
    assert "1.2 Objective" in out
    assert "3. Module Description" in out


# ── Metadata ─────────────────────────────────────────────────────────────────

def test_page_metadata_is_preserved_by_cleaning():
    pages = [
        (3, "\n".join(_body(4) + ["References", "[1] A. Author. 2024."])),
        (4, "Appendix A\nAppendix body."),
    ]
    cleaned = clean_pages(pages)

    assert [page for page, _text in cleaned] == [3, 4]
    assert "Appendix body." in cleaned[1][1]


# ── Real PDF comparison ──────────────────────────────────────────────────────

def _key(doc) -> tuple[str, int, str]:
    md = doc.metadata or {}
    text_hash = hashlib.sha256((doc.page_content or "").encode("utf-8")).hexdigest()
    return str(md.get("title") or ""), int(md.get("page", 0)), text_hash


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _removed_lines(before: str, after: str) -> list[str]:
    before_lines = [ln for ln in before.splitlines() if ln.strip()]
    after_set = {ln.strip() for ln in after.splitlines() if ln.strip()}
    return [ln for ln in before_lines if ln.strip() not in after_set]


def _safe(text: str) -> str:
    return (text or "").encode("ascii", "backslashreplace").decode("ascii")


def _indent(text: str) -> str:
    return "\n".join("    " + line for line in _safe(text).splitlines())


@pytest.mark.parametrize("pdf", sorted(DOCUMENTS.glob("*.pdf")), ids=lambda p: p.stem)
def test_pdf_chunks_before_vs_after_cleaning(pdf: Path, caplog):
    raw_docs = PyMuPDFLoader(str(pdf)).load()
    before = _small_to_big(raw_docs, pdf.stem)
    with caplog.at_level(logging.WARNING, logger="backend.text_clean"):
        after = load_pdf(str(pdf))

    raw_pages = {
        int((doc.metadata or {}).get("page", 0)) + 1: doc.page_content or ""
        for doc in raw_docs
    }
    rule = position_boilerplate_lines(raw_pages, skip_table_blocks=True)

    unmatched_after = set(range(len(after)))
    after_counts = Counter(_key(doc) for doc in after)
    unchanged = 0
    modified: list[tuple] = []
    dropped: list = []

    for b in before:
        key = _key(b)
        if after_counts.get(key, 0) > 0:
            after_counts[key] -= 1
            for idx in sorted(unmatched_after):
                if _key(after[idx]) == key:
                    unmatched_after.remove(idx)
                    break
            unchanged += 1
            continue

        best_idx = None
        best_score = 0.0
        for idx in unmatched_after:
            candidate = after[idx]
            if _key(candidate)[:2] != key[:2]:
                continue
            score = _overlap(b.page_content or "", candidate.page_content or "")
            if score > best_score:
                best_idx, best_score = idx, score
        if best_idx is not None and best_score >= 0.6:
            modified.append((b, after[best_idx], best_score))
            unmatched_after.remove(best_idx)
        else:
            dropped.append(b)

    new = [after[idx] for idx in sorted(unmatched_after)]
    is_2212 = pdf.stem == "2212.10496v1"
    print(
        f"\n[P0] {pdf.name}: before={len(before)} after={len(after)} "
        f"unchanged={unchanged} modified={len(modified)} "
        f"dropped={len(dropped)} new={len(new)} "
        f"dropped%={100 * len(dropped) / max(1, len(before)):.1f}"
    )

    print("  BOILERPLATE RULE OUTPUT (top 30 by occurrences):")
    rule_rows = sorted(
        rule.items(),
        key=lambda item: (-item[1].occurrences, -item[1].pages, item[0]),
    )[:30]
    if not rule_rows:
        print("    none")
    for (band, norm), hit in rule_rows:
        print(
            f"    {band:6} occ={hit.occurrences:<3} "
            f"pages={hit.pages:<3} {_safe(norm)!r}"
        )

    before_pages = Counter(
        int((doc.metadata or {}).get("page", 0)) + 1 for doc in before
    )
    after_pages = Counter(
        int((doc.metadata or {}).get("page", 0)) + 1 for doc in after
    )
    all_pages = sorted(set(before_pages) | set(after_pages))
    print("  page  before  after  flag")
    for page in all_pages:
        n_before = before_pages.get(page, 0)
        n_after = after_pages.get(page, 0)
        flag = "ZERO" if n_before and not n_after else ""
        print(f"  p{page:<3} {n_before:>6} {n_after:>5}  {flag}")
    zero_pages = [
        page for page in all_pages
        if before_pages.get(page, 0) and not after_pages.get(page, 0)
    ]
    print(f"  ZERO PAGES: {zero_pages or 'none'}")

    for doc in dropped:
        md = doc.metadata or {}
        print(f"  DROP p{int(md.get('page', 0)) + 1}: {_safe(doc.page_content[:80])!r}")
    for b, a, score in modified:
        md = b.metadata or {}
        removed = _removed_lines(b.page_content or "", a.page_content or "")
        if is_2212:
            print(
                f"  MOD p{int(md.get('page', 0)) + 1} overlap={score:.2f} "
                f"removed={_safe(str(removed))}"
            )
        else:
            print(
                f"  MOD p{int(md.get('page', 0)) + 1} overlap={score:.2f} "
                f"removed={_safe(str(removed[:2]))}"
            )

    dropped_ids = {id(doc) for doc in dropped}
    modified_ids = {id(doc) for doc, _after, _score in modified}
    for doc in new:
        md = doc.metadata or {}
        same_page = [
            b for b in before
            if _key(b)[:2] == _key(doc)[:2]
        ]
        source = max(
            same_page,
            key=lambda b: _overlap(b.page_content or "", doc.page_content or ""),
            default=None,
        )
        if source is None:
            print(f"  NEW p{int(md.get('page', 0)) + 1}: {_safe(doc.page_content[:80])!r}")
            continue
        kind = (
            "dropped" if id(source) in dropped_ids
            else "modified" if id(source) in modified_ids
            else "unchanged"
        )
        overlap = _overlap(source.page_content or "", doc.page_content or "")
        print(
            f"  NEW p{int(md.get('page', 0)) + 1} vs BEFORE p"
            f"{int((source.metadata or {}).get('page', 0)) + 1} "
            f"source={kind} overlap={overlap:.2f}"
        )
        print("  NEW TEXT:")
        print(_indent(doc.page_content))
        print("  BEFORE TEXT:")
        print(_indent(source.page_content))

    if is_2212:
        later = [
            page for page, count in after_pages.items()
            if page > 10 and count > 0
        ]
        print(f"  APPENDIX CHECK pages>10 surviving: {sorted(later) or 'NONE'}")
        warnings = [
            record.getMessage()
            for record in caplog.records
            if "lettered heading" in record.getMessage()
        ]
        print(f"  LETTERED WARNING FIRED: {'YES' if warnings else 'NO'}")
        for message in warnings:
            print(f"    {_safe(message)}")

    # Every unchanged chunk is an exact title/page/hash match; modified and
    # dropped chunks are expected to differ after cleaning.
    assert unchanged + len(modified) + len(dropped) == len(before)
    assert all(
        (doc.metadata or {}).get("page") is not None
        for doc in after
    )

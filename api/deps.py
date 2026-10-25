"""Shared deps: singleton graph (same build_graph backend/ uses)."""
import os
import re
from functools import lru_cache


@lru_cache(maxsize=1)
def _build_graph_cached():
    from backend.rag_graph import build_graph

    return build_graph()


def get_graph():
    return _build_graph_cached()


def refresh_graph():
    """Rebuild the graph (fresh checkpointer connection).

    Neon Postgres connections go stale when the host DNS/network drops
    mid-session; callers retry once on `psycopg.OperationalError` by
    rebuilding (build_graph itself falls back to the sqlite checkpointer
    if Postgres is still unreachable).
    """
    _build_graph_cached.cache_clear()
    return _build_graph_cached()


_PAPER_CITATION_RE = re.compile(
    r"\[(?P<title>[^\[\]]+?),\s*(?:Page|p\.)\s*(?P<page>\d+)\s*\]"
)

_FIGURE_CITATION_RE = re.compile(
    r"\[(?P<title>[^\[\]]+?),\s*Figure\s*(?P<figure>\d+),\s*(?:Page|p\.)\s*(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)

_REFUSAL_PREFIXES = (
    "i wasn't able to find relevant information",
    "the paper does not state this",
    "i don't know the answer",
)


def _is_no_answer(answer: str) -> bool:
    """True when the answer refused / found nothing usable — sources chips on
    a refusal are just noise (they list pages that didn't support anything)."""
    a = (answer or "").strip().lower()
    return any(a.startswith(p) for p in _REFUSAL_PREFIXES)


def build_sources_list(values: dict) -> list[dict]:
    """Sources chips for the chat UI.

    Grounded-mode default: only chunks the ANSWER actually cites (by title +
    1-indexed page, or [Title, Figure N, Page X] for figures), so irrelevant
    retrieved pages (e.g. references, pp. 8-9) never appear as evidence. Zero
    valid citations (web/off-topic routes, refusal answers) -> fall back to all
    retrieved docs, keeping the old behaviour where citations don't apply.
    """
    docs = values.get("retrieved_docs") or []
    answer = values.get("answer") or ""
    if _is_no_answer(answer):
        return []  # no answer found -> no sources (task rule #4)
    cited = {
        (m.group("title").strip().lower(), int(m.group("page")))
        for m in _PAPER_CITATION_RE.finditer(answer)
    }
    cited_figures = {
        (m.group("title").strip().lower(), int(m.group("figure")))
        for m in _FIGURE_CITATION_RE.finditer(answer)
    }
    session = values.get("session_id") or ""
    figure_prefix = os.environ.get("PAPEER_FIGURE_URL_PREFIX", "/api/figure").rstrip("/")

    entries, by_cited = [], []
    for doc in docs:
        md = doc.metadata or {}
        title = md.get("title", "Unknown")
        page = md.get("page")
        url = md.get("url")
        is_figure = md.get("type") == "figure" and md.get("figure_number") is not None
        entry: dict = {"title": title}
        if page is not None:
            entry["page"] = page + 1
        if url:
            entry["url"] = url
        if is_figure:
            entry["figure"] = md.get("figure_number")
            if session and md.get("figure_id"):
                entry["image_url"] = f"{figure_prefix}/{session}/{md['figure_id']}"
        entries.append(entry)
        if is_figure:
            by_cited.append(
                (title.strip().lower(), int(md["figure_number"])) in cited_figures
            )
        else:
            by_cited.append(
                (entry.get("page") is not None and (title.strip().lower(), entry.get("page")) in cited)
            )

    def _key(e: dict) -> tuple:
        return (e["title"], e.get("page"), e.get("figure"), e.get("url"))

    if (cited or cited_figures) and entries:
        filtered, seen_cited = [], set()
        for e, keep in zip(entries, by_cited):
            if not keep:
                continue
            key = _key(e)
            if key in seen_cited:
                continue
            seen_cited.add(key)
            filtered.append(e)
        return filtered or entries
    # dedupe when returning the unfiltered fallback
    seen, out = set(), []
    for e in entries:
        key = _key(e)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


import os
import re
import math
import threading
import time
from collections import Counter

from dotenv import load_dotenv
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    VectorParams,
)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(override=True)  # .env is source of truth — overrides any stray env vars
# â”€â”€ Config â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

COLLECTION_NAME = "papeer_docs"

base_embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    api_key=os.environ.get("GOOGLE_API_KEY"),
    output_dimensionality=1536,
)

# â”€â”€ Singletons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _build_embedding_store():
    """Neon Postgres ByteStore when DATABASE_URL is set, else local file store."""
    try:
        from backend.postgres_store import get_embedding_byte_store

        pg_store = get_embedding_byte_store(table_name="embedding_cache")
        if pg_store is not None:
            print("[embeddings] Using Neon Postgres byte store (embedding_cache).")
            return pg_store
    except Exception as e:
        print(f"[embeddings] Postgres store unavailable, falling back to file: {e}")
    print("[embeddings] Using LocalFileStore (./embedding_cache/).")
    return LocalFileStore("./embedding_cache/")


embedding_file_store = _build_embedding_store()
embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    embedding_file_store,
    namespace=base_embeddings.model,
    query_embedding_cache=True,
    key_encoder="blake2b",
)

# Opt-in timing: PAPEER_TIMING=1 reports embed latency separately from search.
_orig_embed_query = embeddings.embed_query


def _timed_embed_query(text, *args, **kwargs):
    from backend.timing import timed

    with timed("embed"):
        return _orig_embed_query(text, *args, **kwargs)


embeddings.embed_query = _timed_embed_query

qdrant_client = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ["QDRANT_API_KEY"],
    timeout=120,
)

_collection_ready = False
_collection_lock = threading.Lock()


# â”€â”€ Collection bootstrap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _ensure_collection() -> None:
    """Create the shared collection and payload indexes if they don't exist.

    Runs at most once per process: every payload-index create is a cloud round
    trip (~0.3-0.4 s each), and this used to run 4x per search variant.
    """
    global _collection_ready
    if _collection_ready:
        return
    with _collection_lock:
        if _collection_ready:
            return
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )
        # Idempotent index setup â€” also runs for pre-existing collections.
        for field in (
            "metadata.session_id",
            "metadata.title",
            "metadata.user_id",
            "metadata.figure_id",
            "metadata.paper_id",
        ):
            try:
                qdrant_client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass  # index already exists
        _collection_ready = True


def _get_vectorstore() -> QdrantVectorStore:
    _ensure_collection()
    return QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings,
    )


def _scope_filter(
    scope: str | list[str],
    titles: list[str] | None = None,
    user_id: str | None = None,
) -> Filter:
    """Filter on metadata.session_id for one id or a set of ids (user library).
    Optional `titles` narrows to specific loaded documents; `user_id` enforces
    per-user isolation (server-issued â€” never client-trusted)."""
    if isinstance(scope, str):
        scope = [scope]
    conditions = [
        FieldCondition(key="metadata.session_id", match=MatchAny(any=list(scope)))
    ]
    if titles:
        conditions.append(
            FieldCondition(key="metadata.title", match=MatchAny(any=list(titles)))
        )
    if user_id:
        conditions.append(
            FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id))
        )
    return Filter(must=conditions)


# â”€â”€ Hybrid search: dense (Gemini) + BM25 (in-memory) with RRF fusion â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# No Qdrant schema change / reindex needed: dense runs in Qdrant, BM25 runs over
# the session's chunks fetched via scroll, fused with Reciprocal Rank Fusion.

_STOPWORDS = frozenset(
    "a an the and or but if then else when what which who whom this that these those "
    "am is are was were be been being have has had having do does did doing will would "
    "should can could may might must shall of at by for with about into through during "
    "before after above below to from up down in out on off over under again further "
    "once here there all any both each few more most other some such no nor not only "
    "own same so than too very just as it its he she they them his her their our your "
    "you we us i me my mine kya hai me mein ka ki ke ko se par aur bhi kya".split()
)

_HYBRID_DENSE_K = int(os.environ.get("PAPEER_HYBRID_DENSE_K", "20"))
_HYBRID_BM25_K = int(os.environ.get("PAPEER_HYBRID_BM25_K", "20"))
_HYBRID_RRF_K = int(os.environ.get("PAPEER_HYBRID_RRF_K", "60"))

# â”€â”€ FlashRank reranker (free, local, no API key) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Hybrid fusion se aaye candidates (default 20) ko cross-encoder (TinyBERT)
# se rescore karke top-k nikalta hai. Env se control:
#   PAPEER_RERANK=0 â†’ rerank off (sirf hybrid fusion)
#   PAPEER_RERANK_K â†’ kitne candidates rerank ho (default 20)
_RERANK_ENABLED = os.environ.get("PAPEER_RERANK", "1").strip() != "0"
_MIN_RERANK_SCORE = float(os.environ.get("PAPEER_MIN_SCORE", "0.05"))
_RERANK_CANDIDATES = int(os.environ.get("PAPEER_RERANK_K", "20"))
_ranker = None


def _get_ranker():
    """Lazy singleton â€” pehli query par ~40MB TinyBERT model download hota hai."""
    global _ranker
    if _ranker is None:
        from flashrank import Ranker

        _ranker = Ranker(cache_dir="./flashrank_cache")
    return _ranker


def _flashrank_rerank(query: str, docs: list[Document], k: int) -> list[Document]:
    """Cross-encoder rescoring with a relevance floor. Chunks scoring below
    PAPEER_MIN_SCORE (0-1 rerank scale) are dropped so weak matches never
    reach the LLM. Fail ho to input order me top-k (safe fallback)."""
    if not docs or k <= 0:
        return []
    if not _RERANK_ENABLED:
        return docs[:k]
    try:
        from flashrank import RerankRequest

        passages = [
            {"id": str(i), "text": d.page_content[:2000]} for i, d in enumerate(docs)
        ]
        ranked = _get_ranker().rerank(RerankRequest(query=query, passages=passages))
        scored = [
            (float(r["score"]), int(r["id"]))
            for r in ranked
            if str(r.get("id", "")).isdigit()
        ]
        scored = [(s, i) for s, i in scored if s >= _MIN_RERANK_SCORE]
        if not scored:
            # RELEVANCE-FLOOR FALLBACK: an empty context makes the generator
            # answer "The paper does not state this" even when the paper
            # covers the topic — phrasing mismatches (embedding gap) can push
            # every candidate below the floor. Return the best few raw
            # candidates flagged below_floor; the strict generator +
            # citation-checker still stop junk from being asserted.
            scored = [
                (float(r["score"]), int(r["id"]))
                for r in ranked
                if str(r.get("id", "")).isdigit()
            ]
            scored.sort(key=lambda t: t[0], reverse=True)
            reranked = []
            for score, i in scored[: min(k, 3)]:
                if 0 <= i < len(docs):
                    docs[i].metadata["rerank_score"] = round(score, 4)
                    docs[i].metadata["below_floor"] = True
                    reranked.append(docs[i])
            return reranked if reranked else docs[:k]
        scored.sort(key=lambda t: t[0], reverse=True)
        reranked = []
        for score, i in scored[:k]:
            if 0 <= i < len(docs):
                docs[i].metadata["rerank_score"] = round(score, 4)
                reranked.append(docs[i])
        return reranked if reranked else docs[:k]
    except Exception as e:
        print(f"[rerank] FlashRank failed, using fusion order: {e}")
        return docs[:k]
_HYBRID_MAX_DOCS = int(os.environ.get("PAPEER_HYBRID_MAX_DOCS", "2000"))


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _bm25_scores(
    query_tokens: list[str], docs_tokens: list[list[str]]
) -> list[float]:
    """Classic BM25 (k1=1.2, b=0.75) over tokenized docs."""
    n = len(docs_tokens)
    if n == 0 or not query_tokens:
        return [0.0] * n
    df = Counter()
    for toks in docs_tokens:
        for t in set(toks):
            df[t] += 1
    avgdl = sum(len(t) for t in docs_tokens) / n
    k1, b = 1.2, 0.75
    scores = [0.0] * n
    qtf = Counter(query_tokens)
    for qi, q in enumerate(query_tokens):
        if q not in df:
            continue
        idf = math.log(1 + (n - df[q] + 0.5) / (df[q] + 0.5))
        for di, toks in enumerate(docs_tokens):
            tf = toks.count(q)
            if tf == 0:
                continue
            denom = tf + k1 * (1 - b + b * len(toks) / avgdl)
            scores[di] += idf * (tf * (k1 + 1) / denom) * (1 + 0.1 * (qtf[q] - 1))
    return scores


def _scroll_session_docs(
    scope: str | list[str],
    limit: int,
    titles: list[str] | None = None,
    user_id: str | None = None,
) -> list[Document]:
    """Fetch up to `limit` session chunks as Documents (for BM25)."""
    _ensure_collection()
    docs: list[Document] = []
    offset = None
    while len(docs) < limit:
        points, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=_scope_filter(scope, titles, user_id),
            with_payload=True,
            limit=min(250, limit - len(docs)),
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            docs.append(
                Document(
                    page_content=payload.get("page_content", ""),
                    metadata=payload.get("metadata", {}),
                    id=str(point.id),
                )
            )
        if offset is None:
            break
    return docs


# BM25 corpus cache: one scroll per (scope, titles, user) per TTL instead of
# twice per query variant (~1.9 s each). Invalidated on every write.
_CORPUS_TTL = float(os.environ.get("PAPEER_BM25_CACHE_TTL", "60"))
_corpus_cache: dict[tuple, tuple[float, list[Document]]] = {}
_corpus_lock = threading.Lock()


def _corpus_key(scope: str | list[str], titles: list[str] | None, user_id: str | None) -> tuple:
    scopes = (scope,) if isinstance(scope, str) else tuple(scope)
    return (
        tuple(sorted(s for s in scopes if s)),
        tuple(sorted((t or "").strip() for t in (titles or []))),
        user_id or "",
    )


def invalidate_corpus_cache() -> None:
    """Drop the BM25 corpus cache (call after any add/delete)."""
    with _corpus_lock:
        _corpus_cache.clear()


def _scroll_session_docs_cached(
    scope: str | list[str],
    limit: int,
    titles: list[str] | None = None,
    user_id: str | None = None,
) -> list[Document]:
    key = _corpus_key(scope, titles, user_id)
    now = time.monotonic()
    with _corpus_lock:
        hit = _corpus_cache.get(key)
    if hit is not None and now - hit[0] < _CORPUS_TTL:
        return hit[1]
    docs = _scroll_session_docs(scope, limit, titles=titles, user_id=user_id)
    with _corpus_lock:
        _corpus_cache[key] = (time.monotonic(), docs)
    return docs


def _rrf_fuse(
    ranked_lists: list[list[str]], rrf_k: int = 60
) -> dict[str, float]:
    """Reciprocal Rank Fusion: score(id) = Î£ 1/(k + rank)."""
    fused: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
    return fused


def hybrid_search(
    query: str,
    scope: str | list[str],
    k: int = 4,
    dense_k: int | None = None,
    bm25_k: int | None = None,
    titles: list[str] | None = None,
    user_id: str | None = None,
) -> list[Document]:
    """Hybrid retrieval: dense (Qdrant) + BM25 (in-memory) fused with RRF.

    Falls back to pure dense search when the scope has no scrollable docs.
    Optional `titles` restricts retrieval to specific loaded documents.
    """
    dense_k = dense_k or _HYBRID_DENSE_K
    bm25_k = bm25_k or _HYBRID_BM25_K
    _ensure_collection()

    dense_docs = _get_vectorstore().similarity_search(
        query, k=dense_k, filter=_scope_filter(scope, titles, user_id)
    )
    dense_by_id = {d.id: d for d in dense_docs if d.id}

    session_docs = _scroll_session_docs_cached(scope, _HYBRID_MAX_DOCS, titles=titles, user_id=user_id)
    if not session_docs:
        return dense_docs[:k]

    q_tokens = _tokenize(query)
    doc_tokens = [_tokenize(d.page_content) for d in session_docs]
    scores = _bm25_scores(q_tokens, doc_tokens)
    ranked_bm25 = sorted(
        range(len(session_docs)), key=lambda i: scores[i], reverse=True
    )[:bm25_k]
    bm25_by_id = {session_docs[i].id: session_docs[i] for i in ranked_bm25}

    dense_ranked = [d.id for d in dense_docs if d.id]
    bm25_ranked = [session_docs[i].id for i in ranked_bm25 if session_docs[i].id]
    fused = _rrf_fuse([dense_ranked, bm25_ranked], rrf_k=_HYBRID_RRF_K)

    all_docs = {**bm25_by_id, **dense_by_id}
    fused_ids = sorted(fused, key=lambda i: fused[i], reverse=True)
    over_fetch = max(k * 2, _RERANK_CANDIDATES) if _RERANK_ENABLED else k
    candidates = [all_docs[i] for i in fused_ids[:over_fetch] if i in all_docs]
    if _RERANK_ENABLED:
        return _flashrank_rerank(query, candidates, k)
    return candidates[:k]


def bm25_search(
    query: str,
    scope: str | list[str],
    k: int = 3,
    titles: list[str] | None = None,
    user_id: str | None = None,
) -> list[Document]:
    """Pure keyword (BM25) retrieval — NO cross-encoder veto.

    False-premise questions ("Why does it use GELU?") poison both dense and
    FlashRank scores because the premise token appears nowhere in the paper,
    while the paper's actual counterpart chunk still shares real keywords
    ('activation', 'feed-forward'). The retrieval tool unions these BM25
    candidates into its pool for exactly that case.
    """
    _ensure_collection()
    session_docs = _scroll_session_docs_cached(scope, _HYBRID_MAX_DOCS, titles=titles, user_id=user_id)
    if not session_docs:
        return []
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    doc_tokens = [_tokenize(d.page_content) for d in session_docs]
    scores = _bm25_scores(q_tokens, doc_tokens)
    # ONE parent per parent_id — two children of the same parents block must
    # not burn slots that would push the genuinely-distinct parent out.
    hits, seen_docs, seen_parents = [], set(), set()
    for i in sorted(range(len(session_docs)), key=lambda i: scores[i], reverse=True):
        d = session_docs[i]
        if scores[i] <= 0:
            break
        key = d.id or d.page_content
        if key in seen_docs:
            continue
        seen_docs.add(key)
        meta = getattr(d, "metadata", None) or {}
        parent_key = meta.get("parent_id") or d.page_content
        if parent_key in seen_parents:
            continue
        seen_parents.add(parent_key)
        d.metadata["bm25_score"] = round(float(scores[i]), 4)
        d.metadata["bm25_only"] = True
        hits.append(d)
        if len(hits) >= k:
            break
    return hits


# â”€â”€ Public API â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def add_paper(docs: list[Document], session_id: str, user_id: str | None = None) -> None:
    """Add documents to the shared collection, stamping session_id (and the
    server-issued user_id for per-user isolation) into metadata."""
    _nul_re = re.compile(r"[\x00\x08\x0b\x0c\x0e-\x1f]")
    for doc in docs:
        doc.metadata["session_id"] = session_id
        if user_id:
            doc.metadata["user_id"] = user_id
        if isinstance(doc.page_content, str):
            doc.page_content = _nul_re.sub("", doc.page_content)
        if isinstance(doc.metadata.get("parent_text"), str):
            doc.metadata["parent_text"] = _nul_re.sub("", doc.metadata["parent_text"])
    _get_vectorstore().add_documents(docs)
    invalidate_corpus_cache()


def list_papers(scope: str | list[str], user_id: str | None = None) -> list[str]:
    """Return distinct paper titles for a session (or user library) via scroll."""
    _ensure_collection()
    seen: set[str] = set()
    titles: list[str] = []
    offset = None
    while True:
        points, offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=_scope_filter(scope, None, user_id),
            with_payload=True,
            limit=100,
            offset=offset,
        )
        for point in points:
            title = (point.payload or {}).get("metadata", {}).get("title")
            if title and title not in seen:
                seen.add(title)
                titles.append(title)
        if offset is None:
            break
    return titles


def search(
    query: str,
    scope: str | list[str],
    k: int = 4,
    titles: list[str] | None = None,
    user_id: str | None = None,
) -> list[Document]:
    """Filtered search within a session/library (hybrid dense+BM25, dense fallback).
    Optional `titles` restricts the search to specific loaded documents;
    `user_id` is always applied when issued by the server."""
    _ensure_collection()
    try:
        return hybrid_search(query, scope, k=k, titles=titles, user_id=user_id)
    except Exception as e:
        print(f"[hybrid] fallback to dense search: {e}")
        return _get_vectorstore().similarity_search(
            query,
            k=k,
            filter=_scope_filter(scope, titles, user_id),
        )


def count_chunks(scope: str | list[str], user_id: str | None = None) -> int:
    """Count vectors belonging to a session (or user library)."""
    _ensure_collection()
    return qdrant_client.count(
        collection_name=COLLECTION_NAME,
        count_filter=_scope_filter(scope, None, user_id),
        exact=True,
    ).count


def delete_paper(session_id: str, title: str, user_id: str | None = None) -> int:
    """Delete all chunks for a specific paper in a session. Returns deleted count."""
    _ensure_collection()
    filt = _scope_filter(
        session_id,
        [title],
        user_id,
    )
    before = qdrant_client.count(COLLECTION_NAME, count_filter=filt, exact=True).count
    qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=filt)
    invalidate_corpus_cache()
    return before


def delete_session(session_id: str, user_id: str | None = None) -> int:
    """Delete all chunks for a session. Returns deleted count."""
    _ensure_collection()
    filt = _scope_filter(session_id, None, user_id)
    before = qdrant_client.count(COLLECTION_NAME, count_filter=filt, exact=True).count
    qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=filt)
    invalidate_corpus_cache()
    return before


# ── Figure points ────────────────────────────────────────────────────────────

def figure_owner_filter(figure_id: str, user_id: str) -> Filter:
    """Ownership filter: a figure is only visible to the user who ingested it."""
    return Filter(must=[
        FieldCondition(key="metadata.figure_id", match=MatchValue(value=figure_id)),
        FieldCondition(key="metadata.user_id", match=MatchValue(value=user_id)),
    ])


def get_figure_payload(figure_id: str, user_id: str) -> dict | None:
    """Return the Qdrant payload for a figure iff it belongs to ``user_id``."""
    _ensure_collection()
    points, _ = qdrant_client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=figure_owner_filter(figure_id, user_id),
        with_payload=True,
        limit=1,
    )
    if not points:
        return None
    return dict(points[0].payload or {})


def delete_figure_chunks(session_id: str, title: str, user_id: str | None = None) -> int:
    """Remove only the figure chunks of one paper (used by the backfill script)."""
    _ensure_collection()
    filt = _scope_filter(session_id, [title], user_id)
    filt.must.append(FieldCondition(key="metadata.type", match=MatchValue(value="figure")))
    before = qdrant_client.count(COLLECTION_NAME, count_filter=filt, exact=True).count
    if before:
        qdrant_client.delete(collection_name=COLLECTION_NAME, points_selector=filt)
        invalidate_corpus_cache()
    return before

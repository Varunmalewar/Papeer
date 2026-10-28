"""Stage-by-stage trace of the stock production retrieval pipeline.

The trace is diagnostic only: it does not change retrieval code or thresholds.
It creates one temporary Qdrant session, runs every gold question through the
same stages as ``retrieve_merged``, records page-label and page-text presence,
and deletes only that session's vectors in ``finally``.

``--dry-run`` prints the exact targets and delete calls without importing the
retrieval stack or writing to Qdrant/Neon.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(override=True)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
for _key in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
    os.environ.pop(_key, None)

PDF_DIR = ROOT / "documents"
GOLDENS = ROOT / "goldens_manual_pairs.json"
MANIFEST = ROOT / "ablation_stage_session.json"
DEFAULT_OUTPUT = ROOT / "ablation_stage_trace.json"

COLLECTION_NAME = "papeer_docs"
NEON_TABLE = "embedding_cache"
ABLATE_USER = "lib-" + hashlib.sha256(b"ablate@papeer.local").hexdigest()[:32]
SESSION_PREFIX = "ablate_stage_"
K = 8
BM25_K = 5
MAX_BM25_EXTRAS = 4
STAGE_NAMES = (
    "dense_arm",
    "query_variants",
    "variant_search",
    "variant_bm25",
    "variant_pool",
    "pre_parent_selection",
    "final_parent_expanded",
)
FLAGGED_QIDS = {"Q01", "Q21", "Q31", "Q32", "Q41"}


def _squash(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _pdfs() -> list[Path]:
    return sorted(PDF_DIR.glob("*.pdf"))


def _questions() -> list[dict]:
    raw = json.loads(GOLDENS.read_text(encoding="utf-8"))["docs"]
    by_squash = {_squash(p.name): p for p in _pdfs()}
    out: list[dict] = []
    for name, entry in raw.items():
        pdf = by_squash.get(_squash(name))
        if pdf is None:
            raise SystemExit(f"no PDF matches goldens key {name!r}")
        for local_index, pair in enumerate(entry["pairs"], 1):
            page = pair.get("source_page")
            out.append(
                {
                    "qid": f"Q{len(out) + 1:02d}",
                    "local_id": f"{pdf.stem}#{local_index:02d}",
                    "pdf": pdf,
                    "doc": pdf.stem,
                    "input": pair["input"],
                    "gold_page": int(page) if page is not None else None,
                    "expected_pages": [int(page)] if page is not None else [],
                }
            )
    return out


def _save(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _force_local_cache() -> None:
    import backend.postgres_store as pg_store

    pg_store.get_embedding_byte_store = lambda *args, **kwargs: None


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST}; run without --dry-run first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _raw_page_texts(pdf: Path) -> list[str]:
    from langchain_community.document_loaders import PyMuPDFLoader

    return [(doc.page_content or "") for doc in PyMuPDFLoader(str(pdf)).load()]


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", (text or "").lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _gold_probes(page_text: str, width: int = 120, max_probes: int = 5) -> list[str]:
    norm = _norm(page_text)
    if not norm:
        return []
    if len(norm) <= width or max_probes <= 1:
        return [norm[:width]]
    last_start = len(norm) - width
    starts = sorted({round(i * last_start / (max_probes - 1)) for i in range(max_probes)})
    return [norm[start : start + width] for start in starts]


def _preview(text: str, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:limit]


def _parent_text(doc) -> str:
    meta = getattr(doc, "metadata", None) or {}
    return meta.get("parent_text") or doc.page_content or ""


def _doc_key(doc) -> str:
    meta = getattr(doc, "metadata", None) or {}
    return meta.get("parent_id") or meta.get("parent_text") or doc.page_content


def _rerank_score(doc) -> float:
    meta = getattr(doc, "metadata", None) or {}
    return float(meta.get("rerank_score", 0.0) or 0.0)


def _doc_entry(doc, rank: int, gold_page: int | None, probes: list[str], source: str, variant: str | None) -> dict:
    meta = getattr(doc, "metadata", None) or {}
    metadata_page = meta.get("page")
    human_page = int(metadata_page) + 1 if metadata_page is not None else None
    parent = _parent_text(doc)
    parent_norm = _norm(parent)
    matched = [probe for probe in probes if probe and probe in parent_norm]
    probe_match = matched[0] if matched else None
    return {
        "rank": rank,
        "source": source,
        "variant": variant,
        "title": meta.get("title"),
        "page": human_page,
        "metadata_page": metadata_page,
        "parent_id": meta.get("parent_id"),
        "rerank_score": meta.get("rerank_score"),
        "bm25_only": bool(meta.get("bm25_only")),
        "label_hit": bool(gold_page is not None and human_page == gold_page),
        "text_hit": bool(matched),
        "probe_matches": len(matched),
        "text_fraction": round(len(matched) / len(probes), 3) if probes else 0.0,
        "probe_match": probe_match[:80] if probe_match else None,
        "content_preview": _preview(doc.page_content),
        "parent_preview": _preview(parent),
    }


def _stage_result(docs: list[tuple], gold_page: int | None, probes: list[str], variants: list[str] | None = None) -> dict:
    entries = [
        _doc_entry(doc, rank, gold_page, probes, source, variant)
        for rank, (doc, source, variant) in enumerate(docs, 1)
    ]
    unique = {_doc_key(doc) for doc, _source, _variant in docs}
    parent_norms = [_norm(_parent_text(doc)) for doc, _source, _variant in docs]
    matched_probes = [
        probe for probe in probes
        if probe and any(probe in parent_norm for parent_norm in parent_norms)
    ]
    return {
        "variants": variants,
        "docs": entries,
        "n_docs": len(entries),
        "n_unique_docs": len(unique),
        "probes_total": len(probes),
        "probes_matched": len(matched_probes),
        "text_fraction": round(len(matched_probes) / len(probes), 3) if probes else 0.0,
        "label_present": any(entry["label_hit"] for entry in entries),
        "text_present": bool(matched_probes),
        "either_present": any(entry["label_hit"] for entry in entries) or bool(matched_probes),
    }


def _trace_question(question: dict, manifest: dict, scope_mode: str, raw_page: str, probes: list[str]) -> dict:
    from backend.rag_graph import _clean_docs, _query_variants, _select_evidence
    from backend.vector_store import (
        _get_vectorstore,
        _scope_filter,
        bm25_search,
        search as vs_search,
    )

    scope = [manifest["session_id"]]
    titles = [question["doc"]] if scope_mode == "isolated" else None
    dense_filter = _scope_filter(scope, titles, ABLATE_USER)
    dense_docs = [
        (doc, "dense", None)
        for doc in _get_vectorstore().similarity_search(
            question["input"], k=K, filter=dense_filter
        )
    ]
    variants = _query_variants(question["input"])
    search_docs: list[tuple] = []
    bm25_docs: list[tuple] = []
    for variant in variants:
        for doc in _clean_docs(
            vs_search(
                query=variant,
                scope=scope,
                k=K,
                titles=titles,
                user_id=ABLATE_USER,
            )
        ):
            search_docs.append((doc, "search", variant))
        for doc in _clean_docs(
            bm25_search(
                query=variant,
                scope=scope,
                k=BM25_K,
                titles=titles,
                user_id=ABLATE_USER,
            )
        ):
            bm25_docs.append((doc, "bm25", variant))

    merged: dict[str, tuple] = {}
    for doc, source, variant in search_docs + bm25_docs:
        meta = getattr(doc, "metadata", None) or {}
        key = meta.get("parent_id") or meta.get("parent_text") or doc.page_content
        score = _rerank_score(doc)
        current = merged.get(key)
        if current is None or score > _rerank_score(current[0]):
            merged[key] = (doc, source, variant)

    semantic: list[tuple] = []
    bm25_extras: list[tuple] = []
    seen: set[str] = set()
    for doc, source, variant in sorted(merged.values(), key=lambda item: _rerank_score(item[0]), reverse=True):
        meta = getattr(doc, "metadata", None) or {}
        key = meta.get("parent_id") or doc.page_content
        if key in seen:
            continue
        seen.add(key)
        if meta.get("bm25_only") and len(bm25_extras) < MAX_BM25_EXTRAS:
            bm25_extras.append((doc, source, variant))
        elif len(semantic) < K:
            semantic.append((doc, source, variant))

    pre_parent = bm25_extras + semantic[: max(0, K - len(bm25_extras))]
    expanded = _select_evidence([doc for doc, _source, _variant in pre_parent])
    final_docs = [(doc, "parent", None) for doc in expanded]

    stages = {
        "dense_arm": _stage_result(dense_docs, question["gold_page"], probes),
        "query_variants": _stage_result([], question["gold_page"], probes, variants=variants),
        "variant_search": _stage_result(search_docs, question["gold_page"], probes),
        "variant_bm25": _stage_result(bm25_docs, question["gold_page"], probes),
        "variant_pool": _stage_result(list(merged.values()), question["gold_page"], probes),
        "pre_parent_selection": _stage_result(pre_parent, question["gold_page"], probes),
        "final_parent_expanded": _stage_result(final_docs, question["gold_page"], probes),
    }
    flags = ["known_loss"] if question["qid"] in FLAGGED_QIDS else []
    return {
        "qid": question["qid"],
        "local_id": question["local_id"],
        "doc": question["doc"],
        "input": question["input"],
        "gold_page": question["gold_page"],
        "expected_pages": question["expected_pages"],
        "gold_page_preview": _preview(raw_page),
        "gold_probe_count": len(probes),
        "flags": flags,
        "variants": variants,
        "stages": stages,
    }


def _summarize(questions: list[dict]) -> dict:
    by_stage: dict[str, dict] = {}
    for stage in STAGE_NAMES:
        results = [question["stages"][stage] for question in questions]
        n = len(results)
        by_stage[stage] = {
            "questions": n,
            "label_present": sum(1 for result in results if result["label_present"]),
            "text_present": sum(1 for result in results if result["text_present"]),
            "either_present": sum(1 for result in results if result["either_present"]),
            "rate_label": round(sum(1 for result in results if result["label_present"]) / n, 3) if n else 0.0,
            "rate_text": round(sum(1 for result in results if result["text_present"]) / n, 3) if n else 0.0,
            "rate_either": round(sum(1 for result in results if result["either_present"]) / n, 3) if n else 0.0,
            "avg_text_fraction": round(sum(result["text_fraction"] for result in results) / n, 3) if n else 0.0,
            "avg_docs": round(sum(result["n_docs"] for result in results) / n, 2) if n else 0.0,
        }

    by_doc: dict[str, dict] = {}
    for doc in sorted({question["doc"] for question in questions}):
        doc_questions = [question for question in questions if question["doc"] == doc]
        by_doc[doc] = {}
        for stage in STAGE_NAMES:
            results = [question["stages"][stage] for question in doc_questions]
            n = len(results)
            by_doc[doc][stage] = {
                "questions": n,
                "label_present": sum(1 for result in results if result["label_present"]),
                "text_present": sum(1 for result in results if result["text_present"]),
                "either_present": sum(1 for result in results if result["either_present"]),
                "avg_text_fraction": round(sum(result["text_fraction"] for result in results) / n, 3) if n else 0.0,
            }
    return {
        "n_questions": len(questions),
        "flagged_qids": sorted(
            question["qid"] for question in questions if "known_loss" in question["flags"]
        ),
        "by_stage": by_stage,
        "by_doc": by_doc,
    }


def cmd_setup() -> dict:
    from backend.paper_loader import load_document
    from backend.vector_store import add_paper

    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} already exists; run --teardown first")

    session = f"{SESSION_PREFIX}{int(time.time())}"
    manifest = {
        "session_id": session,
        "user_id": ABLATE_USER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "docs": [],
    }
    _save(MANIFEST, manifest)
    for pdf in _pdfs():
        chunks = load_document(str(pdf))
        if not chunks:
            raise SystemExit(f"load_document produced 0 chunks for {pdf.name}")
        manifest["docs"].append({"pdf": pdf.name, "title": pdf.stem, "chunks": len(chunks)})
        _save(MANIFEST, manifest)
        add_paper(chunks, session, user_id=ABLATE_USER)
        print(f"[setup] {pdf.name}: {len(chunks)} chunks")
    print(f"[setup] session={session} user={ABLATE_USER}")
    return manifest


def cmd_teardown() -> None:
    from backend.vector_store import count_chunks, delete_paper

    if not MANIFEST.exists():
        print("[teardown] no stage manifest; nothing to delete")
        return
    manifest = _load_manifest()
    removed = 0
    user_id = manifest.get("user_id", ABLATE_USER)
    for doc in manifest.get("docs", []):
        removed += delete_paper(manifest["session_id"], doc["title"], user_id=user_id)
    remaining = count_chunks(manifest["session_id"], user_id=user_id)
    for _ in range(5):
        if remaining == 0:
            break
        time.sleep(1)
        remaining = count_chunks(manifest["session_id"], user_id=user_id)
    print(
        f"[teardown] deleted {removed} vectors for {manifest['session_id']}; "
        f"remaining={remaining}"
    )
    if remaining != 0:
        raise SystemExit(f"[teardown] residual vectors remain for {manifest['session_id']}: {remaining}")
    MANIFEST.unlink(missing_ok=True)
    print("[teardown] manifest removed")


def _print_env_targets() -> None:
    db_url = os.environ.get("DATABASE_URL", "")
    qdrant_url = os.environ.get("QDRANT_URL", "")
    db_host = urllib.parse.urlparse(db_url).hostname if db_url else None
    db_name = urllib.parse.urlparse(db_url).path.lstrip("/") if db_url else None
    qdrant_host = urllib.parse.urlparse(qdrant_url).hostname if qdrant_url else None
    print(f"[env] DATABASE_URL host={db_host or 'unset'} db={db_name or 'unset'}")
    print(f"[env] QDRANT_URL host={qdrant_host or 'unset'}")


def _print_dry_run(args: argparse.Namespace) -> None:
    print("[dry-run] no Qdrant writes; no Neon writes; no application code changes")
    print(f"[dry-run] collection={COLLECTION_NAME}")
    print(f"[dry-run] neon_table={NEON_TABLE}")
    print(f"[dry-run] cache_backend={'local' if args.local_cache else 'neon'}")
    print(f"[dry-run] user_id={ABLATE_USER}")
    print(f"[dry-run] session_id={SESSION_PREFIX}<unix_timestamp>")
    print(f"[dry-run] scope={args.scope} k={K} bm25_k={BM25_K}")
    print(f"[dry-run] pace_seconds={args.pace} setup_cooldown_seconds={args.setup_cooldown}")
    print(f"[dry-run] output={args.output}")
    print("[dry-run] stages=" + ", ".join(STAGE_NAMES))
    print("[dry-run] text_probes=up to 5 evenly spaced 120-char windows; fraction logged")
    print("[dry-run] manifest_before_ingest=True; teardown=finally; residual_count_required=0")
    print("[dry-run] flagged_qids=" + ",".join(sorted(FLAGGED_QIDS)))
    print("[dry-run] teardown delete calls:")
    print(
        "[dry-run] delete filter: "
        "metadata.session_id=<session_id> AND metadata.title=<title> "
        f"AND metadata.user_id={ABLATE_USER!r}"
    )
    for pdf in _pdfs():
        print(
            "  delete_paper("
            f"session_id=<session_id>, title={pdf.stem!r}, user_id={ABLATE_USER!r}"
            ")"
        )
    print(f"[dry-run] questions={len(_questions())}")
    _print_env_targets()


def _run(args: argparse.Namespace) -> None:
    if MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} already exists; run --teardown first")

    questions = _questions()
    if args.limit:
        questions = questions[: args.limit]

    if args.local_cache:
        _force_local_cache()

    try:
        cmd_setup()
        manifest = _load_manifest()
        if args.setup_cooldown:
            print(f"[pace] setup cooldown {args.setup_cooldown:.1f}s")
            time.sleep(args.setup_cooldown)
        raw_by_pdf = {pdf: _raw_page_texts(pdf) for pdf in _pdfs()}
        traced: list[dict] = []
        for index, question in enumerate(questions, 1):
            if index > 1 and args.pace:
                time.sleep(args.pace)
            raw_pages = raw_by_pdf[question["pdf"]]
            gold_page = question["gold_page"]
            if gold_page is None or not (1 <= gold_page <= len(raw_pages)):
                raise SystemExit(f"{question['qid']}: invalid gold page {gold_page}")
            raw_page = raw_pages[gold_page - 1]
            probes = _gold_probes(raw_page)
            result = _trace_question(question, manifest, args.scope, raw_page, probes)
            traced.append(result)
            print(
                f"[trace] {index:02d}/{len(questions)} "
                f"label={result['stages']['final_parent_expanded']['label_present']} "
                f"text={result['stages']['final_parent_expanded']['text_present']} "
                f"{question['qid']} {question['doc']} p{gold_page}"
            )

        payload = {
            "schema_version": "1.0",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "mode": "trace",
            "scope": args.scope,
            "k": K,
            "bm25_k": BM25_K,
            "collection": COLLECTION_NAME,
            "neon_table": NEON_TABLE,
            "cache_backend": "local" if args.local_cache else "neon",
            "pace_seconds": args.pace,
            "setup_cooldown_seconds": args.setup_cooldown,
            "user_id": ABLATE_USER,
            "session_id": manifest["session_id"],
            "documents": manifest.get("docs", []),
            "stages": list(STAGE_NAMES),
            "questions": traced,
            "summary": _summarize(traced),
        }
        _save(args.output, payload)
        print(f"[trace] wrote {args.output}")
        for stage, summary in payload["summary"]["by_stage"].items():
            print(
                f"[summary] {stage:22s} "
                f"label={summary['label_present']}/{summary['questions']} "
                f"text={summary['text_present']}/{summary['questions']} "
                f"either={summary['either_present']}/{summary['questions']} "
                f"avg_text_fraction={summary['avg_text_fraction']:.3f}"
            )
        print(f"[summary] flagged_qids={','.join(payload['summary']['flagged_qids'])}")
    finally:
        cmd_teardown()


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace the stock production retrieval pipeline by stage.")
    parser.add_argument("--dry-run", action="store_true", help="print targets and delete calls; write nothing")
    parser.add_argument("--teardown", action="store_true", help="delete vectors from an existing stage manifest and exit")
    parser.add_argument("--scope", choices=("isolated", "pooled"), default="isolated")
    parser.add_argument(
        "--local-cache",
        action="store_true",
        help="use the local embedding cache instead of the Neon embedding_cache table",
    )
    parser.add_argument("--pace", type=float, default=0.0, help="seconds to wait between questions")
    parser.add_argument(
        "--setup-cooldown",
        type=float,
        default=0.0,
        help="seconds to wait after ingestion before the first question",
    )
    parser.add_argument("--limit", type=int, default=0, help="trace only the first N questions (0 = all)")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="trace JSON path (default: ablation_stage_trace.json)",
    )
    args = parser.parse_args()

    if not args.output.is_absolute():
        args.output = ROOT / args.output

    if args.dry_run:
        _print_dry_run(args)
        return
    if args.teardown:
        if args.local_cache:
            _force_local_cache()
        cmd_teardown()
        return
    _run(args)


if __name__ == "__main__":
    main()


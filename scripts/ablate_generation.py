"""End-to-end A/B of retrieval configs: same questions, same generator, same judge.

Only the evidence handed to the generator differs. Every arm uses hybrid search
(dense + BM25 + RRF) with FlashRank rerank inside the vector store.

Arm kinds
  prod   stock `retrieve_merged`: query variants + BM25 rescue, top-3 evidence
  hyb    `hybrid_search` top-k, then an optional relative rerank cutoff: keep a
         chunk only if rerank_score >= rel * best_score (always keeps the best)

Each arm also fixes the chunker (size/overlap); arms that share a chunker share
one temporary Qdrant session, which is deleted in `finally`.

Usage
  uv run python scripts/ablate_generation.py --limit 2            # smoke
  uv run python scripts/ablate_generation.py                      # all arms
  uv run python scripts/ablate_generation.py --arms prod h3 h5r5

Outputs e2_parent_child_ab/results.json (cells are cached, so reruns resume).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(override=True)
for _key in ("LANGCHAIN_TRACING_V2", "LANGCHAIN_TRACING", "LANGSMITH_TRACING"):
    os.environ[_key] = "false"
for _key in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
    os.environ.pop(_key, None)
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")

# The local resolver intermittently fails for the Qdrant host (getaddrinfo 11001);
# retry lookups so one blip does not abort a paid run.
_real_getaddrinfo = socket.getaddrinfo


def _getaddrinfo_retry(*args, **kwargs):
    for attempt in range(8):
        try:
            return _real_getaddrinfo(*args, **kwargs)
        except socket.gaierror:
            if attempt == 7:
                raise
            time.sleep(0.4 * (attempt + 1))


socket.getaddrinfo = _getaddrinfo_retry


def quota_retry(fn, *args, **kwargs):
    """Gemini embeddings are capped at 100 requests/min on the free tier; wait and
    retry on 429 (successful batches are cached in the embedding cache, so a
    retry only re-pays for what has not been embedded yet)."""
    for attempt in range(8):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if "RESOURCE_EXHAUSTED" not in str(e) and "429" not in str(e):
                raise
            wait = 20 + 10 * attempt
            print(f"[quota] 429 from embeddings, sleeping {wait}s (attempt {attempt + 1}/8)", flush=True)
            time.sleep(wait)
    return fn(*args, **kwargs)

PDF_DIR = ROOT / "documents"
GOLDENS = ROOT / "goldens_manual_pairs.json"
OUT_DIR = ROOT / "e2_parent_child_ab"
CACHE = OUT_DIR / "cells.json"
RESULTS = OUT_DIR / "results.json"
METRICS = [
    "Contextual Precision", "Contextual Recall", "Contextual Relevancy",
    "Answer Relevancy", "Faithfulness",
]
USER = "lib-" + hashlib.sha256(b"ablate-gen@papeer.local").hexdigest()[:32]
K_PROD = 8  # candidate k for retrieve_merged (final evidence is 3 for k < 12)

ARMS: dict[str, dict] = {
    "prod":      {"chunk": (1000, 200), "kind": "prod"},
    "h3":        {"chunk": (1000, 200), "kind": "hyb", "k": 3, "rel": 0.0},
    "h5r5":      {"chunk": (1000, 200), "kind": "hyb", "k": 5, "rel": 0.5},
    "h5r7":      {"chunk": (1000, 200), "kind": "hyb", "k": 5, "rel": 0.7},
    "c600_h4":   {"chunk": (600, 120), "kind": "hyb", "k": 4, "rel": 0.0},
    "c600_h6r5": {"chunk": (600, 120), "kind": "hyb", "k": 6, "rel": 0.5},
}


def load_questions(limit: int) -> list[dict]:
    data = json.loads(GOLDENS.read_text(encoding="utf-8"))
    raw = data.get("docs") if isinstance(data.get("docs"), dict) else {
        k: v for k, v in data.items() if isinstance(v, dict) and "pairs" in v
    }
    out = []
    for name, entry in raw.items():
        pdf = PDF_DIR / name
        if not pdf.exists():
            raise SystemExit(f"missing PDF for goldens key {name!r}")
        pairs = entry["pairs"][:limit] if limit else entry["pairs"]
        for i, pair in enumerate(pairs):
            out.append({
                "qid": f"{pdf.stem[:12]}:{i:02d}",
                "pdf": pdf,
                "doc": pdf.stem,
                "input": pair["input"],
                "expected": pair["expected_output"],
                "page": pair.get("source_page"),
            })
    return out


def load_cache() -> dict:
    return json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}


def save_cache(cells: dict) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    CACHE.write_text(json.dumps(cells, ensure_ascii=False), encoding="utf-8")


def ingest(chunk: tuple[int, int], questions: list[dict], session: str) -> None:
    """Chunk the PDFs with the given splitter and index them under `session`."""
    import backend.paper_loader as pl
    from backend.vector_store import add_paper
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    size, overlap = chunk
    pl._splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, add_start_index=True
    )
    for pdf in sorted({q["pdf"] for q in questions}):
        chunks = pl.load_document(str(pdf))
        for i in range(0, len(chunks), 40):
            quota_retry(add_paper, chunks[i:i + 40], session, user_id=USER)
        print(f"[ingest {size}/{overlap}] {pdf.name}: {len(chunks)} chunks")


def retrieve(spec: dict, q: dict, session: str) -> list:
    import backend.rag_graph as rg
    from backend.vector_store import hybrid_search

    scope, titles = [session], [q["doc"]]
    if spec["kind"] == "prod":
        return quota_retry(rg.retrieve_merged, q["input"], K_PROD, scope, titles, USER)
    docs = quota_retry(hybrid_search, q["input"], scope, k=spec["k"], titles=titles, user_id=USER)
    docs = sorted(docs, key=rg._rerank_sort_key)
    rel = spec.get("rel", 0.0)
    if rel > 0 and docs:
        best = (docs[0].metadata or {}).get("rerank_score")
        if best:
            docs = [d for d in docs if ((d.metadata or {}).get("rerank_score") or 0) >= rel * best]
    return docs


def generate(q: dict, docs: list) -> str:
    import backend.rag_graph as rg

    if not docs:
        return "I don't know the answer."
    ctx = rg._format_context_with_citations(docs)
    if len(ctx) > rg._CONTEXT_CHARS:
        ctx = ctx[: rg._CONTEXT_CHARS]
    prompt = rg._INTEGRITY_SYSTEM + f"Context:\n\n{ctx}\n\nQuestion: {q['input']}"
    answer = rg.llm.invoke([{"role": "user", "content": prompt}]).content
    answer = rg._strip_bad_citations(answer, docs)
    return rg._strip_stray_refusal(answer)


def judge(task: tuple) -> tuple:
    import evaluate as ev
    from deepeval.test_case import LLMTestCase

    key, metric_name, case = task
    metric = ev.build_metrics()[METRICS.index(metric_name)]
    # Sync path: DeepEval's async path shares one event loop and deadlocks when
    # several worker threads call measure() at once.
    metric.async_mode = False
    tc = LLMTestCase(
        input=case["input"], actual_output=case["answer"],
        expected_output=case["expected"], retrieval_context=case["contexts"],
    )
    try:
        metric.measure(tc, _show_indicator=False)
        return key, metric_name, {"score": metric.score, "reason": (metric.reason or "")[:400]}
    except Exception as e:  # noqa: BLE001 - recorded, not fatal for the batch
        return key, metric_name, {"score": None, "error": str(e)[:300]}


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def boot_ci(deltas: list[float], n: int = 5000, seed: int = 42) -> tuple[float, float]:
    rng = random.Random(seed)
    means = sorted(mean([rng.choice(deltas) for _ in deltas]) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def score_of(cells: dict, qid: str, arm: str, metric: str):
    return cells.get(f"{qid}|{arm}", {}).get("scores", {}).get(metric, {}).get("score")


def summarize(cells: dict, arms: list[str], qids: list[str]) -> dict:
    out = {"arms": {}, "paired_vs_prod": {}}
    for arm in arms:
        rows = [cells[f"{qid}|{arm}"] for qid in qids if f"{qid}|{arm}" in cells]
        entry = {
            "n": len(rows),
            "avg_ctx_chunks": mean([len(r["contexts"]) for r in rows]),
            "avg_ctx_chars": mean([sum(map(len, r["contexts"])) for r in rows]),
            "page_hit": mean([1.0 if r["page_hit"] else 0.0 for r in rows if r["page_hit"] is not None]),
            "metrics": {},
        }
        for m in METRICS:
            scores = [(r.get("scores") or {}).get(m, {}).get("score") for r in rows]
            entry["metrics"][m] = {
                "mean": mean(scores),
                "pass": sum(1 for s in scores if s is not None and s >= 0.7),
            }
        entry["mean_of_5"] = mean([v["mean"] for v in entry["metrics"].values()])
        out["arms"][arm] = entry
    if "prod" in arms:
        for arm in arms:
            if arm == "prod":
                continue
            d = {}
            for m in METRICS:
                deltas = [
                    score_of(cells, qid, arm, m) - score_of(cells, qid, "prod", m)
                    for qid in qids
                    if score_of(cells, qid, arm, m) is not None
                    and score_of(cells, qid, "prod", m) is not None
                ]
                if deltas:
                    lo, hi = boot_ci(deltas)
                    d[m] = {"mean_delta": round(mean(deltas), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(deltas)}
            out["paired_vs_prod"][arm] = d
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS))
    ap.add_argument("--limit", type=int, default=0, help="questions per document (0 = all)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--keep", action="store_true", help="leave the Qdrant sessions in place")
    args = ap.parse_args()

    from backend.vector_store import delete_paper

    questions = load_questions(args.limit)
    qids = [q["qid"] for q in questions]
    titles = sorted({q["doc"] for q in questions})
    cells = load_cache()
    started = time.time()
    # one Qdrant session per distinct chunker
    sessions: dict[tuple, str] = {}
    for arm in args.arms:
        sessions.setdefault(ARMS[arm]["chunk"], f"ablgen_{int(started)}_{len(sessions)}")
    print(f"[plan] {len(questions)} questions x {len(args.arms)} arms; chunkers={list(sessions)}")
    try:
        for chunk, session in sessions.items():
            if all(f"{q['qid']}|{a}" in cells and cells[f"{q['qid']}|{a}"].get("answer") is not None
                   for a in args.arms if ARMS[a]["chunk"] == chunk for q in questions):
                continue  # everything for this chunker is cached
            ingest(chunk, questions, session)

        for arm in args.arms:
            spec, session = ARMS[arm], sessions[ARMS[arm]["chunk"]]
            for q in questions:
                key = f"{q['qid']}|{arm}"
                if key in cells and cells[key].get("answer") is not None:
                    continue
                docs = retrieve(spec, q, session)
                pages = [int((d.metadata or {}).get("page", -1)) + 1 for d in docs]
                cells[key] = {
                    "input": q["input"], "expected": q["expected"], "arm": arm, "qid": q["qid"],
                    "contexts": [d.page_content for d in docs],
                    "pages": pages,
                    "page_hit": (q["page"] in pages) if q["page"] is not None else None,
                    "answer": generate(q, docs),
                    "scores": {},
                }
                save_cache(cells)
            print(f"[gen] arm {arm} done")

        tasks = [
            (key, m, cell)
            for key, cell in cells.items()
            if cell["arm"] in args.arms and cell["qid"] in qids
            for m in METRICS
            if m not in cell["scores"] or cell["scores"][m].get("score") is None
        ]
        print(f"[judge] {len(tasks)} metric calls on {args.workers} workers")
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(judge, t) for t in tasks]
            for n, fut in enumerate(as_completed(futures), 1):
                key, m, res = fut.result()
                cells[key]["scores"][m] = res
                if n % 10 == 0 or n == len(tasks):
                    save_cache(cells)
                    print(f"[judge] {n}/{len(tasks)}", flush=True)
        save_cache(cells)

        summary = summarize(cells, args.arms, qids)
        summary["seconds"] = round(time.time() - started, 1)
        RESULTS.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({a: {"chunks": round(v["avg_ctx_chunks"], 1), "chars": round(v["avg_ctx_chars"]),
                              "page_hit": v["page_hit"], "mean_of_5": round(v["mean_of_5"], 3),
                              **{m.split()[-1][:6] + m.split()[0][:3]: round(x["mean"], 3) for m, x in v["metrics"].items()}}
                          for a, v in summary["arms"].items()}, indent=1))
    finally:
        if not args.keep:
            for chunk, session in sessions.items():
                try:
                    removed = sum(delete_paper(session, t, user_id=USER) for t in titles)
                    print(f"[teardown] deleted {removed} vectors for {session}")
                except Exception as e:  # noqa: BLE001
                    print(f"[teardown] FAILED for {session}: {e} (delete manually)")


if __name__ == "__main__":
    main()

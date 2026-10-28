"""Retrieval-only ablation harness for Papeer (does not touch app code).

Experiments
-----------
audit   Route the 44 hand-authored questions through the real router and report
        how many avoid the vector store. Without this, retrieval comparisons
        are not representative of production behaviour.

arm     Retrieve with one strategy and score page-level recall / MRR against
        the `source_page` labels in `goldens_manual_pairs.json`. No generation,
        no LLM judge, no DeepEval.

Arms (all k=8, one title filter per question)
  dense          dense vector search only
  dense_rerank   dense overfetch + FlashRank top-8 (relevance floor disabled
                 so the reranker is compared on coverage, not on vetoing docs)
  hybrid         dense + BM25 RRF + FlashRank top-8 (floor disabled)
  production     stock `retrieve_merged`: query variants + hybrid + BM25
                 rescue union, top-3 evidence (what the app actually does;
                 not part of the E1 headline because it returns fewer chunks)

`all` runs setup -> dense, dense_rerank, hybrid, production -> compare ->
teardown, with each arm in its own subprocess (module-level env switches).

Usage
  uv run python scripts/ablate.py all
  uv run python scripts/ablate.py audit
  uv run python scripts/ablate.py setup
  uv run python scripts/ablate.py arm --arm hybrid
  uv run python scripts/ablate.py compare
  uv run python scripts/ablate.py teardown

Outputs: ablation_<arm>.json, ablation_audit.json, ablation_compare.json,
ablation_session.json. All are temporary; delete after the experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(override=True)
# Ablation runs are local experiments; keep them out of LangSmith (the monthly
# trace quota is already exhausted and the 429 noise hides real output).
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
for _key in ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY"):
    os.environ.pop(_key, None)

PDF_DIR = ROOT / "documents"
GOLDENS = ROOT / "goldens_manual_pairs.json"
MANIFEST = ROOT / "ablation_session.json"
AUDIT_OUT = ROOT / "ablation_audit.json"
COMPARE_OUT = ROOT / "ablation_compare.json"
ARMS = ("dense", "dense_rerank", "hybrid", "production")
K = 8
ABLATE_USER = "lib-" + hashlib.sha256(b"ablate@papeer.local").hexdigest()[:32]


# ── Data ─────────────────────────────────────────────────────────────────────

def _squash(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _pdfs() -> list[Path]:
    return sorted(PDF_DIR.glob("*.pdf"))


def _questions() -> list[dict]:
    if not GOLDENS.exists():
        raise SystemExit(f"missing {GOLDENS}")
    data = json.loads(GOLDENS.read_text(encoding="utf-8"))
    raw = data.get("docs") if isinstance(data.get("docs"), dict) else {
        k: v for k, v in data.items() if isinstance(v, dict) and "pairs" in v
    }
    by_squash = {_squash(p.name): p for p in _pdfs()}
    out = []
    for name, entry in raw.items():
        pdf = by_squash.get(_squash(name))
        if pdf is None:
            raise SystemExit(f"no PDF matches goldens key {name!r}")
        for pair in entry["pairs"]:
            page = pair.get("source_page")
            out.append({
                "pdf": pdf,
                "doc": pdf.stem,
                "input": pair["input"],
                "expected_pages": [int(page)] if page is not None else [],
            })
    return out


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST}; run `setup` first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _save(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# ── Setup / teardown ─────────────────────────────────────────────────────────

def cmd_setup(args: argparse.Namespace) -> None:
    from backend.paper_loader import load_document
    from backend.vector_store import add_paper, delete_paper

    if MANIFEST.exists():
        if not args.force:
            print(f"[setup] reusing {MANIFEST}")
            return
        _delete_manifest_vectors()
        MANIFEST.unlink(missing_ok=True)

    session = f"ablate_{int(time.time())}"
    docs_meta = []
    started = time.time()
    for pdf in _pdfs():
        started_doc = time.time()
        chunks = load_document(str(pdf))
        if not chunks:
            raise SystemExit(f"load_document produced 0 chunks for {pdf.name}")
        add_paper(chunks, session, user_id=ABLATE_USER)
        docs_meta.append({
            "pdf": pdf.name,
            "title": pdf.stem,
            "chunks": len(chunks),
            "seconds": round(time.time() - started_doc, 1),
        })
        print(f"[setup] {pdf.name}: {len(chunks)} chunks")
    manifest = {
        "session_id": session,
        "user_id": ABLATE_USER,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 1),
        "docs": docs_meta,
    }
    _save(MANIFEST, manifest)
    print(f"[setup] session={session} user={ABLATE_USER} in {manifest['seconds']}s")


def _delete_manifest_vectors() -> None:
    if not MANIFEST.exists():
        return
    from backend.vector_store import delete_paper

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    removed = 0
    for doc in manifest.get("docs", []):
        removed += delete_paper(manifest["session_id"], doc["title"], user_id=manifest["user_id"])
    print(f"[teardown] deleted {removed} vectors for {manifest.get('session_id')}")


def cmd_teardown(args: argparse.Namespace) -> None:
    _delete_manifest_vectors()
    MANIFEST.unlink(missing_ok=True)
    print("[teardown] manifest removed")


# ── Audit ────────────────────────────────────────────────────────────────────

def cmd_audit(args: argparse.Namespace) -> None:
    from langchain_core.messages import HumanMessage

    from backend.rag_graph import router_node

    questions = _questions()
    if args.limit:
        questions = questions[: args.limit]
    rows = []
    started = time.time()
    for i, q in enumerate(questions, 1):
        state = {
            "messages": [HumanMessage(content=q["input"])],
            "session_id": "audit",
            "doc_scope_ids": ["audit"],
            "doc_titles": [q["doc"]],
            "user_id": ABLATE_USER,
            "query": q["input"],
        }
        route = router_node(state).get("route")
        rows.append({"doc": q["doc"], "input": q["input"], "route": route})
        print(f"[audit] {i:02d}/{len(questions)} {route:12s} {q['input'][:60]}")

    by_doc: dict[str, dict[str, int]] = {}
    for row in rows:
        by_doc.setdefault(row["doc"], {}).setdefault(row["route"], 0)
        by_doc[row["doc"]][row["route"]] += 1
    leaked = [r for r in rows if r["route"] != "paper_qa"]
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seconds": round(time.time() - started, 1),
        "n": len(rows),
        "routes": {r: sum(1 for x in rows if x["route"] == r) for r in sorted({x["route"] for x in rows})},
        "by_doc": by_doc,
        "non_paper_qa": leaked,
        "questions": rows,
    }
    _save(AUDIT_OUT, payload)
    print(f"\n[audit] routes: {payload['routes']}")
    for doc, counts in by_doc.items():
        print(f"[audit] {doc}: {counts}")
    print(f"[audit] {len(leaked)} question(s) bypass the vector store -> {AUDIT_OUT}")


# ── Retrieval arms ───────────────────────────────────────────────────────────

def _retrieve(arm: str, query: str, manifest: dict, title: str, scope_mode: str) -> list:
    from backend.rag_graph import retrieve_merged
    from backend.vector_store import (
        _RERANK_CANDIDATES,
        _flashrank_rerank,
        _get_vectorstore,
        _scope_filter,
        hybrid_search,
    )

    scope = [manifest["session_id"]]
    user = manifest["user_id"]
    titles = [title] if scope_mode == "isolated" else None
    filt = _scope_filter(scope, titles, user)

    if arm == "dense":
        return _get_vectorstore().similarity_search(query, k=K, filter=filt)
    if arm == "dense_rerank":
        candidates = _get_vectorstore().similarity_search(
            query, k=max(K * 2, _RERANK_CANDIDATES), filter=filt
        )
        return _flashrank_rerank(query, candidates, K)
    if arm == "hybrid":
        return hybrid_search(query, scope, k=K, titles=titles, user_id=user)
    if arm == "production":
        return retrieve_merged(query, K, scope, titles, user)
    raise SystemExit(f"unknown arm {arm!r}")


def _recall_at(returned: list[dict], doc: str, expected: set[int], cutoff: int) -> float | None:
    if not expected:
        return None
    pages = {r["page"] for r in returned[:cutoff] if r["title"] == doc}
    return len(expected & pages) / len(expected)


def _row_recall_at(row: dict, cutoff: int) -> float | None:
    known = row.get(f"recall_at_{cutoff}")
    if known is not None:
        return known
    expected = set(row.get("expected_pages") or [])
    if not expected:
        return None
    pages = {
        r["page"] for r in (row.get("returned") or [])[:cutoff]
        if r.get("title") == row.get("doc")
    }
    return len(expected & pages) / len(expected)


def _score_returned(docs: list, expected: list[int], expected_doc: str) -> dict:
    returned = []
    for rank, d in enumerate(docs, 1):
        md = getattr(d, "metadata", None) or {}
        page = md.get("page")
        returned.append({
            "rank": rank,
            "title": md.get("title"),
            "page": int(page) + 1 if page is not None else None,
            "score": md.get("rerank_score"),
            "preview": (d.page_content or "")[:160].replace("\n", " "),
        })
    expected_set = set(expected)
    hits = {
        r["page"] for r in returned
        if r["title"] == expected_doc and r["page"] in expected_set
    }
    recall = len(hits) / len(expected_set) if expected_set else None
    mrr = 0.0
    for r in returned:
        if r["title"] == expected_doc and r["page"] in expected_set:
            mrr = 1.0 / r["rank"]
            break
    return {
        "returned": returned,
        "recall": recall,
        "recall_at_3": _recall_at(returned, expected_doc, expected_set, 3),
        "recall_at_5": _recall_at(returned, expected_doc, expected_set, 5),
        "mrr": mrr,
        "any_hit": bool(hits),
    }


def cmd_arm(args: argparse.Namespace) -> None:
    manifest = _load_manifest()
    questions = _questions()
    if args.limit:
        questions = questions[: args.limit]

    rows = []
    started = time.time()
    for i, q in enumerate(questions, 1):
        docs = _retrieve(args.arm, q["input"], manifest, q["doc"], args.scope)
        scored = _score_returned(docs, q["expected_pages"], q["doc"])
        rows.append({
            "doc": q["doc"],
            "input": q["input"],
            "expected_pages": q["expected_pages"],
            **scored,
        })
        recall = "-" if scored["recall"] is None else f"{scored['recall']:.2f}"
        print(
            f"[{args.arm}] {i:02d}/{len(questions)} recall={recall} "
            f"returned={len(scored['returned'])} {q['input'][:55]}"
        )
    with_recall = [r for r in rows if r["recall"] is not None]
    summary = {
        "n": len(rows),
        "recall": round(sum(r["recall"] for r in with_recall) / len(with_recall), 3)
        if with_recall else None,
        "recall_at_3": round(_mean([r["recall_at_3"] for r in rows if r["recall_at_3"] is not None]) or 0.0, 3),
        "recall_at_5": round(_mean([r["recall_at_5"] for r in rows if r["recall_at_5"] is not None]) or 0.0, 3),
        "mrr": round(sum(r["mrr"] for r in rows) / len(rows), 3) if rows else None,
        "any_hit": round(sum(r["any_hit"] for r in rows) / len(rows), 3) if rows else None,
        "avg_returned": round(sum(len(r["returned"]) for r in rows) / len(rows), 2) if rows else None,
        "seconds": round(time.time() - started, 1),
        "by_doc": {},
    }
    for doc in sorted({r["doc"] for r in rows}):
        sel = [r for r in rows if r["doc"] == doc and r["recall"] is not None]
        if sel:
            summary["by_doc"][doc] = {
                "n": len(sel),
                "recall": round(sum(r["recall"] for r in sel) / len(sel), 3),
                "recall_at_3": round(_mean([r["recall_at_3"] for r in sel if r["recall_at_3"] is not None]) or 0.0, 3),
                "mrr": round(sum(r["mrr"] for r in sel) / len(sel), 3),
            }

    payload = {
        "arm": args.arm,
        "k": K,
        "scope": args.scope,
        "settings": {
            "rerank": os.environ.get("PAPEER_RERANK", "1"),
            "min_rerank_score": os.environ.get("PAPEER_MIN_SCORE", "0.05"),
        },
        "user_id": manifest["user_id"],
        "session_id": manifest["session_id"],
        "questions": rows,
        "summary": summary,
    }
    path = ROOT / f"ablation_{args.arm}{'' if args.scope == 'isolated' else '_' + args.scope}.json"
    _save(path, payload)
    print(f"\n[ablation:{args.arm}] {summary}")
    print(f"[ablation:{args.arm}] -> {path}")


# ── Compare ──────────────────────────────────────────────────────────────────

def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def cmd_compare(args: argparse.Namespace) -> None:
    runs = {}
    suffix = "" if args.scope == "isolated" else "_" + args.scope
    for arm in ARMS:
        path = ROOT / f"ablation_{arm}{suffix}.json"
        if path.exists():
            runs[arm] = json.loads(path.read_text(encoding="utf-8"))
    if len(runs) < 2:
        raise SystemExit("need at least two ablation_<arm>.json files")

    print(f"\nARM SUMMARY (page-level, scope={args.scope})")
    print(f"{'arm':14s} {'recall':>7s} {'r@3':>5s} {'r@5':>5s} {'mrr':>6s} {'any_hit':>8s} {'returned':>9s}")
    for arm, run in runs.items():
        s = run["summary"]
        r3 = s.get("recall_at_3")
        if r3 is None:
            vals = [v for r in run["questions"] if (v := _row_recall_at(r, 3)) is not None]
            r3 = round(_mean(vals), 3) if vals else "-"
        r5 = s.get("recall_at_5")
        if r5 is None:
            vals = [v for r in run["questions"] if (v := _row_recall_at(r, 5)) is not None]
            r5 = round(_mean(vals), 3) if vals else "-"
        print(
            f"{arm:14s} {s['recall'] if s['recall'] is not None else '-':>7} "
            f"{r3:>5} {r5:>5} "
            f"{s['mrr'] if s['mrr'] is not None else '-':>6} "
            f"{s['any_hit'] if s['any_hit'] is not None else '-':>8} "
            f"{s['avg_returned'] if s['avg_returned'] is not None else '-':>9}"
        )

    base = "dense"
    if base not in runs:
        raise SystemExit("dense arm missing; cannot compare")
    base_rows = {(r["doc"], r["input"]): r for r in runs[base]["questions"]}
    comparisons = {}
    for arm in ("dense_rerank", "hybrid"):
        if arm not in runs:
            continue
        run = runs[arm]
        rows = []
        for r in run["questions"]:
            old = base_rows.get((r["doc"], r["input"]))
            if old is None or r["recall"] is None or old["recall"] is None:
                continue
            delta = r["recall"] - old["recall"]
            rows.append({
                "doc": r["doc"],
                "input": r["input"],
                "dense_recall": old["recall"],
                "arm_recall": r["recall"],
                "delta": round(delta, 4),
                "outcome": "win" if delta > 0 else ("loss" if delta < 0 else "tie"),
            })
        comparisons[arm] = {
            "n": len(rows),
            "mean_delta": round(_mean([r["delta"] for r in rows]) or 0.0, 3),
            "wins": sum(1 for r in rows if r["outcome"] == "win"),
            "losses": sum(1 for r in rows if r["outcome"] == "loss"),
            "ties": sum(1 for r in rows if r["outcome"] == "tie"),
            "rows": rows,
        }

    print(f"\nPAIRED VS {base.upper()} (E1 child-level page recall)")
    print(f"{'arm':14s} {'n':>3s} {'delta':>7s} {'win':>5s} {'loss':>5s} {'tie':>5s}")
    for arm, c in comparisons.items():
        print(f"{arm:14s} {c['n']:3d} {c['mean_delta']:>+7.3f} {c['wins']:5d} {c['losses']:5d} {c['ties']:5d}")

    if "production" in runs:
        s = runs["production"]["summary"]
        print(
            "\nSTOCK PIPELINE (not an E1 arm; includes query variants, BM25 rescue "
            f"and parent expansion): recall={s['recall']} mrr={s['mrr']} "
            f"any_hit={s['any_hit']} returned={s['avg_returned']}"
        )
        print("[compare] production page recall is not comparable to child-level arms: "
              "parent expansion reduces the number of distinct pages returned")

    _save(ROOT / f"ablation_compare{suffix}.json", {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "k": K,
        "scope": args.scope,
        "summaries": {arm: run["summary"] for arm, run in runs.items()},
        "vs_dense": comparisons,
        "stock_pipeline": runs.get("production", {}).get("summary"),
    })
    print(f"\n[compare] -> ablation_compare{suffix}.json")
    print("[compare] limitation: n=44, page-level labels, only large effects are detectable")


# ── All ──────────────────────────────────────────────────────────────────────

def cmd_all(args: argparse.Namespace) -> None:
    cmd_setup(args)
    base_env = dict(os.environ)
    base_env["PAPEER_RERANK"] = "1"
    try:
        for arm in ARMS:
            print(f"\n=== arm: {arm} ===")
            env = dict(base_env)
            if arm in ("dense_rerank", "hybrid"):
                # Compare coverage at k, not the reranker's veto behaviour.
                env["PAPEER_MIN_SCORE"] = "-1"
            else:
                env.pop("PAPEER_MIN_SCORE", None)
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "arm", "--arm", arm,
                 "--scope", args.scope],
                cwd=str(ROOT),
                env=env,
            )
            if result.returncode != 0:
                raise SystemExit(f"arm {arm} failed with exit {result.returncode}")
        cmd_compare(args)
    finally:
        if not args.keep:
            cmd_teardown(args)
        else:
            print("[all] --keep set; session left in place (run teardown when done)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("setup", help="ingest the 4 PDFs into a temporary session")
    p.add_argument("--force", action="store_true", help="delete an existing ablation session first")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("teardown", help="delete the temporary session vectors")
    p.set_defaults(func=cmd_teardown)

    p = sub.add_parser("audit", help="route the questions through the real router")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("arm", help="run one retrieval arm")
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--scope", choices=("isolated", "pooled"), default="isolated")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_arm)

    p = sub.add_parser("compare", help="paired comparison of existing arm outputs")
    p.add_argument("--scope", choices=("isolated", "pooled"), default="isolated")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("all", help="setup + every arm + compare + teardown")
    p.add_argument("--scope", choices=("isolated", "pooled"), default="isolated")
    p.add_argument("--force", action="store_true")
    p.add_argument("--keep", action="store_true", help="leave the ablation session in Qdrant")
    p.set_defaults(func=cmd_all)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

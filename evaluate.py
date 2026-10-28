import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(override=True)
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent
DOCUMENTS_DIR = ROOT / "documents"
MANUAL_GOLDENS = ROOT / "goldens_manual_pairs.json"
RESULTS_V4 = ROOT / "eval_results_v4.json"
CHECKPOINT_DB = ROOT / "eval_checkpoints.db"

METRIC_THRESHOLD = 0.7
MAX_CONCURRENT = int(os.environ.get("EVAL_MAX_CONCURRENT", "4"))
THROTTLE_SECONDS = float(os.environ.get("EVAL_THROTTLE_SECONDS", "0.5"))

EVAL_USER = "lib-" + hashlib.sha256(b"eval@papeer.local").hexdigest()[:32]

METRIC_NAMES = [
    "Contextual Precision",
    "Contextual Recall",
    "Contextual Relevancy",
    "Answer Relevancy",
    "Faithfulness",
]


class OpenAIModel(DeepEvalBaseLLM):
    """DeepEval judge backed by OpenAI's chat endpoint."""

    def __init__(self, model: str | None = None):
        self.model_name = model or os.environ.get("DEEPEVAL_JUDGE_MODEL", "gpt-4o-mini")
        self.chat = ChatOpenAI(
            model=self.model_name,
            api_key=os.environ.get("OPENAI_API_KEY"),
            temperature=0,
        )
        super().__init__(model=self.model_name)

    def load_model(self):
        return self.chat

    def generate(self, prompt: str, schema=None):
        if schema:
            for method in ("function_calling", "json_schema"):
                try:
                    structured = self.chat.with_structured_output(schema, method=method)
                except Exception:
                    continue
                for _ in range(3):
                    try:
                        result = self._retry(structured.invoke, prompt)
                    except (ValidationError, OutputParserException):
                        continue  # model returned a malformed schema; ask again
                    if isinstance(result, schema):
                        return result
            raise RuntimeError(
                f"structured output produced no {schema.__name__} (model returned None)"
            )
        return self._retry(self.chat.invoke, prompt).content or ""

    async def a_generate(self, prompt: str, schema=None):
        if schema:
            for method in ("function_calling", "json_schema"):
                try:
                    structured = self.chat.with_structured_output(schema, method=method)
                except Exception:
                    continue
                for _ in range(3):
                    try:
                        result = await self._aretry(structured.ainvoke, prompt)
                    except (ValidationError, OutputParserException):
                        continue  # model returned a malformed schema; ask again
                    if isinstance(result, schema):
                        return result
            raise RuntimeError(
                f"structured output produced no {schema.__name__} (model returned None)"
            )
        res = await self._aretry(self.chat.ainvoke, prompt)
        return res.content or ""

    def get_model_name(self) -> str:
        return self.model_name

    def _is_transient(self, e: Exception) -> bool:
        text = str(e).lower()
        return any(
            marker in text
            for marker in (
                "429",
                "rate limit",
                "conflict",
                "409",
                "duplicate request",
                "524",
                "timeout",
                "internal server error",
                "502",
                "503",
                "connection error",
            )
        )

    def _wait_for(self, e: Exception) -> float:
        text = str(e).lower()
        if "conflict" in text or "409" in text or "duplicate request" in text:
            return 6.0
        if "524" in text or "connection error" in text:
            return 45.0
        return 30.0

    def _retry(self, call, *args, retries: int = 6, **kwargs):
        for attempt in range(retries):
            try:
                return call(*args, **kwargs)
            except Exception as e:
                if not self._is_transient(e) or attempt == retries - 1:
                    raise
                wait = self._wait_for(e)
                print(
                    f"[judge retry] waiting {wait:.0f}s "
                    f"(attempt {attempt + 1}/{retries - 1})..."
                )
                time.sleep(wait)

    async def _aretry(self, call, *args, retries: int = 6, **kwargs):
        for attempt in range(retries):
            try:
                return await call(*args, **kwargs)
            except Exception as e:
                if not self._is_transient(e) or attempt == retries - 1:
                    raise
                wait = self._wait_for(e)
                print(
                    f"[judge retry] waiting {wait:.0f}s "
                    f"(attempt {attempt + 1}/{retries - 1})..."
                )
                await asyncio.sleep(wait)


eval_model = OpenAIModel()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manual_goldens() -> dict:
    if not MANUAL_GOLDENS.exists():
        raise RuntimeError(f"Missing manual goldens file: {MANUAL_GOLDENS}")
    data = json.loads(MANUAL_GOLDENS.read_text(encoding="utf-8"))
    # Two layouts exist: {"docs": {name: entry}} and {name: entry} at top level.
    docs = data.get("docs") if isinstance(data.get("docs"), dict) else {
        key: entry for key, entry in data.items()
        if isinstance(entry, dict) and "pairs" in entry
    }
    if not docs:
        raise RuntimeError(f"{MANUAL_GOLDENS} has no goldens entries with 'pairs'")
    return docs


def manual_entry_for_pdf(pdf: Path, docs: dict) -> tuple[str, dict]:
    digest = file_sha256(pdf)
    for key, entry in docs.items():
        if entry.get("sha256") == digest:
            return key, entry
    for key, entry in docs.items():
        if entry.get("filename") == pdf.name:
            print(
                f"[goldens] {pdf.name}: filename match but hash differs; "
                f"using manual entry {key}"
            )
            return key, entry
    raise RuntimeError(
        f"No manual goldens entry for {pdf.name} "
        f"(sha256 {digest[:12]})"
    )


def clean_pairs(entry: dict) -> list[dict]:
    pairs = []
    for index, pair in enumerate(entry.get("pairs") or [], 1):
        question = str(pair.get("input") or "").strip()
        expected = str(pair.get("expected_output") or "").strip()
        if not question or not expected:
            raise RuntimeError(f"Manual golden {index} is missing input/expected_output")
        page = int(pair.get("source_page") or 0)
        pairs.append(
            {
                "input": question,
                "expected_output": expected,
                "source_page": page,
            }
        )
    if not pairs:
        raise RuntimeError("Manual goldens entry has no pairs")
    return pairs


def sweep_stale_eval_points() -> int:
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=120,
    )
    flt = Filter(
        must=[
            FieldCondition(
                key="metadata.user_id",
                match=MatchValue(value=EVAL_USER),
            )
        ]
    )
    stale = client.count("papeer_docs", exact=True, count_filter=flt).count
    if stale:
        client.delete("papeer_docs", points_selector=flt, wait=True)
    return stale


def run_rag_query(
    graph,
    query: str,
    session_id: str,
    thread_id: str,
    doc_title: str,
) -> tuple[str, list[str], list[dict], list[str]]:
    config = {"configurable": {"thread_id": thread_id}}
    final_state = graph.invoke(
        {
            "messages": [HumanMessage(content=query)],
            "session_id": session_id,
            "doc_scope_ids": [session_id],
            "doc_titles": [doc_title],
            "user_id": EVAL_USER,
            "query": query,
            "route": None,
            "retrieved_docs": [],
            "retrieval_attempts": 0,
            "claim_verdict": None,
            "claim_source": None,
            "superseding_papers": [],
            "answer": None,
            "is_relevant": None,
            "rewrite_count": 0,
        },
        config=config,
    )
    answer = final_state.get("answer") or ""
    docs = final_state.get("retrieved_docs") or []
    retrieval_context = [doc.page_content for doc in docs]
    pages = [
        {
            "title": (doc.metadata or {}).get("title"),
            "page": int((doc.metadata or {}).get("page", 0)) + 1,
        }
        for doc in docs
    ]
    full_context = [doc.page_content for doc in docs]
    return answer, retrieval_context, pages, full_context


def build_metrics() -> list:
    return [
        ContextualPrecisionMetric(threshold=METRIC_THRESHOLD, model=eval_model),
        ContextualRecallMetric(threshold=METRIC_THRESHOLD, model=eval_model),
        ContextualRelevancyMetric(threshold=METRIC_THRESHOLD, model=eval_model),
        AnswerRelevancyMetric(threshold=METRIC_THRESHOLD, model=eval_model),
        FaithfulnessMetric(threshold=METRIC_THRESHOLD, model=eval_model),
    ]


def summarize(results) -> list[dict]:
    summary = []
    for test_result in results.test_results:
        summary.append(
            {
                "input": test_result.input,
                "actual_output": test_result.actual_output,
                "success": test_result.success,
                "metrics": [
                    {
                        "name": m.name,
                        "score": m.score,
                        "passed": m.success,
                        "reason": m.reason,
                    }
                    for m in test_result.metrics_data
                ],
            }
        )
    return summary


def metric_stats(cases: list[dict]) -> dict:
    stats = {}
    for name in METRIC_NAMES:
        scores, passed = [], 0
        for case in cases:
            for metric in case["metrics"]:
                if metric["name"] != name:
                    continue
                if metric["score"] is not None:
                    scores.append(float(metric["score"]))
                passed += int(bool(metric["passed"]))
        stats[name] = {
            "avg": round(sum(scores) / len(scores), 3) if scores else None,
            "pass": passed,
            "n": len(cases),
        }
    return stats


def stat_payload(stats: dict) -> dict:
    return {
        name: {
            "avg": stats[name]["avg"],
            "pass": stats[name]["pass"],
            "n": stats[name]["n"],
        }
        for name in METRIC_NAMES
    }


def fmt_stat(stat: dict) -> str:
    avg = "n/a" if stat["avg"] is None else f"{stat['avg']:.2f}"
    return f"{avg} ({stat['pass']}/{stat['n']})"


def print_table(per_doc: dict, overall: dict) -> None:
    headers = ["Document"] + METRIC_NAMES
    rows = [
        [title] + [fmt_stat(data["stats"][name]) for name in METRIC_NAMES]
        for title, data in per_doc.items()
    ]
    rows.append(["OVERALL"] + [fmt_stat(overall[name]) for name in METRIC_NAMES])
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    line = "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers))
    print("\n" + line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def compute_flags(per_doc: dict, overall: dict) -> list[str]:
    flags = []
    overall_relevancy = overall["Contextual Relevancy"]["avg"]
    if overall_relevancy is None:
        return flags
    for title, data in per_doc.items():
        relevancy = data["stats"]["Contextual Relevancy"]["avg"]
        if relevancy is not None and relevancy < overall_relevancy - 0.10:
            flags.append(
                f"{title}: Contextual Relevancy {relevancy:.2f} is far below "
                f"overall {overall_relevancy:.2f}"
            )
    return flags


def save_results(per_doc: dict, all_cases: list[dict], source: dict) -> tuple[dict, list[str]]:
    overall = metric_stats(all_cases)
    flags = compute_flags(per_doc, overall)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threshold": METRIC_THRESHOLD,
        "source": source,
        "documents": {
            title: {
                "file": data["file"],
                "manual_key": data["manual_key"],
                "pairs": data["pairs"],
                "stats": stat_payload(data["stats"]),
                "cases": data["cases"],
            }
            for title, data in per_doc.items()
        },
        "overall": stat_payload(overall),
        "flags": flags,
    }
    RESULTS_V4.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return overall, flags


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Papeer using goldens_manual_pairs.json."
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=MAX_CONCURRENT,
        help=f"DeepEval max concurrent cases (default: {MAX_CONCURRENT})",
    )
    parser.add_argument(
        "--throttle",
        type=float,
        default=THROTTLE_SECONDS,
        help=f"DeepEval throttle seconds (default: {THROTTLE_SECONDS})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from backend.ingestion import _figure_context
    from backend.paper_loader import load_pdf
    from backend.rag_graph import build_graph
    from backend.vector_store import add_paper

    pdf_paths = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if not pdf_paths:
        raise RuntimeError(f"No PDFs found in {DOCUMENTS_DIR}")

    manual_docs = load_manual_goldens()
    source = {
        "file": MANUAL_GOLDENS.name,
        "documents": len(manual_docs),
        "pairs": sum(len(entry.get("pairs") or []) for entry in manual_docs.values()),
    }
    print(
        f"[goldens] {MANUAL_GOLDENS.name}: "
        f"{source['documents']} docs, {source['pairs']} pairs"
    )

    stale = sweep_stale_eval_points()
    if stale:
        print(f"Swept {stale} stale eval vectors.")

    saved_pg_vars = {}
    for key in ("DATABASE_URL", "POSTGRES_URL", "DATABASE_URL_UNPOOLED"):
        if key in os.environ:
            saved_pg_vars[key] = os.environ.pop(key)
    try:
        for suffix in ("", "-shm", "-wal"):
            path = Path(str(CHECKPOINT_DB) + suffix)
            if path.exists():
                path.unlink()
        graph = build_graph(db_path=str(CHECKPOINT_DB))
    finally:
        os.environ.update(saved_pg_vars)

    per_doc: dict[str, dict] = {}
    ingested_titles: list[str] = []
    all_cases: list[dict] = []
    try:
        for pdf in pdf_paths:
            title = pdf.stem
            manual_key, entry = manual_entry_for_pdf(pdf, manual_docs)
            pairs = clean_pairs(entry)
            print(
                f"\n=== {pdf.name} ===\n"
                f"[goldens] manual key {manual_key}: {len(pairs)} pairs"
            )

            # Same ingestion the API uses: text chunks plus figure chunks (caption,
            # references, optional vision description) when figures are enabled.
            chunks = load_pdf(str(pdf), title=title, figures_ctx=_figure_context(title, EVAL_USER))
            ingested_titles.append(title)
            if not chunks:
                raise RuntimeError(f"load_pdf produced 0 chunks for {pdf.name}")
            session_id = f"eval_{uuid4().hex[:12]}"
            add_paper(chunks, session_id, user_id=EVAL_USER)
            print(f"Ingested {len(chunks)} chunks into session {session_id}")

            test_cases = []
            for index, pair in enumerate(pairs):
                thread_id = f"{session_id}_{index}"
                answer, retrieval_context, pages, full_context = run_rag_query(
                    graph,
                    pair["input"],
                    session_id,
                    thread_id,
                    title,
                )
                test_cases.append(
                    LLMTestCase(
                        input=pair["input"],
                        actual_output=answer,
                        expected_output=pair["expected_output"],
                        retrieval_context=retrieval_context,
                    )
                )
                pair["_pages"] = pages
                pair["_contexts"] = retrieval_context
                pair["_full"] = full_context

            results = evaluate(
                test_cases,
                build_metrics(),
                async_config=AsyncConfig(
                    max_concurrent=args.max_concurrent,
                    throttle_value=args.throttle,
                ),
            )
            cases = summarize(results)
            # DeepEval returns results in completion order, not input order, so
            # attach per-question data by question text (a positional zip mixed
            # up pages/contexts/expected outputs between questions).
            pair_by_input = {pair["input"]: pair for pair in pairs}
            for case in cases:
                pair = pair_by_input[case["input"]]
                case["source_page"] = pair["source_page"]
                case["retrieval_pages"] = pair.get("_pages", [])
                case["expected_output"] = pair["expected_output"]
                case["retrieval_context"] = pair.get("_contexts", [])
                case["retrieval_full_text"] = pair.get("_full", [])
            all_cases.extend(cases)
            per_doc[title] = {
                "file": pdf.name,
                "manual_key": manual_key,
                "pairs": [
                    {
                        "input": pair["input"],
                        "expected_output": pair["expected_output"],
                        "source_page": pair["source_page"],
                    }
                    for pair in pairs
                ],
                "stats": metric_stats(cases),
                "cases": cases,
            }
            save_results(per_doc, all_cases, source)
    finally:
        try:
            if per_doc:
                save_results(per_doc, all_cases, source)
        except Exception as e:
            print(f"Warning: could not save partial results: {e}")
        try:
            from backend.blob_store import delete_paper_files, paper_id_for

            for ingested in ingested_titles:  # figure images written during ingestion
                delete_paper_files(EVAL_USER, paper_id_for(ingested))
        except Exception as e:
            print(f"Warning: could not delete eval figure files: {e}")
        try:
            deleted = sweep_stale_eval_points()
            print(f"\nCleaned up {deleted} eval vectors.")
        except Exception as e:
            print(f"Warning: could not clean up eval session: {e}")

    overall, flags = save_results(per_doc, all_cases, source)
    print_table(per_doc, overall)
    for flag in flags:
        print(f"FLAG: {flag}")
    print(f"\nResults saved to {RESULTS_V4}.")


if __name__ == "__main__":
    main()

import json
import sys
import asyncio
import hashlib
import random
import re
import shutil
import time
from collections import Counter
from pathlib import Path
from uuid import uuid4

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from deepeval.models import DeepEvalBaseEmbeddingModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from deepeval import evaluate
from deepeval.evaluate import AsyncConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.synthesizer import Synthesizer
from deepeval.test_case import LLMTestCase

import os
from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from backend.paper_loader import load_document
from backend.rag_graph import build_graph
from backend.vector_store import add_paper

load_dotenv(override=True)  # .env is source of truth - overrides stray env vars
# DeepEval wraps every task in per-task asyncio.wait_for timeouts; our slow
# judge blows past them and the whole evaluation dies with TimeoutError.
os.environ.setdefault("DEEPEVAL_DISABLE_TIMEOUTS", "1")

DOCUMENTS_DIR    = Path("documents")
PDF_PATHS        = sorted(DOCUMENTS_DIR.glob("*.pdf"))
RESULTS_V1       = Path("eval_results_v1.json")
RESULTS_V2       = Path("eval_results_v2.json")
RESULTS_V3       = Path("eval_results_v3.json")
GOLDENS_CACHE_V3 = Path("goldens_v3.json")
GOLDENS_REJECTED = Path("goldens_rejected.json")
DIAGNOSTICS_V3   = Path("diagnostics_v3.json")

FILTER_VERSION = "v3"
TARGET_GOLDENS = 10
MAX_SYNTH_ATTEMPTS = 3
# Ask for more candidates than we need so one bad batch cannot leave a
# shortfall; rejects are cheap compared with a rerun of the whole eval.
GOLDEN_OVERSAMPLE = 2.0
METRIC_THRESHOLD = 0.7

# --- context filter thresholds -------------------------------------------------
CTX_MIN_WORDS = 25
CTX_MIN_CHARS = 200
CTX_MIN_ALPHA_RATIO = 0.55
CTX_MAX_DIGIT_RATIO = 0.35
CTX_MAX_CITATION_DENSITY = 4.0        # per 100 words
CTX_MIN_SENTENCE_PUNCT_PER_60W = 1.0
CTX_BOILERPLATE_MIN_PAGES = 3
CTX_MAX_CHARS = 1800
CTX_HEADING_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(references|bibliography|acknowledg|appendix|supplementary)\b",
    re.IGNORECASE,
)
CTX_CITATION_RE = re.compile(
    r"(?:et\s+al\.,?\s*(?:19|20)\d{2})|\[\d+(?:\s*,\s*\d+)*\]", re.IGNORECASE
)
# A line that begins a bibliography entry, e.g. "[9] Yamamoto, E., ...".
CTX_REF_ENTRY_RE = re.compile(r"^\s*\[\d+\]\s+\S")
CTX_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?\s*$")

# --- golden filter thresholds --------------------------------------------------
GOLDEN_MISSING_RE = re.compile(
    r"\b(not\s+(?:specified|provided|mentioned|stated|available|reported|discussed)"
    r"|does\s+not\s+(?:mention|state|provide|specify|discuss|report)"
    r"|cannot\s+be\s+determined|insufficient\s+(?:information|detail)"
    r"|unclear\s+from|no\s+information)\b",
    re.IGNORECASE,
)
GOLDEN_HYPOTHETICAL_RE = re.compile(
    r"\b(envision|imagine|scenario|what\s+if|how\s+could\b[^?]*\b(?:reshape|change|impact|affect)"
    r"|in\s+your\s+opinion|would\s+you|do\s+you\s+think|speculat)",
    re.IGNORECASE,
)
GOLDEN_EXTERNAL_RE = re.compile(
    r"\b([A-Z][A-Za-z'\-]+)(?:\s+et\s+al\.?|\s*\(\s*(?:19|20)\d{2}\s*\))"
)
GROUND_MIN_SENTENCE_RECALL = 0.30
GROUND_MIN_COSINE = 0.72
GROUND_MIN_SENTENCE_PASS = 0.60
DUP_COSINE = 0.90
DUP_JACCARD = 0.80

# Eval points live under their own fake user scope (same "lib-<sha256[:32]>"
# format the app uses), so they can never collide with a real user's library
# and a stale sweep can find them all by this one filter.
MAX_CONCURRENT = 2
THROTTLE_SECONDS = 1

EVAL_USER = "lib-" + hashlib.sha256(b"eval@papeer.local").hexdigest()[:32]

METRIC_NAMES = [
    "Contextual Precision",
    "Contextual Recall",
    "Contextual Relevancy",
    "Answer Relevancy",
    "Faithfulness",
]

FILTER_SETTINGS = {
    "version": FILTER_VERSION,
    "target": TARGET_GOLDENS,
    "max_attempts": MAX_SYNTH_ATTEMPTS,
    "ctx_min_words": CTX_MIN_WORDS,
    "ctx_min_chars": CTX_MIN_CHARS,
    "ctx_min_alpha": CTX_MIN_ALPHA_RATIO,
    "ctx_max_digits": CTX_MAX_DIGIT_RATIO,
    "ctx_max_citation_density": CTX_MAX_CITATION_DENSITY,
    "ctx_min_punct_per_60w": CTX_MIN_SENTENCE_PUNCT_PER_60W,
    "ctx_boilerplate_min_pages": CTX_BOILERPLATE_MIN_PAGES,
    "ctx_strip_reference_tail": True,
    "golden_oversample": GOLDEN_OVERSAMPLE,
    "ground_min_sentence_recall": GROUND_MIN_SENTENCE_RECALL,
    "ground_min_cosine": GROUND_MIN_COSINE,
    "ground_min_sentence_pass": GROUND_MIN_SENTENCE_PASS,
    "dup_cosine": DUP_COSINE,
    "dup_jaccard": DUP_JACCARD,
}

_STOPWORDS = set(
    "a an the and or but if then than that this these those of to in on at for "
    "with from by as is are was were be been being it its they their we our you "
    "your he she his her not no also such which who whom whose what when where "
    "why how all any both each few more most other some own same so only very "
    "can will just should could would may might must do does did have has had i "
    "me my".split()
)

_CAP_STOP = {
    w.lower()
    for w in (
        "The This That These Those However Therefore Thus Moreover Furthermore "
        "Additionally Table Figure Figures Section Equation Eq Appendix References "
        "In On For With By As It Its They Their There Here We Our You Your I A An "
        "And But Or If When While Although Because Since Also Both Each Every Some "
        "Many Most Such Other Another One Two Three Four Five First Second Third "
        "Finally Overall Notably Specifically Particularly Generally Typically "
        "Often Usually Then Next Now Indeed Importantly Similarly Likewise Instead "
        "Meanwhile Unlike Compared Given Using Based According Note Answer Question "
        "Paper Yes No Not All Any More Less New Same Different Only Between During "
        "After Before Within Without Across Through Among Around Under Over Above "
        "Below Per Via Step Stage Phase Method Approach Model Results Discussion "
        "Introduction Conclusion Summary Chapter Part".split()
    )
}


def sweep_stale_eval_points() -> int:
    """Delete leftover points from earlier killed eval runs."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    client = QdrantClient(
        url=os.environ["QDRANT_URL"], api_key=os.environ["QDRANT_API_KEY"], timeout=120
    )
    flt = Filter(must=[FieldCondition(key="metadata.user_id", match=MatchValue(value=EVAL_USER))])
    stale = client.count("papeer_docs", exact=True, count_filter=flt).count
    if stale:
        client.delete("papeer_docs", points_selector=flt, wait=True)
    return stale


class OpenAIModel(DeepEvalBaseLLM):
    """DeepEval LLM backed by OpenAI's official endpoint (gpt-4o-mini)."""

    def __init__(self, model: str = None):
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
                except Exception:  # noqa: BLE001 - fall back to the next method
                    continue
                for _ in range(3):
                    result = self._retry(structured.invoke, prompt)
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
                except Exception:  # noqa: BLE001 - fall back to the next method
                    continue
                for _ in range(3):
                    result = await self._aretry(structured.ainvoke, prompt)
                    if isinstance(result, schema):
                        return result
            raise RuntimeError(
                f"structured output produced no {schema.__name__} (model returned None)"
            )
        res = await self._aretry(self.chat.ainvoke, prompt)
        return res.content or ""

    def get_model_name(self) -> str:
        return self.model_name

    def _is_rate_limit(self, e: Exception) -> bool:
        text = str(e).lower()
        return (
            "429" in text
            or "rate limit" in text
            or "conflict" in text
            or "409" in text
            or "duplicate request" in text
            or "524" in text
            or "timeout" in text
            or "internal server error" in text
            or "502" in text
            or "503" in text
            or "connection error" in text
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
            except Exception as e:  # noqa: BLE001 - only transient errors are retried
                if not self._is_rate_limit(e) or attempt == retries - 1:
                    raise
                wait = self._wait_for(e)
                print(f"[judge retry] waiting {wait:.0f}s (attempt {attempt + 1}/{retries - 1})...")
                time.sleep(wait)

    async def _aretry(self, call, *args, retries: int = 6, **kwargs):
        for attempt in range(retries):
            try:
                return await call(*args, **kwargs)
            except Exception as e:  # noqa: BLE001
                if not self._is_rate_limit(e) or attempt == retries - 1:
                    raise
                wait = self._wait_for(e)
                print(f"[judge retry] waiting {wait:.0f}s (attempt {attempt + 1}/{retries - 1})...")
                await asyncio.sleep(wait)


class GeminiEmbeddingModel(DeepEvalBaseEmbeddingModel):
    """Gemini embeddings (used only if no OpenAI key is available)."""

    def __init__(self):
        self.model_name = "gemini-embedding-2"
        self.embedder = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-2",
            api_key=os.environ.get("GOOGLE_API_KEY"),
            output_dimensionality=1536,
        )
        super().__init__(model=self.model_name)

    def load_model(self):
        return self.embedder

    def embed_text(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.embedder.aembed_query(text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.embedder.aembed_documents(texts)

    def get_model_name(self) -> str:
        return self.model_name


class OpenAIEmbeddingModel(DeepEvalBaseEmbeddingModel):
    """OpenAI embeddings used ONLY for grounding/dedup checks in this script.

    Retrieval keeps using the app's Gemini embedder (backend/vector_store.py);
    this model never touches Qdrant.
    """

    def __init__(self):
        self.model_name = "text-embedding-3-small"
        self.embedder = OpenAIEmbeddings(
            model=self.model_name,
            api_key=os.environ.get("OPENAI_API_KEY"),
            dimensions=1536,
        )
        super().__init__(model=self.model_name)

    def load_model(self):
        return self.embedder

    def embed_text(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.embedder.embed_documents(texts)

    async def a_embed_text(self, text: str) -> list[float]:
        return await self.embedder.aembed_query(text)

    async def a_embed_texts(self, texts: list[str]) -> list[list[float]]:
        return await self.embedder.aembed_documents(texts)

    def get_model_name(self) -> str:
        return self.model_name


eval_model = OpenAIModel()
gemini_embeddings = GeminiEmbeddingModel()
synth_embeddings = (
    OpenAIEmbeddingModel() if os.environ.get("OPENAI_API_KEY") else gemini_embeddings
)
# Cache invalidation also covers truncation and the embedding model used for
# grounding/dedup checks (both change which goldens survive).
FILTER_SETTINGS["ctx_max_chars"] = CTX_MAX_CHARS
FILTER_SETTINGS["synth_embedder"] = synth_embeddings.get_model_name()

_EMB_CACHE: dict[str, list[float]] = {}
_EMB_TRANSIENT = (
    "429", "rate limit", "timeout", "timed out", "connection", "overloaded",
    "temporarily", "500", "502", "503", "504",
)


def _embed_batch(batch: list[str], attempts: int = 5) -> list[list[float]]:
    for attempt in range(attempts):
        try:
            return synth_embeddings.embed_texts(batch)
        except Exception as e:  # noqa: BLE001 - only transient errors are retried
            text = str(e).lower()
            if attempt == attempts - 1 or not any(m in text for m in _EMB_TRANSIENT):
                raise
            wait = min(60, 5 * (2 ** attempt))
            print(f"[embeddings] transient error, retrying in {wait}s...")
            time.sleep(wait)
    return []


def _embed_many(texts: list[str]) -> list[list[float]]:
    unique = list(dict.fromkeys(t for t in texts if t))
    todo = [t for t in unique if t not in _EMB_CACHE]
    for start in range(0, len(todo), 64):
        batch = todo[start:start + 64]
        for text, vec in zip(batch, _embed_batch(batch)):
            _EMB_CACHE[text] = vec
    return [_EMB_CACHE.get(t) or _zero_vec() for t in texts]


def _zero_vec() -> list[float]:
    return [0.0] * 1536


def _cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _tokens(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d{1,2}[.)]\s+")


def _numbers(text: str) -> set[str]:
    """Numeric tokens used for grounding checks.

    Years and line-leading list markers are excluded: a paper's publication
    year is rarely repeated in the chunk it was cited from, and "1." / "2."
    are formatting, not claims.
    """
    body = _LIST_MARKER_RE.sub("", text)
    out = set()
    for n in re.findall(r"\d+(?:[.,]\d+)*", body):
        n = n.replace(",", "")
        if _YEAR_RE.match(n):
            continue
        out.add(n)
    return out


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip().lower()


# ------------------------------------------------------------------------------
# Context filter
# ------------------------------------------------------------------------------
def _boilerplate_lines(page_texts: dict[int, list[str]]) -> set[str]:
    counts: Counter[str] = Counter()
    for _page, texts in page_texts.items():
        seen = set()
        for text in texts:
            for line in text.splitlines():
                norm = _norm_line(line)
                if 3 <= len(norm) <= 120 and re.search(r"[a-z]{3}", norm):
                    seen.add(norm)
        for norm in seen:
            counts[norm] += 1
    return {norm for norm, count in counts.items() if count >= CTX_BOILERPLATE_MIN_PAGES}


def _strip_boilerplate(text: str, boiler: set[str]) -> str:
    lines = [ln for ln in text.splitlines() if _norm_line(ln) not in boiler]
    return "\n".join(lines).strip()


def _strip_reference_tail(text: str) -> tuple[str, bool]:
    """Cut a parent block at the first bibliography entry line.

    Chunks often glue the last body paragraph to the reference list; dropping
    the whole block would lose good prose, so keep the head. A cut that leaves
    a fragment ending mid-sentence (mid-URL, mid-citation) is not usable prose.
    """
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if CTX_REF_ENTRY_RE.match(line):
            head = "\n".join(lines[:idx]).strip()
            if not head:
                return "", True
            if not CTX_SENTENCE_END_RE.search(head):
                return "", True
            return head, True
    return text, False


def _context_reject_reason(text: str) -> str | None:
    body = text.strip()
    if not body:
        return "empty"
    if CTX_HEADING_RE.match(body[:120]):
        return "section_heading"
    words = body.split()
    density_base = max(len(words) / 100.0, 1.0)
    if len(CTX_CITATION_RE.findall(body)) / density_base >= CTX_MAX_CITATION_DENSITY:
        return "citation_density"
    if len(words) < CTX_MIN_WORDS or len(body) < CTX_MIN_CHARS:
        return "too_short"
    alpha = sum(ch.isalpha() for ch in body) / len(body)
    if alpha < CTX_MIN_ALPHA_RATIO:
        return "low_alpha"
    digits = sum(ch.isdigit() for ch in body) / len(body)
    if digits > CTX_MAX_DIGIT_RATIO:
        return "high_digits"
    punct = len(re.findall(r"[.!?](?:\s|$)", body))
    if punct / max(len(words) / 60.0, 1.0) < CTX_MIN_SENTENCE_PUNCT_PER_60W:
        return "low_punctuation"
    return None


def build_contexts(pdf: Path) -> tuple[list[dict], dict]:
    """Group child chunks back into parent blocks, strip boilerplate, filter."""
    chunks = load_document(str(pdf))
    parents: dict[str, dict] = {}
    for chunk in chunks:
        md = chunk.metadata or {}
        pid = md.get("parent_id") or f"{md.get('page')}-{len(parents)}"
        text = md.get("parent_text") or chunk.page_content or ""
        parents.setdefault(
            pid, {"text": text, "page": md.get("page", 0)}
        )

    page_texts: dict[int, list[str]] = {}
    for parent in parents.values():
        page_texts.setdefault(int(parent["page"]), []).append(parent["text"])
    boiler = _boilerplate_lines(page_texts)

    stats = {"total": len(parents), "kept": 0, "dropped": Counter()}
    kept: list[dict] = []
    for parent in parents.values():
        text = _strip_boilerplate(parent["text"], boiler)
        text, had_refs = _strip_reference_tail(text)
        if had_refs and not text:
            stats["dropped"]["bibliography"] += 1
            continue
        reason = _context_reject_reason(text)
        if reason is None and text != parent["text"].strip() and len(text) < CTX_MIN_CHARS:
            reason = "boilerplate_only"
        if reason:
            stats["dropped"][reason] += 1
            continue
        if len(text) > CTX_MAX_CHARS:
            cut = text[:CTX_MAX_CHARS]
            end = max(cut.rfind("."), cut.rfind("?"), cut.rfind("!"))
            text = cut[: end + 1] if end > CTX_MAX_CHARS // 2 else cut
        kept.append({"text": text, "page": int(parent["page"])})
        stats["kept"] += 1
    return kept, stats


# ------------------------------------------------------------------------------
# Golden filter
# ------------------------------------------------------------------------------
def _first_page_author_block(pdf: Path) -> str:
    try:
        import fitz

        with fitz.open(str(pdf)) as doc:
            text = doc[0].get_text() if doc.page_count else ""
    except Exception:  # noqa: BLE001 - author block is only a nicety
        return ""
    head = text[:2500]
    match = re.search(r"\babstract\b", head, re.IGNORECASE)
    return head[: match.start()] if match else head


def _entities(sentence: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", sentence)
    out = []
    for i, word in enumerate(words):
        if i == 0 or len(word) < 2 or not word[0].isupper():
            continue
        if word.lower() in _CAP_STOP:
            continue
        out.append(word)
    return out


def _entity_in_context(name: str, context: str) -> bool:
    return re.search(rf"\b{re.escape(name)}(?:s|es)?\b", context, re.IGNORECASE) is not None


def _external_citation_reason(question: str, author_block: str) -> str | None:
    for match in GOLDEN_EXTERNAL_RE.finditer(question):
        surname = match.group(1)
        if surname.lower() in _CAP_STOP:
            continue
        if re.search(rf"\b{re.escape(surname)}\b", author_block, re.IGNORECASE):
            continue
        return f"external_citation:{surname}"
    if re.search(r"\[\d+\]", question):
        return "external_citation:bracket"
    return None


def _ground_sentence(sentence: str, context: str, ctx_tokens: set[str],
                     ctx_numbers: set[str], ctx_vec: list[float],
                     doc_numbers: set[str] | None = None) -> tuple[bool, dict]:
    sent_numbers = _numbers(sentence)
    allowed_numbers = ctx_numbers | (doc_numbers or set())
    missing_numbers = sorted(n for n in sent_numbers if n not in allowed_numbers)
    missing_entities = [e for e in _entities(sentence) if not _entity_in_context(e, context)]
    sent_tokens = set(_tokens(sentence))
    recall = (
        len(sent_tokens & ctx_tokens) / len(sent_tokens) if sent_tokens else 1.0
    )
    cosine = _cosine(_embed_many([sentence])[0], ctx_vec)
    ok = (
        not missing_numbers
        and not missing_entities
        and (recall >= GROUND_MIN_SENTENCE_RECALL or cosine >= GROUND_MIN_COSINE)
    )
    return ok, {
        "recall": round(recall, 3),
        "cosine": round(cosine, 3),
        "missing_numbers": missing_numbers,
        "missing_entities": missing_entities,
    }


def _golden_reject_reason(
    golden, author_block: str, kept_questions: list[tuple[str, list[float]]],
    doc_numbers: set[str] | None = None,
) -> tuple[str | None, dict]:
    question = (golden.input or "").strip()
    expected = (golden.expected_output or "").strip()
    if not question or not expected:
        return "empty", {}

    if GOLDEN_MISSING_RE.search(expected):
        return "admits_missing", {"match": GOLDEN_MISSING_RE.search(expected).group(0)}
    if GOLDEN_HYPOTHETICAL_RE.search(question):
        return "hypothetical", {"match": GOLDEN_HYPOTHETICAL_RE.search(question).group(0)}
    external = _external_citation_reason(question, author_block)
    if external:
        return external.split(":")[0], {"detail": external}

    context = "\n".join(golden.context or [])
    if not context.strip():
        return "no_context", {}

    ctx_tokens = set(_tokens(context))
    ctx_numbers = _numbers(context)
    ctx_vec = _embed_many([context])[0]

    sentences = [s for s in _sentences(expected) if len(s.split()) >= 4]
    if not sentences:
        return "empty", {}

    sentence_results = [
        _ground_sentence(s, context, ctx_tokens, ctx_numbers, ctx_vec, doc_numbers)
        for s in sentences
    ]
    passed = sum(1 for ok, _ in sentence_results if ok)
    ratio = passed / len(sentences)

    name_failed = any(info["missing_entities"] for _, info in sentence_results)
    number_failed = any(info["missing_numbers"] for _, info in sentence_results)
    if number_failed:
        reason = "ungrounded_number"
    elif name_failed:
        reason = "ungrounded_name"
    elif ratio < GROUND_MIN_SENTENCE_PASS:
        reason = "ungrounded_text"
    else:
        reason = None

    detail = {
        "sentences": len(sentences),
        "passed": passed,
        "ratio": round(ratio, 3),
        "name_failed": name_failed,
        "number_failed": number_failed,
        "worst": sorted(
            (
                {
                    "sentence": s[:160],
                    **info,
                }
                for s, (_, info) in zip(sentences, sentence_results)
            ),
            key=lambda item: (item["recall"], item["cosine"]),
        )[:2],
    }
    if reason:
        return reason, detail

    question_vec = _embed_many([question])[0]
    question_tokens = set(_tokens(question))
    for prev_question, prev_vec in kept_questions:
        jaccard = (
            len(question_tokens & set(_tokens(prev_question)))
            / len(question_tokens | set(_tokens(prev_question)))
            if question_tokens
            else 0.0
        )
        if _cosine(question_vec, prev_vec) >= DUP_COSINE or jaccard >= DUP_JACCARD:
            return "duplicate", {
                "similar_to": prev_question,
                "cosine": round(_cosine(question_vec, prev_vec), 3),
                "jaccard": round(jaccard, 3),
            }
    return None, detail


def _synth_goldens(contexts: list[str], source_file: str):
    synthesizer = Synthesizer(model=eval_model)
    return synthesizer.generate_goldens_from_contexts(
        contexts=[[c] for c in contexts],
        include_expected_output=True,
        max_goldens_per_context=1,
        source_files=[source_file] * len(contexts),
        _send_data=False,
    )


def build_goldens_for_doc(pdf: Path) -> tuple[list[dict], dict]:
    contexts, ctx_stats = build_contexts(pdf)
    author_block = _first_page_author_block(pdf)

    stats = {
        "contexts": ctx_stats,
        "generated": 0,
        "kept": 0,
        "shortfall": 0,
        "attempts": 0,
        "dropped": Counter(),
    }
    rejected: list[dict] = []
    if not contexts:
        stats["shortfall"] = TARGET_GOLDENS
        return [], {"stats": stats, "rejected": rejected}

    rng = random.Random(0)
    order = contexts[:]
    rng.shuffle(order)

    kept: list[dict] = []
    kept_questions: list[tuple[str, list[float]]] = []
    doc_numbers = _numbers("\n".join(c["text"] for c in contexts))
    ctx_by_text = {c["text"]: c["page"] for c in contexts}
    cursor = 0
    while len(kept) < TARGET_GOLDENS and stats["attempts"] < MAX_SYNTH_ATTEMPTS:
        remaining = TARGET_GOLDENS - len(kept)
        wanted = max(remaining, int(remaining * GOLDEN_OVERSAMPLE))
        batch = order[cursor: cursor + wanted]
        cursor += len(batch)
        if not batch:
            break
        stats["attempts"] += 1
        try:
            goldens = _synth_goldens([c["text"] for c in batch], pdf.name)
        except Exception as e:  # noqa: BLE001 - report and stop refilling
            print(f"[goldens] synthesis failed for {pdf.name}: {e}")
            break
        stats["generated"] += len(goldens)
        kept_this_batch = 0
        for idx, golden in enumerate(goldens):
            if len(kept) >= TARGET_GOLDENS:
                break
            golden_context = "\n".join(golden.context or [])
            page = ctx_by_text.get(
                golden_context,
                batch[min(idx, len(batch) - 1)]["page"] if batch else 0,
            )
            reason, detail = _golden_reject_reason(
                golden, author_block, kept_questions, doc_numbers
            )
            if reason:
                stats["dropped"][reason] += 1
                rejected.append({
                    "document": pdf.name,
                    "input": golden.input,
                    "expected_output": golden.expected_output,
                    "reason": reason,
                    "detail": detail,
                })
                continue
            kept_questions.append((golden.input, _embed_many([golden.input])[0]))
            kept.append({
                "input": golden.input,
                "expected_output": golden.expected_output,
                "source_page": page,
            })
            kept_this_batch += 1
        if goldens and kept_this_batch == 0:
            # A full batch rejected: the document has no usable contexts left.
            break

    stats["kept"] = len(kept)
    stats["shortfall"] = max(TARGET_GOLDENS - len(kept), 0)
    return kept, {"stats": stats, "rejected": rejected}


# ------------------------------------------------------------------------------
# Golden cache (settings + file hash are part of the key)
# ------------------------------------------------------------------------------
def file_fingerprint(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "filename": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "mtime": int(path.stat().st_mtime),
    }


def load_goldens_cache() -> dict:
    if not GOLDENS_CACHE_V3.exists():
        return {"settings": FILTER_SETTINGS, "docs": {}}
    try:
        cache = json.loads(GOLDENS_CACHE_V3.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("goldens_v3.json is corrupt - regenerating every document.")
        return {"settings": FILTER_SETTINGS, "docs": {}}
    if cache.get("settings") != FILTER_SETTINGS:
        print("Golden filter settings changed - regenerating every document.")
        return {"settings": FILTER_SETTINGS, "docs": {}}
    cache.setdefault("docs", {})
    return cache


def save_goldens_cache(cache: dict) -> None:
    GOLDENS_CACHE_V3.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _clean_pairs(pairs: list[dict]) -> list[dict]:
    return [
        {k: v for k, v in pair.items() if not k.startswith("_")}
        for pair in pairs
    ]


def goldens_for_doc(pdf: Path, cache: dict) -> tuple[list[dict], dict]:
    fp = file_fingerprint(pdf)
    entry = cache["docs"].get(pdf.name)
    if entry and entry.get("sha256") == fp["sha256"] and entry.get("pairs"):
        print(
            f"[goldens] {pdf.name}: cache hit "
            f"({len(entry['pairs'])} pairs, sha {fp['sha256'][:12]})"
        )
        report = entry["report"]
        # The cache is JSON, so Counter drop-reason maps were serialized as
        # plain dicts. Restore them so callers can use .most_common().
        stats = report.setdefault("stats", {})
        stats["dropped"] = Counter(stats.get("dropped") or {})
        contexts = stats.setdefault("contexts", {})
        contexts["dropped"] = Counter(contexts.get("dropped") or {})
        return entry["pairs"], report
    print(f"[goldens] {pdf.name}: building goldens (sha {fp['sha256'][:12]})...")
    pairs, report = build_goldens_for_doc(pdf)
    cache["docs"][pdf.name] = {**fp, "pairs": _clean_pairs(pairs), "report": report}
    save_goldens_cache(cache)
    return pairs, report


def run_rag_query(graph, query: str, session_id: str, thread_id: str,
                  doc_title: str) -> tuple[str, list[str], list[dict], list[str]]:
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
    # Parent text is what the answer prompt actually sees (small-to-big
    # expansion); use it for diagnostics, never for the metric context.
    full_context = [
        (doc.metadata or {}).get("parent_text") or doc.page_content for doc in docs
    ]
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
        summary.append({
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
        })
    return summary


def metric_stats(cases: list[dict]) -> dict:
    stats = {}
    for name in METRIC_NAMES:
        scores, passed = [], 0
        for case in cases:
            for m in case["metrics"]:
                if m["name"] != name:
                    continue
                if m["score"] is not None:
                    scores.append(float(m["score"]))
                passed += int(bool(m["passed"]))
        stats[name] = {
            "avg": round(sum(scores) / len(scores), 3) if scores else None,
            "pass": passed,
            "n": len(cases),
        }
    return stats


def _fmt_stat(stat: dict) -> str:
    avg = "n/a" if stat["avg"] is None else f"{stat['avg']:.2f}"
    return f"{avg} ({stat['pass']}/{stat['n']})"


def print_table(per_doc: dict, overall: dict) -> None:
    headers = ["Document"] + METRIC_NAMES
    rows = [
        [title] + [_fmt_stat(data["stats"][n]) for n in METRIC_NAMES]
        for title, data in per_doc.items()
    ]
    rows.append(["OVERALL"] + [_fmt_stat(overall[n]) for n in METRIC_NAMES])
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print("\n" + line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))


def _stat_payload(stats: dict) -> dict:
    return {
        name: {
            "avg": stats[name]["avg"],
            "pass": stats[name]["pass"],
            "n": stats[name]["n"],
        }
        for name in METRIC_NAMES
    }


def print_drop_table(reports: dict) -> None:
    ctx_reasons = [
        "section_heading", "citation_density", "too_short", "low_alpha",
        "high_digits", "low_punctuation", "boilerplate_only", "empty",
    ]
    golden_reasons = [
        "admits_missing", "hypothetical", "external_citation", "ungrounded_number",
        "ungrounded_name", "ungrounded_text", "duplicate", "no_context", "empty",
    ]
    reasons = [f"ctx:{r}" for r in ctx_reasons] + [f"gold:{r}" for r in golden_reasons]
    headers = ["Document", "ctx total", "ctx kept"] + reasons
    rows = []
    for title, report in reports.items():
        stats = report["stats"]
        ctx = stats["contexts"]
        cells = [title, str(ctx["total"]), str(ctx["kept"])]
        for reason in ctx_reasons:
            cells.append(str(ctx["dropped"].get(reason, 0)))
        for reason in golden_reasons:
            cells.append(str(stats["dropped"].get(reason, 0)))
        rows.append(cells)
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    print("\nDROP COUNTS")
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("-" * sum(widths) + "-" * (2 * (len(widths) - 1)))
    for row in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))


def print_kept_samples(per_doc: dict) -> None:
    rng = random.Random(0)
    print("\nKEPT GOLDEN SAMPLES (3 random per document)")
    for title, data in per_doc.items():
        pairs = data["pairs"]
        print(f"\n=== {title} ({len(pairs)} kept) ===")
        for pair in rng.sample(pairs, min(3, len(pairs))):
            print(f"Q: {pair['input']}")
            print(f"A: {pair['expected_output'][:300].replace(chr(10), ' ')}")
            print(f"   [source page {pair.get('source_page', '?')}]\n")


def print_comparison(v2: dict | None, v3: dict) -> None:
    if not v2:
        print("\n(no v2 results found - skipping comparison)")
        return
    print("\nV2 vs V3 (avg score)")
    headers = ["Document"] + METRIC_NAMES
    rows = []
    for title, data in v3["documents"].items():
        old = v2["documents"].get(title, {}).get("stats", {})
        cells = [title]
        for name in METRIC_NAMES:
            old_avg = old.get(name, {}).get("avg")
            new_avg = data["stats"][name]["avg"]
            if old_avg is None or new_avg is None:
                cells.append("n/a")
            else:
                cells.append(f"{old_avg:.2f}->{new_avg:.2f} ({new_avg - old_avg:+.2f})")
        rows.append(cells)
    old_overall = v2.get("overall", {})
    cells = ["OVERALL"]
    for name in METRIC_NAMES:
        old_avg = old_overall.get(name, {}).get("avg")
        new_avg = v3["overall"][name]["avg"]
        cells.append(
            "n/a" if old_avg is None or new_avg is None
            else f"{old_avg:.2f}->{new_avg:.2f} ({new_avg - old_avg:+.2f})"
        )
    rows.append(cells)
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))


def diagnose_low_recall(per_doc: dict) -> list[dict]:
    diagnostics = []
    for title, data in per_doc.items():
        for case in data["cases"]:
            recall = next(
                (m for m in case["metrics"] if m["name"] == "Contextual Recall"), None
            )
            if not recall or recall["score"] is None or recall["score"] >= METRIC_THRESHOLD:
                continue
            contexts = "\n".join(
                case.get("retrieval_full_text") or case.get("retrieval_context") or []
            )
            ctx_tokens = set(_tokens(contexts))
            missing = []
            for sentence in _sentences(case.get("expected_output") or ""):
                tokens = set(_tokens(sentence))
                if len(tokens) < 4:
                    continue
                if len(tokens & ctx_tokens) / len(tokens) < GROUND_MIN_SENTENCE_RECALL:
                    missing.append(sentence[:220])
            diagnostics.append({
                "document": title,
                "input": case["input"],
                "recall": recall["score"],
                "reason": recall["reason"],
                "retrieval_pages": case.get("retrieval_pages"),
                "expected_output": case.get("expected_output"),
                "actual_output": case.get("actual_output"),
                "unmatched_expected_sentences": missing,
            })
    return diagnostics


def compute_flags(per_doc: dict, overall: dict) -> list[str]:
    flags = []
    overall_rel = overall["Contextual Relevancy"]["avg"]
    if overall_rel is not None:
        for title, data in per_doc.items():
            rel = data["stats"]["Contextual Relevancy"]["avg"]
            if rel is not None and rel < overall_rel - 0.10:
                flags.append(
                    f"{title}: Contextual Relevancy {rel:.2f} is far below "
                    f"overall {overall_rel:.2f}"
                )
    return flags


def save_results(per_doc: dict, all_cases: list[dict],
                 rejected_all: dict) -> tuple[dict, list[str]]:
    """Write v3 results after every document so a crash cannot lose metrics."""
    overall = metric_stats(all_cases)
    flags = compute_flags(per_doc, overall)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "threshold": METRIC_THRESHOLD,
        "settings": FILTER_SETTINGS,
        "documents": {
            title: {
                "file": data["file"],
                "pairs": data["pairs"],
                "stats": _stat_payload(data["stats"]),
                "cases": data["cases"],
            }
            for title, data in per_doc.items()
        },
        "overall": _stat_payload(overall),
        "flags": flags,
    }
    RESULTS_V3.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    GOLDENS_REJECTED.write_text(
        json.dumps(
            {"settings": FILTER_SETTINGS, "documents": rejected_all},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return overall, flags


def main() -> None:
    if not PDF_PATHS:
        raise RuntimeError(f"No PDFs found in {DOCUMENTS_DIR.resolve()}")

    if Path("eval_results.json").exists() and not RESULTS_V1.exists():
        shutil.copyfile("eval_results.json", RESULTS_V1)
        print(f"Preserved previous results as baseline v1 -> {RESULTS_V1}")

    stale = sweep_stale_eval_points()
    if stale:
        print(f"Swept {stale} stale eval vectors.")

    cache = load_goldens_cache()

    saved_pg_vars = {}
    for key in ("DATABASE_URL", "POSTGRES_URL", "DATABASE_URL_UNPOOLED"):
        if key in os.environ:
            saved_pg_vars[key] = os.environ.pop(key)
    try:
        if os.path.exists("eval_checkpoints.db"):
            os.remove("eval_checkpoints.db")
        graph = build_graph(db_path="eval_checkpoints.db")
    finally:
        os.environ.update(saved_pg_vars)

    per_doc: dict[str, dict] = {}
    all_cases: list[dict] = []
    rejected_all: dict[str, list[dict]] = {}
    try:
        for pdf in PDF_PATHS:
            title = pdf.stem
            print(f"\n=== {pdf.name} ===")
            pairs, report = goldens_for_doc(pdf, cache)
            rejected_all[pdf.name] = report["rejected"]
            stats = report["stats"]
            print(
                f"[goldens] contexts {stats['contexts']['kept']}/"
                f"{stats['contexts']['total']} kept; generated {stats['generated']}; "
                f"kept {stats['kept']}; shortfall {stats['shortfall']}; "
                f"attempts {stats['attempts']}"
            )
            if stats["shortfall"]:
                print(
                    f"[goldens] WARNING: {pdf.name} short by "
                    f"{stats['shortfall']} goldens (cap reached)."
                )
            if stats["contexts"]["total"]:
                for reason, count in stats["contexts"]["dropped"].most_common():
                    share = count / stats["contexts"]["total"]
                    if share > 0.5:
                        print(
                            f"[goldens] WARNING: context rule '{reason}' dropped "
                            f"{share:.0%} of contexts in {pdf.name}."
                        )
            if stats["generated"]:
                for reason, count in stats["dropped"].most_common():
                    share = count / stats["generated"]
                    if share > 0.5:
                        print(
                            f"[goldens] WARNING: golden rule '{reason}' dropped "
                            f"{share:.0%} of generated goldens in {pdf.name}."
                        )

            chunks = load_document(str(pdf))
            if not chunks:
                raise RuntimeError(f"load_document produced 0 chunks for {pdf.name}")
            session_id = f"eval_{uuid4().hex[:12]}"
            add_paper(chunks, session_id, user_id=EVAL_USER)
            print(f"Ingested {len(chunks)} chunks into session {session_id}")

            test_cases = []
            for i, pair in enumerate(pairs):
                thread_id = f"{session_id}_{i}"
                answer, retrieval_context, pages, full_context = run_rag_query(
                    graph, pair["input"], session_id, thread_id, title
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

            if not test_cases:
                per_doc[title] = {
                    "file": pdf.name,
                    "pairs": 0,
                    "stats": metric_stats([]),
                    "cases": [],
                    "report": report,
                }
                save_results(per_doc, all_cases, rejected_all)
                continue

            results = evaluate(
                test_cases,
                build_metrics(),
                async_config=AsyncConfig(
                    max_concurrent=MAX_CONCURRENT, throttle_value=THROTTLE_SECONDS
                ),
            )
            cases = summarize(results)
            for case, pair in zip(cases, pairs):
                case["retrieval_pages"] = pair.get("_pages", [])
                case["expected_output"] = pair["expected_output"]
                case["retrieval_context"] = pair.get("_contexts", [])
                case["retrieval_full_text"] = pair.get("_full", [])
            all_cases.extend(cases)
            per_doc[title] = {
                "file": pdf.name,
                "pairs": len(pairs),
                "stats": metric_stats(cases),
                "cases": cases,
                "report": report,
            }
            save_results(per_doc, all_cases, rejected_all)
    finally:
        try:
            if per_doc:
                save_results(per_doc, all_cases, rejected_all)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not save partial results: {e}")
        try:
            deleted = sweep_stale_eval_points()
            print(f"\nCleaned up {deleted} eval vectors.")
        except Exception as e:  # noqa: BLE001
            print(f"Warning: could not clean up eval session: {e}")

    overall, flags = save_results(per_doc, all_cases, rejected_all)

    reports = {title: data.get("report", {"stats": {"contexts": {"total": 0, "kept": 0, "dropped": Counter()}, "generated": 0, "dropped": Counter()}}) for title, data in per_doc.items()}
    print_drop_table(reports)
    print_kept_samples({title: data for title, data in per_doc.items()})
    print_table(per_doc, overall)
    for flag in flags:
        print(f"FLAG: {flag}")

    v2 = None
    if RESULTS_V2.exists():
        try:
            v2 = json.loads(RESULTS_V2.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            v2 = None
    print_comparison(v2, {"documents": per_doc, "overall": overall})

    diagnostics = diagnose_low_recall(per_doc)
    if diagnostics:
        print(f"\nLOW-RECALL DIAGNOSTICS ({len(diagnostics)} cases)")
        for item in diagnostics:
            print(f"- [{item['document']}] {item['input'][:110]}")
            print(f"  recall={item['recall']:.2f} pages={item['retrieval_pages']}")
            if item["unmatched_expected_sentences"]:
                print(f"  unmatched: {item['unmatched_expected_sentences'][0][:160]}")
    else:
        print("\nNo low-recall cases.")

    DIAGNOSTICS_V3.write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nResults saved to {RESULTS_V3}.")
    print(f"Rejected goldens saved to {GOLDENS_REJECTED}.")
    print(f"Low-recall diagnostics saved to {DIAGNOSTICS_V3}.")


if __name__ == "__main__":
    main()

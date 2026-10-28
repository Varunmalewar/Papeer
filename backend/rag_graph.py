import os
import base64
import re
import sqlite3
import warnings
from typing import Annotated

warnings.filterwarnings("ignore", message="The default value of `allowed_objects`")

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolCallId, ToolException, tool
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.prebuilt import InjectedState, ToolNode, tools_condition
from langgraph.types import Command
from pydantic import BaseModel, Field
from tavily import TavilyClient
from backend.llm_config import build_llm
import sys
from pathlib import Path

# Enable direct script execution: python backend/rag_graph.py
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from backend.models import ClaimVerificationResult, PremiseCheck, RelevancyDecision, RouterDecision
from backend.streaming import emit, streaming_active
from backend.timing import timed, timed_node
from backend.vector_store import bm25_search, list_papers, search as vs_search

load_dotenv(override=True)  # .env is source of truth — overrides any stray env vars
llm = build_llm(0)

# â”€â”€ Constants (before first use) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

MAX_RETRIEVAL_ATTEMPTS = 3
MAX_REWRITES = 2


# â”€â”€ State â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _merge_retrieved_docs(existing: list, new: list) -> list:
    """Reducer for retrieved_docs.

    - Tools append their newly fetched docs (may happen twice in one step when
      the model emits parallel tool calls, which `parallel_tool_calls=False`
      only discourages â€” it is not guaranteed).
    - query_rewrite_node sends [] as a RESET marker to clear stale docs.
    """
    if not new:
        return []
    return (existing or []) + new


class RAGState(MessagesState):
    session_id: str
    doc_scope_ids: list[str]
    doc_titles: list[str]
    user_id: str
    query: str
    original_query: str | None
    route: str | None
    retrieved_docs: Annotated[list[Document], _merge_retrieved_docs]
    retrieval_attempts: int
    claim_verdict: str | None
    claim_source: str | None
    superseding_papers: list[dict] | None
    answer: str | None
    is_relevant: bool | None
    rewrite_count: int
    premise_refuted: bool | None
    premise_correction: str | None


# â”€â”€ Router â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a routing assistant for a research paper Q&A system. "
        "Classify the user query into exactly one of four categories:\n\n"
        "  paper_qa â€” Questions about the content of uploaded research papers "
        "(e.g. methods, results, conclusions, authors), or any reference to "
        "'the report', 'the paper', 'the document'.\n"
        "  live_search â€” Questions that require live or current information: "
        "current events, today's weather, live prices, recent news, or anything "
        "where the answer changes over time (e.g. 'Who is the current president?', "
        "'What is the price of gold today?').\n"
        "  verify_claim â€” The user wants to check whether a specific claim or finding "
        "from a paper is still accurate or has been superseded.\n"
        "  direct_answer â€” A stable general knowledge question answerable from training data "
        "with no retrieval needed (e.g. 'What is softmax?', 'Explain backpropagation.').\n\n"
        "CRITICAL RULE: if the user has uploaded documents (any titles listed) and the "
        "question is about a topic those documents could plausibly cover â€” concepts, "
        "definitions, methods, architecture, theory (e.g. 'what is a residual connection "
        "in the transformer?' when a transformer paper is loaded) â€” you MUST route to "
        "paper_qa. Grounded, cited answers are always preferred over memorized ones. "
        "Use direct_answer ONLY when truly no loaded document could cover the topic.\n"
        "Route to live_search ONLY for genuinely real-time information.\n"
        "Route to verify_claim ONLY when the user explicitly asks to verify/fact-check a claim.\n\n"
        "When in doubt between paper_qa and direct_answer, ALWAYS choose paper_qa when documents are loaded.\n"
        "When in doubt between live_search and direct_answer, prefer live_search.\n\n"
        "Return only the route field.",
    ),
    ("human", "{query}"),
])

# APmix proxy ignores `response_format` (json_schema) for Claude and returns
# markdown-fenced ```json text, which fails pydantic parsing. Use function_calling.
router_chain = ROUTER_PROMPT | llm.with_structured_output(RouterDecision, method="function_calling")


def _doc_scope(state: RAGState) -> list[str]:
    """User library ids to search/aggregate: the chat's doc scope, else its own id."""
    scope = state.get("doc_scope_ids") or []
    return list(scope) if scope else [state["session_id"]]


def _doc_titles(state: RAGState) -> list[str]:
    """Titles the user focused on (clicked a loaded doc); empty = whole library."""
    return [t for t in (state.get("doc_titles") or []) if t]


# â”€â”€ Text sanitation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Postgres text/JSON cannot store NUL bytes; gpt-oss occasionally emits \u0000
# in answers and PDF/Tavily text can too. Strip control chars from every string
# that enters the graph state (which the Postgres checkpointer serializes).
_NUL_RE = re.compile(r"[\x00\x08\x0b\x0c\x0e-\x1f]")


def _clean_text(s: str) -> str:
    return _NUL_RE.sub("", s) if isinstance(s, str) else s


# â”€â”€ PII masking (guardrail #5) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Mask obvious PII in content fetched from the untrusted web before it enters
# the graph/LLM. Paper content is untouched: quoting author contact info from
# an uploaded paper is a legitimate use.
_PII_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PII_PHONE_RE = re.compile(r"(?<![\w-])(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}(?![\w-])")
_PII_CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,16}(?!\d)")


def _mask_pii(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = _PII_EMAIL_RE.sub("[email]", text)
    text = _PII_CARD_RE.sub("[number]", text)
    text = _PII_PHONE_RE.sub("[phone]", text)
    return text


# â”€â”€ Prompt-injection scanner (guardrail #1) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Pattern-based first line of defence (LLM-Guard BanSubstrings-style) on the
# user's query. Catches the common override/jailbreak phrasings cheaply; the
# paper context itself is already sandboxed by citation-checked prompts.
_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above|earlier|earlier)\s+instructions", re.I),
    re.compile(r"disregard\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+(?:instructions|prompts|rules)", re.I),
    re.compile(r"(?:reveal|print|show|repeat)\s+(?:me\s+)?(?:your\s+)?(?:the\s+)?(?:system\s+)?(?:prompt|instructions|initial\s+prompt)", re.I),
    re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+", re.I),
    re.compile(r"(?:enter|go\s+into|enable)\s+(?:developer|god|jailbreak|DAN)\s+mode", re.I),
    re.compile(r"pretend\s+(?:that\s+)?you\s+(?:are|have)\s+no\s+(?:restrictions|rules|filters)", re.I),
]
_INJECTION_REPLY = (
    "I can't follow instructions that try to override my behaviour. "
    "Ask me a question about your papers, a claim to verify, or something to search the web for."
)


def query_looks_injected(query: str) -> bool:
    q = query or ""
    return any(p.search(q) for p in _INJECTION_PATTERNS)


# â”€â”€ Citation check (guardrail #2) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# The model may hallucinate a [Title, Page N] label. Drop any citation whose
# (title, page) pair does not exactly match a chunk that was actually
# retrieved, so every bracket in the answer is grounded.
_PAPER_CITATION_RE = re.compile(r"\[(?P<title>[^\[\]]+?),\s*(?:Page|p\.)\s*(?P<page>\d+)\s*\]")

_FIGURE_CITATION_RE = re.compile(
    r"\[(?P<title>[^\[\]]+?),\s*Figure\s*(?P<figure>\d+),\s*(?:Page|p\.)\s*(?P<page>\d+)\s*\]",
    re.IGNORECASE,
)


def _strip_bad_citations(answer: str, docs: list) -> str:
    if not isinstance(answer, str) or not docs:
        return answer
    valid_pages: dict[str, set[int]] = {}
    for d in docs:
        m = d.metadata or {}
        title = (m.get("title") or "").strip()
        if not title:
            continue
        pages = valid_pages.setdefault(title, set())
        page = m.get("page")
        if page is not None:
            try:
                pages.add(int(page) + 1)  # context labels are 1-based
            except (TypeError, ValueError):
                pass

    # Figure citations: valid only when the retrieved set contains that exact
    # (paper title, figure number) pair.
    valid_figures: dict[str, set[int]] = {}
    for d in docs:
        m = d.metadata or {}
        if m.get("type") == "figure" and m.get("figure_number") is not None:
            title = (m.get("title") or "").strip()
            if title:
                valid_figures.setdefault(title, set()).add(int(m["figure_number"]))

    def _repl(mo: re.Match) -> str:
        title = mo.group("title").strip()
        page = int(mo.group("page"))
        # The page regex also matches "[Title, Figure N, Page X]" with the title
        # greedily including ", Figure N"; keep those only when the figure is real.
        fig = re.search(r",?\s*Figure\s+(\d+)\s*$", title, re.IGNORECASE)
        if fig:
            base = title[: fig.start()].strip().rstrip(",").strip()
            allowed_fig = valid_figures.get(base)
            if allowed_fig and int(fig.group(1)) in allowed_fig:
                return mo.group(0)
            return ""
        allowed = valid_pages.get(title)
        if allowed is None:
            return ""          # whole title unknown → fabricated citation
        if allowed and page not in allowed:
            return ""          # wrong page for a real title
        return mo.group(0)

    def _repl_fig(mo: re.Match) -> str:
        title = mo.group("title").strip()
        number = int(mo.group("figure"))
        allowed = valid_figures.get(title)
        if allowed is None or number not in allowed:
            return ""
        return mo.group(0)

    answer = _FIGURE_CITATION_RE.sub(_repl_fig, answer)
    return _PAPER_CITATION_RE.sub(_repl, answer).strip()


_REFUSAL_PREFIXES = (
    "i wasn't able to find relevant information",
    "the paper does not state this",
    "i don't know the answer",
)

_STRAY_REFUSAL_RE = re.compile(
    r"(?:(?<=[.!?])\s+|^)(?:the paper does not state this|i don'?t know the answer)\s*\.?",
    re.IGNORECASE,
)


def _strip_stray_refusal(answer: str) -> str:
    """Drop a refusal sentence that was appended to an otherwise grounded answer.

    The critic occasionally ends a supported answer with "The paper does not
    state this." — the phrase must only stand alone as the whole answer. Pure
    refusals (answer starts with one) are left untouched.
    """
    if not isinstance(answer, str):
        return answer
    text = answer.strip()
    if not text or text.lower().startswith(_REFUSAL_PREFIXES):
        return answer
    cleaned = _STRAY_REFUSAL_RE.sub(" ", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    if len(cleaned.split()) < 8:
        return answer
    return cleaned


def _best_matching_doc(text: str, docs: list) -> tuple:
    """Return (doc, overlap_score) — fraction of the text's 8-grams that appear
    in the doc's text. Score 0 means nothing matched."""
    body = _PAPER_CITATION_RE.sub(" ", text)
    body = re.sub(r"\s+", " ", body).strip().lower()
    if not body or not docs:
        return None, 0.0
    probe = body[:400]
    grams = [probe[i:i + 8] for i in range(0, max(0, len(probe) - 7))]
    if not grams:
        return None, 0.0
    best_doc, best_score = None, 0.0
    for d in docs:
        text_doc = (getattr(d, "page_content", None) or "").lower()
        norm = re.sub(r"\s+", " ", text_doc)
        hit = sum(1 for g in grams if g in norm)
        score = hit / len(grams)
        if score > best_score:
            best_doc, best_score = d, score
    return best_doc, best_score


def _hydrate_correction_citation(fixed_text: str, docs: list) -> str:
    """Premise-correction citations come from the LLM and often name the wrong
    title, so citation validation deletes them. Locate the retrieved chunk the
    correction text actually came from and re-stamp a valid [Title, Page N]."""
    if not isinstance(fixed_text, str) or not docs:
        return fixed_text
    if _PAPER_CITATION_RE.search(fixed_text):
        return fixed_text  # a validated citation already survives
    best_doc, best_score = _best_matching_doc(fixed_text, docs)
    if best_doc is None or best_score <= 0:
        return fixed_text
    md = getattr(best_doc, "metadata", None) or {}
    title = (md.get("title") or "").strip()
    page = md.get("page")
    if not title or page is None:
        return fixed_text
    citation = f"[{title}, Page {int(page) + 1}]"
    if citation in fixed_text:
        return fixed_text
    merged = fixed_text.rstrip() + " " + citation
    return re.sub(r"\s+([.,!?;:])", r"\1", merged)


def _clean_docs(docs: list) -> list:
    for d in docs:
        if isinstance(d.page_content, str):
            d.page_content = _clean_text(d.page_content)
    return docs


@timed_node('route')
def router_node(state: RAGState) -> dict:
    query = state["messages"][-1].content
    try:
        titles = _doc_titles(state) or list_papers(_doc_scope(state))
    except Exception:
        titles = []
    if titles:
        query = (
            f"[The user has these documents loaded in their library: {', '.join(titles)}. "
            "Any question about a topic these documents could plausibly cover â€” or any "
            "reference to 'the report', 'the paper' or 'the document' â€” MUST be routed "
            "to paper_qa. Only real-time info goes to live_search; only explicit "
            "claim-checking goes to verify_claim.]\n\n"
            f"{query}"
        )
    decision: RouterDecision = router_chain.invoke({"query": query})
    # Guardrail: with documents loaded, the LLM alone decides poorly â€” it keeps
    # sending definition-style questions ("what is X?") to direct_answer even
    # though the answer is inside the papers. Enforce deterministically:
    # direct_answer is only reachable when the library is empty.
    if titles and decision.route == "direct_answer":
        decision = decision.model_copy(update={"route": "paper_qa"})
    # The checkpointer keeps these across turns of one chat thread; a stale
    # original_query makes the generator answer the PREVIOUS question.
    return {
        "route": decision.route,
        "original_query": None,
        "premise_refuted": None,
        "premise_correction": None,
    }


# â”€â”€ Tool schemas â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class RetrieverInput(BaseModel):
    query: str = Field(description="Semantic query to search research paper chunks")
    k: int = Field(
        default=8,
        ge=1,
        le=20,
        description=(
            "Number of candidate chunks to retrieve (1-20). Use 6-8 for a "
            "single fact and 12-20 only when the question requires multiple "
            "independent pieces of evidence. The pipeline compacts candidates "
            "before answer generation."
        ),
    )


class WebSearchInput(BaseModel):
    # optimized_query is OPTIONAL on purpose: gpt-oss sometimes confuses this
    # custom tool with Groq's built-in browser_search and emits built-in-style
    # args ({"cursor": 0, "id": 0}). A required field would make Groq reject
    # the whole generation with a 400; an optional one lets the tool run and
    # self-correct via ToolException (fed back to the model as an error
    # ToolMessage so it retries with proper arguments).
    optimized_query: str = Field(
        default="",
        description="Query rewritten and optimized for web search",
    )
    max_results: int = Field(default=3, ge=1, le=10, description="Number of web results to return")


# â”€â”€ Tools â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# Evidence packaging: retrieved chunks go to the generator as-is. Only the
# number of final chunks is tuned here (multi-hop questions get more).

_FINAL_EVIDENCE_K = int(os.environ.get("PAPEER_FINAL_EVIDENCE_K", "3"))
# Primary retrieval: plain hybrid search + rerank, top PRIMARY_K, then drop every
# chunk whose rerank score is below REL_CUTOFF x the best score (dynamic k).
_PRIMARY_K = int(os.environ.get("PAPEER_PRIMARY_K", "5"))
_REL_CUTOFF = float(os.environ.get("PAPEER_REL_CUTOFF", "0.7"))
_MULTIHOP_CANDIDATE_K = int(os.environ.get("PAPEER_MULTIHOP_CANDIDATE_K", "12"))
_MULTIHOP_FINAL_EVIDENCE_K = int(
    os.environ.get("PAPEER_MULTIHOP_FINAL_EVIDENCE_K", "6")
)


def _rerank_sort_key(doc):
    score = (getattr(doc, "metadata", None) or {}).get("rerank_score")
    if score is None:
        return (1, 0.0)
    try:
        return (0, -float(score))
    except (TypeError, ValueError):
        return (1, 0.0)


def _select_evidence(docs: list) -> list:
    """Order reranked chunks best-first and keep them inside the context budget.

    An oversized chunk is skipped rather than ending the loop, so later relevant
    evidence can still use the remaining budget."""
    out: list = []
    total = 0
    for d in sorted(docs, key=_rerank_sort_key):
        size = len(d.page_content or "")
        if total + size > _CONTEXT_CHARS:
            continue
        out.append(d)
        total += size
    return out


_QUERY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "for", "to",
    "with", "and", "or", "what", "why", "how", "does", "do", "did", "which",
    "who", "this", "that", "it", "its", "than", "then", "there", "be", "been",
    "should", "would", "could", "can", "we", "us", "you", "i", "my", "your",
}


def _query_variants(query: str) -> list[str]:
    """Expand the user's question into 2-3 retrieval variants so structured
    content (tables, ablation sections) is not missed by one phrasing."""
    q = (query or "").strip()
    if not q:
        return []
    words = re.findall(r"[A-Za-z0-9]+(?:\*)?", q)
    keywords = " ".join(w for w in words if w.lower() not in _QUERY_STOPWORDS)
    variants = [q]
    if keywords and keywords.lower() != q.lower():
        variants.append(keywords)
    # PREMISE-BRIDGE VARIANT: false-premise questions name a mechanism the
    # paper never uses ("Why does it use GELU?"). That token dominates the
    # query embedding/BM25 while the paper's actual counterpart (ReLU) shares
    # no vocabulary with it. An acronym-stripped variant lets retrieval reach
    # the paragraph that CAN ground a correction.
    acro_stripped = " ".join(
        w for w in words
        if w.lower() not in _QUERY_STOPWORDS and not (w.isupper() and len(w) > 1)
    )
    if acro_stripped and acro_stripped.lower() not in {v.lower() for v in variants}:
        variants.append(acro_stripped)
    low = q.lower()
    if any(t in low for t in (
        "compar", "component", "ablation", "versus", " vs", " mlp", " gn",
        "believe", "adds value", "why should",
    )):
        variants.append(f"{keywords} ablation components table comparison".strip())
    return variants[:4]


@tool(args_schema=RetrieverInput)
def retrieve_from_vectorstore(
    query: str,
    k: int,
    session_id: Annotated[str, InjectedState("session_id")],
    doc_scope_ids: Annotated[list[str], InjectedState("doc_scope_ids")],
    doc_titles: Annotated[list[str], InjectedState("doc_titles")],
    user_id: Annotated[str, InjectedState("user_id")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the uploaded research paper vector store for relevant passages."""
    scope = list(doc_scope_ids) if doc_scope_ids else [session_id]
    titles = [t for t in (doc_titles or []) if t]
    # user_id is issued by the API layer from a verified session; enforced here
    # so client-supplied scopes can never cross into another user's documents.
    docs = retrieve_merged(query, k, scope, titles, user_id)
    if not docs:
        return [ToolMessage(content="No relevant documents found in the vector store.", tool_call_id=tool_call_id)]
    summary = f"Retrieved {len(docs)} section(s) from the vector store."
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        # The retrieved_docs reducer appends; do NOT prepend current_docs here
        # or parallel tool calls would double-append.
        Command(update={"retrieved_docs": docs}),
    ]


def retrieve_merged(query: str, k: int, scope: list[str], titles: list[str], user_id: str | None) -> list:
    """Evidence retrieval shared by the retrieval tool and the deterministic
    direct_retrieval node (query-rewrite retries).

    Primary path is hybrid search + rerank with a relative score cutoff. Only when
    that finds nothing usable (empty, or every candidate under the rerank floor —
    the false-premise / vocabulary-gap case) does it fall back to the query-variant
    + BM25-rescue pool.
    """
    with timed('retrieve_merged'):
        return _retrieve_merged_impl(query, k, scope, titles, user_id)


def _relative_cutoff(docs: list, rel: float) -> list:
    """Best-first docs, keeping only those scoring >= rel x the best rerank score."""
    docs = sorted(docs, key=_rerank_sort_key)
    if rel <= 0 or not docs:
        return docs
    best = (getattr(docs[0], "metadata", None) or {}).get("rerank_score")
    if not best:
        return docs
    return [
        d for d in docs
        if float((getattr(d, "metadata", None) or {}).get("rerank_score") or 0.0) >= rel * float(best)
    ]


def _retrieval_is_weak(docs: list) -> bool:
    return not docs or any((getattr(d, "metadata", None) or {}).get("below_floor") for d in docs)


def _retrieve_merged_impl(query: str, k: int, scope: list[str], titles: list[str], user_id: str | None) -> list:
    candidate_k = max(1, min(k, 20))
    primary_k = _MULTIHOP_FINAL_EVIDENCE_K if candidate_k >= _MULTIHOP_CANDIDATE_K else _PRIMARY_K
    with timed('search'):
        primary = _clean_docs(
            vs_search(query=query, scope=scope, k=primary_k, titles=titles, user_id=user_id or None)
        )
    if not _retrieval_is_weak(primary):
        return _select_evidence(_relative_cutoff(primary, _REL_CUTOFF))
    return _retrieve_variants_impl(query, k, scope, titles, user_id)


def _retrieve_variants_impl(query: str, k: int, scope: list[str], titles: list[str], user_id: str | None) -> list:
    candidate_k = max(1, min(k, 20))
    final_k = (
        _MULTIHOP_FINAL_EVIDENCE_K
        if candidate_k >= _MULTIHOP_CANDIDATE_K
        else _FINAL_EVIDENCE_K
    )
    merged: dict = {}
    for vq in _query_variants(query):
        with timed('search'):
            docs_v = _clean_docs(
                vs_search(
                    query=vq,
                    scope=scope,
                    k=candidate_k,
                    titles=titles,
                    user_id=user_id or None,
                )
            )
        # BM25-only side channel: premise questions poison dense+rerank scores
        # ("GELU" appears nowhere in the paper) while the real counterpart
        # chunk shares keywords ('activation', 'feed-forward'). Small, capped
        # union so keyword hits can never flood the semantic pool.
        with timed('bm25'):
            docs_v += _clean_docs(bm25_search(query=vq, scope=scope, k=5, titles=titles, user_id=user_id or None))
        for d in docs_v:
            meta = getattr(d, "metadata", None) or {}
            key = d.page_content
            score = float(meta.get("rerank_score", 0.0) or 0.0)
            cur = merged.get(key)
            cur_score = float((getattr(cur, "metadata", None) or {}).get("rerank_score", 0.0) or 0.0) if cur else -1.0
            if cur is None or score > cur_score:
                merged[key] = d
    semantic = []
    bm25_extras = []
    seen_keys = set()
    for d in sorted(
        merged.values(),
        key=lambda d: float((getattr(d, "metadata", None) or {}).get("rerank_score", 0.0) or 0.0),
        reverse=True,
    ):
        key = d.page_content
        if key in seen_keys:
            continue
        seen_keys.add(key)
        if (getattr(d, "metadata", None) or {}).get("bm25_only") and len(bm25_extras) < 4:
            bm25_extras.append(d)
        else:
            if len(semantic) < candidate_k:
                semantic.append(d)
    # Semantic/reranked hits remain the default evidence. BM25-only rescue
    # chunks fill genuine gaps but never displace stronger semantic evidence
    # at the start of the answer context.
    docs = semantic[:final_k]
    if len(docs) < final_k:
        docs.extend(bm25_extras[: final_k - len(docs)])
    return _select_evidence(docs)


@tool(args_schema=WebSearchInput)
def web_search(
    optimized_query: str,
    max_results: int,
    state_query: Annotated[str, InjectedState("query")],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> list:
    """Search the web for current or supplementary information using Tavily."""
    # gpt-oss sometimes omits/mangles `optimized_query` (or emits Groq's
    # built-in browser_search-style args) â€” fall back to the user's question
    # instead of crashing the graph with a ToolException.
    query_text = (optimized_query or "").strip() or (state_query or "").strip()
    if not query_text:
        return [
            ToolMessage(
                content=(
                    "web_search received no usable query. Call it again with an "
                    "'optimized_query' string (a concise, keyword-rich rewrite of "
                    "the user's question)."
                ),
                tool_call_id=tool_call_id,
            )
        ]
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    results = client.search(query_text, max_results=max_results)
    if not results.get("results"):
        return [ToolMessage(content="No web results found.", tool_call_id=tool_call_id)]
    web_docs = _clean_docs([
        Document(
            page_content=_mask_pii(r["content"]),
            metadata={"url": r["url"], "title": r.get("title", "Web Result")},
        )
        for r in results["results"]
    ])
    summary = f"Found {len(web_docs)} web result(s) for: {query_text}"
    return [
        ToolMessage(content=summary, tool_call_id=tool_call_id),
        # reducer appends â€” send only the new docs
        Command(update={"retrieved_docs": web_docs}),
    ]


# â”€â”€ Retrieval agent singletons â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

RETRIEVAL_TOOLS = [retrieve_from_vectorstore, web_search]
retrieval_llm = llm.bind_tools(RETRIEVAL_TOOLS, parallel_tool_calls=False)
base_tool_node = ToolNode(RETRIEVAL_TOOLS, handle_tool_errors=True)

# When the user has papers loaded/focused, the retrieval agent gets ONLY the
# vectorstore tool â€” no web search. An earlier answer cited an unrelated .NET
# blog for a paper question; this makes paper answers paper-grounded by
# construction. Web stays available via live_search (explicit questions).
PAPER_ONLY_TOOLS = [retrieve_from_vectorstore]
paper_only_llm = llm.bind_tools(PAPER_ONLY_TOOLS, parallel_tool_calls=False)

RETRIEVE_SYSTEM = (
    "You are a research assistant gathering context to answer a user's question about research papers.\n\n"
    "You have two tools available and full control over how you use them:\n\n"
    "1. retrieve_from_vectorstore â€” searches the uploaded paper collection.\n"
    "   You decide:\n"
    "   - query: the semantic search query (phrase it to best match relevant paper chunks)\n"
    "   - k: how many chunks to retrieve (1â€“10; use 6â€“8, more for broad questions)\n\n"
    "2. web_search â€” searches the live web via Tavily.\n"
    "   You decide:\n"
    "   - optimized_query: rewrite the user's question as a concise, keyword-rich web search query\n"
    "   - max_results: how many results to fetch (1â€“10)\n\n"
    "Choose the right source based on the question:\n"
    "- Questions about the uploaded papers â†’ use retrieve_from_vectorstore\n"
    "- Questions about current events, recent developments, or supplementary information â†’ use web_search\n"
    "- You may call both tools if the question benefits from multiple sources. \n\n"
    "Do NOT produce a final answer. Only call tools to collect context."
)

# System prompt for the live_search-only agent (no vectorstore tool)
LIVE_SEARCH_SYSTEM = (
    "You are an assistant answering a question that needs current/live information.\n"
    "Use the web_search tool to find the answer. Do NOT produce a final answer, only call tools."
)

# Agent bound only to web_search, used by the live_search route
LIVE_SEARCH_TOOLS = [web_search]
live_search_llm = llm.bind_tools(LIVE_SEARCH_TOOLS, parallel_tool_calls=False)
live_search_tool_node = ToolNode(LIVE_SEARCH_TOOLS, handle_tool_errors=True)


# â”€â”€ Relevancy check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

RELEVANCY_CHECK_SYSTEM = (
    "You are evaluating whether retrieved document chunks are relevant enough "
    "to answer a user's question about research papers.\n\n"
    "Return is_relevant=true if the chunks contain information that meaningfully "
    "addresses the question — even partially.\n"
    "Return is_relevant=true ALSO when the chunks CONTRADICT the question's premise "
    "— e.g. the question assumes the paper uses GELU but the chunks say it uses ReLU: "
    "from contradictory context a correction of the premise can be answered.\n"
    "Return is_relevant=false if the chunks are clearly off-topic, contain "
    "no useful information for the question, or discuss unrelated subjects.\n\n"
    "Be fair: partial overlap counts, but generic boilerplate does not. "
    "Off-topic chunks about unrelated topics (compression, regularization, other "
    "methods named nowhere in the question's subject) remain is_relevant=false."
)

relevancy_llm = llm.with_structured_output(RelevancyDecision, method="function_calling")

QUERY_REWRITE_SYSTEM = (
    "You are a query rewriting assistant for a research paper retrieval system. "
    "The previous query failed to retrieve relevant document chunks. "
    "Rewrite the query using more specific or alternative terminology, "
    "domain-specific keywords, or a narrower sub-question.\n\n"
    "IMPORTANT for false-premise questions: if the question names a mechanism, "
    "acronym, or model the paper may not use (e.g. 'Why does it use GELU?', "
    "'How does it compare to BERT?'), DROP that term entirely and rewrite the "
    "query around the underlying CONCEPT the question is really about "
    "(e.g. 'activation function in feed-forward layers'). The written query "
    "must not assume the premise is true.\n\n"
    "Return ONLY the rewritten query as plain text. No explanation, no preamble."
)


# â”€â”€ Nodes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

TOOL_SCHEMA_NUDGE = (
    "Your previous tool call had invalid arguments and was rejected by the API. "
    "Call the tool again using exactly these schemas:\n"
    "- retrieve_from_vectorstore: string 'query' (semantic search), optional int 'k' (1-20)\n"
    "- web_search: string 'optimized_query' (rewritten web query), optional int 'max_results' (1-10)\n"
    "Do not invent parameters such as 'cursor' or 'id'."
)


def _content_of(m) -> str:
    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
    return content or ""


def _history_units(messages):
    """Group messages into units that must stay together.
    An AIMessage with tool_calls plus its following ToolMessages is one unit â€”
    splitting the pair makes Groq reject the request with a validation error."""
    msgs = list(messages)
    units: list[list] = []
    i = 0
    while i < len(msgs):
        m = msgs[i]
        tool_calls = m.get("tool_calls") if isinstance(m, dict) else getattr(m, "tool_calls", None)
        if tool_calls:
            group = [m]
            i += 1
            while i < len(msgs):
                nxt = msgs[i]
                nxt_role = nxt.get("role") if isinstance(nxt, dict) else getattr(nxt, "type", None)
                if nxt_role in ("tool", "ToolMessage"):
                    group.append(nxt)
                    i += 1
                else:
                    break
            units.append(group)
            continue
        units.append([m])
        i += 1
    return units


_HISTORY_NOTE = "[Earlier conversation turns omitted to fit the model's context limit.]"


def _fit_history(messages, max_chars: int):
    """Trim history from the OLDEST side to a character budget (â‰ˆ4 chars/token),
    never splitting tool-call pairs."""
    msgs = list(messages)
    total = sum(len(_content_of(m)) for m in msgs)
    if total <= max_chars:
        return msgs
    keep: list[list] = []
    budget = max_chars
    for unit in reversed(_history_units(msgs)):
        size = sum(len(_content_of(m)) for m in unit)
        if size <= budget:
            keep.insert(0, unit)
            budget -= size
        else:
            break
    if not keep:
        return [msgs[-1]] if msgs else []
    kept = [m for unit in keep for m in unit]
    kept = [{"role": "system", "content": _HISTORY_NOTE}] + kept
    return kept


# Groq gpt-oss-120b on-demand allows 8000 TPM per request; tool schemas + system
# prompts eat ~2k tokens of that, so history/contexts get explicit char budgets
# (â‰ˆ4 chars per token).
_HISTORY_CHARS = int(os.environ.get("PAPEER_HISTORY_CHARS", "12000"))
_CONTEXT_CHARS = int(os.environ.get("PAPEER_CONTEXT_CHARS", "10000"))


def _invoke_tool_llm(bound_llm, messages):
    """Invoke a tool-bound LLM; on Groq tool-call validation failure, retry
    once with a schema nudge. gpt-oss occasionally emits Groq built-in
    browser_search-style args for our custom web_search tool. On a 413/TPM
    (request too large) error, retry once with a heavily trimmed history."""
    try:
        return bound_llm.invoke(messages)
    except Exception as e:
        err = str(e)
        if "tool_use_failed" in err or "tool call validation failed" in err.lower():
            return bound_llm.invoke(list(messages) + [{"role": "system", "content": TOOL_SCHEMA_NUDGE}])
        if "413" in err or "Requested" in err and "TPM" in err:
            return bound_llm.invoke(_fit_history(messages, _HISTORY_CHARS // 2))
        raise


def _clean_messages_content(msgs) -> list:
    """Strip NUL/control chars from message contents and tool-call args
    before they enter the graph state (checkpointer-safe)."""
    out = []
    for m in msgs:
        if isinstance(m.content, str):
            m = m.model_copy(update={"content": _clean_text(m.content)}) if hasattr(m, "model_copy") else m
        tcs = getattr(m, "tool_calls", None)
        if tcs:
            cleaned_tcs = []
            for tc in tcs:
                args = tc.get("args") or {}
                if isinstance(args, dict):
                    args = {k: _clean_text(v) if isinstance(v, str) else v for k, v in args.items()}
                cleaned_tcs.append({**tc, "args": args})
            m = m.model_copy(update={"tool_calls": cleaned_tcs}) if hasattr(m, "model_copy") else m
        out.append(m)
    return out


def _retrieve_system(doc_titles: list[str]) -> str:
    """Retrieval-agent system prompt. When papers are in scope, the paper
    collection is named explicitly and the vectorstore becomes mandatory-first;
    web_search is a fallback, never a substitute."""
    if not doc_titles:
        return RETRIEVE_SYSTEM
    titles = "\n".join(f"- {t}" for t in doc_titles)
    return (
        "You are a research assistant gathering context to answer the user's question.\n\n"
        "The user has these papers in their collection:\n"
        f"{titles}\n\n"
        "TOOLS:\n"
        "1. retrieve_from_vectorstore â€” semantic search over the papers listed above.\n"
        "   - query: the semantic search query (phrase it to best match relevant paper chunks)\n"
        "   - k: how many chunks to retrieve (1â€“10; use 6â€“8, more for broad questions)\n"
        "2. web_search â€” searches the live web via Tavily.\n"
        "   - optimized_query: rewrite the question as a concise, keyword-rich web query\n"
        "   - max_results: how many results to fetch (1â€“10)\n\n"
        "MANDATORY ORDER:\n"
        "- You MUST call retrieve_from_vectorstore first, at least once, no matter how the "
        "question is worded. The listed papers are the primary source.\n"
        "- Web search is DISABLED here: answer only from the papers. If the papers do not "
        "contain the answer, stop calling tools and let the answer step say "
        "\"The paper does not state this.\"\n"
        "- Ignore any documents or answers shown in EARLIER conversation turns; base every "
        "decision on fresh retrieval from this turn only.\n"
        "- NEVER answer a question about the listed papers from web results alone.\n\n"
        "Do NOT produce a final answer. Only call tools to collect context."
    )


@timed_node('agent')
def agent_node(state: RAGState) -> dict:
    current_attempts = state.get("retrieval_attempts", 0)
    # Once at the cap, do NOT call the LLM again. The history already contains tool calls,
    # so the model emits another one even with no tools bound, and Groq rejects it.
    # A plain message lets agent_routing move on to answer generation.
    if current_attempts >= MAX_RETRIEVAL_ATTEMPTS:
        return {"messages": [AIMessage(content="Context gathering complete.")]}
    doc_titles = state.get("doc_titles") or []
    system = _retrieve_system(doc_titles)
    messages = [{"role": "system", "content": system}] + _fit_history(state["messages"], _HISTORY_CHARS)
    # papers in scope -> web tools disabled (paper-grounded answers only)
    bound = paper_only_llm if doc_titles else retrieval_llm
    response = _invoke_tool_llm(bound, messages)
    if getattr(response, "content", ""):
        response = _clean_messages_content([response])[0]
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates


@timed_node('live_search_agent')
def live_search_agent_node(state: RAGState) -> dict:
    """Agent node that only has web_search tool, for the live_search route."""
    current_attempts = state.get("retrieval_attempts", 0)
    if current_attempts >= MAX_RETRIEVAL_ATTEMPTS:
        return {"messages": [AIMessage(content="Web search complete.")]}
    messages = [{"role": "system", "content": LIVE_SEARCH_SYSTEM}] + _fit_history(state["messages"], _HISTORY_CHARS)
    response = _invoke_tool_llm(live_search_llm, messages)
    if getattr(response, "content", ""):
        response = _clean_messages_content([response])[0]
    updates: dict = {"messages": [response]}
    if getattr(response, "tool_calls", None):
        updates["retrieval_attempts"] = current_attempts + 1
    return updates


@timed_node('relevancy_check')
def relevancy_check_node(state: RAGState) -> dict:
    query = state["query"]
    docs = state.get("retrieved_docs") or []
    # Grade all retrieved docs with full content (not truncated)
    doc_snippets = "\n\n---\n\n".join(doc.page_content for doc in docs)
    if not doc_snippets:
        return {"is_relevant": False}
    if len(doc_snippets) > _CONTEXT_CHARS:
        doc_snippets = doc_snippets[:_CONTEXT_CHARS] + "â€¦[chunks truncated]"
    prompt = (
        f"Question: {query}\n\nRetrieved chunks:\n{doc_snippets}\n\n"
        "Are these chunks relevant to answering the question?"
    )
    decision: RelevancyDecision = relevancy_llm.invoke([
        {"role": "system", "content": RELEVANCY_CHECK_SYSTEM},
        {"role": "user", "content": prompt},
    ])
    return {"is_relevant": decision.is_relevant}


@timed_node('query_rewrite')
def query_rewrite_node(state: RAGState) -> dict:
    original_query = state["query"]
    rewrite_count = state.get("rewrite_count", 0)
    response = llm.invoke([
        {"role": "system", "content": QUERY_REWRITE_SYSTEM},
        {"role": "user", "content": f"Original query: {original_query}\n\nWrite an improved search query."},
    ])
    rewritten = _clean_text(response.content.strip())
    return {
        "messages": [HumanMessage(content=rewritten)],
        "query": rewritten,
        # preserve the user's ORIGINAL question: after retries, generate_answer
        # must answer what the user actually asked, not the retry's phrasing
        "original_query": state.get("original_query") or state.get("query") or original_query,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "rewrite_count": rewrite_count + 1,
        "is_relevant": None,
    }


@timed_node('direct_retrieval')
def direct_retrieval_node(state: RAGState) -> dict:
    """Deterministic retry after a query rewrite.

    The old flow handed the rewritten query to the agent LLM, which kept
    re-searching the ORIGINAL premise-laden phrasing (gpt-4o-mini ignored the
    rewritten HumanMessage). Retrieval after a rewrite must be deterministic:
    run retrieve_merged with the rewritten query and go straight to the
    relevancy gate.
    """
    scope = list(state.get("doc_scope_ids") or []) or [state.get("session_id")]
    titles = [t for t in (state.get("doc_titles") or []) if t]
    docs = retrieve_merged(state.get("query", ""), 8, scope, titles, state.get("user_id"))
    return {
        "retrieved_docs": docs,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1,
    }


CLAIM_ANALYSIS_PROMPT = (
    "You are a research fact-checker. Given a claim from a research paper and "
    "a set of recent web and arXiv search results, determine:\n"
    "1. Has this claim been superseded, significantly challenged, or updated by more recent work?\n"
    "2. Identify up to 3 papers from the provided results that supersede or update the claim.\n\n"
    "Rules:\n"
    "- Use ONLY titles and URLs that appear verbatim in the provided search results.\n"
    "- Prefer arXiv paper links (arxiv.org) over general web links when available.\n"
    "- For each superseding paper, write one sentence explaining how it supersedes the claim.\n"
    "- If the claim still holds, set is_superseded=false and return an empty superseding_papers list.\n"
    "- verdict_summary should be 1-2 sentences suitable for display to the user."
)

# gpt-oss on Groq intermittently answers in plain text when tool choice is
# forced ("Tool choice is required, but model did not call a tool", code
# tool_use_failed), which 400s the whole graph. json_mode (response_format
# json_object) is reliable with this model, so verification uses it instead
# of function_calling â€” the exact JSON schema is spelled out in the prompt.
verification_llm = llm.with_structured_output(ClaimVerificationResult, method="json_mode")

VERIFY_JSON_REMINDER = (
    "Your previous reply was not valid JSON for this schema. "
    "Return ONLY a single JSON object with EXACTLY these keys:\n"
    '{"is_superseded": <bool>, "verdict_summary": "<string>", '
    '"superseding_papers": [{"title": "<string>", "url": "<string>", '
    '"summary": "<string>"}]}\n'
    "No other keys, no markdown fences, no commentary."
)


def _invoke_verification(prompt: str) -> ClaimVerificationResult:
    """Invoke the verification LLM, retrying once on schema/parse failures."""
    messages = [{"role": "user", "content": prompt}]
    try:
        return verification_llm.invoke(messages)
    except Exception:
        return verification_llm.invoke(
            list(messages) + [{"role": "user", "content": VERIFY_JSON_REMINDER}]
        )


@timed_node('verify_claim')
def verify_claim_node(state: RAGState) -> dict:
    claim = state["messages"][-1].content
    tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])

    # General web search for recent work superseding the claim
    general_results = tavily_client.search(
        f"recent research superseding: {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # arXiv-targeted search via web to get paper titles and links
    arxiv_results = tavily_client.search(
        f"site:arxiv.org {claim[:200]}",
        max_results=5,
    ).get("results", [])

    # Build context block
    lines = ["=== General Web Search Results ==="]
    for r in general_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    lines.append("=== arXiv Paper Search Results ===")
    for r in arxiv_results:
        lines.append(
            f"Title: {r.get('title', '')}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r.get('content', '')[:300]}\n"
        )

    context = "\n".join(lines)

    prompt = (
        f"{CLAIM_ANALYSIS_PROMPT}\n\n"
        "Respond ONLY with a single JSON object matching this schema:\n"
        '{"is_superseded": <bool>, "verdict_summary": "<string>", '
        '"superseding_papers": [{"title": "<string>", "url": "<string>", '
        '"summary": "<string>"}]}\n\n'
        f"Claim to verify:\n{claim}\n\n"
        f"Search Results:\n{context}"
    )
    result: ClaimVerificationResult = _invoke_verification(prompt)

    papers_dicts = [p.model_dump() for p in result.superseding_papers[:3]]
    return {
        "claim_verdict": _clean_text(result.verdict_summary),
        "claim_source": papers_dicts[0]["url"] if papers_dicts else None,
        "superseding_papers": papers_dicts,
    }


def _format_context_with_citations(docs: list[Document]) -> str:
    """Format retrieved docs with [Title, Page N] / [Title, Figure N, Page N]
    headers for citation in answers."""
    parts = []
    for doc in docs:
        md = doc.metadata or {}
        title = md.get("title", "Document")
        page = md.get("page")
        url = md.get("url")
        if md.get("type") == "figure" and md.get("figure_number") is not None:
            number = int(md["figure_number"])
            header = (
                f"[{title}, Figure {number}, Page {page + 1}]"
                if page is not None else f"[{title}, Figure {number}]"
            )
        elif url:
            header = f"[{title}]({url})"
        elif page is not None:
            header = f"[{title}, Page {page + 1}]"
        else:
            header = f"[{title}]"
        parts.append(f"{header}:\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _invoke_with_optional_figures(prompt: str, docs: list):
    """Attach original images for top-ranked figure hits when the model is
    multimodal and PAPEER_ATTACH_FIGURE_IMAGES=1 (off by default: costs tokens)."""
    if os.environ.get("PAPEER_ATTACH_FIGURE_IMAGES", "0").strip() != "1":
        return llm.invoke([{"role": "user", "content": prompt}]).content
    blocks: list[dict] = [{"type": "text", "text": prompt}]
    try:
        from backend.blob_store import get_figure_store
        from backend.figure_extractor import media_type_for
    except Exception:
        return llm.invoke([{"role": "user", "content": prompt}]).content
    for doc in docs or []:
        md = getattr(doc, "metadata", None) or {}
        if md.get("type") != "figure" or not md.get("image_path"):
            continue
        try:
            data = get_figure_store().get(md["image_path"])
            b64 = base64.b64encode(data).decode("ascii")
            mime = media_type_for(md.get("image_ext") or "png")
            blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        except Exception:
            continue
        if len(blocks) >= 3:  # text + at most two figure images
            break
    return llm.invoke([{"role": "user", "content": blocks}]).content


_INTEGRITY_SYSTEM = (
    "You are a rigorous, honest research assistant answering questions about ONE paper. Rules:\n"
    "- Answer the question exactly as asked. Do NOT reinterpret it as a defence of the paper.\n"
    "- If the question challenges a claim, evaluate honestly. \"This is not proven\" is an "
    "acceptable and expected answer.\n"
    "- If the paper lists something as future work or a limitation, call it \"suggestive, not "
    "proven\". Never use \"strong proof\" or \"compelling evidence\" unless the paper reports a "
    "direct experiment for exactly that claim.\n"
    "- If two parts of the paper disagree (e.g. abstract vs table), state both values and say "
    "the paper does not explain the difference. Do NOT invent an explanation such as "
    "\"average vs best-case\".\n"
    "- Check inferences for mathematical validity. Note: h_hybrid = min(h_classical, h_mlp, "
    "h_gnn) is always <= h_classical, so it cannot be tighter than Euclidean; the min "
    "preserves admissibility only and does NOT make the heuristic more informed.\n"
    "- FALSE-PREMISE QUESTIONS (highest priority): if the question asserts a detail the "
    "paper contradicts (wrong named mechanism like GELU/BERT, wrong attribution, wrong "
    "scope), you MUST NOT answer \"The paper does not state this.\" Instead, explicitly "
    "correct the premise: point out which assumption is wrong, say what the paper "
    "ACTUALLY states, and cite the [Title, Page N] of the chunk containing that "
    "counterpart. A question naming a mechanism the paper never mentions is ANSWERABLE "
    "whenever the paper describes the correct counterpart and it sits in the context.\n"
    "- Otherwise, if the paper genuinely does not state something, say exactly: "
    "\"The paper does not state this.\"\n"
    "- Quote exact table/section names and numbers.\n"
    "- Cite your sources using the [Title, Page N] labels provided with each chunk. "
    "For figure chunks cite them as [Title, Figure N, Page X] (keep the figure "
    "number and page from that chunk's header). "
    "EVERY paragraph must end with its own citation naming the SPECIFIC chunk that "
    "paragraph is grounded in — never reuse one citation across paragraphs that are "
    "built from different chunks/pages. Set-to-negative-infinity / softmax mechanisms "
    "and top-k mechanics each belong to the chunk that literally contains them.\n"
    "- Use ONLY chunks DIRECTLY about the question. If a chunk discusses a related but "
    "distinct mechanism (e.g. encoder self-attention alignment vs decoder masking, "
    "residual connections vs layer norm), do NOT blend its wording or claims into the "
    "answer — ignoring it entirely is correct.\n"
    "- If information comes from a web source, cite it as [Title](URL).\n"
    "- MATH FORMATTING: use standard LaTeX delimiters only â€” $$ on separate lines for block "
    "equations and single $ for inline math. Never wrap equations in \\[...\\] or \\(...\\).\n\n"
)

_CRITIC_SYSTEM = (
    "You are a strict correctness critic for an answer about a research paper. Given the "
    "question and the retrieved paper context, revise the DRAFT answer:\n"
    "- Remove or soften every claim not supported by a quote-equivalent in the context "
    "(an unsupported explanation like \"average vs best-case\" must be deleted).\n"
    "- Flag any mathematically invalid inference as invalid and correct it (e.g. a min of "
    "heuristics that includes the admissible one cannot be more informed than it).\n"
    "- Replace proof-level language (\"strong proof\", \"compelling evidence\") with "
    "\"suggestive, not proven\" when there is no direct experiment.\n"
    "- If the paper does not state something, write: \"The paper does not state this.\"\n"
    "- Never append \"The paper does not state this.\" to a draft that contains supported "
    "content; delete the unsupported sentences instead. Use that sentence ONLY when the "
    "entire draft is unsupported, and then output it alone.\n"
    "- Keep grounded numbers, table names, and citations exactly as they are.\n"
    "Output ONLY the revised answer, no commentary."
)


def _critic_pass(query: str, context: str, draft: str) -> str:
    """Second-pass verification before the answer is returned."""
    prompt = (
        f"Question: {query}\n\n"
        f"Retrieved paper context:\n\n{context}\n\n"
        f"Draft answer:\n\n{draft}"
    )
    try:
        revised = llm.invoke([
            {"role": "system", "content": _CRITIC_SYSTEM},
            {"role": "user", "content": prompt},
        ]).content
        return revised if isinstance(revised, str) and revised.strip() else draft
    except Exception as e:
        print(f"[critic] failed, keeping draft: {e}")
        return draft


# ── Critic gate: only pay for the second LLM pass when the draft is risky ────
_CRITIC_MAX_WORDS = int(os.environ.get("PAPEER_CRITIC_MAX_WORDS", "90"))
_CRITIC_MIN_GROUNDING = float(os.environ.get("PAPEER_CRITIC_MIN_GROUNDING", "0.45"))
_CRITIC_LOG = os.environ.get("PAPEER_CRITIC_LOG", "1").strip() != "0"

# Numbers inside equations are not factual claims the critic can check, so the
# numeric trigger only looks at prose (math delimiters stripped first).
_MATH_SPAN_RE = re.compile(r"\$\$.*?\$\$|\$[^$\n]*\$|\\\(.*?\\\)|\\\[.*?\\\]", re.DOTALL)
_NUM_CLAIM_RE = re.compile(r"\d")
_CAUSAL_RE = re.compile(
    r"\b(?:because|due to|therefore|thus|hence|since|so that|leads? to|"
    r"results? in|caused by|as a result|in order to)\b",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[a-zA-Z]{4,}")


def _lexical_grounding(text: str, context: str) -> float:
    """Fraction of the draft's distinct content words that appear in context."""
    words = {w.lower() for w in _WORD_RE.findall(text)}
    if not words:
        return 1.0
    ctx = context.lower()
    return sum(1 for w in words if w in ctx) / len(words)


def _needs_critic(
    draft: str,
    context: str,
    fabricated_citation: bool = False,
) -> tuple[bool, list[str], float]:
    """Decide whether the critic pass is worth one more LLM call."""
    reasons: list[str] = []
    if fabricated_citation:
        reasons.append("fabricated_citation")
    if len(draft.split()) > _CRITIC_MAX_WORDS:
        reasons.append("long")
    prose = _MATH_SPAN_RE.sub(" ", draft)
    # page/figure numbers inside citations are not claims
    prose = _FIGURE_CITATION_RE.sub(" ", prose)
    prose = _PAPER_CITATION_RE.sub(" ", prose)
    if _NUM_CLAIM_RE.search(prose):
        reasons.append("numbers")
    if _CAUSAL_RE.search(prose):
        reasons.append("causal")
    grounding = _lexical_grounding(draft, context)
    if grounding < _CRITIC_MIN_GROUNDING:
        reasons.append("low_grounding")
    return bool(reasons), reasons, grounding


# Streaming policy: buffer (no token stream) when the answer is likely to need
# the critic or spans several documents, so the user never sees uncorrected text.
_STREAM_BUFFER_Q_RE = re.compile(
    r"\b(?:how many|how much|how large|what percentage|number of|dimension|"
    r"why\s+(?:does|do|is|are|did|was))\b",
    re.IGNORECASE,
)


def _buffer_stream(state: RAGState) -> bool:
    if state.get("premise_refuted"):
        return True
    docs = state.get("retrieved_docs") or []
    if len(docs) > 6:
        return True
    if len({(getattr(d, "metadata", None) or {}).get("title") for d in docs}) > 1:
        return True
    q = state.get("original_query") or state.get("query") or ""
    return bool(_STREAM_BUFFER_Q_RE.search(q))


def _stream_answer(prompt: str) -> str:
    """Generate while emitting token deltas; falls back to a normal call."""
    parts: list[str] = []
    try:
        for chunk in llm.stream([{"role": "user", "content": prompt}]):
            text = chunk.content if isinstance(chunk.content, str) else ""
            if text:
                parts.append(text)
                emit({"type": "delta", "text": text})
    except Exception as e:
        print(f"[stream] token stream failed: {e}")
        if not parts:
            return llm.invoke([{"role": "user", "content": prompt}]).content
    return "".join(parts)


_PREMISE_SYSTEM = (
    "Task: FALSE-PREMISE DETECTION for a research-paper Q&A assistant.\n"
    "The question may assert a detail as if the paper states it. Check it against the context.\n"
    "Steps:\n"
    "1. Extract the questioned detail. Presupposition counts: 'Why does the paper use X?' / "
    "'Why is it N?' / 'Why does the encoder do Y?' all ASSERT X / N / Y. Open questions "
    "('Which/What/How many ... does the paper use?') assert nothing.\n"
    "2. Locate what the context actually states for the SAME slot.\n"
    "3. premise_refuted=true when either:\n"
    "   (a) NUMERIC: the context gives a different number for that slot; or\n"
    "   (b) MECHANISM: the asserted technique is absent from the ENTIRE context while "
    "the context names a DIFFERENT technique performing the same architectural role "
    "(the asserted term never appearing is expected in a false premise — judge the "
    "role, not the word).\n"
    "   premise_refuted=false when the claim matches the context, the context has no "
    "information about that slot, or the question asserts nothing (open question).\n"
    "4. Correction (only when refuted): answer directly; state the paper's actual "
    "value/technique using the context's own wording; explicitly say the asserted "
    "detail is not what the paper uses. For a numeric premise, the correction MUST "
    "include the paper's actual number. End with a citation copied char-for-char from "
    "a context header, e.g. [Exact Title From Header, Page 4].\n"
    "Rules:\n"
    "- Use only words/numbers present in the context; never invent or infer beyond it.\n"
    "- Never reuse this instruction's phrasing as the answer; be specific to this question.\n"
    "- If the real value/technique is not in the context, use premise_refuted=false."
)

_premise_llm = llm.with_structured_output(PremiseCheck, method="function_calling")

# Minimum 8-gram overlap between a correction and the retrieved chunks.
# Measured empirically: genuine grounded corrections score >0.35; parroted or
# invented ones land near 0.0-0.1.
_PREMISE_MIN_GROUNDING = float(os.environ.get("PAPEER_PREMISE_MIN_GROUNDING", "0.2"))


@timed_node('premise_guard')
def premise_guard_node(state: RAGState) -> dict:
    """Deterministic false-premise check: beats integrity-LLM refusals, which
    reliably pattern-match even when the correct counterpart sits in context."""
    if state.get("route") != "paper_qa":
        return {}
    docs = state.get("retrieved_docs") or []
    if not docs:
        return {"premise_refuted": False}
    q = state.get("original_query") or state["query"]
    context = _format_context_with_citations(docs)
    if len(context) > _CONTEXT_CHARS:
        context = context[:_CONTEXT_CHARS]
    prompt = f"User question: {q}\n\nRetrieved paper context:\n\n{context}"
    try:
        check = _premise_llm.invoke([
            {"role": "system", "content": _PREMISE_SYSTEM},
            {"role": "user", "content": prompt},
        ])
    except Exception as e:
        print(f"[premise_guard] check failed, falling through: {e}")
        return {"premise_refuted": False}
    corr = (getattr(check, "correction", "") or "").strip()
    if os.environ.get("PAPEER_DEBUG"):
        print(f"[premise_guard] refuted={getattr(check, 'premise_refuted', None)!r} corr={corr!r} relu_in_ctx={'relu' in context.lower()} q={q!r}")
    if getattr(check, "premise_refuted", False) and corr:
        # GROUNDING GATE: a correction that doesn't overlap the retrieved text is
        # hallucinated/parroted — refuse to surface it; let normal generation run.
        _, score = _best_matching_doc(corr, docs)
        if os.environ.get("PAPEER_DEBUG"):
            print(f"[premise_guard] grounding={score:.3f}")
        if score >= _PREMISE_MIN_GROUNDING:
            return {"premise_refuted": True, "premise_correction": corr}
    return {"premise_refuted": False}


@timed_node('generate_answer')
def generate_answer_node(state: RAGState) -> dict:
    route = state.get("route")
    # After a rewrite loop, state["query"] holds the RETRY phrasing used for
    # retrieval — the visible answer must address the user's ORIGINAL question.
    query = state.get("original_query") or state["query"]

    if route in ("paper_qa", "live_search"):
        if state.get("premise_refuted") and (state.get("premise_correction") or "").strip():
            fixed = re.sub(r"\s+", " ", state["premise_correction"]).strip()
            fixed_bad = _strip_bad_citations(fixed, state.get("retrieved_docs") or [])
            fixed_bad = _hydrate_correction_citation(fixed_bad, state.get("retrieved_docs") or [])
            fixed_bad = _strip_stray_refusal(fixed_bad)
            return {"answer": fixed_bad, "messages": [AIMessage(content=fixed_bad)]}
        if state.get("is_relevant") is False and state.get("rewrite_count", 0) >= MAX_REWRITES:
            answer = (
                "I wasn't able to find relevant information in the uploaded papers "
                "to answer your question. You may want to rephrase your question "
                "or upload additional papers."
            )
        else:
            docs = state.get("retrieved_docs") or []
            if not docs:
                answer = "I don't know the answer."
            else:
                context = _format_context_with_citations(docs)
                if len(context) > _CONTEXT_CHARS:
                    context = context[:_CONTEXT_CHARS] + "â€¦[context truncated]"
                prompt = (
                    _INTEGRITY_SYSTEM
                    + f"Context:\n\n{context}\n\n"
                    + f"Question: {query}"
                )
                # Stream tokens only when no critic pass is expected; risky
                # answers stay buffered and arrive in a single `done` payload.
                streamed = streaming_active() and not _buffer_stream(state)
                with timed("generate_llm"):
                    if streamed:
                        answer = _stream_answer(prompt)
                    else:
                        answer = _invoke_with_optional_figures(prompt, docs)
                streamed_text = answer if streamed else None
                if isinstance(answer, str) and answer.strip():
                    if os.environ.get("PAPEER_DEBUG"):
                        ix = context.lower().find("relu")
                        snip = context[max(ix - 150, 0):ix + 260] if ix >= 0 else ""
                        print(f"--- [gen] query={query!r}\n[gen draft] {answer!r}\n[relu in ctx: {ix >= 0}] {snip!r}")
                    # guardrail #2 first: stripping tells us whether the draft
                    # invented a citation (a critic-worthy signal by itself).
                    draft = answer
                    stripped = _strip_bad_citations(draft, docs)
                    fabricated = stripped != draft.strip()
                    needs_critic, reasons, grounding = _needs_critic(
                        stripped, context, fabricated
                    )
                    if _CRITIC_LOG:
                        verdict = "fired" if needs_critic else "skipped"
                        print(
                            f"[critic] {verdict} reason={','.join(reasons) or '-'} "
                            f"words={len(stripped.split())} grounding={grounding:.2f}",
                            flush=True,
                        )
                    answer = stripped
                    if needs_critic:
                        # second-pass critic: strip overclaims/unsupported explanations
                        with timed("critic"):
                            answer = _critic_pass(query, context, answer)
                        # The critic may append a refusal sentence to a grounded
                        # answer; only a standalone refusal is legitimate.
                        answer = _strip_stray_refusal(answer)
                        if os.environ.get("PAPEER_DEBUG"):
                            print(f"[critic revised] {answer!r}")
                # guardrail #2 (again): the critic may introduce new citations
                answer = _strip_bad_citations(answer, docs)
                answer = _strip_stray_refusal(answer)
                if streamed_text is not None and answer != streamed_text:
                    # tokens already reached the client; swap the final text
                    emit({"type": "replace", "text": answer})

    elif route == "verify_claim":
        verdict = state.get("claim_verdict", "")
        papers = state.get("superseding_papers") or []
        claim_text = state["query"]
        if papers:
            papers_block = "\n\n".join(
                f"{i + 1}. **{p['title']}**\n   {p['summary']}\n   Link: {p['url']}"
                for i, p in enumerate(papers)
            )
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"**Superseding Papers:**\n\n{papers_block}\n\n"
                f"---\n"
                f"*You can load any of these papers into your knowledge base "
                f"to continue your research with the latest findings.*"
            )
        else:
            answer = (
                f"**Claim Verification Result**\n\n"
                f"> {claim_text}\n\n"
                f"**Verdict:** {verdict}\n\n"
                f"*No papers directly superseding this claim were found in recent literature.*"
            )

    else:  # direct_answer
        prompt = (
            "Answer from your knowledge.\n\n"
            "MATH FORMATTING: use standard LaTeX delimiters only â€” $$ on separate lines "
            "for block equations and single $ for inline math. Never wrap equations in "
            "\\[...\\] or \\(...\\).\n\n"
            f"Question: {query}"
        )
        with timed("generate_llm"):
            answer = llm.invoke([{"role": "user", "content": prompt}]).content

    return {"answer": _clean_text(answer) if isinstance(answer, str) else answer,
            "messages": [AIMessage(content=_clean_text(answer) if isinstance(answer, str) else answer)]}


# â”€â”€ Graph â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def route_query(state: RAGState) -> str:
    return state["route"]


def agent_routing(state: RAGState) -> str:
    # Always execute pending tool calls first â€” shortcutting here would leave
    # an AIMessage with tool_calls unmatched by ToolMessages in the checkpointer,
    # corrupting history for all future turns in the same session.
    tc = tools_condition(state)
    if tc == "tools":
        return "retrieval"
    if state.get("retrieval_attempts", 0) >= MAX_RETRIEVAL_ATTEMPTS:
        return "generate_answer"
    return "relevancy_check"


def live_search_routing(state: RAGState) -> str:
    """Routing for the live_search agent â€” only uses web_search tool node."""
    tc = tools_condition(state)
    if tc == "tools":
        return "live_search_tools"
    return "generate_answer"


def after_relevancy_routing(state: RAGState) -> str:
    if state.get("is_relevant", False):
        return "generate_answer"
    if state.get("rewrite_count", 0) < MAX_REWRITES:
        return "query_rewrite"
    return "generate_answer"


def _resolve_postgres_url() -> str | None:
    for key in ("DATABASE_URL", "POSTGRES_URL", "DATABASE_URL_UNPOOLED"):
        url = os.environ.get(key, "").strip()
        if url:
            return url
    return None


_pg_checkpointer_cm = None


def build_graph(db_path: str = "checkpoints.db"):
    global _pg_checkpointer_cm
    pg_url = _resolve_postgres_url()
    if pg_url:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver

            _pg_checkpointer_cm = PostgresSaver.from_conn_string(pg_url)
            checkpointer = _pg_checkpointer_cm.__enter__()
            checkpointer.setup()
            print("[checkpointer] Using Neon Postgres checkpointer.")
        except Exception as e:
            print(f"[checkpointer] Postgres failed, falling back to sqlite: {e}")
            conn = sqlite3.connect(db_path, check_same_thread=False)
            checkpointer = SqliteSaver(conn)
    else:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        checkpointer = SqliteSaver(conn)

    graph = StateGraph(RAGState)
    graph.add_node("router", router_node)
    graph.add_node("agent_node", agent_node)
    graph.add_node("retrieval", base_tool_node)
    graph.add_node("relevancy_check", relevancy_check_node)
    graph.add_node("query_rewrite", query_rewrite_node)
    graph.add_node("direct_retrieval", direct_retrieval_node)
    graph.add_node("verify_claim", verify_claim_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.add_node("premise_guard", premise_guard_node)
    # Live search nodes
    graph.add_node("live_search_agent", live_search_agent_node)
    graph.add_node("live_search_tools", live_search_tool_node)

    graph.set_entry_point("router")

    graph.add_conditional_edges(
        "router",
        route_query,
        {
            "paper_qa": "agent_node",
            "live_search": "live_search_agent",
            "verify_claim": "verify_claim",
            "direct_answer": "generate_answer",
        },
    )

    graph.add_conditional_edges(
        "agent_node",
        agent_routing,
        {
            "retrieval": "retrieval",
            "relevancy_check": "relevancy_check",
            "generate_answer": "premise_guard",
        },
    )
    graph.add_edge("retrieval", "agent_node")

    # Live search loop: agent â†’ tools â†’ agent, then â†’ generate_answer
    graph.add_conditional_edges(
        "live_search_agent",
        live_search_routing,
        {
            "live_search_tools": "live_search_tools",
            "generate_answer": "generate_answer",
        },
    )
    graph.add_edge("live_search_tools", "live_search_agent")

    graph.add_conditional_edges(
        "relevancy_check",
        after_relevancy_routing,
        {"query_rewrite": "query_rewrite", "generate_answer": "premise_guard"},
    )
    graph.add_edge("premise_guard", "generate_answer")
    graph.add_edge("query_rewrite", "direct_retrieval")
    graph.add_edge("direct_retrieval", "relevancy_check")

    graph.add_edge("verify_claim", "generate_answer")

    # REFUSAL GUARD: if the whole paper_qa pipeline produced "The paper does
    # not state this." / "I don't know..." AND there is rewrite budget left,
    # force one more rewrite + deterministic retrieval. False-premise questions
    # (GELU-style) reach the gate with junk context and relevancy votes true on
    # the premise-contradiction clause; the only reliable signal that retrieval
    # starved is the final answer itself.
    def routing_after_answer(state: RAGState) -> str:
        answer = (state.get("answer") or "").strip().lower()
        refused = answer.startswith(_REFUSAL_PREFIXES)
        if refused and state.get("rewrite_count", 0) < MAX_REWRITES:
            return "query_rewrite"
        return "end"

    graph.add_conditional_edges(
        "generate_answer",
        routing_after_answer,
        {"query_rewrite": "query_rewrite", "end": END},
    )

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    print("Building RAG graph...")
    compiled = build_graph(":memory:")
    print("Graph built successfully!")
    print(f"Nodes ({len(compiled.nodes)}): {list(compiled.nodes.keys())}")

    output_path = Path(_project_root) / "graph_built.png"
    try:
        png_data = compiled.get_graph().draw_mermaid_png()
        output_path.write_bytes(png_data)
        print(f"Saved graph visualization to: {output_path}")
    except Exception as e:
        print(f"Could not render graph image: {e}")


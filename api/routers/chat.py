"""POST /chat — run the RAG graph to completion, return answer + sources.
POST /chat/stream — same graph, SSE token stream (`delta`/`replace`/`done`).
GET  /chat/{session_id}/history — restore chat UI from the checkpointer.
POST /chat/rename — LLM-generated 3-5 word session title (same as the web UI).
"""
import asyncio
import json
import os
import re

import psycopg
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from backend.llm_config import build_llm

from ..auth import require_api_key, require_user
from ..deps import build_sources_list, get_graph, refresh_graph
from ..rate_limit import rate_limit
from ..schemas import (
    ChatRequest,
    ChatResponse,
    HistoryResponse,
    RenameRequest,
    RenameResponse,
)

router = APIRouter()


def _input_state(body: ChatRequest, user_id: str) -> dict:
    return {
        "messages": [{"role": "user", "content": body.query}],
        "session_id": body.session_id,
        "doc_scope_ids": body.doc_scope_ids or [body.session_id],
        "doc_titles": body.doc_titles or [],
        "user_id": user_id,
        "query": body.query,
        "route": None,
        "retrieved_docs": [],
        "retrieval_attempts": 0,
        "claim_verdict": None,
        "claim_source": None,
        "superseding_papers": [],
        "answer": None,
        "is_relevant": None,
        "rewrite_count": 0,
    }


def _call_with_recovery(op, config):
    """Run a checkpointer-backed graph op; rebuild the graph once if the
    Neon Postgres connection went stale (`the connection is closed`)."""
    try:
        return op(get_graph())
    except psycopg.OperationalError:
        # Any operational failure (closed SSL socket, DNS drop, pool
        # exhaustion) means the cached checkpointer connection is dead.
        # Rebuild once, then let a second failure surface.
        return op(refresh_graph())


@router.post("", response_model=ChatResponse,
             dependencies=[Depends(require_api_key), Depends(rate_limit)])
def chat(body: ChatRequest, user_id: str = Depends(require_user)) -> ChatResponse:
    graph = get_graph()
    input_state = _input_state(body, user_id)
    config = {"configurable": {"thread_id": body.session_id}}

    # Guardrail #1: pattern-based prompt-injection block before the graph runs.
    from backend.rag_graph import query_looks_injected, _INJECTION_REPLY

    if query_looks_injected(body.query):
        return ChatResponse(answer=_INJECTION_REPLY, route="blocked", sources=[])

    def _run(graph):
        # ponytail: non-streaming; SSE later if latency demands it.
        from backend.timing import timed

        with timed("request_total"):
            return graph.invoke(input_state, config)

    try:
        final = _call_with_recovery(_run, config)
    except Exception as e:  # noqa: BLE001 - surfaced as a friendly 429 below
        if type(e).__name__ == "RateLimitError" and type(e).__module__.split(".")[0] in ("groq", "openai"):
            m = re.search(r"try again in (\d+(?:\.\d+)?)m", str(e))
            wait = int(float(m.group(1)) + 0.9) if m else 20
            raise HTTPException(
                status_code=429,
                detail=(
                    f"AI rate limit reached (free tier daily cap). "
                    f"Try again in about {wait} min, or upgrade your LLM provider tier."
                ),
            ) from e
        raise
    return ChatResponse(
        answer=final.get("answer") or "No response generated.",
        route=final.get("route"),
        sources=build_sources_list(final),
    )


def _sse(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/stream",
             dependencies=[Depends(require_api_key), Depends(rate_limit)])
async def chat_stream(body: ChatRequest, user_id: str = Depends(require_user)):
    """SSE chat: `delta` tokens, optional `replace` (critic rewrite), `done`.

    Answers that may need the critic are buffered by the generate node and
    arrive only in `done`; safe answers stream. If the client disconnects
    mid-stream we stop sending tokens, but the graph runs in a worker thread that
    cannot be cancelled, so the finished turn can still be saved to history.
    """
    from backend.rag_graph import _INJECTION_REPLY, query_looks_injected
    from backend.streaming import reset_emitter, set_emitter

    graph = get_graph()
    input_state = _input_state(body, user_id)
    config = {"configurable": {"thread_id": body.session_id}}
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _emit(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    async def _run() -> None:
        token = set_emitter(_emit)
        try:
            if query_looks_injected(body.query):
                await queue.put({
                    "type": "done",
                    "answer": _INJECTION_REPLY,
                    "route": "blocked",
                    "sources": [],
                })
                return
            final = await asyncio.to_thread(
                _call_with_recovery,
                lambda g: g.invoke(input_state, config),
                config,
            )
            answer = final.get("answer") or "No response generated."
            await queue.put({
                "type": "done",
                "answer": answer,
                "route": final.get("route"),
                "sources": build_sources_list(final),
            })
        except Exception as e:  # noqa: BLE001 - delivered to the client as SSE
            await queue.put({"type": "error", "error": str(e)})
        finally:
            reset_emitter(token)
            await queue.put(None)

    async def _events():
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                yield _sse(event)
        finally:
            # client disconnected (or stream finished): stop waiting for the graph run
            # (its worker thread still finishes)
            if not task.done():
                task.cancel()

    task = asyncio.create_task(_run())
    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


_RENAME_PROMPT = (
    "Generate a concise 3-5 word title for a research chat session "
    "based on the user's first message. Return only the title, "
    "no punctuation at the end, no quotes."
)


@router.post("/rename", response_model=RenameResponse,
             dependencies=[Depends(require_api_key), Depends(rate_limit)])
def rename_session(body: RenameRequest, user_id: str = Depends(require_user)) -> RenameResponse:
    llm = build_llm(0)
    try:
        resp = llm.invoke([
            {"role": "system", "content": _RENAME_PROMPT},
            {"role": "user", "content": body.query[:500]},
        ])
        return RenameResponse(name=resp.content.strip() or "New Session")
    except Exception:
        return RenameResponse(name="New Session")


@router.get("/{session_id}/history", response_model=HistoryResponse,
            dependencies=[Depends(require_api_key), Depends(rate_limit)])
def chat_history(session_id: str, user_id: str = Depends(require_user)) -> HistoryResponse:
    """Rebuild the chat UI from the LangGraph checkpointer (same as the web UI)."""
    config = {"configurable": {"thread_id": session_id}}

    def _run(graph):
        return graph.get_state(config)

    state = _call_with_recovery(_run, config)
    import re

    ctrl = re.compile(r"[\x00\x08\x0b\x0c\x0e-\x1f]")
    messages = []
    if state and state.values:
        for msg in state.values.get("messages", []):
            type_name = type(msg).__name__
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content = ctrl.sub("", content)
            if type_name == "HumanMessage":
                if content.strip():
                    messages.append({"role": "user", "content": content})
            elif type_name in ("AIMessage", "AIMessageChunk"):
                # skip empty tool-call/router intermediates â€” only real answers
                if content.strip():
                    messages.append({"role": "assistant", "content": content})
    return HistoryResponse(messages=messages)


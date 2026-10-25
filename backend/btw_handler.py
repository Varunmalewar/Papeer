import os
from typing import Generator

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from backend.llm_config import build_llm
from tavily import TavilyClient

load_dotenv(override=True)  # .env is source of truth — overrides any stray env vars
llm = build_llm(0)

# Single prompt that handles both general knowledge and web-augmented answers.
# The LLM decides inline whether to mention it needs web data; we always
# attempt a lightweight web search and include results if available.
_ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant answering a side question outside the user's "
     "main research session. Be concise and accurate.\n\n"
     "{web_context}"),
    ("human", "{query}"),
])


def handle_btw(query: str) -> Generator[str, None, None]:
    """Off-topic side channel â€” never touches the vector store or checkpointer.

    Attempts a lightweight web search; falls back to general knowledge if it
    fails or returns nothing.
    """
    web_context = ""
    try:
        client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
        results = client.search(query, max_results=3)
        hits = results.get("results") or []
        if hits:
            context = "\n\n".join(r["content"] for r in hits)
            sources = "\n".join(f"- {r['url']}" for r in hits)
            web_context = (
                f"Web search results (use if relevant, ignore if not):\n{context}\n\nSources:\n{sources}"
            )
    except Exception:
        pass  # fall back to general knowledge

    for chunk in (_ANSWER_PROMPT | llm).stream({"query": query, "web_context": web_context}):
        if chunk.content:
            yield chunk.content

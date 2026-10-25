"""Shared chat-LLM factory for the Papeer pipeline.

Provider: OpenAI (gpt-4o-mini) — paid tier, no free-host rate/TPD caps.
Env overrides: PAPEER_LLM_MODEL (falls back to Xkiro if no OpenAI key).
"""
import os

from langchain_openai import ChatOpenAI


def build_llm(temperature: float = 0.0):
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=os.environ.get("PAPEER_LLM_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            temperature=temperature,
        )
    # Legacy fallback: Xkiro OpenAI-compatible proxy (qwen).
    return ChatOpenAI(
        model=os.environ.get("PAPEER_LLM_MODEL", "qwen/qwen3.8-max:free"),
        base_url=os.environ.get("PAPEER_LLM_BASE_URL", "https://api.xkiro.com/v1"),
        api_key=os.environ.get("XKIRO_API_KEY") or os.environ.get("GROQ_API_KEY"),
        temperature=temperature,
    )

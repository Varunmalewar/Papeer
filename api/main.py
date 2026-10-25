"""Papeer HTTP API: FastAPI wrapper over backend/ RAG graph + vector store."""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

load_dotenv(override=True)  # .env is source of truth — overrides any stray env vars
from api.routers import btw, chat, health, papers  # noqa: E402

app = FastAPI(title="Papeer API", version="0.1.0")

# Browser (Next.js dev server / production origin) calls the API directly,
# so CORS must allow the web app origin. Comma-separated list in WEB_ORIGINS,
# defaulting to the local dev origins.
_web_origins_env = os.getenv(
    "WEB_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
)
allow_origins = [o.strip() for o in _web_origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(btw.router, prefix="/btw", tags=["btw"])
app.include_router(papers.router, prefix="/papers", tags=["papers"])

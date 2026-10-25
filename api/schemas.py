"""Request/response schemas for the Papeer API."""
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    doc_scope_ids: list[str] | None = None
    doc_titles: list[str] | None = None


class ChatResponse(BaseModel):
    answer: str
    route: str | None = None
    sources: list[dict] = []


class BtwRequest(BaseModel):
    query: str = Field(min_length=1)


class HistoryMsg(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    messages: list[HistoryMsg] = []


class RenameRequest(BaseModel):
    query: str = Field(min_length=1)


class RenameResponse(BaseModel):
    name: str


class SessionCreateResponse(BaseModel):
    session_id: str


class PapersResponse(BaseModel):
    titles: list[str] = []


class StatusResponse(BaseModel):
    chunks: int

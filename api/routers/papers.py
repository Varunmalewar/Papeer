"""Paper endpoints: upload / list / status / delete (same backend functions as the web UI).

Every endpoint requires `X-Papeer-User` â€” a server-issued per-user scope id set
by the Next.js proxy after Neon Auth session verification. All Qdrant reads and
writes are additionally filtered by it, so client-supplied session/scope ids
cannot reach another user's documents.
"""
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, Response, UploadFile

from ..auth import require_api_key, require_user
from ..deps import get_graph  # noqa: F401 â€” keeps graph warm on import order
from ..rate_limit import rate_limit
from ..schemas import PapersResponse, StatusResponse
from backend.ingestion import (
    delete_paper_assets,
    ingest_arxiv,
    ingest_pdf_bytes,
    ingest_webpage,
)

router = APIRouter()

_ALLOWED_EXTS = {".pdf", ".txt", ".md", ".markdown"}
_MAX_BYTES = 50 * 1024 * 1024  # ponytail: fixed cap; env knob only if asked


def _validate_upload(ext: str, data: bytes, filename: str) -> None:
    """Content-level checks beyond the extension/size gate. Raises 400."""
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")
    if ext == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise HTTPException(
                status_code=400,
                detail=f"'{filename}' is not a valid PDF (missing %PDF header).",
            )
    elif ext in (".txt", ".md"):
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(
                status_code=400,
                detail=f"'{filename}' is not readable text (UTF-8 required).",
            )


@router.post("/upload",
             dependencies=[Depends(require_api_key), Depends(rate_limit)])
def upload_paper(session_id: str, file: UploadFile = File(...),
                 user_id: str = Depends(require_user)) -> StatusResponse:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in _ALLOWED_EXTS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
    data = file.file.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 50MB).")
    _validate_upload(ext, data, file.filename or "file")
    try:
        if ext == ".pdf":
            docs = ingest_pdf_bytes(data, file.filename or "file.pdf", session_id, user_id)
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(data)
                tmp_path = tmp.name
            try:
                from backend.paper_loader import load_document
                from backend.vector_store import add_paper

                docs = load_document(tmp_path)
                for doc in docs:
                    doc.metadata["title"] = Path(file.filename).stem
                add_paper(docs, session_id, user_id=user_id)
            finally:
                Path(tmp_path).unlink(missing_ok=True)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not extract text from '{file.filename}': {e}",
        )
    if not docs:
        raise HTTPException(
            status_code=400,
            detail=f"No readable text found in '{file.filename}'.",
        )
    from backend.vector_store import count_chunks

    return StatusResponse(chunks=count_chunks(session_id, user_id=user_id))


@router.post("/arxiv",
             dependencies=[Depends(require_api_key), Depends(rate_limit)])
def upload_arxiv(session_id: str, query: str,
                 user_id: str = Depends(require_user)) -> StatusResponse:
    from backend.vector_store import count_chunks

    ingest_arxiv(query, session_id, user_id)
    return StatusResponse(chunks=count_chunks(session_id, user_id=user_id))


@router.post("/webpage",
             dependencies=[Depends(require_api_key), Depends(rate_limit)])
def upload_webpage(session_id: str, url: str,
                   user_id: str = Depends(require_user)) -> StatusResponse:
    from backend.vector_store import count_chunks

    ingest_webpage(url, session_id, user_id)
    return StatusResponse(chunks=count_chunks(session_id, user_id=user_id))


@router.get("/{session_id}/figures/{figure_id}",
            dependencies=[Depends(require_api_key), Depends(rate_limit)])
def get_figure(session_id: str, figure_id: str,
               user_id: str = Depends(require_user)) -> Response:
    """Serve a figure image â€” only to the user who owns the paper."""
    from backend.blob_store import get_figure_store
    from backend.figure_extractor import media_type_for
    from backend.vector_store import get_figure_payload

    payload = get_figure_payload(figure_id, user_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Figure not found.")
    md = payload.get("metadata") or {}
    image_path = md.get("image_path")
    if not image_path:
        raise HTTPException(status_code=404, detail="Figure image unavailable.")
    try:
        data = get_figure_store().get(image_path)
    except Exception:
        raise HTTPException(status_code=404, detail="Figure image unavailable.")
    return Response(
        content=data,
        media_type=media_type_for(md.get("image_ext") or "png"),
        headers={"Cache-Control": "private, max-age=3600"},
    )


def _scope(session_id: str, scope_ids: str | None) -> list[str]:
    """User library scope: the primary id plus any extra (legacy) session ids."""
    extra = [s.strip() for s in (scope_ids or "").split(",") if s.strip()]
    return list(dict.fromkeys([session_id, *extra]))


@router.get("/{session_id}", response_model=PapersResponse,
            dependencies=[Depends(require_api_key), Depends(rate_limit)])
def list_session_papers(session_id: str, scope_ids: str | None = Query(None),
                        user_id: str = Depends(require_user)) -> PapersResponse:
    from backend.vector_store import list_papers

    return PapersResponse(titles=list_papers(_scope(session_id, scope_ids), user_id=user_id))


@router.get("/{session_id}/status", response_model=StatusResponse,
            dependencies=[Depends(require_api_key), Depends(rate_limit)])
def session_status(session_id: str, scope_ids: str | None = Query(None),
                   user_id: str = Depends(require_user)) -> StatusResponse:
    from backend.vector_store import count_chunks

    return StatusResponse(chunks=count_chunks(_scope(session_id, scope_ids), user_id=user_id))


@router.delete("/{session_id}",
              dependencies=[Depends(require_api_key), Depends(rate_limit)])
def delete_session(session_id: str,
                   user_id: str = Depends(require_user)) -> dict:
    from backend.vector_store import delete_session as vs_delete_session
    from backend.vector_store import list_papers

    titles = list_papers([session_id], user_id=user_id)
    deleted = vs_delete_session(session_id, user_id=user_id)
    for title in titles:
        delete_paper_assets(title, user_id)
    return {"deleted_chunks": deleted}


@router.delete("/{session_id}/papers",
              dependencies=[Depends(require_api_key), Depends(rate_limit)])
def delete_paper(session_id: str, title: str,
                 user_id: str = Depends(require_user)) -> dict:
    from backend.vector_store import delete_paper as vs_delete_paper

    deleted = vs_delete_paper(session_id, title, user_id=user_id)
    delete_paper_assets(title, user_id)
    return {"deleted_chunks": deleted}


@router.post("/sessions", dependencies=[Depends(require_api_key)])
def create_session() -> dict:
    return {"session_id": str(uuid.uuid4())}

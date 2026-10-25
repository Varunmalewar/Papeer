"""POST /btw â€” off-topic side channel (no vector store)."""
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ..auth import require_api_key
from ..rate_limit import rate_limit
from ..schemas import BtwRequest
from backend.btw_handler import handle_btw

router = APIRouter()


@router.post("", dependencies=[Depends(require_api_key), Depends(rate_limit)])
def btw(body: BtwRequest):
    return StreamingResponse(handle_btw(body.query), media_type="text/plain")

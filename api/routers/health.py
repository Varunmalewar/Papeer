"""GET /health — unauthenticated liveness probe for Docker/orchestrators."""
from fastapi import APIRouter

router = APIRouter()


@router.get("")
def health() -> dict:
    return {"status": "ok"}

"""In-memory sliding-window rate limiter (stdlib only, no new dependency).

ponytail: single-process memory. Upgrade to Redis when running >1 replica.
"""
import os
import time
from collections import deque

from fastapi import HTTPException, Request

_hits: dict[str, deque[float]] = {}

WINDOW_S = 60.0


def _limit_per_min() -> int:
    try:
        return max(1, int(os.environ.get("PAPEER_RATE_LIMIT_PER_MIN", "20")))
    except ValueError:
        return 20


def rate_limit(request: Request) -> None:
    auth = request.headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        key = "jwt:" + auth.split(None, 1)[1][-12:]  # per-token bucket, no token stored
    else:
        # X-Papeer-User is only honoured after auth passed (require_api_key runs first)
        key = request.headers.get("X-Papeer-User") or request.headers.get("X-API-Key") or (request.client.host if request.client else "?")
    now = time.monotonic()
    window = _hits.setdefault(key, deque())
    while window and now - window[0] > WINDOW_S:
        window.popleft()
    if len(window) >= _limit_per_min():
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again in a minute.")
    window.append(now)

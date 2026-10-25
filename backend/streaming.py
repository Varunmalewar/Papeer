"""Per-request token emitter for SSE streaming.

The graph nodes are sync and run inside a worker thread; the HTTP layer sets a
thread-safe emitter in the request context and the generate node calls
``emit()`` as tokens arrive. When no emitter is set (normal JSON requests,
tests) every helper is a no-op.
"""
from __future__ import annotations

import contextvars
from typing import Callable

Emitter = Callable[[dict], None]

_emitter: contextvars.ContextVar[Emitter | None] = contextvars.ContextVar(
    "papeer_stream_emitter", default=None
)


def set_emitter(fn: Emitter):
    return _emitter.set(fn)


def reset_emitter(token) -> None:
    _emitter.reset(token)


def streaming_active() -> bool:
    return _emitter.get() is not None


def emit(event: dict) -> None:
    fn = _emitter.get()
    if fn is None:
        return
    try:
        fn(event)
    except Exception:
        pass  # a broken client must never break generation

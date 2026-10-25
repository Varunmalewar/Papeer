"""Opt-in per-stage timing instrumentation.

Set ``PAPEER_TIMING=1`` to print one ``[timing] <label>=<ms>ms`` line per stage.
Disabled by default; every helper is a no-op when the flag is off.
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from functools import wraps


def enabled() -> bool:
    return os.environ.get("PAPEER_TIMING", "").strip() == "1"


def log(label: str, ms: float, detail: str = "") -> None:
    suffix = f" {detail}" if detail else ""
    print(f"[timing] {label}={ms:.0f}ms{suffix}", flush=True)


@contextmanager
def timed(label: str, detail: str = ""):
    if not enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        log(label, (time.perf_counter() - start) * 1000.0, detail)


def timed_node(label: str):
    """Wrap a graph node so its wall time is reported as ``[timing] <label>``."""

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            with timed(label):
                return fn(*args, **kwargs)

        return wrapper

    return decorator

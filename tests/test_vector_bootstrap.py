"""Guard tests for the offline vector-store bootstrap switch."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from backend import vector_store


ROOT = Path(__file__).resolve().parents[1]


def test_guard_leaves_qdrant_client_unset():
    assert os.environ["PAPEER_SKIP_VECTOR_BOOTSTRAP"] == "1"
    assert vector_store.qdrant_client is None


def test_guard_rejects_vector_access_with_clear_error():
    with pytest.raises(RuntimeError, match="PAPEER_SKIP_VECTOR_BOOTSTRAP"):
        vector_store._ensure_collection()


def test_guard_prevents_create_engine_and_qdrant_client_calls():
    script = r"""
import os
os.environ["PAPEER_SKIP_VECTOR_BOOTSTRAP"] = "1"

import sqlalchemy

def _boom(*args, **kwargs):
    raise AssertionError("database/Qdrant bootstrap was attempted")

sqlalchemy.create_engine = _boom

import qdrant_client
qdrant_client.QdrantClient = _boom

import backend.vector_store as vs

assert vs.qdrant_client is None
try:
    vs._ensure_collection()
except RuntimeError as exc:
    assert "PAPEER_SKIP_VECTOR_BOOTSTRAP" in str(exc)
else:
    raise AssertionError("_ensure_collection did not raise")

print("guarded import ok")
"""
    env = os.environ.copy()
    env["PAPEER_SKIP_VECTOR_BOOTSTRAP"] = "1"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "guarded import ok" in result.stdout

"""Pluggable blob storage for ingested artefacts (figure images, source PDFs).

Keys are always POSIX-style relative paths that START with the owning
``user_id`` — user isolation is structural, so a path can never escape its
owner's subtree. Swapping to S3 only requires a new ``BlobStore`` subclass;
callers depend on the abstract interface only.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

_KEY_RE = re.compile(r"^[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*$")


def validate_key(key: str) -> str:
    """Reject absolute paths, traversal, backslashes and empty segments."""
    if not isinstance(key, str) or not key.strip():
        raise ValueError("storage key must be a non-empty string")
    key = key.strip().replace("\\", "/")
    if key.startswith("/") or ".." in key.split("/") or not _KEY_RE.match(key):
        raise ValueError(f"invalid storage key: {key!r}")
    return key


class BlobStore(ABC):
    """Minimal object-storage contract: save / get / delete / prefix delete."""

    @abstractmethod
    def save(self, key: str, data: bytes) -> str: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int: ...


class LocalBlobStore(BlobStore):
    """Filesystem backend rooted at ``root`` (created on demand)."""

    def __init__(self, root: str | os.PathLike) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / validate_key(key)).resolve()
        if self.root != path and self.root not in path.parents:
            raise ValueError(f"storage key escapes root: {key!r}")
        return path

    def save(self, key: str, data: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        return validate_key(key)

    def get(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def delete(self, key: str) -> bool:
        path = self._path(key)
        if path.is_file():
            path.unlink()
            return True
        return False

    def delete_prefix(self, prefix: str) -> int:
        """Delete every file under ``prefix`` (a user/paper directory)."""
        prefix = validate_key(prefix)
        base = self._path(prefix)
        if not base.exists():
            return 0
        if base.is_file():
            base.unlink()
            return 1
        count = sum(1 for p in base.rglob("*") if p.is_file())
        shutil.rmtree(base, ignore_errors=True)
        return count


def paper_id_for(title: str) -> str:
    """Stable short id for a paper title (same title -> same id everywhere)."""
    norm = re.sub(r"\s+", " ", (title or "").strip().lower())
    return hashlib.md5(norm.encode("utf-8")).hexdigest()[:16]


def figure_key(user_id: str, paper_id: str, figure_id: str, ext: str) -> str:
    ext = (ext or "png").lower().lstrip(".")
    return validate_key(f"{user_id}/{paper_id}/figures/{figure_id}.{ext}")


def pdf_key(user_id: str, paper_id: str) -> str:
    return validate_key(f"{user_id}/{paper_id}/paper.pdf")


def paper_prefix(user_id: str, paper_id: str) -> str:
    return validate_key(f"{user_id}/{paper_id}")


def _store_from_env(var: str, default: str) -> BlobStore:
    # S3 support plugs in here: read PAPEER_BLOB_BACKEND and return an
    # S3BlobStore(bucket=..., prefix=...) — no caller changes required.
    return LocalBlobStore(os.environ.get(var, default))


_figure_store: BlobStore | None = None
_pdf_store: BlobStore | None = None


def get_figure_store() -> BlobStore:
    global _figure_store
    if _figure_store is None:
        _figure_store = _store_from_env("PAPEER_FIGURE_DIR", "./data/figures")
    return _figure_store


def get_pdf_store() -> BlobStore:
    global _pdf_store
    if _pdf_store is None:
        _pdf_store = _store_from_env("PAPEER_PDF_DIR", "./data/pdfs")
    return _pdf_store


def delete_paper_files(user_id: str, paper_id: str) -> int:
    """Cascade: remove the paper's images AND its persisted PDF."""
    removed = get_figure_store().delete_prefix(paper_prefix(user_id, paper_id))
    removed += get_pdf_store().delete_prefix(paper_prefix(user_id, paper_id))
    return removed

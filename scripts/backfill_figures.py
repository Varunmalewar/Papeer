"""Backfill figure chunks for papers already ingested.

Figures are extracted from the *persisted* source PDF, so only papers ingested
after PDF persistence was enabled (or arXiv papers that can be re-fetched) can
be backfilled. Papers without a stored PDF are reported as needing a re-upload.

Usage:
    python scripts/backfill_figures.py --user lib-XXXX [--session lib-XXXX]
        [--title "Paper title"] [--describe] [--dry-run]

The script is idempotent: existing figure chunks for a paper are deleted before
the freshly extracted ones are indexed. Text chunks are never touched.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.blob_store import get_figure_store, get_pdf_store, paper_id_for, pdf_key
from backend.figure_extractor import (
    DescriptionCache,
    FigureIngestContext,
    figures_enabled,
    ingest_figure_docs,
    vision_model_name,
)
from backend.vector_store import add_paper, delete_figure_chunks, list_papers


def _ctx(user: str, title: str, describe: bool) -> FigureIngestContext:
    return FigureIngestContext(
        user_id=user,
        paper_id=paper_id_for(title),
        store=get_figure_store(),
        describe=describe,
        vision_model=vision_model_name() if describe else "",
        cache=DescriptionCache() if describe else None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill figure chunks for ingested papers.")
    ap.add_argument("--user", required=True, help="Library id owning the papers (X-Papeer-User).")
    ap.add_argument("--session", default=None, help="Session id (defaults to the user's library).")
    ap.add_argument("--title", default=None, help="Only backfill this exact paper title.")
    ap.add_argument("--describe", action="store_true", help="Also generate vision descriptions.")
    ap.add_argument("--dry-run", action="store_true", help="Report what would change.")
    args = ap.parse_args()

    if not figures_enabled():
        print("PAPEER_FIGURES=0 — figure ingestion is disabled.")
        return 1

    session = args.session or args.user
    titles = list_papers([session], user_id=args.user)
    if args.title:
        titles = [t for t in titles if t.strip().lower() == args.title.strip().lower()]
    if not titles:
        print(f"No papers found for session {session!r}.")
        return 0

    pdfs = get_pdf_store()
    added = missing = failed = 0
    for title in titles:
        key = pdf_key(args.user, paper_id_for(title))
        if not pdfs.exists(key):
            print(f"[skip] {title!r}: no persisted PDF — re-upload this paper once.")
            missing += 1
            continue
        if args.dry_run:
            print(f"[dry-run] {title!r}: PDF found, figures would be re-extracted.")
            added += 1
            continue
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdfs.get(key))
                tmp_path = tmp.name
            docs = ingest_figure_docs(tmp_path, title, _ctx(args.user, title, args.describe))
            if not docs:
                print(f"[none] {title!r}: no captioned figures found.")
                continue
            removed = delete_figure_chunks(session, title, user_id=args.user)
            add_paper(docs, session, user_id=args.user)
            print(f"[ok] {title!r}: +{len(docs)} figure chunks (removed {removed} old).")
            added += 1
        except Exception as e:
            print(f"[fail] {title!r}: {e}")
            failed += 1
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    print(f"\nDone. papers={len(titles)} backfilled={added} missing_pdf={missing} failed={failed}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())

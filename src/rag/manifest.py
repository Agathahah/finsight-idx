"""
Index manifest: records how the vector store was built.

The October 2026 audit found that the deployed demo was answering from an
index built by a buggy chunker, and nothing in the app could tell. The
manifest (``<persist_dir>/index_manifest.json``) stores the settings that
must match at query time. The retriever refuses an index whose schema
version or embedding model differs from the code, because the vectors
would be meaningless for the query encoder.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from loguru import logger

# Bump when chunking, extraction or metadata change in a way that needs a re-index.
#   1: original chunker (near-duplicate chunks, broken page numbers)
#   2: fixed chunker, header/footer removal, OCR fallback, `extraction` metadata
#   3: line-aware chunker without gaps (v2 skipped 1.8% of the text), chunks
#      prefixed with section, page, unit and year columns
#   4: one extra chunk per table row ("Label: 2023 = x; 2022 = y (unit)"),
#      `chunk_type` metadata
INDEX_SCHEMA_VERSION = 4
MANIFEST_NAME = "index_manifest.json"


class IndexVersionError(RuntimeError):
    """The vector store was built with settings the current code cannot use."""


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest_path(persist_dir: str | Path) -> Path:
    return Path(persist_dir) / MANIFEST_NAME


def load_manifest(persist_dir: str | Path) -> Optional[dict]:
    path = manifest_path(persist_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def record_document(
    persist_dir: str | Path,
    document_id: str,
    entry: dict,
    *,
    embed_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    """Add or replace one document in the manifest and return the manifest."""
    current = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "embed_model": embed_model,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }
    manifest = load_manifest(persist_dir) or {**current, "documents": {}}
    others = {k: v for k, v in manifest.get("documents", {}).items() if k != document_id}
    changed = {k: (manifest.get(k), v) for k, v in current.items() if manifest.get(k) != v}
    if changed and others:
        raise IndexVersionError(
            f"Index settings changed {changed} but {len(others)} other document(s) were "
            "indexed with the old settings. Delete the vector store and re-index every document."
        )
    manifest.update(current)
    manifest.setdefault("documents", {})[document_id] = {
        **entry,
        "indexed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    manifest_path(persist_dir).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def strict_mode() -> bool:
    return os.getenv("FINSIGHT_STRICT_INDEX", "0").lower() in {"1", "true", "yes"}


def check_compatible(
    persist_dir: str | Path,
    embed_model: str,
    strict: Optional[bool] = None,
) -> Optional[dict]:
    """Validate the manifest before serving queries.

    Raises IndexVersionError when the index cannot be used. A missing manifest
    is an error in strict mode (deployments) and a warning otherwise (tests,
    notebooks).
    """
    strict = strict_mode() if strict is None else strict
    manifest = load_manifest(persist_dir)
    if manifest is None:
        msg = (f"No {MANIFEST_NAME} in {persist_dir}: the index was built by an older "
               "version or by hand. Re-index with RAGIndexer.index_pdf(..., force_reindex=True).")
        if strict:
            raise IndexVersionError(msg)
        logger.warning(msg)
        return None
    if manifest.get("schema_version") != INDEX_SCHEMA_VERSION:
        raise IndexVersionError(
            f"Index schema v{manifest.get('schema_version')} but code expects "
            f"v{INDEX_SCHEMA_VERSION}. Re-index the documents."
        )
    if manifest.get("embed_model") != embed_model:
        raise IndexVersionError(
            f"Index was embedded with {manifest.get('embed_model')!r} but the retriever "
            f"uses {embed_model!r}. Query and document vectors would not be comparable."
        )
    return manifest

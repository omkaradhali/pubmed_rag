"""
parents.py — Sidecar store for parent chunks (v0.2 parent-child schema).

Parents are not embedded — they live in data/parents.jsonl and are loaded
into a process-wide dict keyed by chunk_id. retrieve.py uses get_parent()
to swap a child hit's text for its parent's text before passing context
to the generator. See D-042 and [[parent-child-chunking-explained]].

Design notes:
  * ChromaDB requires embeddings on every row, so parents cannot live in
    the same collection without polluting the vector space. Keeping them
    in a sidecar JSONL is cleaner and matches the helix-rag pattern.
  * The store is lazy-loaded on first call to get_parent(). Tests can
    inject a custom path via load_parents(path) or reset via clear_cache().
  * For PMC full-text (future) this same module is the swap point — replace
    the JSONL backend with SQLite or Postgres without changing the public API.

Public API:
    PARENTS_PATH                  — default sidecar path (env-overridable)
    save_parents(parents, path)   — write parents to JSONL
    append_parents(parents, path) — append parents to JSONL (incremental mode)
    load_parents(path)            — load JSONL into the module cache; returns dict
    get_parent(chunk_id)          — fetch a parent by chunk_id (lazy-loads on miss)
    get_all_parents(path)         — return all parent dicts as a list (for BM25 index)
    clear_cache()                 — reset the module cache (tests, hot reload)
"""

import json
import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)

PARENTS_PATH = Path(os.getenv("PARENTS_PATH", "data/parents.jsonl"))

# Process-wide cache keyed by chunk_id. None = not loaded yet; {} = loaded but
# empty (no parents file). Lazy loading avoids reading a large file at import
# time and keeps tests fast.
_cache: dict[str, dict] | None = None


# Persistence
def save_parents(parents: list[dict], path: str | os.PathLike = PARENTS_PATH) -> None:
    """
    Write parents to a JSONL file, overwriting any existing contents.

    Used by pipeline.py's full-rebuild path. Resets the in-memory cache so
    subsequent get_parent() calls reflect the new file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for parent in parents:
            f.write(json.dumps(parent) + "\n")
    _logger.info("Saved %d parents to %s", len(parents), path)
    clear_cache()


def append_parents(parents: list[dict], path: str | os.PathLike = PARENTS_PATH) -> None:
    """
    Append parents to a JSONL file, creating it if missing.

    Used by pipeline.py's incremental path. Resets the in-memory cache so
    the next lookup sees the new entries.
    """
    if not parents:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for parent in parents:
            f.write(json.dumps(parent) + "\n")
    _logger.info("Appended %d parents to %s", len(parents), path)
    clear_cache()


# Cache and lookup
def load_parents(path: str | os.PathLike = PARENTS_PATH) -> dict[str, dict]:
    """
    Load parents.jsonl into the module cache and return the dict.

    Idempotent: subsequent calls hit the cache. Use clear_cache() to force
    a re-read.

    If the file does not exist, the cache is set to {} so get_parent()
    raises KeyError consistently instead of FileNotFoundError. This matches
    the v0.1 corpus state before parents.jsonl exists.
    """
    global _cache
    if _cache is not None:
        return _cache

    path = Path(path)
    if not path.exists():
        _logger.warning("Parents file %s not found — using empty cache.", path)
        _cache = {}
        return _cache

    cache: dict[str, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parent = json.loads(line)
            cache[parent["chunk_id"]] = parent

    _logger.info("Loaded %d parents from %s", len(cache), path)
    _cache = cache
    return _cache


def get_parent(chunk_id: str, path: str | os.PathLike = PARENTS_PATH) -> dict:
    """
    Return the parent dict for a chunk_id, lazy-loading the file on first call.

    Raises:
        KeyError: if chunk_id is not in the parent store.
    """
    cache = load_parents(path)
    return cache[chunk_id]


def get_all_parents(path: str | os.PathLike = PARENTS_PATH) -> list[dict]:
    """Return all parent dicts as a list, lazy-loading the file on first call.

    Used by retrieve.py to build the BM25 index over all parent texts. Order
    matches the JSONL file order — stable across calls as long as the cache
    is warm.
    """
    return list(load_parents(path).values())


def clear_cache() -> None:
    """Reset the in-memory parents cache. Use after writes or in test setup."""
    global _cache
    _cache = None

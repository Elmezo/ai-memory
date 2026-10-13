"""Elasticsearch access: client, non-destructive mapping management, errors."""

from __future__ import annotations

import logging
import time
from typing import Any

from elasticsearch import (
    ApiError,
    ConflictError,
    ConnectionError,
    NotFoundError,
    RequestError,
    Elasticsearch,
    TransportError,
)

from config import settings

log = logging.getLogger("memory.store")

# NOTE: ``project_id`` in the pre-existing index is mapped as
# ``text`` + ``keyword`` sub-field (not pure keyword). Changing a field type
# requires a reindex, so we deliberately do NOT touch it here and query
# ``project_id.keyword`` with a fallback. See README "Migration" for the v2
# reindex procedure. All other fields below are added non-destructively.
FULL_PROPERTIES: dict[str, Any] = {
    "memory_id": {"type": "keyword"},
    "user_id": {"type": "keyword"},
    "tenant_id": {"type": "keyword"},
    "agent_id": {"type": "keyword"},
    "type": {"type": "keyword"},
    "content": {"type": "text"},
    "content_norm": {"type": "keyword"},  # normalized content for exact dedupe
    "importance": {"type": "float"},
    "confidence": {"type": "float"},
    "source": {"type": "keyword"},
    "tags": {"type": "keyword"},
    "metadata": {"type": "object", "dynamic": True},
    "status": {"type": "keyword"},
    "created_at": {"type": "date"},
    "updated_at": {"type": "date"},
    "last_accessed_at": {"type": "date"},
    "expires_at": {"type": "date"},
}

VECTOR_PROPERTY = {
    "type": "dense_vector",
    "dims": settings.EMBEDDING_DIMS,
    "index": True,
    "similarity": "cosine",
}


class StoreError(Exception):
    """Mapped storage error with an HTTP status code and safe message."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def map_es_error(exc: Exception, action: str = "operation") -> StoreError:
    if isinstance(exc, NotFoundError):
        return StoreError(404, "Memory not found")
    if isinstance(exc, (ConnectionError, TransportError)) and not isinstance(
        exc, (ApiError, RequestError)
    ):
        return StoreError(503, f"Search backend unavailable during {action}")
    if isinstance(exc, ConflictError):
        return StoreError(409, f"Version conflict during {action}")
    if isinstance(exc, RequestError):
        return StoreError(400, f"Invalid request during {action}")
    if isinstance(exc, ApiError):
        return StoreError(502, f"Search backend error during {action}")
    if isinstance(exc, StoreError):
        return exc
    log.exception("Unexpected storage error during %s", action)
    return StoreError(500, f"Unexpected error during {action}")


def create_client() -> Elasticsearch:
    return Elasticsearch(
        settings.ES_URL,
        request_timeout=settings.ES_TIMEOUT,
        max_retries=settings.ES_MAX_RETRIES,
        retry_on_timeout=True,
    )


def wait_for_es(es: Elasticsearch, timeout: float = 60.0) -> None:
    """Block until ES responds (used at startup / in Docker)."""
    deadline = time.time() + timeout
    delay = settings.ES_RETRY_BACKOFF
    last: Exception | None = None
    while time.time() < deadline:
        try:
            es.info()
            return
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(delay)
    raise StoreError(503, f"Elasticsearch not reachable at {settings.ES_URL}: {last}")


def ensure_index(es: Elasticsearch) -> dict[str, Any]:
    """Create the index if missing; otherwise ADD missing fields only.

    Never deletes data. Returns a report of what was done.
    """
    index = settings.ES_INDEX
    report: dict[str, Any] = {"index": index, "created": False, "added": []}
    try:
        if not es.indices.exists(index=index):
            properties = dict(FULL_PROPERTIES)
            properties["project_id"] = {"type": "keyword"}
            body: dict[str, Any] = {"mappings": {"properties": properties}}
            if settings.EMBEDDING_PROVIDER != "none":
                body["mappings"]["properties"]["embedding"] = dict(VECTOR_PROPERTY)
            es.indices.create(index=index, body=body)
            report["created"] = True
            return report

        # When ``index`` is an alias, ES returns the response keyed by the real
        # index name, not the alias. Use the first (and only) key instead.
        mapping_resp = es.indices.get_mapping(index=index)
        real_key = next(iter(mapping_resp))
        existing = mapping_resp[real_key]["mappings"].get("properties", {})
        additions: dict[str, Any] = {}
        for field, mapping in FULL_PROPERTIES.items():
            if field not in existing:
                additions[field] = mapping
        if settings.EMBEDDING_PROVIDER != "none" and "embedding" not in existing:
            try:
                additions["embedding"] = dict(VECTOR_PROPERTY)
            except Exception:  # noqa: BLE001
                log.warning("Skipping vector mapping (unsupported by ES version)")
        if additions:
            es.indices.put_mapping(index=index, properties=additions)
            report["added"] = sorted(additions)
        return report
    except StoreError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise map_es_error(exc, "index initialization")

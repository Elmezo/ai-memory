"""AI Memory Server — FastAPI REST API (model-agnostic memory service)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from elasticsearch import Elasticsearch
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from auth import Identity, enforce_scope, get_identity, validate_token_config
from config import settings
from embeddings import get_provider
from models import (
    MemoryInput,
    MemoryInteractionRequest,
    MemoryRecord,
    MemorySearch,
    MemoryUpdate,
)
from service import MemoryService
from store import StoreError, create_client, ensure_index, wait_for_es

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("memory.api")

service: MemoryService | None = None
es_client: Elasticsearch | None = None


def _service() -> MemoryService:
    if service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return service


@asynccontextmanager
async def lifespan(app: FastAPI):
    global service, es_client
    if settings.AUTH_MODE != "disabled":
        # Fail fast on a bad token configuration instead of silently dropping
        # malformed entries (which could otherwise weaken tenant isolation).
        validate_token_config(
            settings.API_TOKENS, require_non_empty=True, source="API_TOKENS"
        )
    es_client = create_client()
    wait_for_es(es_client)
    report = ensure_index(es_client)
    log.info("index ensure report: %s", report)
    try:
        embedder = get_provider(
            settings.EMBEDDING_PROVIDER,
            settings.EMBEDDING_DIMS,
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("Embedding provider unavailable (%s); continuing without vectors", exc)
        embedder = None
    service = MemoryService(es_client, embedder)
    yield
    try:
        es_client.close()
    except Exception:  # noqa: BLE001
        pass


app = FastAPI(title="AI Memory Server", version="1.0.0", lifespan=lifespan)


# --------------------------------------------------------------------- #
# middleware + error handlers (safe messages, BOM-safe, size-limited)
# --------------------------------------------------------------------- #
@app.middleware("http")
async def limit_body_size(request: Request, call_next):
    if request.method in {"POST", "PATCH", "PUT"}:
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413, content={"detail": "Request body too large"}
            )
    return await call_next(request)


@app.exception_handler(StoreError)
async def store_error_handler(_: Request, exc: StoreError):
    return JSONResponse(status_code=exc.status, content={"detail": exc.message})


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(Exception)
async def unhandled_handler(_: Request, exc: Exception):
    log.exception("Unhandled error: %s", type(exc).__name__)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


def _handle_store(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except StoreError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.message)


# --------------------------------------------------------------------- #
# endpoints
# --------------------------------------------------------------------- #
@app.get("/health")
def health():
    assert es_client is not None
    try:
        info = es_client.info()
        es_status: Any = {
            "reachable": True,
            "version": info["version"]["number"],
            "cluster": info.get("cluster_name"),
        }
        try:
            count = es_client.count(index=settings.ES_INDEX).get("count")
            es_status["index_docs"] = count
        except Exception:  # noqa: BLE001
            es_status["index_docs"] = None
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=503, detail="Elasticsearch unavailable")
    return {
        "status": "ok",
        "version": app.version,
        "elasticsearch": es_status,
        "index": settings.ES_INDEX,
        "auth_mode": settings.AUTH_MODE,
        "embedding_provider": settings.EMBEDDING_PROVIDER,
    }


@app.post("/memories", status_code=201)
def save_memory(data: MemoryInput, identity: Identity = Depends(get_identity)):
    if not data.user_id or not data.tenant_id:
        raise HTTPException(
            status_code=422,
            detail="user_id and tenant_id are required fields for POST /memories",
        )
    enforce_scope(identity, data.tenant_id, data.user_id)
    result = _handle_store(_service().create, data)
    if result.get("status") == "deduplicated":
        return JSONResponse(status_code=200, content=result)
    return JSONResponse(status_code=201, content=result)


@app.post("/memories/search")
def search_memories(data: MemorySearch, identity: Identity = Depends(get_identity)):
    if not data.user_id or not data.tenant_id:
        raise HTTPException(
            status_code=422,
            detail="user_id and tenant_id are required fields for POST /memories/search",
        )
    enforce_scope(identity, data.tenant_id, data.user_id)
    return _handle_store(_service().search, data)


@app.post("/memories/interact")
def interact_memories(data: MemoryInteractionRequest, identity: Identity = Depends(get_identity)):
    if not data.user_id or not data.tenant_id:
        raise HTTPException(
            status_code=422,
            detail="user_id and tenant_id are required fields for POST /memories/interact",
        )
    enforce_scope(identity, data.tenant_id, data.user_id)
    try:
        return _service().interact(
            query=data.query,
            tenant_id=data.tenant_id,
            user_id=data.user_id,
            agent_id=data.agent_id or "default",
            project_id=data.project_id or "default",
            mem_type=data.type,
            status=data.status,
            tags=data.tags,
            limit=data.limit,
            mode=data.mode,
            include_expired=data.include_expired,
            include_inactive=data.include_inactive,
            max_steps=data.max_steps,
            max_searches=data.max_searches,
            max_candidates=data.max_candidates,
            timeout=data.timeout,
            add_relevance_threshold=data.add_relevance_threshold,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/memories")
def list_memories(
    tenant_id: str = Query(...),
    user_id: str = Query(...),
    agent_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    type: str | None = Query(default=None),
    status: str | None = Query(default=None),
    tags: list[str] | None = Query(default=None),
    limit: int = Query(default=5, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
    include_expired: bool = Query(default=False),
    include_inactive: bool = Query(default=False),
    identity: Identity = Depends(get_identity),
):
    # Strip a possible BOM from query-string values (e.g. ?user_id=﻿x).
    tenant_id = tenant_id.replace("\ufeff", "").strip()
    user_id = user_id.replace("\ufeff", "").strip()
    enforce_scope(identity, tenant_id, user_id)
    return _handle_store(
        _service().list_memories,
        tenant_id,
        user_id,
        agent_id,
        project_id,
        type,
        status,
        tags,
        limit,
        offset,
        include_expired,
        include_inactive,
    )


@app.get("/memories/{memory_id}", response_model=MemoryRecord)
def get_memory(
    memory_id: str,
    user_id: str = Query(...),
    tenant_id: str = Query(...),
    identity: Identity = Depends(get_identity),
):
    user_id = user_id.replace("\ufeff", "").strip()
    tenant_id = tenant_id.replace("\ufeff", "").strip()
    enforce_scope(identity, tenant_id, user_id)
    return _handle_store(_service().get, memory_id, user_id, tenant_id)


@app.patch("/memories/{memory_id}")
def update_memory(
    memory_id: str,
    data: MemoryUpdate,
    user_id: str = Query(...),
    tenant_id: str = Query(...),
    identity: Identity = Depends(get_identity),
):
    user_id = user_id.replace("\ufeff", "").strip()
    tenant_id = tenant_id.replace("\ufeff", "").strip()
    enforce_scope(identity, tenant_id, user_id)
    return _handle_store(_service().update, memory_id, user_id, tenant_id, data)


@app.delete("/memories/{memory_id}")
def delete_memory(
    memory_id: str,
    user_id: str = Query(...),
    tenant_id: str = Query(...),
    identity: Identity = Depends(get_identity),
):
    user_id = user_id.replace("\ufeff", "").strip()
    tenant_id = tenant_id.replace("\ufeff", "").strip()
    enforce_scope(identity, tenant_id, user_id)
    return _handle_store(_service().delete, memory_id, user_id, tenant_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=settings.API_HOST, port=settings.API_PORT)

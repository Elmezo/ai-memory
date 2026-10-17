"""Model Context Protocol (MCP) server for AI Memory Server.

Exposes tools for AI agents using the current MCP Python SDK (mcp 2.x MCPServer).
Tools:
  - memory_health
  - memory_save
  - memory_search
  - memory_get
  - memory_update
  - memory_delete
  - memory_list

Uses the shared service layer (MemoryService) directly to avoid duplicating logic.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from config import settings
from embeddings import get_provider
from mcp_auth import (
    build_auth_settings,
    build_token_verifier,
    http_transport_enabled,
    require_http_auth_config,
)
from models import MemoryInput, MemorySearch, MemoryStatus, MemoryType, MemoryUpdate
from service import MemoryService
from store import StoreError, create_client, ensure_index, wait_for_es

log = logging.getLogger("memory.mcp")

# The stdio transport is a local, process-bound pipe: no network auth needed.
# The streamable-http transport listens on TCP, so it is wired with the SDK's
# bearer-token authentication from the start and refuses to start misconfigured.
_HTTP_AUTH = http_transport_enabled()

# Initialize MCP server
mcp = MCPServer(
    name="ai-memory-server",
    title="AI Memory Server",
    description="Model-agnostic memory service for AI agents and multi-tenant applications.",
    version="1.0.0",
    auth=build_auth_settings() if _HTTP_AUTH else None,
    token_verifier=build_token_verifier() if _HTTP_AUTH else None,
)

_service: MemoryService | None = None


def _annotations(
    title: str,
    *,
    read_only: bool,
    destructive: bool,
    idempotent: bool,
) -> ToolAnnotations:
    """Build MCP tool annotations that accurately describe a tool's behavior.

    ``open_world_hint`` is always ``False``: every tool operates on this
    server's own Elasticsearch-backed memory store (a closed domain), never on
    arbitrary external entities.
    """
    return ToolAnnotations(
        title=title,
        read_only_hint=read_only,
        destructive_hint=destructive,
        idempotent_hint=idempotent,
        open_world_hint=False,
    )


def get_service() -> MemoryService:
    global _service
    if _service is None:
        es_client = create_client()
        wait_for_es(es_client, timeout=10.0)
        ensure_index(es_client)
        try:
            embedder = get_provider(
                settings.EMBEDDING_PROVIDER,
                settings.EMBEDDING_DIMS,
                model=settings.EMBEDDING_MODEL,
                base_url=settings.OLLAMA_BASE_URL,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Embedding provider unavailable: %s", exc)
            embedder = None
        _service = MemoryService(es_client, embedder)
    return _service


@mcp.tool(
    name="memory_health",
    title="Memory Server Health",
    description=(
        "Check the health and connectivity of the AI Memory Server and its "
        "Elasticsearch backend. Returns the server status, Elasticsearch version "
        "and cluster name, index document count, embedding provider/model, and the "
        "active memory scope. Use this to verify the service is reachable before "
        "other operations. Read-only: it never modifies stored data."
    ),
    annotations=_annotations(
        "Memory Server Health", read_only=True, destructive=False, idempotent=True
    ),
)
def memory_health() -> dict[str, Any]:
    svc = get_service()
    try:
        info = svc.es.info()
        count = svc.es.count(index=svc.index).get("count")
        return {
            "status": "ok",
            "elasticsearch": {
                "reachable": True,
                "version": info["version"]["number"],
                "cluster": info.get("cluster_name"),
                "index_docs": count,
            },
            "index": svc.index,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_model": settings.EMBEDDING_MODEL,
            "scope": {
                "tenant_id": settings.MEMORY_TENANT_ID or "(agent-supplied)",
                "user_id": settings.MEMORY_USER_ID or "(agent-supplied)",
                "agent_id": settings.MEMORY_AGENT_ID,
                "project_id": settings.MEMORY_PROJECT_ID,
            },
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc)}


def _scope_tenant(agent_value: str) -> str:
    """Return the env-injected tenant if set, else fall back to agent-supplied value."""
    return settings.MEMORY_TENANT_ID or agent_value


def _scope_user(agent_value: str) -> str:
    """Return the env-injected user if set, else fall back to agent-supplied value."""
    return settings.MEMORY_USER_ID or agent_value


def _scope_agent(agent_value: str) -> str:
    return settings.MEMORY_AGENT_ID or agent_value


def _scope_project(agent_value: str) -> str:
    return settings.MEMORY_PROJECT_ID or agent_value


@mcp.tool(
    name="memory_save",
    title="Save Memory",
    description=(
        "Save a new long-term memory for the current agent/user, or return an "
        "existing equivalent memory instead of duplicating it (exact and fuzzy "
        "deduplication). Returns the saved or deduplicated memory plus any "
        "possible conflicts. This is an additive write: it never overwrites or "
        "deletes unrelated memories. tenant_id and user_id are injected from the "
        "server environment when configured; agent-supplied values are used only "
        "as a fallback."
    ),
    annotations=_annotations(
        "Save Memory", read_only=False, destructive=False, idempotent=False
    ),
)
def memory_save(
    content: str,
    user_id: str = "",
    tenant_id: str = "",
    agent_id: str = "",
    project_id: str = "",
    type: str = "fact",
    importance: float = 0.5,
    confidence: float = 1.0,
    source: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    status: str = "active",
    expires_at: str | None = None,
) -> dict[str, Any]:
    svc = get_service()
    try:
        mem_type = MemoryType(type)
    except ValueError:
        mem_type = MemoryType.fact

    try:
        mem_status = MemoryStatus(status)
    except ValueError:
        mem_status = MemoryStatus.active

    mem_input = MemoryInput(
        user_id=_scope_user(user_id),
        tenant_id=_scope_tenant(tenant_id),
        agent_id=_scope_agent(agent_id),
        project_id=_scope_project(project_id),
        type=mem_type,
        content=content,
        importance=importance,
        confidence=confidence,
        source=source,
        tags=tags or [],
        metadata=metadata or {},
        status=mem_status,
        expires_at=expires_at,  # pydantic parses ISO strings
    )
    try:
        return svc.create(mem_input)
    except StoreError as exc:
        return {"error": exc.message, "status_code": exc.status}


@mcp.tool(
    name="memory_search",
    title="Search Memories",
    description=(
        "Search stored memories within the caller's tenant/user scope using "
        "keyword (BM25), semantic (vector), or hybrid (RRF + cross-encoder "
        "rerank) retrieval, with optional filters and ranking. Returns a ranked "
        "list of matching memories and a total count. Read-only: it does not "
        "create, update, or delete memories."
    ),
    annotations=_annotations(
        "Search Memories", read_only=True, destructive=False, idempotent=True
    ),
)
def memory_search(
    query: str,
    user_id: str = "",
    tenant_id: str = "",
    agent_id: str | None = None,
    project_id: str | None = None,
    type: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
    limit: int = 5,
    offset: int = 0,
    include_expired: bool = False,
    include_inactive: bool = False,
    mode: str = "keyword",
    explain: bool = False,
) -> dict[str, Any]:
    svc = get_service()
    mem_type = None
    if type:
        try:
            mem_type = MemoryType(type)
        except ValueError:
            pass

    mem_status = None
    if status:
        try:
            mem_status = MemoryStatus(status)
        except ValueError:
            pass

    if mode not in {"keyword", "semantic", "hybrid"}:
        mode = "keyword"

    params = MemorySearch(
        user_id=_scope_user(user_id),
        tenant_id=_scope_tenant(tenant_id),
        query=query,
        agent_id=agent_id,
        project_id=project_id,
        type=mem_type,
        status=mem_status,
        tags=tags,
        limit=limit,
        offset=offset,
        include_expired=include_expired,
        include_inactive=include_inactive,
        mode=mode,  # type: ignore
        explain=explain,
    )
    try:
        return svc.search(params)
    except StoreError as exc:
        return {"error": exc.message, "status_code": exc.status}


@mcp.tool(
    name="memory_get",
    title="Get Memory",
    description=(
        "Retrieve a single stored memory by its memory_id, enforcing tenant and "
        "user isolation (a memory outside the caller's scope is reported as not "
        "found). Returns the full memory record. Read-only; the only side effect "
        "is updating the memory's last-accessed timestamp."
    ),
    annotations=_annotations(
        "Get Memory", read_only=True, destructive=False, idempotent=True
    ),
)
def memory_get(memory_id: str, user_id: str = "", tenant_id: str = "") -> dict[str, Any]:
    svc = get_service()
    try:
        return svc.get(
            memory_id=memory_id,
            user_id=_scope_user(user_id),
            tenant_id=_scope_tenant(tenant_id),
        )
    except StoreError as exc:
        return {"error": exc.message, "status_code": exc.status}


@mcp.tool(
    name="memory_update",
    title="Update Memory",
    description=(
        "Update selected fields (content, importance, confidence, type, tags, "
        "metadata, status, expiry, source) of an existing memory, scoped to the "
        "caller's tenant and user. Only the fields you provide are changed; "
        "omitted fields are left untouched. Returns the update status and the "
        "list of changed fields. This modifies stored data but is not "
        "destructive: it never deletes the memory."
    ),
    annotations=_annotations(
        "Update Memory", read_only=False, destructive=False, idempotent=True
    ),
)
def memory_update(
    memory_id: str,
    user_id: str = "",
    tenant_id: str = "",
    content: str | None = None,
    importance: float | None = None,
    confidence: float | None = None,
    type: str | None = None,
    source: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    status: str | None = None,
    expires_at: str | None = None,
) -> dict[str, Any]:
    svc = get_service()
    mem_type = None
    if type:
        try:
            mem_type = MemoryType(type)
        except ValueError:
            pass

    mem_status = None
    if status:
        try:
            mem_status = MemoryStatus(status)
        except ValueError:
            pass

    patch = MemoryUpdate(
        content=content,
        importance=importance,
        confidence=confidence,
        type=mem_type,
        source=source,
        tags=tags,
        metadata=metadata,
        status=mem_status,
        expires_at=expires_at,
    )
    try:
        return svc.update(
            memory_id=memory_id,
            user_id=_scope_user(user_id),
            tenant_id=_scope_tenant(tenant_id),
            patch=patch,
        )
    except StoreError as exc:
        return {"error": exc.message, "status_code": exc.status}


@mcp.tool(
    name="memory_delete",
    title="Delete Memory",
    description=(
        "Permanently delete a single memory by its memory_id, scoped to the "
        "caller's tenant and user. DESTRUCTIVE and irreversible: the memory "
        "cannot be recovered after deletion. Only the matching in-scope memory "
        "is removed; deleting an already-deleted memory has no additional effect."
    ),
    annotations=_annotations(
        "Delete Memory", read_only=False, destructive=True, idempotent=True
    ),
)
def memory_delete(memory_id: str, user_id: str = "", tenant_id: str = "") -> dict[str, Any]:
    svc = get_service()
    try:
        return svc.delete(
            memory_id=memory_id,
            user_id=_scope_user(user_id),
            tenant_id=_scope_tenant(tenant_id),
        )
    except StoreError as exc:
        return {"error": exc.message, "status_code": exc.status}


@mcp.tool(
    name="memory_list",
    title="List Memories",
    description=(
        "List memories for the caller's tenant/user scope with pagination and "
        "optional filters (agent, project, type, status, tags), sorted by "
        "updated_at, created_at, or importance. Returns the page of memories, "
        "the total match count, and the offset. Read-only: it does not modify "
        "stored data."
    ),
    annotations=_annotations(
        "List Memories", read_only=True, destructive=False, idempotent=True
    ),
)
def memory_list(
    tenant_id: str = "",
    user_id: str = "",
    agent_id: str | None = None,
    project_id: str | None = None,
    type: str | None = None,
    status: str | None = None,
    tags: list[str] | None = None,
    limit: int = 5,
    offset: int = 0,
    include_expired: bool = False,
    include_inactive: bool = False,
    sort: str = "updated_at",
) -> dict[str, Any]:
    svc = get_service()
    try:
        return svc.list_memories(
            tenant_id=_scope_tenant(tenant_id),
            user_id=_scope_user(user_id),
            agent_id=agent_id,
            project_id=project_id,
            mem_type=type,
            status=status,
            tags=tags,
            limit=limit,
            offset=offset,
            include_expired=include_expired,
            include_inactive=include_inactive,
            sort=sort,
        )
    except StoreError as exc:
        return {"error": exc.message, "status_code": exc.status}



def main():
    """Entrypoint for running the MCP server."""
    transport = settings.MCP_TRANSPORT
    if transport == "stdio":
        asyncio.run(mcp.run_stdio_async())
    elif transport in {"streamable-http", "http"}:
        # Fail fast if the network transport would be exposed without a valid
        # bearer-token configuration and an explicit tenant/user scope.
        require_http_auth_config()
        log.info(
            "MCP streamable-http listening on %s:%s (bearer auth required)",
            settings.MCP_HOST,
            settings.MCP_PORT,
        )
        asyncio.run(
            mcp.run_streamable_http_async(
                host=settings.MCP_HOST,
                port=settings.MCP_PORT,
            )
        )
    else:
        raise ValueError(f"Unknown MCP_TRANSPORT={transport!r}. Use stdio or streamable-http")


if __name__ == "__main__":
    main()

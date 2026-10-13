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

from config import settings
from embeddings import get_provider
from models import MemoryInput, MemorySearch, MemoryStatus, MemoryType, MemoryUpdate
from service import MemoryService
from store import StoreError, create_client, ensure_index, wait_for_es

log = logging.getLogger("memory.mcp")

# Initialize MCP server
mcp = MCPServer(
    name="ai-memory-server",
    title="AI Memory Server",
    description="Model-agnostic memory service for AI agents and multi-tenant applications.",
    version="1.0.0",
)

_service: MemoryService | None = None


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
    description="Check health of the AI Memory Server and Elasticsearch backend.",
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
    description=(
        "Save a new memory or deduplicate against existing memories. "
        "tenant_id and user_id are injected from the server environment when configured; "
        "values supplied by the agent are used only as fallback."
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
    description="Search memories using keyword, semantic, or hybrid search with filters and ranking.",
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
    description="Retrieve a specific memory by its ID, enforcing tenant and user isolation.",
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
    description="Update fields of an existing memory safely within tenant and user scope.",
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
    description="Permanently delete a memory within tenant and user scope.",
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
    description="List memories with pagination and filters by tenant, user, agent, status, etc.",
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
        asyncio.run(
            mcp.run_streamable_http_async(
                host=settings.API_HOST,
                port=settings.MCP_PORT,
            )
        )
    else:
        raise ValueError(f"Unknown MCP_TRANSPORT={transport!r}. Use stdio or streamable-http")


if __name__ == "__main__":
    main()

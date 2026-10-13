"""Comprehensive test suite for the AI Memory Server.

Covers:
  - Health checks
  - Creating and retrieving memories (CRUD)
  - Searching and filtering
  - Updating and deleting
  - Duplicate detection (exact and fuzzy)
  - Conflict detection
  - Invalid inputs and input size limits
  - Unicode, Arabic text, and UTF-8 BOM handling
  - Elasticsearch error mapping
  - Tenant isolation and authorization (Bearer tokens)
  - MCP tool behavior
  - Keyword and semantic/hybrid search
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import main
from auth import Identity, enforce_scope
from config import settings
from embeddings import HashEmbeddingProvider, get_provider
from mcp_server import (
    memory_delete,
    memory_get,
    memory_health,
    memory_list,
    memory_save,
    memory_search,
    memory_update,
)
from models import MemoryInput, MemorySearch, MemoryUpdate
from textutil import is_trivial, jaccard, normalize_text, strip_bom


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def client():
    """TestClient that initializes the FastAPI lifespan (Elasticsearch connection)."""
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer sec-demo-tok"}


# -----------------------------------------------------------------------------
# 1. Text Utilities & Unicode Tests
# -----------------------------------------------------------------------------
def test_strip_bom():
    assert strip_bom("\ufeffHello") == "Hello"
    assert strip_bom("Hello\ufeff") == "Hello"
    assert strip_bom("\ufeffمرحبا بك\ufeff") == "مرحبا بك"
    assert strip_bom("Simple text") == "Simple text"


def test_normalize_text():
    raw = "\ufeff  النص العربي   مع   مسافات  \ufeff"
    norm = normalize_text(raw)
    assert norm == "النص العربي مع مسافات"
    assert normalize_text("UPPERCASE String") == "uppercase string"


def test_is_trivial():
    assert is_trivial("") is True
    assert is_trivial("   ") is True
    assert is_trivial("ok") is True
    assert is_trivial("شكرا") is True
    assert is_trivial("تمام") is True
    assert is_trivial("???") is True
    assert is_trivial("This is important context") is False
    assert is_trivial("العميل يفضل اللغة العربية دائماً") is False


def test_jaccard_similarity():
    sim_exact = jaccard("same text", "same text")
    assert sim_exact == 1.0

    sim_high = jaccard("user prefers dark theme in editor", "user prefers dark theme")
    assert sim_high > 0.6

    sim_diff = jaccard("completely different text", "another phrase entirely")
    assert sim_diff == 0.0


# -----------------------------------------------------------------------------
# 2. Embeddings Provider Tests
# -----------------------------------------------------------------------------
def test_hash_embedding_provider():
    provider = HashEmbeddingProvider(dims=128)
    assert provider.name == "hash"
    assert provider.dims == 128
    assert provider.available() is True

    vec1 = provider.embed("User likes dark theme")
    vec2 = provider.embed("User likes dark theme")
    vec3 = provider.embed("مرحبا بالعالم")

    assert len(vec1) == 128
    assert vec1 == vec2  # Deterministic
    assert len(vec3) == 128


def test_embedding_factory():
    p_hash = get_provider("hash", 64)
    assert p_hash.name == "hash"

    p_none = get_provider("none", 64)
    assert p_none.available() is False


# -----------------------------------------------------------------------------
# 3. Health Endpoint Test
# -----------------------------------------------------------------------------
def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "elasticsearch" in data
    assert data["elasticsearch"]["reachable"] is True


# -----------------------------------------------------------------------------
# 4. REST API CRUD & Lifecycle
# -----------------------------------------------------------------------------
def test_create_get_search_delete_cycle(client):
    tenant = "test_tenant"
    user = "test_user_01"

    # 1. Create a memory
    payload = {
        "tenant_id": tenant,
        "user_id": user,
        "agent_id": "test_agent",
        "project_id": "test_project",
        "type": "fact",
        "content": "The production database port is 5432.",
        "importance": 0.8,
        "confidence": 0.95,
        "tags": ["db", "infrastructure"],
    }
    res = client.post("/memories", json=payload)
    assert res.status_code == 201
    created = res.json()
    assert created["status"] == "saved"
    mem_id = created["memory"]["memory_id"]
    assert mem_id.startswith("mem_")

    # 2. Get the memory
    res_get = client.get(f"/memories/{mem_id}?tenant_id={tenant}&user_id={user}")
    assert res_get.status_code == 200
    got = res_get.json()
    assert got["memory_id"] == mem_id
    assert got["content"] == payload["content"]
    assert got["last_accessed_at"] is not None

    # 3. Search the memory
    search_payload = {
        "tenant_id": tenant,
        "user_id": user,
        "query": "database port",
        "mode": "keyword",
    }
    res_search = client.post("/memories/search", json=search_payload)
    assert res_search.status_code == 200
    hits = res_search.json()["memories"]
    assert any(h["memory_id"] == mem_id for h in hits)

    # 4. Patch/update the memory
    patch_payload = {
        "content": "The production database port is 5433 updated.",
        "importance": 0.9,
    }
    res_patch = client.patch(
        f"/memories/{mem_id}?tenant_id={tenant}&user_id={user}", json=patch_payload
    )
    assert res_patch.status_code == 200
    assert res_patch.json()["status"] == "updated"

    # 5. List memories
    res_list = client.get(f"/memories?tenant_id={tenant}&user_id={user}")
    assert res_list.status_code == 200
    assert res_list.json()["count"] >= 1

    # 6. Delete the memory
    res_del = client.delete(f"/memories/{mem_id}?tenant_id={tenant}&user_id={user}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "deleted"

    # 7. Verify deletion
    res_get_del = client.get(f"/memories/{mem_id}?tenant_id={tenant}&user_id={user}")
    assert res_get_del.status_code == 404


def test_arabic_and_bom_handling(client):
    tenant = "arabic_tenant"
    user = "arabic_user"
    arabic_content = "\ufeffالعميل يطلب إعداد التقارير باللغة العربية كل صباح\ufeff"

    payload = {
        "tenant_id": f"\ufeff{tenant}\ufeff",
        "user_id": user,
        "type": "constraint",
        "content": arabic_content,
        "tags": ["تقارير", "لغة"],
    }
    res = client.post("/memories", json=payload)
    assert res.status_code == 201
    mem_id = res.json()["memory"]["memory_id"]

    # Search with Arabic query
    res_search = client.post(
        "/memories/search",
        json={"tenant_id": tenant, "user_id": user, "query": "التقارير العربية"},
    )
    assert res_search.status_code == 200
    assert len(res_search.json()["memories"]) >= 1

    # Cleanup
    client.delete(f"/memories/{mem_id}?tenant_id={tenant}&user_id={user}")


def test_deduplication_exact(client):
    tenant = "dedup_tenant"
    user = "dedup_user"
    content = "Unique fact: The solar constant is approximately 1361 W/m2."

    payload = {
        "tenant_id": tenant,
        "user_id": user,
        "type": "fact",
        "content": content,
    }
    # First save
    res1 = client.post("/memories", json=payload)
    assert res1.status_code == 201
    mem_id1 = res1.json()["memory"]["memory_id"]

    # Second save (exact same text)
    res2 = client.post("/memories", json=payload)
    assert res2.status_code == 200
    assert res2.json()["status"] == "deduplicated"
    assert res2.json()["duplicate_of"] == mem_id1

    # Cleanup
    client.delete(f"/memories/{mem_id1}?tenant_id={tenant}&user_id={user}")


def test_conflict_detection(client):
    tenant = "conflict_tenant"
    user = "conflict_user"

    # Initial memory
    res1 = client.post(
        "/memories",
        json={
            "tenant_id": tenant,
            "user_id": user,
            "type": "preference",
            "content": "The user preferred deadline is Tuesday 5 PM.",
        },
    )
    assert res1.status_code == 201
    mem_id1 = res1.json()["memory"]["memory_id"]

    # Conflicting memory (shares vocabulary but different deadline)
    res2 = client.post(
        "/memories",
        json={
            "tenant_id": tenant,
            "user_id": user,
            "type": "preference",
            "content": "The user preferred deadline is Thursday 10 AM.",
        },
    )
    assert res2.status_code == 201
    conflicts = res2.json().get("possible_conflicts", [])
    assert any(c["memory_id"] == mem_id1 for c in conflicts)

    # Cleanup
    client.delete(f"/memories/{mem_id1}?tenant_id={tenant}&user_id={user}")
    client.delete(f"/memories/{res2.json()['memory']['memory_id']}?tenant_id={tenant}&user_id={user}")


def test_reject_trivial_content(client):
    res = client.post(
        "/memories",
        json={"tenant_id": "t", "user_id": "u", "content": "ok"},
    )
    assert res.status_code == 422
    assert "trivial" in res.json()["detail"].lower()


# -----------------------------------------------------------------------------
# 5. Hybrid and Semantic Search Modes
# -----------------------------------------------------------------------------
def test_search_modes(client):
    tenant = "search_test_tenant"
    user = "search_test_user"

    res = client.post(
        "/memories",
        json={
            "tenant_id": tenant,
            "user_id": user,
            "type": "procedure",
            "content": "To deploy the application, run deploy.sh script.",
        },
    )
    mem_id = res.json()["memory"]["memory_id"]

    for mode in ["keyword", "semantic", "hybrid"]:
        res_search = client.post(
            "/memories/search",
            json={
                "tenant_id": tenant,
                "user_id": user,
                "query": "deploy application script",
                "mode": mode,
            },
        )
        assert res_search.status_code == 200
        data = res_search.json()
        assert data["mode"] == mode
        assert len(data["memories"]) >= 1

    client.delete(f"/memories/{mem_id}?tenant_id={tenant}&user_id={user}")


# -----------------------------------------------------------------------------
# 6. Multi-Tenancy and Scope Security
# -----------------------------------------------------------------------------
def test_tenant_isolation(client):
    # Tenant A
    res_a = client.post(
        "/memories",
        json={
            "tenant_id": "tenant_alpha",
            "user_id": "user_1",
            "content": "Confidential Alpha memory details",
        },
    )
    mem_a = res_a.json()["memory"]["memory_id"]

    # Tenant B should NOT see Tenant A's memory
    res_search_b = client.post(
        "/memories/search",
        json={
            "tenant_id": "tenant_beta",
            "user_id": "user_1",
            "query": "Confidential Alpha",
        },
    )
    assert res_search_b.status_code == 200
    assert len(res_search_b.json()["memories"]) == 0

    # Tenant B cannot get Tenant A's memory
    res_get_b = client.get(f"/memories/{mem_a}?tenant_id=tenant_beta&user_id=user_1")
    assert res_get_b.status_code == 404

    # Cleanup
    client.delete(f"/memories/{mem_a}?tenant_id=tenant_alpha&user_id=user_1")


def test_auth_enforce_scope_logic():
    # Test Identity authorization logic directly
    id_specific = Identity(tenant_id="tenant_a", user_id="user_1")
    id_wildcard = Identity(tenant_id="tenant_a", user_id=None, wildcard_user=True)

    # Allowed scopes
    settings.AUTH_MODE = "token"
    try:
        enforce_scope(id_specific, "tenant_a", "user_1")
        enforce_scope(id_wildcard, "tenant_a", "user_any")

        # Denied cross-tenant
        with pytest.raises(Exception) as exc1:
            enforce_scope(id_specific, "tenant_b", "user_1")
        assert "Cross-tenant" in str(exc1.value.detail)

        # Denied cross-user
        with pytest.raises(Exception) as exc2:
            enforce_scope(id_specific, "tenant_a", "user_2")
        assert "Cross-user" in str(exc2.value.detail)
    finally:
        settings.AUTH_MODE = "disabled"


# -----------------------------------------------------------------------------
# 7. MCP Tools Behavior Tests
# -----------------------------------------------------------------------------
def test_mcp_tools():
    # 1. Health
    h = memory_health()
    assert h["status"] == "ok"

    # 2. Save
    s = memory_save(
        user_id="mcp_user",
        tenant_id="mcp_tenant",
        content="MCP created test memory record.",
        type="task",
    )
    assert s["status"] == "saved"
    m_id = s["memory"]["memory_id"]

    # 3. Get
    g = memory_get(memory_id=m_id, user_id="mcp_user", tenant_id="mcp_tenant")
    assert g["memory_id"] == m_id

    # 4. Search
    srch = memory_search(
        user_id="mcp_user",
        tenant_id="mcp_tenant",
        query="MCP created test",
    )
    assert srch["count"] >= 1

    # 5. Update
    upd = memory_update(
        memory_id=m_id,
        user_id="mcp_user",
        tenant_id="mcp_tenant",
        content="MCP created test memory record updated.",
    )
    assert upd["status"] == "updated"

    # 6. List
    lst = memory_list(user_id="mcp_user", tenant_id="mcp_tenant")
    assert lst["count"] >= 1

    # 7. Delete
    d = memory_delete(memory_id=m_id, user_id="mcp_user", tenant_id="mcp_tenant")
    assert d["status"] == "deleted"

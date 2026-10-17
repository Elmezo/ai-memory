# AI Memory Server

A robust, model-agnostic, multi-tenant AI Memory service built with **Python 3.11**, **FastAPI**, **Elasticsearch 9.2.0**, and the official **Model Context Protocol (MCP)** SDK.

---

## 1. Architecture Overview

```
                      +------------------------------------------+
                      |         AI Agents / LLM Clients          |
                      +------------------------------------------+
                                     |                |
                HTTP REST (Port 8000)|                | MCP (stdio / HTTP 8001)
                                     v                v
                      +------------------------------------------+
                      |    FastAPI REST API    |   MCP Server    |
                      |       (main.py)        | (mcp_server.py) |
                      +------------------------+-----------------+
                                     \                /
                                      \              /
                                  +---------------------+
                                  |    MemoryService    |
                                  |    (service.py)     |
                                  +---------------------+
                                     /               \
                                    /                 \
                  +-----------------------+     +-----------------------+
                  |  Embedding Providers  |     |   Elasticsearch 9.2   |
                  |  (hash / ST / none)   |     |    (agent_memory)     |
                  +-----------------------+     +-----------------------+
```

### Key Highlights
- **Model-Agnostic**: Compatible with any LLM, agent framework, or backend service.
- **Multi-Tenancy & Authorization**: Tenant & user scoped querying with token-based authorization and cross-tenant isolation enforcement.
- **Lifecycle Management**: Exact and fuzzy deduplication, vocabulary-overlap conflict detection, access tracking (`last_accessed_at`), and status filtering.
- **Search Capabilities**: Keyword (BM25), Semantic Vector (dense_vector BBQ-HNSW), and Hybrid search with configurable weights and boosts (importance, confidence, recency).
- **Dual Interface**: Full REST API + Model Context Protocol (MCP) server exposing tools for agents.
- **Enterprise-Ready**: UTF-8 BOM sanitization, size limits, structured error handling, non-destructive index migrations, and Docker Compose deployment.

---

## 2. Project Structure

```
D:\ai-memory\memory-api\
├── auth.py                  # Token authentication and tenant/user scope enforcement
├── config.py                # Environment configuration settings
├── embeddings.py            # Embedding provider interfaces (hash, ST, none)
├── main.py                  # FastAPI REST API application & lifespan
├── mcp_server.py            # Model Context Protocol server exposing memory tools
├── models.py                # Pydantic models & canonical memory schemas
├── service.py               # MemoryService business logic (CRUD, search, dedup)
├── store.py                 # Elasticsearch client, non-destructive mapping, errors
├── textutil.py              # Text normalization, UTF-8 BOM, Jaccard similarity
├── test_all.py              # Automated test suite (16 comprehensive tests)
├── Dockerfile               # Container build definition
├── docker-compose.yml       # Stack definition (Elasticsearch + memory-api)
├── .dockerignore            # Docker ignore file
├── .env.example             # Documented environment variables template
└── requirements.txt         # Production dependencies
```

---

## 3. Prerequisites

- **Python**: 3.11+
- **Elasticsearch**: 9.x (or 8.x with vector search support).
- **Docker & Docker Compose**: (Optional, for containerized run).

---

## 4. Local Installation & Startup

### Step 1: Install Dependencies
```powershell
cd D:\ai-memory\memory-api
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Configure Environment
Copy `.env.example` to `.env` (or configure system environment variables):
```powershell
Copy-Item .env.example .env
```

### Step 3: Run Elasticsearch
Ensure Elasticsearch 9.2 is running at `http://localhost:9200`.

### Step 4: Run the REST API
```powershell
.\.venv\Scripts\python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Swagger UI will be accessible at: `http://localhost:8000/docs`

### Step 5: Run the MCP Server
- **stdio mode** (for local agent runtimes like Claude Desktop / Cursor):
  ```powershell
  .\.venv\Scripts\python mcp_server.py
  ```
- **Streamable HTTP mode** (remote agent access):
  Set `MCP_TRANSPORT=streamable-http` and `MCP_PORT=8001`, then run `python mcp_server.py`.

---

## 5. Docker Compose Deployment

To run the complete stack (Elasticsearch 9.2 + AI Memory Server):

```bash
docker compose up -d --build
```

### Verify Status
```bash
docker compose ps
curl http://localhost:8000/health
```

### Shutdown
```bash
docker compose down
```
*(Data is persisted in the `es_data` volume).*

---

## 6. API Endpoints & Request Examples

### 1. Health Check
`GET /health`
```bash
curl -s http://localhost:8000/health
```
**Response:**
```json
{
  "status": "ok",
  "version": "1.0.0",
  "elasticsearch": {
    "reachable": true,
    "version": "9.2.0",
    "cluster": "docker-cluster",
    "index_docs": 1
  },
  "index": "agent_memory",
  "auth_mode": "disabled",
  "embedding_provider": "hash"
}
```

### 2. Create / Save Memory (English & Arabic)
`POST /memories`

**English Example:**
```bash
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "user_id": "user_001",
    "agent_id": "erp-agent",
    "project_id": "sales",
    "type": "fact",
    "content": "The production database port is 5432.",
    "importance": 0.8,
    "confidence": 0.95,
    "tags": ["db", "infrastructure"]
  }'
```

**Arabic Example (with UTF-8 BOM safe handling):**
```bash
curl -X POST http://localhost:8000/memories \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "user_id": "user_001",
    "agent_id": "assistant",
    "project_id": "finance",
    "type": "preference",
    "content": "العميل يفضل استلام الفواتير شهرياً باللغة العربية.",
    "importance": 0.9,
    "confidence": 1.0,
    "tags": ["فواتير", "تفضيلات"]
  }'
```

**Deduplication Response:**
If an identical or near-identical memory is posted within the same tenant & user scope, HTTP 200 is returned with `"status": "deduplicated"` and the existing `duplicate_of` memory ID, updating the `last_accessed_at` timestamp.

### 3. Search Memories (Keyword, Semantic, Hybrid)
`POST /memories/search`

```bash
curl -X POST http://localhost:8000/memories/search \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "user_id": "user_001",
    "query": "الفواتير باللغة العربية",
    "mode": "hybrid",
    "limit": 5
  }'
```

### 4. Get Memory
`GET /memories/{memory_id}?tenant_id={tenant_id}&user_id={user_id}`
```bash
curl -s "http://localhost:8000/memories/mem_001?tenant_id=demo&user_id=user_001"
```

### 5. Update Memory
`PATCH /memories/{memory_id}?tenant_id={tenant_id}&user_id={user_id}`
```bash
curl -X PATCH "http://localhost:8000/memories/mem_001?tenant_id=demo&user_id=user_001" \
  -H "Content-Type: application/json" \
  -d '{
    "importance": 0.95,
    "content": "The user strongly prefers Arabic responses."
  }'
```

### 6. Delete Memory
`DELETE /memories/{memory_id}?tenant_id={tenant_id}&user_id={user_id}`
```bash
curl -X DELETE "http://localhost:8000/memories/mem_001?tenant_id=demo&user_id=user_001"
```

### 7. List Memories
`GET /memories?tenant_id={tenant_id}&user_id={user_id}&limit=10&offset=0`
```bash
curl -s "http://localhost:8000/memories?tenant_id=demo&user_id=user_001&limit=10"
```

---

## 7. MCP (Model Context Protocol) Server Setup

The MCP server connects agent tools directly to `MemoryService`:

### Tools Provided:
1. `memory_health()`
2. `memory_save(user_id, tenant_id, content, ...)`
3. `memory_search(user_id, tenant_id, query, mode, ...)`
4. `memory_get(memory_id, user_id, tenant_id)`
5. `memory_update(memory_id, user_id, tenant_id, ...)`
6. `memory_delete(memory_id, user_id, tenant_id)`
7. `memory_list(tenant_id, user_id, ...)`

### Client Configuration Example (Claude Desktop / Cursor)

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "ai-memory": {
      "command": "D:\\ai-memory\\memory-api\\.venv\\Scripts\\python.exe",
      "args": ["D:\\ai-memory\\memory-api\\mcp_server.py"],
      "env": {
        "ES_URL": "http://localhost:9200",
        "ES_INDEX": "agent_memory",
        "MCP_TRANSPORT": "stdio"
      }
    }
  }
}
```

---

## 8. Security & Multi-Tenancy

- **Development Mode**: `AUTH_MODE=disabled`. No tokens required, but user and tenant scope parameters are strictly enforced to prevent accidental cross-tenant data leaks.
- **Production Mode**: `AUTH_MODE=token`.
  - Pass header: `Authorization: Bearer <token>`.
  - Configured via `API_TOKENS`:
    `API_TOKENS=tok1:tenantA/user1,tok2:tenantB/*`
  - Rejects cross-tenant requests with `403 Cross-tenant access denied`.
  - Rejects cross-user requests with `403 Cross-user access denied`.
- **MCP scope injection**: set `MEMORY_TENANT_ID` / `MEMORY_USER_ID` to lock the
  MCP server to a fixed identity so agents cannot forge their scope.
- **Elasticsearch auth/TLS**: the local stack disables ES security and binds
  `9200` to `127.0.0.1`. For a secured cluster set `ES_USERNAME`,
  `ES_PASSWORD`, `ES_CA_CERTS`, and `ES_VERIFY_CERTS`.
- **Containers**: the image runs as a non-root user (`appuser`); Compose
  services use `no-new-privileges` and publish ports on loopback only.
- See [../SECURITY.md](../SECURITY.md) for the full policy and reporting process.

> MCP tools advertise `ToolAnnotations` (`readOnlyHint`, `destructiveHint`,
> `idempotentHint`, `openWorldHint`) describing their real behavior —
> `memory_delete` is destructive, `memory_save`/`memory_update` are writes, and
> the rest are read-only.

---

## 9. Automated Testing

Run the full automated test suite:
```powershell
.\.venv\Scripts\pytest -v test_all.py
```

### Tested Suites:
- UTF-8 and BOM strip testing
- Normalization and Arabic tokenization
- Jaccard similarity & trivial content rejection
- Embedding providers (`hash`, `none`)
- Health endpoint validation
- Full CRUD cycle
- Arabic content & query testing
- Exact deduplication
- Vocabulary-overlap conflict detection
- Keyword, semantic, and hybrid search
- Tenant isolation and cross-tenant rejection
- Direct token scope authorization checks
- MCP tool execution suite
- MCP tool annotations (read-only / destructive / idempotent hints) — `test_mcp_annotations.py`

---

## 10. License

MIT — see [../LICENSE](../LICENSE).

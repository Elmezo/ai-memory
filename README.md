# 🧠 AI Memory Server — Long-term Memory for AI Agents via MCP

A model-agnostic, production-ready, multi-tenant memory service designed for AI Coding Agents (such as **OpenCode**, Claude Desktop, Cursor, etc.). Built on **FastAPI**, **Elasticsearch 9.2.0**, and **Ollama (`bge-m3`)** with native support for the **Model Context Protocol (MCP)**.

---

## ✨ Key Features

- **Hybrid Semantic Search**: Combines BM25 keyword matching with multilingual dense vector retrieval (`bge-m3`, 1024 dimensions) supporting Arabic and English.
- **Model Context Protocol (MCP)**: Native `stdio` and HTTP transports enabling AI agents to query and store memories automatically during coding sessions.
- **Multi-Tenant Scope Isolation**: Strict tenant, user, project, and agent scoping with server-side identity injection to prevent scope spoofing.
- **Intelligent Lifecycle & Deduplication**: Normalization, Jaccard/fuzzy deduplication, recency boosts, and conflict detection.
- **One-Click Automation**: Single-click scripts for Windows (`run.bat` and `stop.bat`) handling Docker, Elasticsearch, Ollama, API startup, and health diagnostics.
- **100% Local & Privacy-First**: Operates completely offline without relying on third-party cloud APIs.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────┐
│     AI Agent (OpenCode / Client)       │
└──────────────────┬─────────────────────┘
                   │ MCP (stdio)
                   ▼
┌────────────────────────────────────────┐
│           MCP Server Layer             │
│        (mcp_server.py - 7 tools)       │
└──────────────────┬─────────────────────┘
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│     Memory Service      │   │   Ollama (bge-m3)       │
│  (CRUD, Dedup, Ranking) │   │ (1024-dim Multilingual) │
└───────────┬─────────────┘   └───────────┬─────────────┘
            │                             │
            └──────────────┬──────────────┘
                           ▼
┌────────────────────────────────────────┐
│       Elasticsearch 9.2.0 (Docker)     │
│   Index: agent_memory (HNSW BBQ Index) │
└────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- [Docker Desktop](https://www.docker.com/)
- [Ollama](https://ollama.ai/) with `bge-m3` installed:
  ```bash
  ollama pull bge-m3
  ```
- Python 3.11+

### 2. Start Everything (One-Click)
Double-click `run.bat` (or run in PowerShell):
```powershell
.\run.bat
```
This automatically:
1. Starts Docker Desktop (if not running).
2. Launches Elasticsearch 9.2.0 via Docker Compose.
3. Checks Ollama and verifies `bge-m3`.
4. Starts the Memory REST API (`http://localhost:8000/docs`).
5. Verifies all 7 MCP tools.

### 3. Stop Everything Cleanly
Double-click `stop.bat` (or run in PowerShell):
```powershell
.\stop.bat
```

---

## 🔌 OpenCode Configuration

Add the server to your OpenCode configuration (`~/.config/opencode/opencode.jsonc`):

```jsonc
{
  "mcp": {
    "ai-memory": {
      "type": "local",
      "command": [
        "D:\\ai-memory\\memory-api\\.venv\\Scripts\\python.exe",
        "D:\\ai-memory\\memory-api\\mcp_server.py"
      ],
      "cwd": "D:\\ai-memory\\memory-api",
      "environment": {
        "ES_URL": "http://localhost:9200",
        "ES_INDEX": "agent_memory",
        "EMBEDDING_PROVIDER": "ollama",
        "EMBEDDING_MODEL": "bge-m3",
        "EMBEDDING_DIMS": "1024",
        "OLLAMA_BASE_URL": "http://localhost:11434",
        "MEMORY_TENANT_ID": "opencode_tenant",
        "MEMORY_USER_ID": "opencode_user",
        "MEMORY_AGENT_ID": "opencode",
        "MEMORY_PROJECT_ID": "default",
        "AUTH_MODE": "disabled",
        "MCP_TRANSPORT": "stdio"
      },
      "timeout": 90000
    }
  }
}
```

---

## 🛠️ MCP Tools

| Tool | Description |
|------|-------------|
| `memory_health` | Checks status of Elasticsearch, Ollama, and index statistics |
| `memory_save` | Saves or deduplicates memories with importance & confidence scores |
| `memory_search` | Executes keyword, semantic, or hybrid search with recency boost |
| `memory_get` | Retrieves a specific memory by ID with strict scope enforcement |
| `memory_update` | Updates fields of an existing memory |
| `memory_delete` | Permanently deletes a scoped memory |
| `memory_list` | Lists memories with pagination, status filters, and sorting |

---

## 📄 License
MIT License

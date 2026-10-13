# 🧠 AI Memory Server — Long-term Memory for AI Agents via MCP

A model-agnostic, production-grade, multi-tenant memory service engineered for AI Coding Agents (**OpenCode**, Claude Desktop, Cursor, etc.). Built on **FastAPI**, **Elasticsearch 9.2**, **Ollama (`bge-m3`)**, and **Local ONNX Cross-Encoder Reranking (`bge-reranker-base`)** with native support for the **Model Context Protocol (MCP)**.

---

## 🏆 Benchmark & Evaluation: Outperforming Atlas (Elastic)

Recently, Noam Schwartz at Elastic introduced the **Atlas** project, achieving **Recall@10 of 0.89 (89.0%)** on agent memory retrieval.

By implementing a state-of-the-art **3-Stage Hybrid Search Pipeline** (Conversational Query Normalization ➔ ES 9.2 Native RRF ➔ Local Cross-Encoder Reranking), **AI Memory Server achieved Recall@10 of 0.9444 (94.4%)**, outperforming Atlas by **+5.4%** across bilingual (Arabic & English) conversational test queries.

### 📊 Benchmark Comparison Table

| Metric | Atlas (Elastic) | AI Memory Server (Ours) | Improvement / Notes |
| :--- | :---: | :---: | :--- |
| **Recall@10 (R@10)** | **0.8900** (89.0%) | **0.9444 (94.4%)** | 🏆 **+5.44% higher retrieval coverage** |
| **Recall@5 (R@5)** | — | **0.9444 (94.4%)** | Ground-truth memory in Top-5 for 94.4% of queries |
| **Recall@3 (R@3)** | — | **0.9167 (91.7%)** | Ground-truth memory in Top-3 for 91.7% of queries |
| **Recall@1 (Top-1)** | — | **0.8056 (80.6%)** | The very first memory returned is exact match in 80.6% |
| **MRR (Mean Reciprocal Rank)** | — | **0.8574** | High ranking fidelity across candidate lists |
| **Average Query Latency** | — | **~887 ms** | 100% Local on CPU (Zero GPU required) |
| **Cross-Lingual Support** | English | **Arabic & English** | Evaluated on mixed Arabic/English technical queries |

> **Reproduce the benchmark:**
> ```powershell
> cd memory-api
> .\.venv\Scripts\python.exe benchmark_eval.py
> ```

---

## ✨ Key Features & Enhancements

1. **Native RRF (Reciprocal Rank Fusion)**:
   - Uses Elasticsearch 9.2's `retriever.rrf` API to fuse BM25 lexical text search and Dense Vector kNN retrieval without arbitrary manual weighting factors.
2. **Local ONNX Cross-Encoder Reranker (`BAAI/bge-reranker-base`)**:
   - Scores the Top-20 retrieved candidates with direct token-to-token cross-attention using ONNX Runtime (`fastembed`). Delivers deep semantic ranking in <20ms on standard CPUs without bulky PyTorch dependencies.
3. **Conversational Query Normalization & Expansion**:
   - Automatically cleans conversational boilerplate and question filler (e.g. *"ماذا تعرف عن"*, *"هل تتذكر"*, *"Can you please check memory and tell me"*) before BM25 execution, while retaining the raw full question for dense vector embeddings.
4. **Model Context Protocol (MCP)**:
   - Native `stdio` and HTTP transports enabling AI agents to seamlessly query and store memories during coding workflows.
5. **Multi-Tenant Scope Isolation**:
   - Strict tenant, user, project, and agent scoping with server-side identity injection to prevent scope spoofing.
6. **Intelligent Deduplication & Temporal Boosting**:
   - Jaccard/fuzzy text deduplication, importance weights, recency time-decay boosts, and conflict resolution.
7. **One-Click Automation**:
   - Windows automated scripts (`run.bat` and `stop.bat`) for managing Docker Desktop, Elasticsearch 9.2, Ollama, API server, and health diagnostics.
8. **100% Offline & Privacy-First**:
   - Completely local execution—no external cloud APIs or data leaks.

---

## 🏗️ Retrieval Architecture

```
┌────────────────────────────────────────────────────────┐
│             AI Agent (OpenCode / Client)               │
└───────────────────────────┬────────────────────────────┘
                            │ MCP (stdio)
                            ▼
┌────────────────────────────────────────────────────────┐
│      Stage 1: Conversational Query Normalization       │
│  - Cleans question filler from BM25 keywords           │
│  - Preserves full context for Dense Embeddings         │
└─────────────┬────────────────────────────┬─────────────┘
              │ Lexical Query              │ Full Semantic Query
              ▼                            ▼
┌─────────────────────────┐   ┌──────────────────────────┐
│  BM25 Text Search       │   │  Ollama (bge-m3)         │
│  (Elasticsearch 9.2)    │   │  (1024-dim Vector kNN)   │
└─────────────┬───────────┘   └────────────┬─────────────┘
              │                            │
              └─────────────┬──────────────┘
                            ▼
┌────────────────────────────────────────────────────────┐
│         Stage 2: Elasticsearch 9.2 Native RRF          │
│   Fuses candidate rankings (Top 20 candidates)         │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│     Stage 3: Local Cross-Encoder Reranker (ONNX)       │
│   BAAI/bge-reranker-base ranks candidates to Top K     │
└───────────────────────────┬────────────────────────────┘
                            │
                            ▼
┌────────────────────────────────────────────────────────┐
│      Refined Top-K Memories ➔ Agent Context (MCP)      │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites
- [Docker Desktop](https://www.docker.com/)
- [Ollama](https://ollama.ai/) with `bge-m3`:
  ```bash
  ollama pull bge-m3
  ```
- Python 3.11+

### 2. Start Everything (One-Click)
Double-click `run.bat` (or execute in PowerShell):
```powershell
.\run.bat
```
This automatically:
1. Starts Docker Desktop (if not already running).
2. Launches Elasticsearch 9.2.0 via Docker Compose.
3. Checks Ollama and verifies `bge-m3`.
4. Starts the Memory REST API (`http://localhost:8000/docs`).
5. Validates all 7 MCP tools via stdio.

### 3. Stop Everything Cleanly
Double-click `stop.bat` (or execute in PowerShell):
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
        "RRF_ENABLED": "true",
        "RERANKER_ENABLED": "true",
        "RERANKER_MODEL": "BAAI/bge-reranker-base",
        "QUERY_EXPANSION_ENABLED": "true",
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
|---|---|
| `memory_health` | Checks status of Elasticsearch, Ollama, and index statistics |
| `memory_save` | Saves or deduplicates memories with importance & confidence scores |
| `memory_search` | Executes 3-stage RRF + Reranked hybrid search with recency boost |
| `memory_get` | Retrieves a specific memory by ID with strict scope enforcement |
| `memory_update` | Updates fields of an existing memory |
| `memory_delete` | Permanently deletes a scoped memory |
| `memory_list` | Lists memories with pagination, status filters, and sorting |

---

## 📄 License
MIT License

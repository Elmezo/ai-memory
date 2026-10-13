@echo off
:: Start AI Memory REST API with Ollama semantic embeddings
:: Run this before starting OpenCode

cd /d D:\ai-memory\memory-api

set ES_URL=http://localhost:9200
set EMBEDDING_PROVIDER=ollama
set EMBEDDING_MODEL=bge-m3
set EMBEDDING_DIMS=1024
set OLLAMA_BASE_URL=http://localhost:11434
set AUTH_MODE=disabled
set MCP_TRANSPORT=stdio

echo Starting AI Memory REST API on http://localhost:8000 ...
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000 --log-level info

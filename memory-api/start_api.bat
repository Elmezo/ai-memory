@echo off
:: Start AI Memory REST API with Ollama semantic embeddings
:: Run this before starting OpenCode
::
:: This is a LOCAL DEVELOPMENT profile: it binds to loopback and disables auth.
:: For production, do NOT copy these values. Instead set:
::   AUTH_MODE=token
::   API_TOKENS=<secret>:<tenant>/<user>[,...]   (never commit real tokens)
::   API_HOST=127.0.0.1                            (terminate TLS in a reverse proxy)
::   ES_USERNAME / ES_PASSWORD / ES_CA_CERTS / ES_VERIFY_CERTS   (secured Elasticsearch)
:: and put secrets in the environment or a secret manager, not in this file.

cd /d %~dp0

set ES_URL=http://localhost:9200
set EMBEDDING_PROVIDER=ollama
set EMBEDDING_MODEL=bge-m3
set EMBEDDING_DIMS=1024
set OLLAMA_BASE_URL=http://localhost:11434
:: --- dev-only defaults (see notes above before exposing this service) ---
set AUTH_MODE=disabled
set API_HOST=127.0.0.1
set MCP_TRANSPORT=stdio

echo Starting AI Memory REST API on http://127.0.0.1:8000 ...
.venv\Scripts\python.exe -m uvicorn main:app --host %API_HOST% --port 8000 --log-level info

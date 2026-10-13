"""Central configuration for the AI Memory Server.

All settings come from environment variables with safe development defaults.
Production must override secrets and enable authentication (see .env.example).
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv
    # Load .env from current directory or parent directory
    env_path = find_dotenv(usecwd=True)
    if not env_path:
        env_path = str(Path(__file__).resolve().parent / ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path, override=False)
except ImportError:
    pass


def _getenv(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def _getbool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _getfloat(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _getint(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


class Settings:
    # Elasticsearch
    ES_URL: str = _getenv("ES_URL", "http://localhost:9200")
    ES_INDEX: str = _getenv("ES_INDEX", "agent_memory")
    ES_TIMEOUT: int = _getint("ES_TIMEOUT", 10)
    ES_MAX_RETRIES: int = _getint("ES_MAX_RETRIES", 5)
    ES_RETRY_BACKOFF: float = _getfloat("ES_RETRY_BACKOFF", 1.0)

    # API
    API_HOST: str = _getenv("API_HOST", "0.0.0.0")
    API_PORT: int = _getint("API_PORT", 8000)
    MAX_BODY_BYTES: int = _getint("MAX_BODY_BYTES", 1_000_000)

    # Auth: "disabled" (local dev) or "token" (production-like)
    AUTH_MODE: str = _getenv("AUTH_MODE", "disabled").strip().lower()
    # Format: "TOKEN:tenant/user,TOKEN2:tenant2/*"  ("*" = any user in tenant)
    API_TOKENS: str = _getenv("API_TOKENS", "")

    # Lifecycle / deduplication
    FUZZY_DEDUP_ENABLED: bool = _getbool("FUZZY_DEDUP_ENABLED", True)
    FUZZY_THRESHOLD: float = _getfloat("FUZZY_THRESHOLD", 0.9)
    CONFLICT_THRESHOLD: float = _getfloat("CONFLICT_THRESHOLD", 0.4)
    MIN_CONTENT_CHARS: int = _getint("MIN_CONTENT_CHARS", 3)

    # Search / ranking
    DEFAULT_LIMIT: int = _getint("DEFAULT_LIMIT", 5)
    MAX_LIMIT: int = _getint("MAX_LIMIT", 50)
    BOOST_IMPORTANCE: float = _getfloat("BOOST_IMPORTANCE", 1.0)
    BOOST_CONFIDENCE: float = _getfloat("BOOST_CONFIDENCE", 0.5)
    BOOST_RECENCY: float = _getfloat("BOOST_RECENCY", 0.3)

    # Embeddings / semantic search
    EMBEDDING_PROVIDER: str = _getenv("EMBEDDING_PROVIDER", "ollama").strip().lower()
    EMBEDDING_MODEL: str = _getenv("EMBEDDING_MODEL", "bge-m3")
    EMBEDDING_DIMS: int = _getint("EMBEDDING_DIMS", 1024)
    OLLAMA_BASE_URL: str = _getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    # Hybrid Search Fusion: RRF (Reciprocal Rank Fusion)
    RRF_ENABLED: bool = _getbool("RRF_ENABLED", True)
    RRF_WINDOW_SIZE: int = _getint("RRF_WINDOW_SIZE", 50)
    RRF_CONSTANT: int = _getint("RRF_CONSTANT", 60)

    # Cross-Encoder Reranking
    RERANKER_ENABLED: bool = _getbool("RERANKER_ENABLED", True)
    RERANKER_MODEL: str = _getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
    RERANK_TOP_K: int = _getint("RERANK_TOP_K", 20)

    # Query Normalization & Conversational Cleanup
    QUERY_EXPANSION_ENABLED: bool = _getbool("QUERY_EXPANSION_ENABLED", True)

    # MCP
    MCP_TRANSPORT: str = _getenv("MCP_TRANSPORT", "stdio").strip().lower()
    MCP_PORT: int = _getint("MCP_PORT", 8001)

    # MCP scope injection — set these in the MCP server's environment so the
    # agent cannot forge its own tenant/user identity.
    MEMORY_TENANT_ID: str = _getenv("MEMORY_TENANT_ID", "opencode_tenant")
    MEMORY_USER_ID: str = _getenv("MEMORY_USER_ID", "opencode_user")
    MEMORY_AGENT_ID: str = _getenv("MEMORY_AGENT_ID", "opencode")
    MEMORY_PROJECT_ID: str = _getenv("MEMORY_PROJECT_ID", "default")


settings = Settings()

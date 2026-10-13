"""Embedding provider interface for semantic / hybrid search.

The core service works WITHOUT any embedding model (keyword search).
Semantic search is opt-in via a provider. The default ``hash`` provider is a
local, dependency-free, deterministic encoder intended for development and
tests only — it is NOT a real semantic model. A real model (e.g.
sentence-transformers) can be plugged in without changing the REST API by
implementing :class:`EmbeddingProvider` and selecting it with
``EMBEDDING_PROVIDER``.
"""

from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    name: str = "base"
    dims: int = 0

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a unit-norm vector of length ``dims``."""
        raise NotImplementedError

    def available(self) -> bool:
        return True


class DisabledEmbeddingProvider(EmbeddingProvider):
    """Used when EMBEDDING_PROVIDER=none: semantic search is unavailable."""

    name = "none"

    def __init__(self, dims: int = 0) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        raise RuntimeError("Semantic search is disabled (EMBEDDING_PROVIDER=none)")

    def available(self) -> bool:
        return False


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic hashing encoder (dev/test fallback, NOT semantic).

    Projects hashed token buckets into a unit-norm float vector. Similar
    strings tend to share buckets, giving a weak similarity signal that lets
    the hybrid pipeline be exercised end-to-end without external models or
    paid APIs. Do NOT use for production semantic quality.
    """

    name = "hash"

    def __init__(self, dims: int = 384) -> None:
        self.dims = dims

    def embed(self, text: str) -> list[float]:
        import unicodedata

        norm = unicodedata.normalize("NFKC", text.replace("\ufeff", "")).casefold()
        vec = [0.0] * self.dims
        for token in norm.split():
            h = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            for i in range(3):  # 3 buckets per token reduces collisions
                vec[(h >> (i * 11)) % self.dims] += 1.0 + 0.1 * i
        length = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / length for v in vec]


class SentenceTransformerProvider(EmbeddingProvider):
    """Optional real semantic model (requires ``sentence-transformers``).

    Not a hard dependency. Install it separately and set
    EMBEDDING_PROVIDER=sentence-transformers plus EMBEDDING_MODEL.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str, dims: int) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install it or use EMBEDDING_PROVIDER=hash/none."
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.dims = dims or self._model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True).tolist()
        if len(vec) != self.dims:
            raise RuntimeError(f"Embedding dim mismatch: {len(vec)} != {self.dims}")
        return vec


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Real multilingual semantic embeddings via a local Ollama server.

    Supports Arabic and English out of the box with models like ``bge-m3``
    (1024 dims) or ``nomic-embed-text`` (768 dims).  Requires Ollama running
    at ``OLLAMA_BASE_URL`` (default ``http://localhost:11434``).

    Set ``EMBEDDING_PROVIDER=ollama``, ``EMBEDDING_MODEL=bge-m3``, and
    ``EMBEDDING_DIMS=1024`` in your environment.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "bge-m3",
        dims: int = 1024,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.model = model
        self.dims = dims
        self._base_url = base_url.rstrip("/")

    def available(self) -> bool:
        try:
            import httpx
            r = httpx.get(f"{self._base_url}/api/tags", timeout=5.0)
            return r.status_code == 200
        except Exception:
            return False

    def embed(self, text: str) -> list[float]:
        """Call Ollama /api/embeddings and return a float vector.

        The first call per model may be slow (model load). Subsequent calls
        are fast. Timeout is set to 60 s to accommodate cold starts.
        """
        import unicodedata
        import httpx

        # Normalize input: strip BOM, NFKC-normalize
        text = unicodedata.normalize("NFKC", text.replace("\ufeff", "")).strip()
        if not text:
            raise ValueError("Cannot embed empty text")

        try:
            r = httpx.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=60.0,
            )
            r.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"Ollama embedding request failed: {exc.response.status_code} {exc.response.text}"
            ) from exc
        except httpx.RequestError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self._base_url}: {exc}"
            ) from exc

        data = r.json()
        vec: list[float] = data.get("embedding") or data.get("embeddings", [])
        if not vec:
            raise RuntimeError(f"Ollama returned no embedding. Response: {data}")
        if len(vec) != self.dims:
            raise RuntimeError(
                f"Ollama embedding dim mismatch: got {len(vec)}, expected {self.dims}. "
                f"Check EMBEDDING_DIMS matches the model output."
            )
        return vec


def get_provider(name: str, dims: int, model: str = "", base_url: str = "") -> EmbeddingProvider:
    name = (name or "hash").lower()
    if name == "none":
        return DisabledEmbeddingProvider(dims)
    if name in {"hash", "local", "dev"}:
        return HashEmbeddingProvider(dims)
    if name in {"sentence-transformers", "st", "sbert"}:
        import os

        return SentenceTransformerProvider(
            model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"), dims
        )
    if name in {"ollama"}:
        import os

        return OllamaEmbeddingProvider(
            model=model or os.getenv("EMBEDDING_MODEL", "bge-m3"),
            dims=dims,
            base_url=base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    raise ValueError(
        f"Unknown EMBEDDING_PROVIDER={name!r} (use hash|none|sentence-transformers|ollama)"
    )

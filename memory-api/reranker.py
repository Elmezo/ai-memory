"""Local Cross-Encoder Reranker for AI Memory Server.

Re-ranks top candidate documents retrieved by hybrid search (RRF / BM25 + dense vector)
using a cross-encoder model like ``BAAI/bge-reranker-base`` or ``BAAI/bge-reranker-v2-m3-int8``.
"""

from __future__ import annotations

import logging
from typing import Any

from config import settings

log = logging.getLogger("memory.reranker")

_reranker_instance: Reranker | None = None


class Reranker:
    """Cross-encoder reranker wrapping FastEmbed or falling back gracefully."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.model_name = model_name
        self._encoder: Any = None
        self._available: bool = False
        self._init_model()

    def _init_model(self) -> None:
        if not settings.RERANKER_ENABLED:
            log.info("Reranker is disabled by configuration (RERANKER_ENABLED=false)")
            return

        try:
            from fastembed.rerank.cross_encoder import TextCrossEncoder  # type: ignore

            log.info("Loading cross-encoder reranker model: %s ...", self.model_name)
            self._encoder = TextCrossEncoder(model_name=self.model_name)
            self._available = True
            log.info("Reranker model %s initialized successfully", self.model_name)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not initialize cross-encoder reranker (%s). Running without reranking.", exc)
            self._encoder = None
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available and self._encoder is not None

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Rerank candidates based on cross-encoder relevance scores.

        Each candidate dict must contain a ``content`` field (or ``_source.content``).
        Returns the top_k candidates reordered with an updated ``rerank_score``.
        """
        if not candidates or len(candidates) <= 1 or not self.is_available:
            return candidates[:top_k]

        docs_text: list[str] = []
        for c in candidates:
            text = c.get("content")
            if not text and "_source" in c:
                text = c["_source"].get("content", "")
            docs_text.append(str(text or ""))

        try:
            scores = list(self._encoder.rerank(query, docs_text))
            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)

            # Sort descending by rerank score
            reranked = sorted(candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)
            return reranked[:top_k]
        except Exception as exc:  # noqa: BLE001
            log.warning("Reranking failed (%s); returning original candidate ranking.", exc)
            return candidates[:top_k]


def get_reranker() -> Reranker:
    global _reranker_instance
    if _reranker_instance is None:
        _reranker_instance = Reranker(model_name=settings.RERANKER_MODEL)
    return _reranker_instance

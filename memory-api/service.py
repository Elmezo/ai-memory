"""Memory business logic shared by the REST API and the MCP server."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from config import settings
from embeddings import EmbeddingProvider
from models import MemoryInput, MemorySearch, MemoryUpdate
from reranker import get_reranker
from store import StoreError, map_es_error
from textutil import clean_conversational_query, is_trivial, jaccard, normalize_text

log = logging.getLogger("memory.service")

_TRIVIAL_LOG_LIMIT = 80


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _snippet(content: str, limit: int = 200) -> str:
    return content if len(content) <= limit else content[:limit] + "…"


class MemoryService:
    def __init__(self, es: Any, embedder: EmbeddingProvider | None = None) -> None:
        self.es = es
        self.embedder = embedder
        self.index = settings.ES_INDEX

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    def _scope_filters(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str | None = None,
        project_id: str | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = [
            {"term": {"tenant_id": tenant_id}},
            {"term": {"user_id": user_id}},
        ]
        if agent_id and agent_id != "default":
            filters.append({"term": {"agent_id": agent_id}})
        if project_id and project_id != "default":
            # ``project_id`` is text+keyword in the legacy mapping and keyword
            # in fresh indexes; match either shape.
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"project_id": project_id}},
                            {"term": {"project_id.keyword": project_id}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        return filters

    def _visibility_filters(
        self,
        status: str | None,
        include_inactive: bool,
        include_expired: bool,
        tags: list[str] | None = None,
        mem_type: str | None = None,
    ) -> list[dict[str, Any]]:
        filters: list[dict[str, Any]] = []
        if mem_type:
            filters.append({"term": {"type": mem_type}})
        if status:
            filters.append({"term": {"status": status}})
        elif not include_inactive:
            # Legacy docs have no status field: treat missing as active.
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"term": {"status": "active"}},
                            {"bool": {"must_not": {"exists": {"field": "status"}}}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        if tags:
            filters.append({"terms": {"tags": tags}})
        if not include_expired:
            filters.append(
                {
                    "bool": {
                        "should": [
                            {"bool": {"must_not": {"exists": {"field": "expires_at"}}}},
                            {"range": {"expires_at": {"gt": "now"}}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        return filters

    def _find_scoped(self, memory_id: str, user_id: str, tenant_id: str) -> dict:
        try:
            result = self.es.search(
                index=self.index,
                size=1,
                query={
                    "bool": {
                        "filter": [
                            {"term": {"memory_id": memory_id}},
                            {"term": {"user_id": user_id}},
                            {"term": {"tenant_id": tenant_id}},
                        ]
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise map_es_error(exc, "memory lookup")
        hits = result["hits"]["hits"]
        if not hits:
            raise StoreError(404, "Memory not found")
        return hits[0]

    @staticmethod
    def _to_record(hit: dict[str, Any], explain: bool = False) -> dict[str, Any]:
        record = dict(hit["_source"])
        record["score"] = hit.get("_score")
        if explain and "_explanation" in hit:
            record["explanation"] = hit["_explanation"]
        return record

    def _embed_or_none(self, text: str) -> list[float] | None:
        if self.embedder is None or not self.embedder.available():
            return None
        try:
            return self.embedder.embed(text)
        except Exception:  # noqa: BLE001
            log.warning("Embedding failed; storing memory without vector")
            return None

    # ------------------------------------------------------------------ #
    # lifecycle: dedupe + conflict detection
    # ------------------------------------------------------------------ #
    def _find_duplicates(
        self, data: MemoryInput, norm: str
    ) -> tuple[dict | None, list[dict], list[dict]]:
        """Return (exact_dup, fuzzy_dup_candidates, conflict_candidates).

        Dedupe is scoped to tenant+user and weighted by project/agent/type —
        never a bare global content hash.
        """
        scope = self._scope_filters(data.tenant_id, data.user_id)
        try:
            result = self.es.search(
                index=self.index,
                size=15,
                query={
                    "bool": {
                        "filter": scope,
                        "should": [
                            {"term": {"content_norm": norm}},
                            {"match": {"content": data.content}},
                        ],
                        "minimum_should_match": 0,
                    }
                },
            )
        except Exception as exc:  # noqa: BLE001
            raise map_es_error(exc, "duplicate check")
        hits = result["hits"]["hits"]

        exact: dict | None = None
        fuzzy: list[dict] = []
        conflicts: list[dict] = []
        for hit in hits:
            src = hit["_source"]
            other_norm = src.get("content_norm") or normalize_text(
                src.get("content", "")
            )
            same_type = src.get("type", "fact") == str(data.type.value)
            same_scope_detail = (
                src.get("project_id", "default") == data.project_id
                and src.get("agent_id", "default") == data.agent_id
            )
            if other_norm == norm and same_type and same_scope_detail:
                exact = hit
                break
            sim = jaccard(norm, other_norm)
            entry = {
                "memory_id": src.get("memory_id"),
                "content": _snippet(src.get("content", "")),
                "similarity": round(sim, 3),
            }
            if (
                settings.FUZZY_DEDUP_ENABLED
                and sim >= settings.FUZZY_THRESHOLD
                and same_type
            ):
                fuzzy.append(entry)
            elif sim >= settings.CONFLICT_THRESHOLD and other_norm != norm:
                # Shares vocabulary but says something different: flag it
                # instead of silently overwriting.
                conflicts.append(entry)
        fuzzy_hit = None
        if exact is None and fuzzy:
            best_id = fuzzy[0]["memory_id"]
            for hit in hits:
                if hit["_source"].get("memory_id") == best_id:
                    fuzzy_hit = hit
                    break
        return exact or fuzzy_hit, fuzzy, conflicts[:3]

    def _touch(self, doc_id: str, fields: dict[str, Any]) -> None:
        try:
            self.es.update(
                index=self.index, id=doc_id, doc=fields, refresh="wait_for"
            )
        except Exception as exc:  # noqa: BLE001
            raise map_es_error(exc, "memory touch")

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def create(self, data: MemoryInput) -> dict[str, Any]:
        content = (data.content or "").strip()
        if is_trivial(content, settings.MIN_CONTENT_CHARS):
            raise StoreError(
                422, "Memory content is empty, trivial, or malformed; not saved"
            )
        norm = normalize_text(content)
        exact, fuzzy, conflicts = self._find_duplicates(data, norm)
        if exact is not None:
            src = exact["_source"]
            now_iso = utcnow().isoformat()
            self._touch(
                exact["_id"],
                {"updated_at": now_iso, "last_accessed_at": now_iso},
            )
            src = dict(src)
            src["updated_at"] = now_iso
            src["last_accessed_at"] = now_iso
            return {
                "status": "deduplicated",
                "memory": src,
                "duplicate_of": src.get("memory_id"),
                "fuzzy_matches": fuzzy,
                "possible_conflicts": conflicts,
            }

        timestamp = utcnow()
        timestamp_iso = timestamp.isoformat()
        memory_id = f"mem_{uuid4().hex}"
        document = data.model_dump(mode="json")
        document.update(
            {
                "memory_id": memory_id,
                "type": str(data.type.value),
                "status": str(data.status.value),
                "content": content,
                "content_norm": norm,
                "created_at": timestamp_iso,
                "updated_at": timestamp_iso,
                "last_accessed_at": timestamp_iso,
            }
        )
        vector = self._embed_or_none(content)
        if vector is not None:
            document["embedding"] = vector
        try:
            self.es.index(
                index=self.index, id=memory_id, document=document, refresh="wait_for"
            )
        except Exception as exc:  # noqa: BLE001
            raise map_es_error(exc, "memory indexing")
        log.info(
            "saved memory %s (tenant=%s user=%s len=%d)",
            memory_id,
            data.tenant_id,
            data.user_id,
            len(content),
        )
        return {
            "status": "saved",
            "memory": document,
            "possible_conflicts": conflicts,
        }

    def get(self, memory_id: str, user_id: str, tenant_id: str) -> dict[str, Any]:
        hit = self._find_scoped(memory_id, user_id, tenant_id)
        now_iso = utcnow().isoformat()
        self._touch(hit["_id"], {"last_accessed_at": now_iso})
        record = self._to_record(hit)
        record["last_accessed_at"] = now_iso
        return record

    def update(
        self, memory_id: str, user_id: str, tenant_id: str, patch: MemoryUpdate
    ) -> dict[str, Any]:
        hit = self._find_scoped(memory_id, user_id, tenant_id)
        changes = patch.model_dump(exclude_none=True, mode="json")
        if "type" in changes and changes["type"] is not None:
            changes["type"] = str(patch.type.value)  # type: ignore[union-attr]
        if "status" in changes and changes["status"] is not None:
            changes["status"] = str(patch.status.value)  # type: ignore[union-attr]
        if "content" in changes:
            new_content = (changes["content"] or "").strip()
            if is_trivial(new_content, settings.MIN_CONTENT_CHARS):
                raise StoreError(422, "Updated content is empty or trivial")
            changes["content"] = new_content
            changes["content_norm"] = normalize_text(new_content)
            vector = self._embed_or_none(new_content)
            if vector is not None:
                changes["embedding"] = vector
        if "tags" in changes and isinstance(changes["tags"], list):
            from textutil import strip_bom as _strip

            changes["tags"] = [
                _strip(t) for t in changes["tags"] if isinstance(t, str) and _strip(t)
            ][:32]
        if not changes:
            raise StoreError(400, "No changes provided")
        changes["updated_at"] = utcnow().isoformat()
        try:
            self.es.update(
                index=self.index, id=hit["_id"], doc=changes, refresh="wait_for"
            )
        except Exception as exc:  # noqa: BLE001
            raise map_es_error(exc, "memory update")
        return {"status": "updated", "memory_id": memory_id, "changes": sorted(changes)}

    def delete(self, memory_id: str, user_id: str, tenant_id: str) -> dict[str, Any]:
        hit = self._find_scoped(memory_id, user_id, tenant_id)
        try:
            self.es.delete(index=self.index, id=hit["_id"], refresh="wait_for")
        except Exception as exc:  # noqa: BLE001
            raise map_es_error(exc, "memory deletion")
        return {"status": "deleted", "memory_id": memory_id}

    # ------------------------------------------------------------------ #
    # listing + search
    # ------------------------------------------------------------------ #
    def list_memories(
        self,
        tenant_id: str,
        user_id: str,
        agent_id: str | None = None,
        project_id: str | None = None,
        mem_type: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
        limit: int = 5,
        offset: int = 0,
        include_expired: bool = False,
        include_inactive: bool = False,
        sort: str = "updated_at",
    ) -> dict[str, Any]:
        limit = max(1, min(limit, settings.MAX_LIMIT))
        filters = self._scope_filters(tenant_id, user_id, agent_id, project_id)
        filters += self._visibility_filters(
            status, include_inactive, include_expired, tags, mem_type
        )
        sort_field = sort if sort in {"updated_at", "created_at", "importance"} else "updated_at"
        try:
            result = self.es.search(
                index=self.index,
                size=limit,
                from_=offset,
                query={"bool": {"filter": filters}},
                sort=[{sort_field: {"order": "desc"}}],
            )
        except Exception as exc:  # noqa: BLE001
            raise map_es_error(exc, "memory listing")
        total = result["hits"]["total"]["value"]
        return {
            "count": len(result["hits"]["hits"]),
            "total": total,
            "offset": offset,
            "memories": [self._to_record(h) for h in result["hits"]["hits"]],
        }

    def search(self, params: MemorySearch) -> dict[str, Any]:
        query_text = (params.query or "").strip()
        if not query_text:
            raise StoreError(422, "Search query is empty")
        limit = max(1, min(params.limit, settings.MAX_LIMIT))

        # Query Expansion & Conversational Normalization
        cleaned_query = (
            clean_conversational_query(query_text)
            if settings.QUERY_EXPANSION_ENABLED
            else query_text
        )

        filters = self._scope_filters(
            params.tenant_id, params.user_id, params.agent_id, params.project_id
        )
        filters += self._visibility_filters(
            str(params.status.value) if params.status else None,
            params.include_inactive,
            params.include_expired,
            params.tags,
            str(params.type.value) if params.type else None,
        )

        # Keyword matching: prioritize cleaned keywords, fallback to raw query
        match_clauses: list[dict[str, Any]] = [
            {"match": {"content": {"query": cleaned_query, "boost": 1.5}}}
        ]
        if cleaned_query != query_text:
            match_clauses.append({"match": {"content": {"query": query_text, "boost": 1.0}}})

        keyword_query: dict[str, Any] = {
            "bool": {
                "should": match_clauses,
                "minimum_should_match": 1,
                "filter": filters,
            }
        }

        # Importance / confidence / recency ranking on top of BM25.
        if settings.BOOST_IMPORTANCE or settings.BOOST_CONFIDENCE or settings.BOOST_RECENCY:
            functions: list[dict[str, Any]] = []
            if settings.BOOST_IMPORTANCE:
                functions.append(
                    {
                        "field_value_factor": {
                            "field": "importance",
                            "factor": settings.BOOST_IMPORTANCE,
                            "missing": 0.5,
                        }
                    }
                )
            if settings.BOOST_CONFIDENCE:
                functions.append(
                    {
                        "field_value_factor": {
                            "field": "confidence",
                            "factor": settings.BOOST_CONFIDENCE,
                            "missing": 1.0,
                        }
                    }
                )
            if settings.BOOST_RECENCY:
                functions.append(
                    {
                        "gauss": {
                            "updated_at": {"origin": "now", "scale": "30d", "decay": 0.5}
                        },
                        "weight": settings.BOOST_RECENCY,
                    }
                )
            keyword_query = {
                "function_score": {
                    "query": keyword_query,
                    "functions": functions,
                    "score_mode": "sum",
                    "boost_mode": "sum",
                }
            }

        # Candidate limit for Reranker (fetch more candidates to re-rank top_k)
        candidate_limit = (
            max(limit, settings.RERANK_TOP_K)
            if settings.RERANKER_ENABLED
            else limit
        )

        knn: dict[str, Any] | None = None
        semantic_warning: str | None = None
        vector = None
        if params.mode in {"semantic", "hybrid"}:
            vector = self._embed_or_none(query_text)
            if vector is None:
                semantic_warning = (
                    "Semantic search unavailable "
                    f"(provider={self.embedder.name if self.embedder else 'none'}); "
                    "fell back to keyword search"
                )
            else:
                knn = {
                    "field": "embedding",
                    "query_vector": vector,
                    "k": candidate_limit,
                    "num_candidates": max(candidate_limit * 2, 50),
                    "filter": {"bool": {"filter": filters}},
                }
                if params.mode == "semantic":
                    keyword_query = {"bool": {"filter": filters}}

        # Execute Search with native RRF when in hybrid mode
        result: dict[str, Any]
        used_rrf = False

        if params.mode == "hybrid" and knn is not None and settings.RRF_ENABLED:
            # Modern Elasticsearch 9.2 Retriever API with Reciprocal Rank Fusion
            retriever_body: dict[str, Any] = {
                "index": self.index,
                "size": candidate_limit,
                "retriever": {
                    "rrf": {
                        "retrievers": [
                            {"standard": {"query": keyword_query}},
                            {"knn": knn},
                        ],
                        "rank_window_size": settings.RRF_WINDOW_SIZE,
                        "rank_constant": settings.RRF_CONSTANT,
                    }
                },
            }
            if params.explain:
                retriever_body["explain"] = True
            try:
                result = self.es.search(**retriever_body)
                used_rrf = True
            except Exception as exc:  # noqa: BLE001
                log.info("Native RRF retriever returned (%s); falling back to hybrid query.", exc)
                used_rrf = False

        if not used_rrf:
            search_kwargs: dict[str, Any] = {
                "index": self.index,
                "size": candidate_limit,
                "from_": 0 if settings.RERANKER_ENABLED else params.offset,
                "query": keyword_query,
            }
            if knn is not None:
                search_kwargs["knn"] = knn
            if params.explain:
                search_kwargs["explain"] = True
            try:
                result = self.es.search(**search_kwargs)
            except Exception as exc:  # noqa: BLE001
                # Missing vector mapping on legacy indexes: retry as keyword.
                if knn is not None and "embedding" in str(exc).lower():
                    semantic_warning = (
                        "Vector mapping missing on this index; fell back to keyword. "
                        "See README migration notes"
                    )
                    search_kwargs.pop("knn", None)
                    search_kwargs["query"] = {
                        "bool": {
                            "must": [{"match": {"content": cleaned_query}}],
                            "filter": filters,
                        }
                    }
                    try:
                        result = self.es.search(**search_kwargs)
                    except Exception as exc2:  # noqa: BLE001
                        raise map_es_error(exc2, "memory search")
                else:
                    raise map_es_error(exc, "memory search")

        hits = result["hits"]["hits"]
        total = result["hits"]["total"]["value"] if isinstance(result["hits"].get("total"), dict) else len(hits)
        records = [self._to_record(h, explain=params.explain) for h in hits]

        # Stage 3: Cross-Encoder Reranking
        if settings.RERANKER_ENABLED and records:
            reranker = get_reranker()
            if reranker.is_available:
                records = reranker.rerank(query_text, records, top_k=limit)
            else:
                records = records[params.offset : params.offset + limit]
        else:
            records = records[params.offset : params.offset + limit]

        response: dict[str, Any] = {
            "count": len(records),
            "total": total,
            "offset": params.offset,
            "mode": params.mode,
            "rrf": used_rrf,
            "cleaned_query": cleaned_query if cleaned_query != query_text else None,
            "memories": records,
        }
        if semantic_warning:
            response["warning"] = semantic_warning
        return response


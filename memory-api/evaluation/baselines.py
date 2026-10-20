"""Evaluation baselines."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from config import settings
from models import MemorySearch


@dataclass
class BaselineResult:
    name: str
    retrieved_ids: list[str] = field(default_factory=list)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0
    num_es_queries: int = 0
    num_reranker_calls: int = 0
    candidates_processed: int = 0
    candidate_details: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def _example_to_search_params(example: Any, limit: int) -> MemorySearch:
    return MemorySearch(
        user_id=example.user_id,
        tenant_id=example.tenant_id,
        query=example.query,
        agent_id=example.agent_id if example.agent_id and example.agent_id != "default" else None,
        project_id=example.project_id if example.project_id and example.project_id != "default" else None,
        limit=limit,
        mode="hybrid",
    )


def run_search_baseline(service: Any, example: Any, limit: int = 10) -> BaselineResult:
    start = time.time()
    params = _example_to_search_params(example, limit)
    result = service.search(params)
    latency = (time.time() - start) * 1000.0
    ids = [str(m.get("memory_id")) for m in (result.get("memories") or []) if m.get("memory_id")]
    traj = [
        {
            "step": 0,
            "action_type": "SEARCH",
            "query": example.query,
            "result_count": result.get("count", len(ids)),
        }
    ]
    details = [
        {
            "memory_id": m.get("memory_id"),
            "score": m.get("score"),
            "rerank_score": m.get("rerank_score"),
        }
        for m in (result.get("memories") or [])
    ]
    return BaselineResult(
        name="search",
        retrieved_ids=ids,
        trajectory=traj,
        latency_ms=latency,
        num_es_queries=1,
        num_reranker_calls=1 if settings.RERANKER_ENABLED else 0,
        candidates_processed=len(ids),
        candidate_details=details,
    )


def run_interaction_baseline(
    service: Any,
    example: Any,
    limit: int = 10,
    max_steps: int | None = None,
    add_relevance_threshold: float | None = None,
) -> BaselineResult:
    start = time.time()
    result = service.interact(
        query=example.query,
        tenant_id=example.tenant_id,
        user_id=example.user_id,
        agent_id=example.agent_id,
        project_id=example.project_id,
        limit=limit,
        max_steps=max_steps,
        add_relevance_threshold=add_relevance_threshold,
    )
    latency = (time.time() - start) * 1000.0
    ids = [str(m.get("memory_id")) for m in (result.get("working_memory") or []) if m.get("memory_id")]
    traj = result.get("trajectory") or []
    searches = sum(1 for a in traj if str(a.get("action_type") or a.get("action") or "").upper() == "SEARCH")
    details = [
        {
            "memory_id": c.get("memory_id"),
            "score": c.get("score"),
            "rerank_score": c.get("rerank_score"),
        }
        for c in (result.get("retrieved_candidates") or [])
    ]
    # attach ADD/SKIP decision from trajectory
    decisions = {}
    reasons = {}
    for a in traj:
        t = str(a.get("action_type") or a.get("action") or "").upper()
        if t in {"ADD", "SKIP"} and a.get("memory_id") is not None:
            decisions[str(a.get("memory_id"))] = t
            reasons[str(a.get("memory_id"))] = a.get("reason")
    for d in details:
        d["decision"] = decisions.get(str(d.get("memory_id")), "NOT_RETRIEVED")
        d["reason"] = reasons.get(str(d.get("memory_id")))
    return BaselineResult(
        name="interaction",
        retrieved_ids=ids,
        trajectory=traj,
        latency_ms=latency,
        num_es_queries=searches if searches else 1,
        num_reranker_calls=searches,
        candidates_processed=len(result.get("retrieved_candidates") or []),
        candidate_details=details,
        extra={"rewards": result.get("rewards", [])},
    )


BASELINES: dict[str, Callable[[Any, Any, int], BaselineResult]] = {
    "search": lambda service, example, limit=10: run_search_baseline(service, example, limit),
    "interaction": lambda service, example, limit=10: run_interaction_baseline(service, example, limit),
}


def get_baseline(name: str):
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline {name!r}. Available: {sorted(BASELINES)}")
    return BASELINES[name]

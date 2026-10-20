"""Metrics for memory evaluation."""

from __future__ import annotations

import math
import statistics
from typing import Any, Iterable

from evaluation.dataset import canonical_relevant_set, canonicalize_memory_id


def _safe_div(num: float, den: float) -> float | None:
    if den == 0:
        return None
    return num / den


def retrieval_metrics(
    retrieved: list[str],
    relevant: list[str] | set[str],
    k_values: Iterable[int] = (1, 3, 5, 10),
    equivalence: dict[str, str] | None = None,
) -> dict[str, Any]:
    relevant_set = canonical_relevant_set(relevant, equivalence)
    retrieved = [canonicalize_memory_id(str(x), equivalence) for x in (retrieved or [])]
    result: dict[str, Any] = {}
    ks = list(k_values)

    for k in ks:
        topk = retrieved[:k]
        hits = len(set(topk) & relevant_set)
        if not relevant_set:
            result[f"recall@{k}"] = None
            result[f"precision@{k}"] = None
        else:
            result[f"recall@{k}"] = hits / len(relevant_set)
            result[f"precision@{k}"] = hits / k if k else 0.0
        # NDCG@k
        if not relevant_set:
            result[f"ndcg@{k}"] = None
        else:
            dcg = 0.0
            for i, mid in enumerate(topk, 1):
                rel = 1.0 if mid in relevant_set else 0.0
                dcg += rel / math.log2(i + 1)
            ideal_hits = min(len(relevant_set), k)
            idcg = 0.0
            for i in range(1, ideal_hits + 1):
                idcg += 1.0 / math.log2(i + 1)
            result[f"ndcg@{k}"] = (dcg / idcg) if idcg else 0.0

    # MRR
    if not relevant_set:
        result["mrr"] = None
    else:
        rr = 0.0
        for idx, mid in enumerate(retrieved, 1):
            if mid in relevant_set:
                rr = 1.0 / idx
                break
        result["mrr"] = rr

    result["num_retrieved"] = len(retrieved)
    result["num_relevant"] = len(relevant_set)
    return result


def decision_metrics(
    trajectory: list[dict[str, Any]],
    relevant: list[str] | set[str],
    equivalence: dict[str, str] | None = None,
) -> dict[str, Any]:
    relevant_set = canonical_relevant_set(relevant, equivalence)
    trajectory = trajectory or []

    def _action_type(a: dict[str, Any]) -> str:
        return str(a.get("action_type") or a.get("action") or "").upper()

    searches = [a for a in trajectory if _action_type(a) == "SEARCH"]
    adds = [a for a in trajectory if _action_type(a) == "ADD"]
    skips = [a for a in trajectory if _action_type(a) == "SKIP"]
    stops = [a for a in trajectory if _action_type(a) == "STOP"]

    add_ids = [str(a.get("memory_id")) for a in adds if a.get("memory_id") is not None]
    add_ids_canonical = [canonicalize_memory_id(mid, equivalence) for mid in add_ids]
    # duplicate additions (canonical)
    seen_adds: set[str] = set()
    duplicate_adds = 0
    for mid in add_ids_canonical:
        if mid in seen_adds:
            duplicate_adds += 1
        seen_adds.add(mid)

    add_hits = len(set(add_ids_canonical) & relevant_set)
    if add_ids:
        add_precision = add_hits / len(add_ids)
    else:
        add_precision = 1.0  # vacuous: no false positives
    if relevant_set:
        add_recall = add_hits / len(relevant_set)
    else:
        add_recall = None
    redundant_add_rate = (duplicate_adds / len(add_ids)) if add_ids else 0.0

    skip_correct = sum(
        1 for a in skips
        if canonicalize_memory_id(str(a.get("memory_id")), equivalence) not in relevant_set
    )
    if skips:
        skip_accuracy = skip_correct / len(skips)
    else:
        skip_accuracy = 1.0

    # redundant searches: repeated query strings
    search_queries = [str(a.get("query")) for a in searches if a.get("query") is not None]
    seen_queries: set[str] = set()
    redundant_searches = 0
    for q in search_queries:
        if q in seen_queries:
            redundant_searches += 1
        seen_queries.add(q)
    if searches:
        redundant_search_rate = redundant_searches / len(searches)
    else:
        redundant_search_rate = 0.0

    # useful searches: at least one result and not redundant
    useful_searches = 0
    for i, a in enumerate(searches):
        rc = a.get("result_count")
        try:
            has_results = (rc is not None and int(rc) > 0)
        except (TypeError, ValueError):
            has_results = False
        q = str(a.get("query")) if a.get("query") is not None else None
        is_dup = q is not None and search_queries[:i].count(q) > 0
        if has_results and not is_dup:
            useful_searches += 1
    if searches:
        useful_search_rate = useful_searches / len(searches)
    else:
        useful_search_rate = 0.0

    # per-search added relevant counts (segment trajectory)
    segments: list[int] = []
    current = 0
    for a in trajectory:
        t = _action_type(a)
        if t == "SEARCH":
            if segments:
                segments[-1] = current
            segments.append(0)
            current = 0
        elif t == "ADD" and a.get("memory_id") is not None:
            if canonicalize_memory_id(str(a.get("memory_id")), equivalence) in relevant_set:
                current += 1
    if segments:
        segments[-1] = current
    last_search_added_relevant = segments[-1] if segments else 0

    premature_stop = False
    unnecessary_continuation = False
    if stops:
        if relevant_set and not relevant_set.issubset(set(add_ids_canonical)):
            premature_stop = True
        if len(searches) > 1 and last_search_added_relevant == 0:
            unnecessary_continuation = True

    if stops:
        correct_stop = (not premature_stop) and (not unnecessary_continuation)
    else:
        correct_stop = None

    return {
        "add_count": len(adds),
        "skip_count": len(skips),
        "search_count": len(searches),
        "stop_count": len(stops),
        "add_precision": add_precision,
        "add_recall": add_recall,
        "redundant_add_rate": redundant_add_rate,
        "skip_accuracy": skip_accuracy,
        "useful_search_rate": useful_search_rate,
        "redundant_search_rate": redundant_search_rate,
        "premature_stop": premature_stop,
        "unnecessary_continuation": unnecessary_continuation,
        "correct_stop": correct_stop,
    }


def trajectory_metrics(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory = trajectory or []
    counts: dict[str, int] = {}
    for a in trajectory:
        t = str(a.get("action_type") or a.get("action") or "").upper()
        counts[t] = counts.get(t, 0) + 1

    retrieved_ids: set[str] = set()
    added_ids: set[str] = set()
    search_queries: list[str] = []
    for a in trajectory:
        t = str(a.get("action_type") or a.get("action") or "").upper()
        mid = a.get("memory_id")
        if t in {"SEARCH", "ADD", "SKIP"} and mid is not None:
            # retrieved candidates are not explicitly listed; track added only here
            pass
        if t == "ADD" and mid is not None:
            added_ids.add(str(mid))
        if t == "SEARCH" and a.get("query") is not None:
            search_queries.append(str(a.get("query")))

    duplicate_searches = 0
    seen_q: set[str] = set()
    for q in search_queries:
        if q in seen_q:
            duplicate_searches += 1
        seen_q.add(q)

    add_ids_list = [str(a.get("memory_id")) for a in trajectory
                    if str(a.get("action_type") or a.get("action") or "").upper() == "ADD"
                    and a.get("memory_id") is not None]
    seen_a: set[str] = set()
    duplicate_additions = 0
    for mid in add_ids_list:
        if mid in seen_a:
            duplicate_additions += 1
        seen_a.add(mid)

    premature = False
    unnecessary = False
    # These are filled by decision_metrics; recompute minimal here.
    stops = [a for a in trajectory if str(a.get("action_type") or a.get("action") or "").upper() == "STOP"]
    if stops:
        pass

    return {
        "search_count": counts.get("SEARCH", 0),
        "add_count": counts.get("ADD", 0),
        "skip_count": counts.get("SKIP", 0),
        "stop_count": counts.get("STOP", 0),
        "total_actions": len(trajectory),
        "unique_memories_added": len(added_ids),
        "duplicate_searches": duplicate_searches,
        "duplicate_additions": duplicate_additions,
    }


def aggregate(values: list[float]) -> dict[str, float | None]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"mean": None, "median": None, "p95": None, "stdev": None, "n": 0}
    vals_sorted = sorted(vals)
    p95_idx = max(0, math.ceil(0.95 * len(vals_sorted)) - 1)
    try:
        stdev = statistics.stdev(vals_sorted) if len(vals_sorted) > 1 else 0.0
    except statistics.StatisticsError:
        stdev = 0.0
    return {
        "mean": statistics.fmean(vals_sorted),
        "median": statistics.median(vals_sorted),
        "p95": vals_sorted[p95_idx],
        "stdev": stdev,
        "n": len(vals_sorted),
    }

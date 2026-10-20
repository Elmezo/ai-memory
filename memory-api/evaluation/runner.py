"""CLI runner for memory evaluation."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Allow enabling process rewards before config import.
os.environ.setdefault("MEMORY_ENABLE_PROCESS_REWARD", "true")

from config import settings
from evaluation.baselines import BASELINES, BaselineResult, get_baseline, run_interaction_baseline
from evaluation.dataset import (
    EvalExample,
    canonical_relevant_set,
    canonicalize_memory_id,
    load_dataset,
)
from evaluation.metrics import (
    aggregate,
    decision_metrics,
    retrieval_metrics,
    trajectory_metrics,
)
from models import MemoryInput
from service import MemoryService
from store import create_client, ensure_index, wait_for_es


def seed_memories(es: Any, path: str | Path) -> None:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Memories file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            mid = doc.get("memory_id")
            if not mid:
                raise ValueError("Each seeded memory must include memory_id")
            # Remove memory_id from document body? Keep it; it is part of schema.
            es.index(index=settings.ES_INDEX, id=mid, document=doc, refresh="wait_for")


def _isolation_violations(example: EvalExample, result: BaselineResult, service: Any) -> int:
    # Best-effort: if service can fetch memory, check scope. Here we rely on
    # retrieved ids only; violations counted when a retrieved memory's scope
    # differs. Without fetching, we count 0 and note limitation in docs.
    return 0


def evaluate_example(
    service: Any,
    example: EvalExample,
    baseline_name: str,
    k_values: list[int],
    limit: int,
    add_relevance_threshold: float | None = None,
) -> dict[str, Any]:
    if baseline_name == "interaction":
        baseline_fn = lambda svc, ex, lim: run_interaction_baseline(
            svc, ex, lim, add_relevance_threshold=add_relevance_threshold
        )
    else:
        baseline_fn = get_baseline(baseline_name)
    result: BaselineResult = baseline_fn(service, example, limit)

    rmetrics = retrieval_metrics(
        result.retrieved_ids, example.relevant_memory_ids, k_values, example.equivalent_memory_ids
    )
    dmetrics = decision_metrics(
        result.trajectory, example.relevant_memory_ids, example.equivalent_memory_ids
    )
    tmetrics = trajectory_metrics(result.trajectory)

    relevant_canonical = canonical_relevant_set(
        example.relevant_memory_ids, example.equivalent_memory_ids
    )
    candidate_details = []
    for d in (result.candidate_details or []):
        mid = d.get("memory_id")
        canon = canonicalize_memory_id(str(mid), example.equivalent_memory_ids) if mid is not None else None
        item = dict(d)
        item["relevant"] = bool(canon in relevant_canonical) if canon is not None else False
        candidate_details.append(item)

    # Reward breakdown (if present)
    rewards = result.extra.get("rewards", []) if result.extra else []
    total_reward = sum(float(r.get("overall_score", 0.0) or 0.0) for r in rewards) if rewards else None
    avg_reward = (total_reward / len(rewards)) if (rewards and total_reward is not None) else None

    return {
        "query_id": example.id,
        "query": example.query,
        "baseline": baseline_name,
        "retrieved_ids": result.retrieved_ids,
        "relevant_ids": example.relevant_memory_ids,
        "retrieval": rmetrics,
        "decisions": dmetrics,
        "trajectory": tmetrics,
        "trajectory_raw": result.trajectory,
        "candidate_details": candidate_details,
        "performance": {
            "latency_ms": result.latency_ms,
            "num_es_queries": result.num_es_queries,
            "num_reranker_calls": result.num_reranker_calls,
            "candidates_processed": result.candidates_processed,
        },
        "rewards": {
            "per_action": rewards,
            "total": total_reward,
            "average": avg_reward,
            "count": len(rewards),
        },
    }


def run_evaluation(
    dataset_path: str,
    baseline_name: str,
    output_dir: str,
    k_values: list[int],
    limit: int,
    memories_path: str | None = None,
    reset_index: bool = False,
    add_relevance_threshold: float | None = None,
    calibration_fraction: float | None = None,
    run_split: str = "all",
) -> dict[str, Any]:
    es = create_client()
    wait_for_es(es)
    if reset_index:
        try:
            es.indices.delete(index=settings.ES_INDEX, ignore_unavailable=True)
        except Exception:
            pass
    ensure_index(es)
    if memories_path:
        seed_memories(es, memories_path)

    embedder = None
    try:
        from embeddings import get_provider

        embedder = get_provider(
            settings.EMBEDDING_PROVIDER,
            settings.EMBEDDING_DIMS,
            model=settings.EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    except Exception:
        embedder = None
    service = MemoryService(es, embedder)

    examples = load_dataset(dataset_path)
    if calibration_fraction is not None and 0.0 < calibration_fraction < 1.0:
        ordered = sorted(examples, key=lambda e: e.id)
        cut = int(len(ordered) * calibration_fraction)
        calibration = ordered[:cut]
        test = ordered[cut:]
        if run_split == "calibration":
            examples = calibration
        elif run_split == "test":
            examples = test
        # all: keep all

    per_query: list[dict[str, Any]] = []
    trajectories: list[dict[str, Any]] = []
    for ex in examples:
        qresult = evaluate_example(
            service, ex, baseline_name, k_values, limit, add_relevance_threshold
        )
        per_query.append(qresult)
        trajectories.append(
            {
                "query_id": ex.id,
                "actions": qresult["trajectory_raw"],
                "metrics": {
                    "retrieval": qresult["retrieval"],
                    "decisions": qresult["decisions"],
                    "trajectory": qresult["trajectory"],
                },
                "latency_ms": qresult["performance"]["latency_ms"],
            }
        )

    # Aggregate retrieval metrics
    retrieval_agg: dict[str, Any] = {}
    for k in k_values:
        vals = [q["retrieval"].get(f"recall@{k}") for q in per_query]
        retrieval_agg[f"recall@{k}"] = aggregate(vals)
        vals = [q["retrieval"].get(f"precision@{k}") for q in per_query]
        retrieval_agg[f"precision@{k}"] = aggregate(vals)
        vals = [q["retrieval"].get(f"ndcg@{k}") for q in per_query]
        retrieval_agg[f"ndcg@{k}"] = aggregate(vals)
    retrieval_agg["mrr"] = aggregate([q["retrieval"].get("mrr") for q in per_query])

    decision_agg: dict[str, Any] = {}
    for key in [
        "add_precision",
        "add_recall",
        "redundant_add_rate",
        "skip_accuracy",
        "useful_search_rate",
        "redundant_search_rate",
    ]:
        decision_agg[key] = aggregate([q["decisions"].get(key) for q in per_query])
    decision_agg["premature_stop_rate"] = aggregate(
        [1.0 if q["decisions"].get("premature_stop") else 0.0 for q in per_query]
    )
    decision_agg["unnecessary_continuation_rate"] = aggregate(
        [1.0 if q["decisions"].get("unnecessary_continuation") else 0.0 for q in per_query]
    )
    decision_agg["correct_stop_rate"] = aggregate(
        [1.0 if q["decisions"].get("correct_stop") else 0.0 for q in per_query
         if q["decisions"].get("correct_stop") is not None]
    )

    traj_agg: dict[str, Any] = {}
    for key in ["search_count", "add_count", "skip_count", "total_actions", "duplicate_searches", "duplicate_additions"]:
        traj_agg[key] = aggregate([q["trajectory"].get(key) for q in per_query])

    perf_latencies = [q["performance"]["latency_ms"] for q in per_query]
    perf_agg = {
        "mean_latency_ms": aggregate(perf_latencies)["mean"],
        "median_latency_ms": aggregate(perf_latencies)["median"],
        "p95_latency_ms": aggregate(perf_latencies)["p95"],
        "mean_es_queries": aggregate([q["performance"]["num_es_queries"] for q in per_query])["mean"],
        "mean_reranker_calls": aggregate([q["performance"]["num_reranker_calls"] for q in per_query])["mean"],
        "mean_candidates_processed": aggregate([q["performance"]["candidates_processed"] for q in per_query])["mean"],
    }

    # Score distribution for ADD gating analysis
    score_buckets: dict[str, dict[str, list[float]]] = {
        "relevant": {"scores": [], "rerank_scores": []},
        "irrelevant": {"scores": [], "rerank_scores": []},
    }
    for q in per_query:
        for d in q.get("candidate_details", []) or []:
            bucket = "relevant" if d.get("relevant") else "irrelevant"
            if d.get("score") is not None:
                score_buckets[bucket]["scores"].append(float(d["score"]))
            if d.get("rerank_score") is not None:
                score_buckets[bucket]["rerank_scores"].append(float(d["rerank_score"]))
    score_distribution = {}
    for bucket, vals in score_buckets.items():
        score_distribution[bucket] = {
            "score": aggregate(vals["scores"]),
            "rerank_score": aggregate(vals["rerank_scores"]),
        }

    reward_totals = [q["rewards"]["total"] for q in per_query if q["rewards"]["total"] is not None]
    reward_agg = {
        "mean_total_reward": aggregate(reward_totals)["mean"],
        "mean_per_action_reward": aggregate(
            [q["rewards"]["average"] for q in per_query if q["rewards"]["average"] is not None]
        )["mean"],
        "queries_with_rewards": sum(1 for q in per_query if q["rewards"]["count"] > 0),
    }

    run_report = {
        "run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "baseline": baseline_name,
            "dataset": str(dataset_path),
            "memories": memories_path,
            "num_queries": len(examples),
            "k_values": k_values,
            "limit": limit,
            "add_relevance_threshold": add_relevance_threshold,
        },
        "retrieval": {
            f"recall@{k}": retrieval_agg[f"recall@{k}"]["mean"] for k in k_values
        } | {
            f"precision@{k}": retrieval_agg[f"precision@{k}"]["mean"] for k in k_values
        } | {
            f"ndcg@{k}": retrieval_agg[f"ndcg@{k}"]["mean"] for k in k_values
        } | {"mrr": retrieval_agg["mrr"]["mean"]},
        "retrieval_aggregate": retrieval_agg,
        "decisions": {
            "add_precision": decision_agg["add_precision"]["mean"],
            "add_recall": decision_agg["add_recall"]["mean"],
            "redundant_add_rate": decision_agg["redundant_add_rate"]["mean"],
            "skip_accuracy": decision_agg["skip_accuracy"]["mean"],
            "useful_search_rate": decision_agg["useful_search_rate"]["mean"],
            "redundant_search_rate": decision_agg["redundant_search_rate"]["mean"],
            "premature_stop_rate": decision_agg["premature_stop_rate"]["mean"],
            "unnecessary_continuation_rate": decision_agg["unnecessary_continuation_rate"]["mean"],
            "correct_stop_rate": decision_agg["correct_stop_rate"]["mean"],
        },
        "decision_aggregate": decision_agg,
        "trajectory": {
            "avg_searches": traj_agg["search_count"]["mean"],
            "avg_adds": traj_agg["add_count"]["mean"],
            "avg_skips": traj_agg["skip_count"]["mean"],
            "avg_total_actions": traj_agg["total_actions"]["mean"],
            "avg_duplicate_searches": traj_agg["duplicate_searches"]["mean"],
            "avg_duplicate_additions": traj_agg["duplicate_additions"]["mean"],
        },
        "trajectory_aggregate": traj_agg,
        "performance": perf_agg,
        "rewards": reward_agg,
        "score_distribution": score_distribution,
        "per_query": per_query,
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_path = out_dir / "run.json"
    with run_path.open("w", encoding="utf-8") as f:
        json.dump(run_report, f, indent=2, ensure_ascii=False)

    traj_path = out_dir / "trajectories.jsonl"
    with traj_path.open("w", encoding="utf-8") as f:
        for t in trajectories:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    return run_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run memory evaluation")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL")
    parser.add_argument("--output", default="evaluation/results", help="Output directory")
    parser.add_argument("--baseline", default="interaction", choices=sorted(BASELINES.keys()))
    parser.add_argument("--k", default="1,3,5,10", help="Comma-separated K values")
    parser.add_argument("--limit", type=int, default=10, help="Retrieval limit per query")
    parser.add_argument("--memories", default=None, help="Path to memories JSONL for seeding")
    parser.add_argument("--reset-index", action="store_true", help="Delete and recreate index before seeding")
    parser.add_argument("--threshold", type=float, default=None, help="Single ADD relevance threshold")
    parser.add_argument("--thresholds", default=None, help="Comma-separated ADD relevance thresholds")
    parser.add_argument("--calibration-fraction", type=float, default=None, help="Fraction of queries used for calibration")
    parser.add_argument("--run-split", choices=["all", "calibration", "test"], default="all", help="Which split to evaluate")
    args = parser.parse_args(argv)

    k_values = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    if args.thresholds:
        thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    elif args.threshold is not None:
        thresholds = [args.threshold]
    else:
        thresholds = [None]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(thresholds) == 1:
        report = run_evaluation(
            dataset_path=args.dataset,
            baseline_name=args.baseline,
            output_dir=args.output,
            k_values=k_values,
            limit=args.limit,
            memories_path=args.memories,
            reset_index=args.reset_index,
            add_relevance_threshold=thresholds[0],
            calibration_fraction=args.calibration_fraction,
            run_split=args.run_split,
        )
        print(json.dumps(report["run"], indent=2))
        print(json.dumps(report["retrieval"], indent=2))
        print(json.dumps(report["decisions"], indent=2))
        return 0

    # Multi-threshold sweep
    comparison = {"baselines": [], "metrics": {}}
    reports = {}
    for t in thresholds:
        subdir = out_dir / f"threshold_{t:g}"
        report = run_evaluation(
            dataset_path=args.dataset,
            baseline_name=args.baseline,
            output_dir=str(subdir),
            k_values=k_values,
            limit=args.limit,
            memories_path=args.memories,
            reset_index=(args.reset_index and t == thresholds[0]),
            add_relevance_threshold=t,
            calibration_fraction=args.calibration_fraction,
            run_split=args.run_split,
        )
        reports[t] = report
        comparison["baselines"].append(f"t={t:g}")

    # Build comparison summary
    summary = {}
    for t in thresholds:
        r = reports[t]
        summary[t] = {
            "recall@5": r["retrieval"].get("recall@5"),
            "mrr": r["retrieval"].get("mrr"),
            "add_precision": r["decisions"].get("add_precision"),
            "add_recall": r["decisions"].get("add_recall"),
            "correct_stop_rate": r["decisions"].get("correct_stop_rate"),
            "redundant_add_rate": r["decisions"].get("redundant_add_rate"),
        }
    with (out_dir / "threshold_sweep.json").open("w", encoding="utf-8") as f:
        json.dump({"thresholds": thresholds, "results": summary}, f, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
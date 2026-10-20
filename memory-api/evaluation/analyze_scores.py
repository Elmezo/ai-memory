"""Score distribution and ADD error analysis for evaluation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluation.dataset import canonical_relevant_set, canonicalize_memory_id, load_dataset


def analyze_run(run_path: str, dataset_path: str | None = None) -> dict[str, Any]:
    run = json.loads(Path(run_path).read_text(encoding="utf-8"))
    eqmap: dict[str, dict[str, str]] = {}
    if dataset_path:
        for ex in load_dataset(dataset_path):
            eqmap[ex.id] = ex.equivalent_memory_ids or {}

    false_positives: list[dict[str, Any]] = []
    false_negatives: list[dict[str, Any]] = []
    for q in run.get("per_query", []):
        eq = eqmap.get(q["query_id"], {})
        rel_canonical = canonical_relevant_set(q.get("relevant_ids", []), eq)
        for c in q.get("candidate_details", []) or []:
            mid = c.get("memory_id")
            canon = canonicalize_memory_id(str(mid), eq) if mid is not None else None
            relevant = bool(canon in rel_canonical) if canon is not None else False
            decision = c.get("decision")
            redundant = "redundant" in str(c.get("reason") or "")
            if decision == "ADD" and not relevant:
                false_positives.append(
                    {
                        "query_id": q["query_id"],
                        "query": q["query"],
                        "memory_id": mid,
                        "score": c.get("score"),
                        "rerank_score": c.get("rerank_score"),
                        "reason": c.get("reason"),
                    }
                )
            if relevant and decision != "ADD" and not redundant:
                false_negatives.append(
                    {
                        "query_id": q["query_id"],
                        "query": q["query"],
                        "memory_id": mid,
                        "score": c.get("score"),
                        "rerank_score": c.get("rerank_score"),
                        "decision": decision,
                        "reason": c.get("reason"),
                    }
                )

    # Scale comparison
    has_rerank = any(
        c.get("rerank_score") is not None
        for q in run.get("per_query", [])
        for c in (q.get("candidate_details") or [])
    )
    scale_info = {
        "rerank_scores_present": has_rerank,
        "note": (
            "Reranker scores and Elasticsearch scores use different scales; "
            "thresholds are only meaningful for the score actually thresholded."
            if has_rerank
            else "No reranker scores present; thresholds apply to Elasticsearch scores."
        ),
    }

    return {
        "run": run.get("run", {}),
        "score_distribution": run.get("score_distribution", {}),
        "scale_info": scale_info,
        "false_positive_adds": false_positives,
        "false_negative_adds": false_negatives,
        "counts": {
            "false_positive_adds": len(false_positives),
            "false_negative_adds": len(false_negatives),
        },
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Analyze evaluation run scores")
    parser.add_argument("--run", required=True, help="Path to run.json")
    parser.add_argument("--dataset", default=None, help="Path to dataset JSONL")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    args = parser.parse_args(argv)

    result = analyze_run(args.run, args.dataset)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY
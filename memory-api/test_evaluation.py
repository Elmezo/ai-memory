"""Tests for evaluation harness."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from evaluation.dataset import EvalExample, load_dataset
from evaluation.dataset import canonicalize_memory_id
from evaluation.metrics import aggregate, decision_metrics, retrieval_metrics, trajectory_metrics


def test_retrieval_metrics_basic():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "d"]
    m = retrieval_metrics(retrieved, relevant, k_values=[1, 2, 4])
    assert m["recall@1"] == 0.0
    assert m["recall@2"] == 0.5
    assert m["recall@4"] == 1.0
    assert m["precision@2"] == 0.5
    assert m["mrr"] == 0.5
    assert m["ndcg@4"] is not None


def test_retrieval_metrics_empty_ground_truth():
    m = retrieval_metrics(["a", "b"], [], k_values=[3])
    assert m["recall@3"] is None


def test_canonical_equivalence_recall():
    m = retrieval_metrics(["mem_006"], ["mem_001"], k_values=[5], equivalence={"mem_006": "mem_001"})
    assert m["recall@5"] == 1.0
    assert m["mrr"] == 1.0


def test_canonical_decision_add():
    traj = [
        {"action_type": "SEARCH", "query": "q", "result_count": 1},
        {"action_type": "ADD", "memory_id": "mem_006"},
        {"action_type": "STOP"},
    ]
    dm = decision_metrics(traj, ["mem_001"], equivalence={"mem_006": "mem_001"})
    assert dm["add_precision"] == 1.0
    assert dm["add_recall"] == 1.0


def test_canonical_cycle_validation():
    try:
        canonicalize_memory_id("a", {"a": "b", "b": "a"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_decision_metrics_add_and_stop():
    traj = [
        {"action_type": "SEARCH", "query": "q", "result_count": 2},
        {"action_type": "ADD", "memory_id": "a"},
        {"action_type": "SKIP", "memory_id": "b"},
        {"action_type": "STOP"},
    ]
    dm = decision_metrics(traj, {"a"})
    assert dm["add_precision"] == 1.0
    assert dm["add_recall"] == 1.0
    assert dm["skip_accuracy"] == 1.0
    assert dm["correct_stop"] is True
    assert dm["premature_stop"] is False


def test_decision_metrics_premature_stop():
    traj = [
        {"action_type": "SEARCH", "query": "q", "result_count": 2},
        {"action_type": "ADD", "memory_id": "a"},
        {"action_type": "STOP"},
    ]
    dm = decision_metrics(traj, {"a", "b"})
    assert dm["premature_stop"] is True
    assert dm["correct_stop"] is False


def test_decision_metrics_redundant_search():
    traj = [
        {"action_type": "SEARCH", "query": "q", "result_count": 2},
        {"action_type": "SEARCH", "query": "q", "result_count": 1},
        {"action_type": "STOP"},
    ]
    dm = decision_metrics(traj, set())
    assert dm["redundant_search_rate"] == 0.5


def test_trajectory_metrics_counts():
    traj = [
        {"action_type": "SEARCH", "query": "q"},
        {"action_type": "ADD", "memory_id": "a"},
        {"action_type": "ADD", "memory_id": "a"},
        {"action_type": "STOP"},
    ]
    tm = trajectory_metrics(traj)
    assert tm["search_count"] == 1
    assert tm["add_count"] == 2
    assert tm["duplicate_additions"] == 1
    assert tm["total_actions"] == 4


def test_aggregate():
    agg = aggregate([1.0, 2.0, 3.0, 4.0])
    assert agg["mean"] == 2.5
    assert agg["n"] == 4
    assert agg["p95"] is not None


def test_dataset_load_and_validate():
    data = [
        {"id": "q1", "query": "x", "tenant_id": "t", "user_id": "u", "relevant_memory_ids": ["m1"]},
    ]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "d.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in data), encoding="utf-8")
        exs = load_dataset(path)
        assert len(exs) == 1
        assert exs[0].id == "q1"


def test_eval_example_validation():
    try:
        EvalExample(id="", query="q", tenant_id="t", user_id="u")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
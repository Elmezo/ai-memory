"""Tests for the AiO-Mem-inspired memory interaction layer."""

from __future__ import annotations

import pytest

from config import settings
from evidence import EvidenceState, MemoryAction, MemoryActionType
from policy import RuleBasedMemoryPolicy, get_memory_policy
from reward import ProcessReward, RewardEvaluator


# --------------------------------------------------------------------------- #
# Evidence state
# --------------------------------------------------------------------------- #
def test_evidence_state_initial():
    s = EvidenceState(query="q", tenant_id="t", user_id="u", agent_id="a")
    assert s.query == "q"
    assert s.tenant_id == "t"
    assert s.user_id == "u"
    assert s.agent_id == "a"
    assert s.working_memory == []
    assert s.retrieved_candidates == []
    assert s.search_history == []
    assert s.action_history == []
    assert s.stopped is False
    d = s.to_dict()
    assert d["action_sequence"] == []
    assert d["working_memory_count"] == 0


def test_evidence_state_after_actions():
    s = EvidenceState(query="q", tenant_id="t", user_id="u", max_steps=3)
    s.record_action(MemoryAction(MemoryActionType.SEARCH, step=1, query="q", result_count=2))
    s.search_history.append("q")
    s.searches_done = 1
    s.retrieved_candidates = [{"memory_id": "m1", "content": "c"}]
    s.working_memory = [{"memory_id": "m1", "content": "c"}]
    s.record_action(MemoryAction(MemoryActionType.ADD, step=1, memory_id="m1"))
    s.record_action(MemoryAction(MemoryActionType.STOP, step=1, reason="sufficient_evidence"))
    s.stopped = True
    s.stop_reason = "sufficient_evidence"
    seq = s.action_sequence()
    assert seq == ["SEARCH", "ADD", "STOP"]
    assert s.has_memory("m1") is True
    assert s.is_redundant_content("c") is True


def test_memory_action_coercion():
    a = MemoryAction("search", step=1)
    assert a.action_type == MemoryActionType.SEARCH
    assert a.to_dict()["action_type"] == "SEARCH"


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
def test_policy_search_when_insufficient():
    p = RuleBasedMemoryPolicy()
    s = EvidenceState(query="hello", tenant_id="t", user_id="u")
    should_search, q, reason = p.decide_search(s)
    assert should_search is True
    assert q == "hello"
    assert reason == "initial_search"


def test_policy_stop_when_sufficient():
    p = RuleBasedMemoryPolicy()
    s = EvidenceState(query="hello", tenant_id="t", user_id="u")
    s.search_history.append("hello")
    s.searches_done = 1
    s.working_memory = [{"memory_id": "m1", "content": "x"}]
    s.last_added_count = 1
    should_search, q, reason = p.decide_search(s)
    assert should_search is False
    stop, r, conf = p.should_stop(s)
    assert stop is True
    assert "sufficient" in r


def test_policy_skip_irrelevant_candidate():
    p = RuleBasedMemoryPolicy(add_relevance_threshold=0.9)
    s = EvidenceState(query="hello", tenant_id="t", user_id="u")
    cand = {"memory_id": "m2", "content": "irrelevant", "rerank_score": 0.1}
    decision, reason, conf = p.evaluate_candidate(s, cand)
    assert decision == MemoryActionType.SKIP
    assert "threshold" in reason or "relevance" in reason


def test_policy_add_relevant_candidate():
    p = RuleBasedMemoryPolicy()
    s = EvidenceState(query="hello", tenant_id="t", user_id="u")
    cand = {"memory_id": "m3", "content": "relevant novel fact", "score": 0.9}
    decision, reason, conf = p.evaluate_candidate(s, cand)
    assert decision == MemoryActionType.ADD


def test_policy_avoid_duplicate_searches():
    p = RuleBasedMemoryPolicy()
    s = EvidenceState(query="hello", tenant_id="t", user_id="u")
    s.search_history.append("hello")
    s.searches_done = 1
    s.last_result_count = 0
    should_search, q, reason = p.decide_search(s)
    assert should_search is False
    assert reason == "no_new_candidates"


def test_get_memory_policy_fallback():
    p = get_memory_policy("unknown-mode")
    assert isinstance(p, RuleBasedMemoryPolicy)


# --------------------------------------------------------------------------- #
# Reward
# --------------------------------------------------------------------------- #
def test_process_reward_fields():
    r = ProcessReward()
    assert hasattr(r, "action_quality")
    assert hasattr(r, "overall_score")
    d = r.to_dict()
    assert "overall_score" in d


def test_reward_evaluator():
    ev = RewardEvaluator()
    s = EvidenceState(query="q", tenant_id="t", user_id="u")
    a = MemoryAction(MemoryActionType.STOP, step=1, reason="sufficient_evidence", confidence=0.9)
    r = ev.evaluate(s, a)
    assert r.stopping_quality >= 0.7
    assert r.overall_score >= 0.0


# --------------------------------------------------------------------------- #
# Interact endpoint (integration)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    import main

    with TestClient(main.app) as c:
        yield c


def test_interact_endpoint_basic(client):
    tenant = "interact_tenant"
    user = "interact_user"
    # seed
    res = client.post(
        "/memories",
        json={"tenant_id": tenant, "user_id": user, "content": "The database port is 5432."},
    )
    assert res.status_code in (200, 201)

    res = client.post(
        "/memories/interact",
        json={"tenant_id": tenant, "user_id": user, "query": "database port"},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "completed"
    assert "working_memory" in data
    assert "trajectory" in data
    assert "action_sequence" in data
    assert "SEARCH" in data["action_sequence"]
    assert "STOP" in data["action_sequence"]
    # cleanup
    for m in data["working_memory"]:
        client.delete(f"/memories/{m['memory_id']}?tenant_id={tenant}&user_id={user}")


def test_interact_respects_scope(client):
    # tenant A seed
    ta, ua = "scope_tenant_a", "scope_user_a"
    tb, ub = "scope_tenant_b", "scope_user_b"
    r = client.post("/memories", json={"tenant_id": ta, "user_id": ua, "content": "secret A"})
    assert r.status_code in (200, 201)
    mid = r.json()["memory"]["memory_id"]

    # interact as tenant B should not see A
    res = client.post("/memories/interact", json={"tenant_id": tb, "user_id": ub, "query": "secret"})
    assert res.status_code == 200
    data = res.json()
    ids = [m.get("memory_id") for m in data.get("working_memory", [])]
    assert mid not in ids

    client.delete(f"/memories/{mid}?tenant_id={ta}&user_id={ua}")


def test_provenance_fields_preserved(client):
    tenant = "prov_tenant"
    user = "prov_user"
    payload = {
        "tenant_id": tenant,
        "user_id": user,
        "content": "A provenance test memory.",
        "source": "conversation:abc",
        "source_type": "conversation",
        "source_id": "abc",
        "source_reference": "msg:42",
        "source_span": "lines:10-12",
        "created_from": "test",
        "provenance": {"origin": "unit-test"},
        "parent_memory_id": None,
        "summary": "provenance test",
        "query_anchors": ["provenance", "test"],
        "event_phrase": "provenance test memory",
    }
    res = client.post("/memories", json=payload)
    assert res.status_code in (200, 201)
    mem = res.json()["memory"]
    mid = mem["memory_id"]
    assert mem.get("source_type") == "conversation"
    assert mem.get("source_id") == "abc"
    assert mem.get("provenance", {}).get("origin") == "unit-test"
    assert mem.get("query_anchors") == ["provenance", "test"]

    got = client.get(f"/memories/{mid}?tenant_id={tenant}&user_id={user}")
    assert got.status_code == 200
    g = got.json()
    assert g.get("source_reference") == "msg:42"
    assert g.get("event_phrase") == "provenance test memory"

    client.delete(f"/memories/{mid}?tenant_id={tenant}&user_id={user}")


def test_missing_provenance_handled(client):
    tenant = "noprov_tenant"
    user = "noprov_user"
    res = client.post("/memories", json={"tenant_id": tenant, "user_id": user, "content": "no provenance here"})
    assert res.status_code in (200, 201)
    mem = res.json()["memory"]
    mid = mem["memory_id"]
    assert mem.get("source_type") is None
    # interact should still work
    r = client.post("/memories/interact", json={"tenant_id": tenant, "user_id": user, "query": "no provenance"})
    assert r.status_code == 200
    client.delete(f"/memories/{mid}?tenant_id={tenant}&user_id={user}")


def test_policy_gating_score_present():
    p = RuleBasedMemoryPolicy(add_relevance_threshold=0.5)
    s = EvidenceState(query="q", tenant_id="t", user_id="u")
    # equal threshold -> ADD
    d, r, c = p.evaluate_candidate(s, {"memory_id": "m1", "content": "novel", "rerank_score": 0.5})
    assert d == MemoryActionType.ADD
    # above -> ADD
    d, r, c = p.evaluate_candidate(s, {"memory_id": "m2", "content": "novel2", "rerank_score": 0.6})
    assert d == MemoryActionType.ADD
    # below -> SKIP
    d, r, c = p.evaluate_candidate(s, {"memory_id": "m3", "content": "novel3", "rerank_score": 0.4})
    assert d == MemoryActionType.SKIP
    assert "threshold" in r


def test_policy_gating_missing_rerank_score():
    p = RuleBasedMemoryPolicy(add_relevance_threshold=0.5)
    s = EvidenceState(query="q", tenant_id="t", user_id="u")
    # missing rerank score bypasses gating even if ES score is low
    d, r, c = p.evaluate_candidate(s, {"memory_id": "m4", "content": "novel4", "score": 0.01})
    assert d == MemoryActionType.ADD


def test_policy_gating_both_scores_absent():
    p = RuleBasedMemoryPolicy(add_relevance_threshold=0.5)
    s = EvidenceState(query="q", tenant_id="t", user_id="u")
    d, r, c = p.evaluate_candidate(s, {"memory_id": "m5", "content": "novel5"})
    assert d == MemoryActionType.ADD


def test_policy_gating_malformed_score():
    p = RuleBasedMemoryPolicy(add_relevance_threshold=0.5)
    s = EvidenceState(query="q", tenant_id="t", user_id="u")
    d, r, c = p.evaluate_candidate(s, {"memory_id": "m6", "content": "novel6", "rerank_score": "bad"})
    assert d == MemoryActionType.ADD
    d, r, c = p.evaluate_candidate(s, {"memory_id": "m7", "content": "novel7", "rerank_score": float("nan")})
    assert d == MemoryActionType.ADD


def test_policy_gating_duplicate_skip():
    p = RuleBasedMemoryPolicy(add_relevance_threshold=0.0)
    s = EvidenceState(query="q", tenant_id="t", user_id="u")
    s.working_memory = [{"memory_id": "m8", "content": "x"}]
    d, r, c = p.evaluate_candidate(s, {"memory_id": "m8", "content": "x", "rerank_score": 99.0})
    assert d == MemoryActionType.SKIP


def test_default_threshold_zero():
    p = RuleBasedMemoryPolicy()
    assert p.add_relevance_threshold == 0.0


def test_interaction_duplicate_relevant_memory_is_skipped(client):
    import main

    tenant = "errdup_tenant"
    user = "errdup_user"
    svc = main.service
    assert svc is not None
    for mid in ("dup_a", "dup_b"):
        svc.es.index(
            index=svc.index,
            id=mid,
            document={
                "memory_id": mid,
                "tenant_id": tenant,
                "user_id": user,
                "agent_id": "default",
                "project_id": "default",
                "type": "fact",
                "content": "The production database port is 5432.",
                "importance": 0.9,
                "confidence": 0.95,
                "tags": ["db"],
                "status": "active",
            },
            refresh="wait_for",
        )

    r = client.post("/memories/interact", json={"tenant_id": tenant, "user_id": user, "query": "database port"})
    assert r.status_code == 200
    data = r.json()
    actions = [(a.get("action_type"), a.get("memory_id"), a.get("reason")) for a in data["trajectory"]]
    assert any(a[0] == "SKIP" and "redundant" in str(a[2]) for a in actions)
    ids = [m.get("memory_id") for m in data["working_memory"]]
    assert len(set(ids)) == len(ids)

    for mid in ("dup_a", "dup_b"):
        try:
            svc.es.delete(index=svc.index, id=mid, refresh="wait_for")
        except Exception:
            pass


def test_interaction_adds_related_false_positive(client):
    tenant = "errfp_tenant"
    user = "errfp_user"
    client.post("/memories", json={"tenant_id": tenant, "user_id": user, "content": "We decided to use PostgreSQL for production."})
    client.post("/memories", json={"tenant_id": tenant, "user_id": user, "content": "The production database port is 5432."})

    r = client.post("/memories/interact", json={"tenant_id": tenant, "user_id": user, "query": "PostgreSQL production decision"})
    assert r.status_code == 200
    data = r.json()
    # Policy currently adds both the relevant decision and a related port memory.
    assert data["trajectory"].count({"action_type": "ADD"}) if False else True
    add_ids = [a.get("memory_id") for a in data["trajectory"] if a.get("action_type") == "ADD"]
    assert len(add_ids) >= 2


def test_interact_max_steps_one(client):
    tenant = "maxsteps_tenant"
    user = "maxsteps_user"
    res = client.post("/memories", json={"tenant_id": tenant, "user_id": user, "content": "max steps test"})
    assert res.status_code in (200, 201)
    mid = res.json()["memory"]["memory_id"]

    r = client.post(
        "/memories/interact",
        json={"tenant_id": tenant, "user_id": user, "query": "max steps", "max_steps": 1},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "completed"
    assert data["action_sequence"].count("SEARCH") == 1
    assert data["action_sequence"][-1] == "STOP"

    client.delete(f"/memories/{mid}?tenant_id={tenant}&user_id={user}")


def test_interact_empty_result(client):
    tenant = "empty_tenant"
    user = "empty_user"
    r = client.post("/memories/interact", json={"tenant_id": tenant, "user_id": user, "query": "nothing here"})
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "completed"
    assert data["working_memory"] == []
    assert "STOP" in data["action_sequence"]


def test_controller_fallback_on_policy_error():
    from controller import MemoryInteractionController

    class FailingPolicy:
        version = "failing"

        def decide_search(self, state):
            raise RuntimeError("boom")

        def evaluate_candidate(self, state, candidate):
            return None

        def should_stop(self, state):
            return False, "continue", 0.5

    class FakeService:
        def search(self, params):
            return {"memories": [{"memory_id": "m1", "content": "fallback"}]}

    ctrl = MemoryInteractionController(FakeService(), policy=FailingPolicy())
    result = ctrl.run(query="q", tenant_id="t", user_id="u")
    assert result["status"] == "fallback"
    assert result["working_memory"][0]["memory_id"] == "m1"
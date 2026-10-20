"""Memory interaction controller: orchestrates evidence state, policy, retrieval.

This is the bridge between storage/retrieval and the explicit
search/add/skip/stop decision model. It never bypasses service-level
authorization: all retrieval goes through ``MemoryService.search``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from config import settings
from evidence import EvidenceState, MemoryAction, MemoryActionType
from models import MemorySearch, MemoryStatus, MemoryType
from policy import MemoryPolicy, RuleBasedMemoryPolicy, get_memory_policy
from reward import RewardEvaluator

log = logging.getLogger("memory.controller")


class MemoryInteractionController:
    def __init__(
        self,
        service: Any,
        policy: MemoryPolicy | None = None,
        reward_evaluator: RewardEvaluator | None = None,
        policy_mode: str | None = None,
        add_relevance_threshold: float | None = None,
    ) -> None:
        self.service = service
        if policy is not None:
            self.policy = policy
        else:
            self.policy = get_memory_policy(policy_mode)
            if add_relevance_threshold is not None and isinstance(self.policy, RuleBasedMemoryPolicy):
                self.policy.add_relevance_threshold = add_relevance_threshold
        self.reward_evaluator = (
            reward_evaluator
            if reward_evaluator is not None
            else (RewardEvaluator() if settings.MEMORY_ENABLE_PROCESS_REWARD else None)
        )

    def _coerce_type(self, value: Any) -> Any:
        if value is None or isinstance(value, MemoryType):
            return value
        try:
            return MemoryType(str(value))
        except ValueError:
            return None

    def _coerce_status(self, value: Any) -> Any:
        if value is None or isinstance(value, MemoryStatus):
            return value
        try:
            return MemoryStatus(str(value))
        except ValueError:
            return None

    def run(
        self,
        *,
        query: str,
        tenant_id: str,
        user_id: str,
        agent_id: str = "default",
        project_id: str = "default",
        mem_type: Any = None,
        status: Any = None,
        tags: list[str] | None = None,
        limit: int = 5,
        mode: str = "hybrid",
        include_expired: bool = False,
        include_inactive: bool = False,
        max_steps: int | None = None,
        max_searches: int | None = None,
        max_candidates: int | None = None,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if max_steps is None:
            max_steps = settings.MEMORY_MAX_STEPS
        if max_searches is None:
            max_searches = settings.MEMORY_MAX_SEARCHES
        if max_candidates is None:
            max_candidates = settings.MEMORY_MAX_CANDIDATES
        if timeout is None:
            timeout = settings.MEMORY_INTERACTION_TIMEOUT_SECONDS

        state = EvidenceState(
            query=query,
            tenant_id=tenant_id,
            user_id=user_id,
            agent_id=agent_id or "default",
            project_id=project_id or "default",
            max_steps=max_steps,
            max_searches=max_searches,
            max_candidates=max_candidates,
        )

        mem_type = self._coerce_type(mem_type)
        status = self._coerce_status(status)

        rewards: list[dict[str, Any]] = []
        start = time.time()

        try:
            while (not state.stopped) and state.step < state.max_steps:
                if timeout and (time.time() - start) > float(timeout):
                    state.stopped = True
                    state.stop_reason = "timeout"
                    state.stop_confidence = 0.5
                    action = MemoryAction(
                        MemoryActionType.STOP, step=state.step, reason="timeout", confidence=0.5
                    )
                    state.record_action(action)
                    break

                state.step += 1

                should_search, search_query, search_reason = self.policy.decide_search(state)

                if should_search:
                    q = search_query or state.query
                    t0 = time.time()
                    params = MemorySearch(
                        user_id=user_id,
                        tenant_id=tenant_id,
                        query=q,
                        agent_id=agent_id if agent_id and agent_id != "default" else None,
                        project_id=project_id if project_id and project_id != "default" else None,
                        type=mem_type,
                        status=status,
                        tags=tags,
                        limit=limit,
                        offset=0,
                        include_expired=include_expired,
                        include_inactive=include_inactive,
                        mode=mode if mode in {"keyword", "semantic", "hybrid"} else "hybrid",
                        explain=False,
                    )
                    try:
                        result = self.service.search(params)
                    except Exception as exc:  # noqa: BLE001
                        log.warning("memory search failed during interaction: %s", exc)
                        result = {"memories": [], "total": 0}
                    latency_ms = (time.time() - t0) * 1000.0

                    raw_candidates = result.get("memories", []) or []
                    seen = {c.get("memory_id") for c in state.retrieved_candidates}
                    new_candidates: list[dict[str, Any]] = []
                    for c in raw_candidates:
                        mid = c.get("memory_id")
                        if mid in seen:
                            continue
                        seen.add(mid)
                        new_candidates.append(dict(c))

                    capacity = state.max_candidates * max(1, state.max_searches)
                    if len(state.retrieved_candidates) + len(new_candidates) > capacity:
                        allowed = max(0, capacity - len(state.retrieved_candidates))
                        new_candidates = new_candidates[:allowed]

                    state.retrieved_candidates.extend(new_candidates)
                    state.search_history.append(q)
                    state.searches_done += 1
                    state.last_result_count = len(new_candidates)

                    search_action = MemoryAction(
                        MemoryActionType.SEARCH,
                        step=state.step,
                        query=q,
                        reason=search_reason,
                        result_count=len(raw_candidates),
                        latency_ms=latency_ms,
                        affected_memory_ids=[c.get("memory_id") for c in new_candidates],
                    )
                    if self.reward_evaluator is not None:
                        r = self.reward_evaluator.evaluate(state, search_action, {"result_count": len(raw_candidates)})
                        rewards.append(r.to_dict())
                        search_action.metadata["reward"] = r.to_dict()
                    state.record_action(search_action)

                    if settings.MEMORY_ENABLE_TRAJECTORY_LOGGING:
                        log.info(
                            "memory interaction step=%d action=SEARCH results=%d new=%d latency_ms=%.1f",
                            state.step,
                            len(raw_candidates),
                            len(new_candidates),
                            latency_ms,
                        )

                    added_this = 0
                    for cand in new_candidates:
                        if len(state.working_memory) >= state.max_candidates:
                            skip_action = MemoryAction(
                                MemoryActionType.SKIP,
                                step=state.step,
                                memory_id=cand.get("memory_id"),
                                reason="candidate_capacity_reached",
                                confidence=0.0,
                                affected_memory_ids=[cand.get("memory_id")],
                            )
                            state.record_action(skip_action)
                            if self.reward_evaluator is not None:
                                r = self.reward_evaluator.evaluate(state, skip_action, {})
                                rewards.append(r.to_dict())
                                skip_action.metadata["reward"] = r.to_dict()
                            continue

                        decision, reason, conf = self.policy.evaluate_candidate(state, cand)
                        if decision == MemoryActionType.ADD:
                            state.working_memory.append(dict(cand))
                            added_this += 1
                            act = MemoryAction(
                                MemoryActionType.ADD,
                                step=state.step,
                                memory_id=cand.get("memory_id"),
                                reason=reason,
                                confidence=float(conf) if conf is not None else None,
                                affected_memory_ids=[cand.get("memory_id")],
                            )
                        else:
                            act = MemoryAction(
                                MemoryActionType.SKIP,
                                step=state.step,
                                memory_id=cand.get("memory_id"),
                                reason=reason,
                                confidence=float(conf) if conf is not None else None,
                                affected_memory_ids=[cand.get("memory_id")],
                            )
                        state.record_action(act)
                        if self.reward_evaluator is not None:
                            r = self.reward_evaluator.evaluate(state, act, {})
                            rewards.append(r.to_dict())
                            act.metadata["reward"] = r.to_dict()
                        if settings.MEMORY_ENABLE_TRAJECTORY_LOGGING:
                            log.info(
                                "memory interaction step=%d action=%s memory_id=%s reason=%s",
                                state.step,
                                act.action_type.value,
                                act.memory_id,
                                act.reason,
                            )

                    state.last_added_count = added_this

                stop, stop_reason, stop_conf = self.policy.should_stop(state)
                if stop:
                    state.stopped = True
                    state.stop_reason = stop_reason
                    state.stop_confidence = float(stop_conf) if stop_conf is not None else None
                    stop_action = MemoryAction(
                        MemoryActionType.STOP,
                        step=state.step,
                        reason=stop_reason,
                        confidence=state.stop_confidence,
                    )
                    state.record_action(stop_action)
                    if self.reward_evaluator is not None:
                        r = self.reward_evaluator.evaluate(state, stop_action, {})
                        rewards.append(r.to_dict())
                        stop_action.metadata["reward"] = r.to_dict()
                    if settings.MEMORY_ENABLE_TRAJECTORY_LOGGING:
                        log.info(
                            "memory interaction action=STOP reason=%s confidence=%s",
                            stop_reason,
                            state.stop_confidence,
                        )

            if not state.stopped:
                state.stopped = True
                state.stop_reason = state.stop_reason or "max_steps_reached"
                state.stop_confidence = state.stop_confidence or 0.5

        except Exception as exc:  # noqa: BLE001
            log.exception("memory interaction failed; falling back to retrieval: %s", exc)
            try:
                params = MemorySearch(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    query=query,
                    agent_id=agent_id if agent_id and agent_id != "default" else None,
                    project_id=project_id if project_id and project_id != "default" else None,
                    type=mem_type,
                    status=status,
                    tags=tags,
                    limit=limit,
                    mode=mode if mode in {"keyword", "semantic", "hybrid"} else "hybrid",
                )
                result = self.service.search(params)
                working = result.get("memories", []) or []
            except Exception:  # noqa: BLE001
                working = []
            return {
                "status": "fallback",
                "reason": "policy_error",
                "query": query,
                "working_memory": working,
                "trajectory": [],
                "action_sequence": [],
                "stopped_reason": "policy_error",
                "steps": state.step,
                "rewards": [],
                "evidence_state": None,
            }

        return {
            "status": "completed",
            "query": query,
            "working_memory": state.working_memory,
            "retrieved_candidates": state.retrieved_candidates,
            "search_history": state.search_history,
            "trajectory": state.trajectory(),
            "action_sequence": state.action_sequence(),
            "stopped_reason": state.stop_reason,
            "stop_confidence": state.stop_confidence,
            "steps": state.step,
            "rewards": rewards if settings.MEMORY_ENABLE_PROCESS_REWARD else [],
            "evidence_state": state.to_dict(),
        }

    def interact(self, **kwargs: Any) -> dict[str, Any]:
        return self.run(**kwargs)


MemoryController = MemoryInteractionController
MemoryInteraction = MemoryInteractionController
Controller = MemoryInteractionController

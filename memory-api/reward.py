"""Process-level reward / evaluation interface (advisory, non-training).

This module defines a clean abstraction for evaluating memory interaction
trajectories. It is intentionally deterministic and rule-based. It does NOT
train any model and does not alter production behavior unless explicitly
enabled.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from evidence import EvidenceState, MemoryAction, MemoryActionType


@dataclass
class ProcessReward:
    action_quality: float = 0.0
    evidence_relevance: float = 0.0
    evidence_coverage: float = 0.0
    memory_faithfulness: float = 0.0
    redundancy_penalty: float = 0.0
    stopping_quality: float = 0.0
    overall_score: float = 0.0
    details: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.details is None:
            self.details = {}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


class RewardEvaluator:
    """Deterministic, advisory reward evaluator."""

    def evaluate(
        self,
        state: EvidenceState,
        action: MemoryAction,
        result: dict[str, Any] | None = None,
    ) -> ProcessReward:
        result = result or {}
        reward = ProcessReward()
        at = action.action_type

        if at == MemoryActionType.SEARCH:
            rc = action.result_count or 0
            if rc > 0:
                reward.evidence_relevance = min(1.0, rc / max(1, state.max_candidates))
                reward.action_quality = 0.8
            else:
                reward.action_quality = 0.2
                reward.redundancy_penalty = 0.0
            # Penalize repeated identical searches
            if state.search_history.count(action.query) > 1 if action.query else False:
                reward.redundancy_penalty = 0.5
                reward.action_quality = max(0.0, reward.action_quality - 0.3)

        elif at == MemoryActionType.ADD:
            reward.evidence_relevance = max(0.0, min(1.0, action.confidence or 0.5))
            reward.memory_faithfulness = 0.9 if action.reason != "added_without_provenance" else 0.6
            reward.action_quality = 0.9
            if "redundant" in (action.reason or ""):
                reward.redundancy_penalty = 0.8

        elif at == MemoryActionType.SKIP:
            reason = action.reason or ""
            if "redundant" in reason or "duplicate" in reason or "low" in reason or "below" in reason:
                reward.action_quality = 0.8
            else:
                reward.action_quality = 0.6
            reward.redundancy_penalty = 0.0

        elif at == MemoryActionType.STOP:
            reason = action.reason or ""
            if "sufficient" in reason or "budget" in reason:
                reward.stopping_quality = 0.9
            elif "premature" in reason:
                reward.stopping_quality = 0.3
            else:
                reward.stopping_quality = 0.7
            reward.action_quality = 0.8

        # Coverage: fraction of working memory relative to max candidates
        if state.max_candidates:
            reward.evidence_coverage = min(
                1.0, len(state.working_memory) / max(1, state.max_candidates)
            )

        # Overall
        reward.overall_score = max(
            0.0,
            min(
                1.0,
                (
                    reward.action_quality
                    + reward.evidence_relevance
                    + reward.evidence_coverage
                    + reward.memory_faithfulness
                    + reward.stopping_quality
                    - reward.redundancy_penalty
                )
                / 5.0,
            ),
        )
        reward.details = {"action_type": at.value, "reason": action.reason}
        return reward

    def evaluate_trajectory(self, state: EvidenceState) -> list[ProcessReward]:
        return [self.evaluate(state, a) for a in state.action_history]


# Aliases
Reward = ProcessReward
ProcessRewardEvaluator = RewardEvaluator

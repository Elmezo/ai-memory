"""Memory policy interface and deterministic rule-based implementation.

The rest of the system depends on :class:`MemoryPolicy` rather than a concrete
policy, so a future LLM- or RL-based policy can be dropped in without touching
storage, retrieval, or transport layers.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Tuple

from config import settings
from evidence import EvidenceState, MemoryAction, MemoryActionType

log = logging.getLogger("memory.policy")


class MemoryPolicy(ABC):
    """Abstract memory interaction policy."""

    version: str = "abstract"

    @abstractmethod
    def decide_search(self, state: EvidenceState) -> Tuple[bool, str | None, str]:
        """Return ``(should_search, query_or_none, reason)``."""
        raise NotImplementedError

    @abstractmethod
    def evaluate_candidate(
        self, state: EvidenceState, candidate: dict[str, Any]
    ) -> Tuple[MemoryActionType, str, float]:
        """Return ``(ADD/SKIP, reason, confidence)`` for one candidate."""
        raise NotImplementedError

    @abstractmethod
    def should_stop(self, state: EvidenceState) -> Tuple[bool, str, float]:
        """Return ``(should_stop, reason, confidence)``."""
        raise NotImplementedError


class RuleBasedMemoryPolicy(MemoryPolicy):
    """Deterministic, dependency-free policy used by default."""

    version: str = getattr(settings, "MEMORY_POLICY_VERSION", "rule_based_v1")

    def __init__(
        self,
        max_steps: int | None = None,
        max_searches: int | None = None,
        max_candidates: int | None = None,
        add_relevance_threshold: float | None = None,
    ) -> None:
        self.max_steps = max_steps if max_steps is not None else settings.MEMORY_MAX_STEPS
        self.max_searches = (
            max_searches if max_searches is not None else settings.MEMORY_MAX_SEARCHES
        )
        self.max_candidates = (
            max_candidates if max_candidates is not None else settings.MEMORY_MAX_CANDIDATES
        )
        self.add_relevance_threshold = (
            add_relevance_threshold
            if add_relevance_threshold is not None
            else settings.MEMORY_ADD_RELEVANCE_THRESHOLD
        )

    # ------------------------------------------------------------------ #
    def decide_search(self, state: EvidenceState) -> Tuple[bool, str | None, str]:
        if state.stopped:
            return False, None, "already_stopped"
        if state.searches_done >= (state.max_searches or self.max_searches):
            return False, None, "max_searches_reached"
        if state.searches_done >= (state.max_steps or self.max_steps):
            return False, None, "max_steps_reached"
        if not state.search_history:
            return True, state.query, "initial_search"
        if state.last_result_count == 0:
            return False, None, "no_new_candidates"
        # No new query synthesis in the rule-based policy: stop after first pass.
        return False, None, "rule_based_single_pass"

    def evaluate_candidate(
        self, state: EvidenceState, candidate: dict[str, Any]
    ) -> Tuple[MemoryActionType, str, float]:
        memory_id = candidate.get("memory_id")
        content = str(candidate.get("content", "") or "")

        if state.has_memory(memory_id):
            return MemoryActionType.SKIP, "already_in_working_memory", 1.0

        if state.is_redundant_content(content):
            return MemoryActionType.SKIP, "redundant_content", 0.9

        score = candidate.get("rerank_score")
        if score is None:
            # No reranker score is available (for example a single-candidate
            # result). Do not apply the reranker threshold to the unrelated
            # Elasticsearch score; allow the candidate and record the ES score.
            relevance = None
            try:
                es_score = float(candidate.get("score")) if candidate.get("score") is not None else None
            except (TypeError, ValueError):
                es_score = None
            if es_score is not None and not math.isfinite(es_score):
                es_score = None
        else:
            try:
                relevance = float(score)
            except (TypeError, ValueError):
                relevance = None
            if relevance is not None and not math.isfinite(relevance):
                relevance = None
            es_score = None

        # Normalize cross-encoder scores loosely: treat very negative as low.
        if (
            relevance is not None
            and self.add_relevance_threshold
            and relevance < self.add_relevance_threshold
        ):
            return MemoryActionType.SKIP, "below_relevance_threshold", relevance

        if relevance is None:
            relevance = 1.0 if es_score is None else es_score

        # Capacity guard
        if len(state.working_memory) >= (state.max_candidates or self.max_candidates):
            return MemoryActionType.SKIP, "candidate_capacity_reached", relevance

        confidence = candidate.get("confidence")
        try:
            conf = float(confidence) if confidence is not None else relevance
        except (TypeError, ValueError):
            conf = relevance
        conf = max(0.0, min(1.0, conf)) if conf == conf else 0.5
        if conf > 1.0:
            # Scores can exceed 1; clamp for metadata only is too aggressive,
            # but confidence field is bounded [0,1] in storage. Keep raw in meta.
            pass

        provenance = candidate.get("provenance") or candidate.get("source")
        if not provenance and not candidate.get("source_type"):
            # Still add, but mark provenance as unavailable (never fabricate).
            return MemoryActionType.ADD, "added_without_provenance", conf

        return MemoryActionType.ADD, "relevant_novel_evidence", conf

    def should_stop(self, state: EvidenceState) -> Tuple[bool, str, float]:
        if state.stopped:
            return True, state.stop_reason or "already_stopped", state.stop_confidence or 1.0
        if state.step >= (state.max_steps or self.max_steps):
            return True, "max_steps_reached", 1.0
        if state.searches_done >= (state.max_searches or self.max_searches):
            return True, "search_budget_exhausted", 0.9
        if state.searches_done > 0 and state.last_added_count == 0:
            return True, "no_new_evidence", 0.8
        if state.searches_done > 0:
            # Rule-based policy performs a single retrieval pass.
            return True, "sufficient_evidence", 0.85
        return False, "continue", 0.5


def get_memory_policy(mode: str | None = None) -> MemoryPolicy:
    mode = (mode or settings.MEMORY_POLICY_MODE or "rule_based").strip().lower()
    if mode in {"rule", "rule_based", "deterministic"}:
        return RuleBasedMemoryPolicy()
    # Future policies are not implemented yet; fail safe to rule-based.
    log.warning("Unknown MEMORY_POLICY_MODE=%r; falling back to rule_based", mode)
    return RuleBasedMemoryPolicy()


# Aliases
MemoryControllerPolicy = MemoryPolicy
Policy = MemoryPolicy
RuleBasedPolicy = RuleBasedMemoryPolicy

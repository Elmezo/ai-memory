"""Explicit evidence state and memory action abstractions (AiO-Mem inspired).

These structures make memory operations traceable, observable decisions
rather than hidden implementation details. They are framework-agnostic and
do not depend on Elasticsearch or any transport.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from textutil import jaccard, normalize_text


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryActionType(str, Enum):
    SEARCH = "SEARCH"
    ADD = "ADD"
    SKIP = "SKIP"
    STOP = "STOP"


ActionType = MemoryActionType
MemoryActionKind = MemoryActionType


def coerce_action_type(value: Any) -> MemoryActionType:
    if isinstance(value, MemoryActionType):
        return value
    text = str(value).strip().upper()
    try:
        return MemoryActionType(text)
    except ValueError:
        aliases = {
            "SEARCH_MEMORY": MemoryActionType.SEARCH,
            "ADD_MEMORY": MemoryActionType.ADD,
            "SKIP_MEMORY": MemoryActionType.SKIP,
            "STOP_SEARCH": MemoryActionType.STOP,
        }
        return aliases.get(text, MemoryActionType.SEARCH)


@dataclass
class MemoryAction:
    action_type: MemoryActionType
    step: int = 0
    query: str | None = None
    memory_id: str | None = None
    reason: str | None = None
    confidence: float | None = None
    affected_memory_ids: list[str] = field(default_factory=list)
    result_count: int | None = None
    latency_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_utcnow_iso)

    def __post_init__(self) -> None:
        self.action_type = coerce_action_type(self.action_type)
        if self.affected_memory_ids is None:
            self.affected_memory_ids = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["action_type"] = self.action_type.value
        return data


@dataclass
class EvidenceState:
    query: str
    tenant_id: str = ""
    user_id: str = ""
    agent_id: str = "default"
    project_id: str = "default"
    working_memory: list[dict[str, Any]] = field(default_factory=list)
    retrieved_candidates: list[dict[str, Any]] = field(default_factory=list)
    search_history: list[str] = field(default_factory=list)
    action_history: list[MemoryAction] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    step: int = 0
    max_steps: int = 3
    max_searches: int = 2
    max_candidates: int = 10
    searches_done: int = 0
    last_result_count: int = 0
    last_added_count: int = 0
    stopped: bool = False
    stop_reason: str | None = None
    stop_confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.working_memory is None:
            self.working_memory = []
        if self.retrieved_candidates is None:
            self.retrieved_candidates = []
        if self.search_history is None:
            self.search_history = []
        if self.action_history is None:
            self.action_history = []
        if self.provenance is None:
            self.provenance = {}
        if self.metadata is None:
            self.metadata = {}

    @property
    def working_memory_ids(self) -> set[str]:
        return {str(m.get("memory_id")) for m in self.working_memory if m.get("memory_id")}

    def has_memory(self, memory_id: str | None) -> bool:
        if not memory_id:
            return False
        return str(memory_id) in self.working_memory_ids

    def is_redundant_content(self, content: str, threshold: float = 0.8) -> bool:
        norm = normalize_text(content or "")
        for mem in self.working_memory:
            other = normalize_text(str(mem.get("content", "")))
            if jaccard(norm, other) >= threshold:
                return True
        return False

    def record_action(self, action: MemoryAction) -> None:
        self.action_history.append(action)

    def trajectory(self) -> list[dict[str, Any]]:
        return [a.to_dict() for a in self.action_history]

    def action_sequence(self) -> list[str]:
        return [a.action_type.value for a in self.action_history]

    def to_dict(self, include_content: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "query": self.query,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "project_id": self.project_id,
            "step": self.step,
            "max_steps": self.max_steps,
            "max_searches": self.max_searches,
            "max_candidates": self.max_candidates,
            "searches_done": self.searches_done,
            "last_result_count": self.last_result_count,
            "last_added_count": self.last_added_count,
            "stopped": self.stopped,
            "stop_reason": self.stop_reason,
            "stop_confidence": self.stop_confidence,
            "search_history": list(self.search_history),
            "action_sequence": self.action_sequence(),
            "trajectory": self.trajectory(),
            "working_memory_count": len(self.working_memory),
            "retrieved_candidates_count": len(self.retrieved_candidates),
            "provenance": dict(self.provenance),
        }
        if include_content:
            data["working_memory"] = [dict(m) for m in self.working_memory]
            data["retrieved_candidates"] = [dict(c) for c in self.retrieved_candidates]
        else:
            data["working_memory"] = [
                {"memory_id": m.get("memory_id")} for m in self.working_memory
            ]
            data["retrieved_candidates"] = [
                {"memory_id": c.get("memory_id")} for c in self.retrieved_candidates
            ]
        return data


MemoryEvidenceState = EvidenceState
MemoryState = EvidenceState

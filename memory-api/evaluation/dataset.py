"""Dataset loading and validation for memory evaluation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EvalExample:
    id: str
    query: str
    tenant_id: str
    user_id: str
    agent_id: str = "default"
    project_id: str = "default"
    relevant_memory_ids: list[str] = field(default_factory=list)
    equivalent_memory_ids: dict[str, str] = field(default_factory=dict)
    expected_answer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.id:
            raise ValueError("example.id is required")
        if not self.query:
            raise ValueError("example.query is required")
        if not self.tenant_id:
            raise ValueError("example.tenant_id is required")
        if not self.user_id:
            raise ValueError("example.user_id is required")
        if self.relevant_memory_ids is None:
            self.relevant_memory_ids = []
        if not isinstance(self.relevant_memory_ids, list):
            raise ValueError("relevant_memory_ids must be a list")
        if self.equivalent_memory_ids is None:
            self.equivalent_memory_ids = {}
        if not isinstance(self.equivalent_memory_ids, dict):
            raise ValueError("equivalent_memory_ids must be a dict")
        for key, value in self.equivalent_memory_ids.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("equivalent_memory_ids keys and values must be strings")
        # Validate equivalence relationships, including chains and cycles.
        for key in self.equivalent_memory_ids:
            canonicalize_memory_id(key, self.equivalent_memory_ids)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvalExample":
        ex = cls(
            id=str(data.get("id", "")),
            query=str(data.get("query", "")),
            tenant_id=str(data.get("tenant_id", "")),
            user_id=str(data.get("user_id", "")),
            agent_id=str(data.get("agent_id", "default")),
            project_id=str(data.get("project_id", "default")),
            relevant_memory_ids=list(data.get("relevant_memory_ids", []) or []),
            equivalent_memory_ids=dict(data.get("equivalent_memory_ids", {}) or {}),
            expected_answer=data.get("expected_answer"),
            metadata=dict(data.get("metadata", {}) or {}),
        )
        ex.validate()
        return ex


def load_dataset(path: str | Path) -> list[EvalExample]:
    examples: list[EvalExample] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"Line {line_no} is not a JSON object")
            examples.append(EvalExample.from_dict(data))
    return examples


def canonicalize_memory_id(memory_id: str, equivalence: dict[str, str] | None = None) -> str:
    """Resolve a memory ID through an equivalence mapping.

    Handles missing mappings, self-references, chains, and cycles. Raises
    ``ValueError`` on cyclic mappings instead of looping forever.
    """
    if not memory_id:
        return memory_id
    equivalence = equivalence or {}
    current = memory_id
    seen: set[str] = set()
    while current in equivalence:
        nxt = equivalence[current]
        if nxt == current:
            return current
        if current in seen:
            raise ValueError(f"Cyclic equivalent_memory_ids mapping at {current!r}")
        seen.add(current)
        current = nxt
        if len(seen) > len(equivalence) + 1:
            raise ValueError("Equivalent memory chain is too long")
    return current


def canonicalize_ids(ids: list[str] | None, equivalence: dict[str, str] | None = None) -> list[str]:
    return [canonicalize_memory_id(str(x), equivalence) for x in (ids or [])]


def canonical_relevant_set(relevant: list[str] | set[str] | None, equivalence: dict[str, str] | None = None) -> set[str]:
    return {canonicalize_memory_id(str(x), equivalence) for x in (relevant or [])}

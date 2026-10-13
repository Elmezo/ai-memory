"""Pydantic request/response models and the canonical memory schema."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from textutil import strip_bom


def _clean_str(value: str | None) -> str | None:
    if value is None:
        return None
    return strip_bom(value)


class MemoryType(str, Enum):
    preference = "preference"
    fact = "fact"
    decision = "decision"
    task = "task"
    entity = "entity"
    conversation_summary = "conversation_summary"
    procedure = "procedure"
    constraint = "constraint"


class MemoryStatus(str, Enum):
    active = "active"
    inactive = "inactive"
    archived = "archived"


class MemoryBase(BaseModel):
    """Base fields shared by MemoryInput and MemorySearch.

    ``user_id`` and ``tenant_id`` accept empty strings so that the MCP scope
    injection mechanism can supply them after model construction.  The REST API
    endpoint validators must enforce non-empty values independently.
    """

    user_id: str = Field(default="", max_length=128)
    tenant_id: str = Field(default="", max_length=128)
    agent_id: str = Field(default="default", max_length=128)
    project_id: str = Field(default="default", max_length=128)

    @field_validator("user_id", "tenant_id", "agent_id", "project_id", mode="before")
    @classmethod
    def _strip_ids(cls, v: Any) -> Any:
        return strip_bom(v) if isinstance(v, str) else v


class MemoryInput(MemoryBase):
    """Payload for POST /memories."""

    type: MemoryType = Field(default=MemoryType.fact)
    content: str = Field(min_length=1, max_length=10000)
    importance: float = Field(default=0.5, ge=0, le=1)
    confidence: float = Field(default=1.0, ge=0, le=1)
    source: str | None = Field(default=None, max_length=256)
    tags: list[str] = Field(default_factory=list, max_length=32)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: MemoryStatus = Field(default=MemoryStatus.active)
    expires_at: datetime | None = None

    @field_validator("content", "source", mode="before")
    @classmethod
    def _strip_text(cls, v: Any) -> Any:
        return strip_bom(v) if isinstance(v, str) else v

    @field_validator("tags", mode="before")
    @classmethod
    def _clean_tags(cls, v: Any) -> Any:
        if v is None:
            return []
        return [strip_bom(t) for t in v if isinstance(t, str) and strip_bom(t)][:32]


class MemoryUpdate(BaseModel):
    """Payload for PATCH /memories/{memory_id}. Scope comes from query params."""

    content: str | None = Field(default=None, max_length=10000)
    importance: float | None = Field(default=None, ge=0, le=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    type: MemoryType | None = None
    source: str | None = Field(default=None, max_length=256)
    tags: list[str] | None = Field(default=None, max_length=32)
    metadata: dict[str, Any] | None = None
    status: MemoryStatus | None = None
    expires_at: datetime | None = None

    @field_validator("content", "source", mode="before")
    @classmethod
    def _strip_text(cls, v: Any) -> Any:
        return strip_bom(v) if isinstance(v, str) else v


class MemorySearch(BaseModel):
    """Payload for POST /memories/search.

    ``user_id`` and ``tenant_id`` are required for REST-API calls but can be
    empty strings when the MCP server injects scope from environment variables.
    An empty value means "use whatever scope the server has configured".
    The service layer MUST validate that at least a tenant is resolvable.
    """

    user_id: str = Field(default="", max_length=128)
    tenant_id: str = Field(default="", max_length=128)
    query: str = Field(min_length=1, max_length=1000)
    agent_id: str | None = None
    project_id: str | None = None
    type: MemoryType | None = None
    status: MemoryStatus | None = None
    tags: list[str] | None = None
    limit: int = Field(default=5, ge=1, le=50)
    offset: int = Field(default=0, ge=0)
    include_expired: bool = False
    include_inactive: bool = False
    mode: Literal["keyword", "semantic", "hybrid"] = "keyword"
    explain: bool = False

    @field_validator(
        "user_id", "tenant_id", "agent_id", "project_id", "query", mode="before"
    )
    @classmethod
    def _strip_text(cls, v: Any) -> Any:
        return strip_bom(v) if isinstance(v, str) else v


class MemoryRecord(BaseModel):
    """Canonical memory document as stored in Elasticsearch."""

    memory_id: str
    user_id: str
    tenant_id: str
    agent_id: str = "default"
    project_id: str = "default"
    type: str = "fact"
    content: str = ""
    importance: float = 0.5
    confidence: float = 1.0
    source: str | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    status: str = "active"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_accessed_at: datetime | None = None
    expires_at: datetime | None = None
    score: float | None = None
    explanation: dict[str, Any] | None = None

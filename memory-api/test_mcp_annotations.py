"""Regression tests for MCP tool metadata (annotations).

These tests do not require Elasticsearch; they only inspect the tool
definitions that the MCP server advertises to clients. They guard against
tools silently losing their annotations or mis-declaring their behavior.
"""

from __future__ import annotations

import asyncio

import mcp_server

# Expected annotation hints per tool: (read_only, destructive, idempotent).
EXPECTED = {
    "memory_health": (True, False, True),
    "memory_save": (False, False, False),
    "memory_search": (True, False, True),
    "memory_get": (True, False, True),
    "memory_update": (False, False, True),
    "memory_delete": (False, True, True),
    "memory_list": (True, False, True),
}


def _tools():
    return {t.name: t for t in asyncio.run(mcp_server.mcp.list_tools())}


def test_all_expected_tools_are_registered():
    tools = _tools()
    assert set(EXPECTED).issubset(tools.keys())


def test_every_tool_has_annotations():
    for name, tool in _tools().items():
        assert tool.annotations is not None, f"{name} is missing MCP tool annotations"


def test_annotation_hints_match_behavior():
    for name, (read_only, destructive, idempotent) in EXPECTED.items():
        ann = _tools()[name].annotations
        assert ann.read_only_hint is read_only, name
        assert ann.destructive_hint is destructive, name
        assert ann.idempotent_hint is idempotent, name
        assert ann.open_world_hint is False, name
        assert ann.title, f"{name} annotations should include a title"


def test_read_only_tools_are_not_destructive():
    for name, tool in _tools().items():
        if tool.annotations.read_only_hint:
            assert tool.annotations.destructive_hint is False, name


def test_only_delete_is_destructive():
    destructive = {
        name for name, t in _tools().items() if t.annotations.destructive_hint
    }
    assert destructive == {"memory_delete"}


def test_mutating_tools_are_not_read_only():
    for name in {"memory_save", "memory_update", "memory_delete"}:
        assert _tools()[name].annotations.read_only_hint is False


def test_annotations_serialize_with_camel_case_keys():
    ann = _tools()["memory_delete"].annotations
    dumped = ann.model_dump(by_alias=True, exclude_none=True)
    assert dumped["readOnlyHint"] is False
    assert dumped["destructiveHint"] is True
    assert "openWorldHint" in dumped

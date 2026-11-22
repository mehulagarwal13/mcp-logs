"""Issue 2: the MCP tool catalogue comes from the server's real registration, and the
frontend's mock-mode catalogue cannot drift from it."""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.core.exceptions import PermissionDeniedError
from app.core.observability import service as observability_service
from app.core.observability.catalogue import parameters_from_schema
from app.core.observability.schemas import McpToolSpec
from app.mcp.catalogue import describe_registered_tools
from app.mcp.servers import main as mcp_assembly
from app.shared.schemas import ActorKind, Identity
from scripts import export_mcp_tool_catalogue as export_script

_EXPECTED_TOOLS = {
    "ask_question",
    "configure_sso",
    "create_access_rule",
    "create_invitation",
    "create_project",
    "generate_postmortem",
    "investigate_incident",
    "propose_runbook_update",
    "search_recent_changes",
    "search_similar_incidents",
}


def _reader(*, allowed: bool = True) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        permissions=frozenset({"observability:read"} if allowed else set()),
    )


# ------------------------------------------------------- the real registration


def test_description_reads_the_real_registry_not_a_list():
    specs = asyncio.run(describe_registered_tools(mcp_assembly.mcp_server))
    names = {s.name for s in specs}
    assert names == _EXPECTED_TOOLS
    for spec in specs:
        assert spec.description.strip(), f"{spec.name} has no description"
        assert spec.input_schema.get("type") == "object", spec.name
    # it is exactly what the MCP protocol's tools/list serves
    served = asyncio.run(mcp_assembly.mcp_server.list_tools())
    assert names == {t.name for t in served}


def test_registering_a_new_tool_changes_the_catalogue_without_any_other_edit():
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="probe")

    @server.tool()
    async def probe_tool(count: int, label: str | None = None, verbose: bool = False) -> str:
        """Probe docstring."""
        return "ok"

    (spec,) = asyncio.run(describe_registered_tools(server))
    assert spec.name == "probe_tool" and spec.description == "Probe docstring."
    params = {p.name: p for p in parameters_from_schema(spec.input_schema)}
    assert (params["count"].type, params["count"].required) == ("number", True)
    assert (params["label"].type, params["label"].required) == ("string", False)
    assert (params["verbose"].type, params["verbose"].default) == ("boolean", "false")


# ------------------------------------------------------- parameter derivation


def test_parameters_derivation_handles_types_optionals_and_defaults():
    schema = {
        "type": "object",
        "required": ["query", "limit"],
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "default": 10},
            "since": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "tags": {"type": "array"},
            "options": {"type": "object"},
            "flag": {"type": "boolean", "default": True},
            "mode": {"type": "string", "default": "fast"},
            "weird": {},
        },
    }
    p = {x.name: x for x in parameters_from_schema(schema)}
    assert (p["query"].type, p["query"].required, p["query"].description) == (
        "string",
        True,
        "What to search for.",
    )
    assert (p["limit"].type, p["limit"].required, p["limit"].default) == ("number", True, "10")
    assert (p["since"].type, p["since"].required, p["since"].default) == ("string", False, None)
    assert (p["tags"].type, p["options"].type) == ("array", "object")
    assert (p["flag"].default, p["mode"].default) == ("true", "fast")
    assert p["weird"].type == "string" and p["weird"].required is False


def test_no_properties_means_no_parameters():
    assert parameters_from_schema({"type": "object"}) == []


# --------------------------------------------------------------- mock drift guard


def test_the_mock_catalogue_snapshot_matches_the_live_registration():
    """Fails if a tool is added/removed/changed without regenerating the snapshot
    (`python scripts/export_mcp_tool_catalogue.py`)."""
    expected = export_script.render(asyncio.run(export_script.build_snapshot()))
    current = export_script.SNAPSHOT.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert current == expected, "mock MCP tool catalogue is stale"
    names = {t["name"] for t in json.loads(current)}
    assert names == _EXPECTED_TOOLS


# ----------------------------------------------------------------- service layer


@pytest.mark.asyncio
async def test_catalogue_requires_observability_read():
    with pytest.raises(PermissionDeniedError):
        await observability_service.list_mcp_tool_catalogue(None, _reader(allowed=False))


@pytest.mark.asyncio
async def test_catalogue_maps_rows_to_definitions_with_parameters(monkeypatch):
    now = datetime.now(UTC)
    rows = [
        SimpleNamespace(
            name="create_project",
            description="Create a project.",
            input_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}, "is_default": {"type": "boolean"}},
            },
            synced_at=now,
        )
    ]

    async def fake_list(session):
        return rows

    monkeypatch.setattr(observability_service.repository, "list_tool_catalogue", fake_list)
    (definition,) = await observability_service.list_mcp_tool_catalogue(None, _reader())
    assert definition.name == "create_project" and definition.status == "available"
    assert definition.synced_at == now
    assert [(p.name, p.required) for p in definition.parameters] == [
        ("name", True),
        ("is_default", False),
    ]


@pytest.mark.asyncio
async def test_sync_replaces_the_stored_set_with_exactly_what_was_registered(monkeypatch):
    captured = {}

    async def fake_replace(session, tools):
        captured["tools"] = tools

    monkeypatch.setattr(observability_service.repository, "replace_tool_catalogue", fake_replace)
    count = await observability_service.sync_mcp_tool_catalogue(
        None, [McpToolSpec(name="a", description="A", input_schema={"type": "object"})]
    )
    assert count == 1
    assert captured["tools"] == [
        {"name": "a", "description": "A", "input_schema": {"type": "object"}}
    ]


def test_the_route_is_registered_and_distinct_from_the_statistics_route():
    from app.api.main import app

    paths = set(app.openapi()["paths"])
    assert "/observability/mcp/tools" in paths and "/observability/mcp" in paths

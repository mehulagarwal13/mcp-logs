"""Pydantic contracts for core/observability.

Owned by: core/observability. `McpRequestLog` is the read-side view of one
`mcp_requests` row (`app.database.models.mcp_models.McpRequest`) -- the same
name-mirrors-the-ORM-row convention `agents.schemas.AgentExecution` already
uses for its own table.

This schema (and the module writing it, `core.observability.service`) lives
under `core/`, not `mcp/`, even though `mcp_requests` is conceptually an
mcp-motivated table (DATABASE_DESIGN.md lists it under "mcp/ -- owned
tables") -- see `app.database.models.mcp_models.McpRequest`'s own docstring
for why: `pyproject.toml`'s import-linter contract forbids `app.mcp` from
importing `app.database` in any form, including for its own logging table,
so *something* core-side has to hold write access on mcp/'s behalf. This
mirrors `core.incidents.service.record_investigation_result` in reverse:
that function is core/incidents writing agents-motivated content into its
own table; this module is core writing mcp-motivated content into a table
mcp/ conceptually owns but cannot touch directly.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class McpRequestLog(BaseModel):
    """One row of `mcp_requests`, as read back after
    `service.record_mcp_request`."""

    model_config = ConfigDict(from_attributes=True, frozen=True)

    id: uuid.UUID
    organization_id: uuid.UUID | None = None
    tool_name: str
    identity: str
    request_summary: dict | None
    status_code: int | None
    latency_ms: int | None
    occurred_at: datetime


class McpToolStats(BaseModel):
    """Aggregated latency/error metrics for one MCP tool over some time
    window -- Milestone 10's "MCP latency metrics" dashboard requirement
    (PROJECT_PLAN.md section 10), scoped to the requesting organization:
    `mcp_requests` carries an owning `organization_id`, and both the query
    and row-level security restrict the aggregate to the caller's own rows.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str
    request_count: int
    error_count: int
    avg_latency_ms: float | None
    max_latency_ms: int | None


class McpToolSpec(BaseModel):
    """One registered tool as the MCP server reports it (`tools/list` shape)."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str = ""
    input_schema: dict[str, Any]


class McpToolParameter(BaseModel):
    """One input parameter, derived from the tool's JSON Schema."""

    model_config = ConfigDict(frozen=True)

    name: str
    type: Literal["string", "number", "boolean", "object", "array"]
    required: bool
    description: str = ""
    default: str | None = None


class McpToolDefinition(BaseModel):
    """A registered MCP tool for the catalogue page: what the server actually
    exposes, NOT usage statistics (see `McpToolStats`). `status` is
    "available" for every tool in the server's latest published registration;
    `synced_at` says when that was published, so a stale catalogue is visible.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any]
    parameters: list[McpToolParameter]
    status: Literal["available"] = "available"
    synced_at: datetime

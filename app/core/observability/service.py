"""Public interface for core/observability -- `record_mcp_request` (the
original write path), `get_mcp_dashboard` (a tenant-scoped read path over
the same table) and `purge_expired_mcp_requests` (retention). Exists solely
so `mcp/` (which cannot import `app.database` at all -- see `core.observability.schemas`'s module
docstring) has a `core`-side function to call for its own request logging,
the same "call the owning layer's service.py, never its repository.py
directly" convention every other cross-module call in this codebase follows.

`record_mcp_request` has no `require_permission` gate: this is internal
observability bookkeeping triggered on every MCP tool call regardless of
outcome, not a user-initiated business mutation -- the same reasoning
`agents.repository.insert_agent_execution` has no permission gate of its
own either. `get_mcp_dashboard` (new) IS gated, since unlike the write path
it has a real caller-facing surface (a REST endpoint) that should not be
open to just anyone with a valid session.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.observability import repository
from app.core.observability.catalogue import parameters_from_schema
from app.core.observability.redaction import minimize_request_summary
from app.core.observability.schemas import (
    McpRequestLog,
    McpToolDefinition,
    McpToolSpec,
    McpToolStats,
)
from app.core.users.service import require_permission
from app.shared.config.settings import get_settings
from app.shared.schemas import Identity

_OBSERVABILITY_READ_PERMISSION = "observability:read"


async def record_mcp_request(
    session: AsyncSession,
    *,
    tool_name: str,
    identity: str,
    request_summary: dict | None,
    status_code: int | None,
    latency_ms: int | None,
    organization_id: uuid.UUID | None = None,
) -> McpRequestLog:
    """Log one completed MCP tool call to `mcp_requests`. See
    `app.mcp.dispatch.run_mcp_tool`, the sole caller -- every MCP tool call
    logs exactly once, after it completes, successfully or not.

    `organization_id` is the caller's organization (None only when the call
    failed before any identity was resolved); the caller must also have set
    the tenant context on `session` for it. `request_summary` is minimized
    here -- raw request content is never persisted (`redaction`).
    """
    row = await repository.insert_mcp_request(
        session,
        organization_id=organization_id,
        tool_name=tool_name,
        identity=identity,
        request_summary=minimize_request_summary(request_summary),
        status_code=status_code,
        latency_ms=latency_ms,
    )
    return McpRequestLog.model_validate(row)


async def get_mcp_dashboard(
    session: AsyncSession, actor: Identity, *, since: datetime | None = None
) -> list[McpToolStats]:
    """Per-tool MCP latency/error aggregate for the actor's OWN organization,
    optionally windowed by `since` (Milestone 10's "MCP latency metrics"
    dashboard requirement, PROJECT_PLAN.md section 10).

    Gated by `observability:read` and scoped to `actor.organization_id`
    (explicitly in the query, and again by row-level security): one tenant
    never sees another tenant's tool usage.
    """
    require_permission(actor, _OBSERVABILITY_READ_PERMISSION)

    rows = await repository.get_mcp_tool_stats(session, actor.organization_id, since=since)
    return [
        McpToolStats(
            tool_name=row.tool_name,
            request_count=row.request_count,
            error_count=int(row.error_count or 0),
            avg_latency_ms=float(row.avg_latency_ms) if row.avg_latency_ms is not None else None,
            max_latency_ms=row.max_latency_ms,
        )
        for row in rows
    ]


async def purge_expired_mcp_requests(
    session: AsyncSession, *, now: datetime | None = None, retention_days: int | None = None
) -> int:
    """Retention: delete every `mcp_requests` row older than the configured
    retention (`Settings.mcp_request_retention_days`), for all tenants and the
    unattributed rows. Returns the number of rows deleted. Operator-run
    (`scripts/purge_mcp_requests.py`); no `Identity` is involved because this
    is platform housekeeping, not a tenant action.
    """
    days = (
        retention_days if retention_days is not None else get_settings().mcp_request_retention_days
    )
    cutoff = (now or datetime.now(UTC)) - timedelta(days=days)
    return await repository.purge_expired(session, cutoff)


async def sync_mcp_tool_catalogue(session: AsyncSession, tools: list[McpToolSpec]) -> int:
    """Publish the MCP server's registered tools (called by the MCP server at
    startup). Replaces the stored catalogue with exactly `tools`. Internal
    bookkeeping with no `Identity`, like `record_mcp_request`."""
    await repository.replace_tool_catalogue(
        session,
        [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ],
    )
    return len(tools)


async def list_mcp_tool_catalogue(
    session: AsyncSession, actor: Identity
) -> list[McpToolDefinition]:
    """The tools the MCP server has registered, with descriptions, input
    schemas and derived parameter lists. Gated by `observability:read` (the
    permission the MCP Tools page already requires). Tool metadata is
    platform-wide and contains no tenant data."""
    require_permission(actor, _OBSERVABILITY_READ_PERMISSION)
    rows = await repository.list_tool_catalogue(session)
    return [
        McpToolDefinition(
            name=row.name,
            description=row.description,
            input_schema=row.input_schema,
            parameters=parameters_from_schema(row.input_schema),
            synced_at=row.synced_at,
        )
        for row in rows
    ]

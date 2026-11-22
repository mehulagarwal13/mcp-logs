"""Persistence for core/observability -- `mcp_requests` only
(DATABASE_DESIGN.md's "mcp/ -- owned tables"; see `core.observability.
schemas`'s module docstring for why this write access lives under core/
rather than mcp/). Pure data access, same discipline as every other
repository.py in this codebase: one statement per function, ORM rows in/out,
no business rules, no ORM->Pydantic mapping.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import case, delete, func, insert, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.mcp_models import McpRequest, McpToolCatalogueEntry

# A request without a recorded status_code (a call that failed before
# `mcp.dispatch.run_mcp_tool` could even map an outcome to one) is counted as
# an error -- absence of a status here means "something went wrong before
# normal completion," not "unknown, ignore it," matching this table's own
# "logged after it completes, successfully or not" guarantee (see
# `insert_mcp_request`'s docstring: every call is logged exactly once).
_ERROR_STATUS_CASE = case(
    (McpRequest.status_code.is_(None), 1),
    (McpRequest.status_code >= 400, 1),
    else_=0,
)


async def get_mcp_tool_stats(
    session: AsyncSession, organization_id: uuid.UUID, *, since: datetime | None = None
) -> list[Any]:
    """Aggregate ONE organization's `mcp_requests` by `tool_name`: count, error
    count, average and max latency -- backs the observability dashboard
    (`core.observability.service.get_mcp_dashboard`).

    Filtered by `organization_id` here AND by row-level security (the caller's
    tenant context): either alone would isolate tenants, both is deliberate.

    Returns raw SQLAlchemy `Row` objects (named-tuple-like, with `.tool_name`
    /`.request_count`/`.error_count`/`.avg_latency_ms`/`.max_latency_ms`
    attributes matching the query's `label()`s) rather than ORM instances --
    an aggregate query has no single `McpRequest` row to map back onto, so
    the usual "return ORM rows, let service.py map to Pydantic" convention
    doesn't apply verbatim; `service.py` builds `McpToolStats` directly from
    these labeled columns instead.
    """
    stmt = select(
        McpRequest.tool_name.label("tool_name"),
        func.count().label("request_count"),
        func.sum(_ERROR_STATUS_CASE).label("error_count"),
        func.avg(McpRequest.latency_ms).label("avg_latency_ms"),
        func.max(McpRequest.latency_ms).label("max_latency_ms"),
    ).where(McpRequest.organization_id == organization_id).group_by(McpRequest.tool_name)
    if since is not None:
        stmt = stmt.where(McpRequest.occurred_at >= since)

    result = await session.execute(stmt)
    return list(result.all())


async def insert_mcp_request(
    session: AsyncSession,
    *,
    organization_id: uuid.UUID | None,
    tool_name: str,
    identity: str,
    request_summary: dict | None,
    status_code: int | None,
    latency_ms: int | None,
) -> McpRequest:
    """Log one completed MCP tool call. Unlike `agents.repository.
    insert_agent_execution`, this is not a "create running, update later"
    pair -- every field here is already known by the time a tool call has
    finished (successfully or not), so this is the only write this table
    ever needs per call. See `McpRequest`'s own docstring.

    A plain `INSERT` with a client-generated id and timestamp, deliberately
    without `RETURNING`/refresh: row-level security applies the SELECT policy
    to returned rows, and an unattributed row (`organization_id` NULL, a call
    that failed before any identity existed) is insertable but, by design,
    never visible to the session that wrote it.
    """
    row = McpRequest(
        id=uuid.uuid4(),
        organization_id=organization_id,
        tool_name=tool_name,
        identity=identity,
        request_summary=request_summary,
        status_code=status_code,
        latency_ms=latency_ms,
        occurred_at=datetime.now(UTC),
    )
    await session.execute(
        insert(McpRequest).values(
            id=row.id,
            organization_id=row.organization_id,
            tool_name=row.tool_name,
            identity=row.identity,
            request_summary=row.request_summary,
            status_code=row.status_code,
            latency_ms=row.latency_ms,
            occurred_at=row.occurred_at,
        )
    )
    return row


async def purge_expired(session: AsyncSession, cutoff: datetime) -> int:
    """Delete every `mcp_requests` row older than `cutoff`, across all
    tenants (and the unattributed ones), through the `SECURITY DEFINER`
    function `purge_mcp_requests` -- the only cross-tenant path, narrowed to
    exactly this one deletion. Returns the number of rows removed."""
    result = await session.execute(
        text("SELECT purge_mcp_requests(:cutoff)"), {"cutoff": cutoff}
    )
    return int(result.scalar_one())


async def replace_tool_catalogue(session: AsyncSession, tools: list[dict[str, Any]]) -> None:
    """Make `mcp_tool_catalogue` exactly `tools` (name/description/input_schema
    dicts): upsert each, delete every row not in the list."""
    names = [tool["name"] for tool in tools]
    if names:
        stmt = pg_insert(McpToolCatalogueEntry).values(tools)
        stmt = stmt.on_conflict_do_update(
            index_elements=[McpToolCatalogueEntry.name],
            set_={
                "description": stmt.excluded.description,
                "input_schema": stmt.excluded.input_schema,
                "synced_at": func.now(),
            },
        )
        await session.execute(stmt)
        await session.execute(
            delete(McpToolCatalogueEntry).where(McpToolCatalogueEntry.name.not_in(names))
        )
    else:
        await session.execute(delete(McpToolCatalogueEntry))


async def list_tool_catalogue(session: AsyncSession) -> list[McpToolCatalogueEntry]:
    result = await session.execute(
        select(McpToolCatalogueEntry).order_by(McpToolCatalogueEntry.name)
    )
    return list(result.scalars().all())

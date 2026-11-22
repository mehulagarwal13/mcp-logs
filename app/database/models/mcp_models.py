"""The mcp_requests table records MCP request telemetry, mainly to measure tool latency.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class McpRequest(Base):
    """One logged MCP tool call -- observability record backing the MCP
    latency metrics referenced throughout PROJECT_PLAN.md/DATABASE_DESIGN.md.

    Unlike `AgentExecution`, this is not a running->succeeded/failed
    lifecycle row: a tool call's outcome, latency, and status are all known
    by the time anything is written, so `mcp.repository.insert_mcp_request`
    performs exactly one insert per call, after it completes (successfully
    or not) -- there is no "running" state to record separately.

    Tenant isolation: `organization_id` is the owning organization. Row-level
    security (migration `f1b6d3a8c5e2`) lets a session see, insert and delete only
    rows whose `organization_id` equals its tenant context; there is no UPDATE
    policy and `ekip_app` has no UPDATE privilege (the log is append-only). It
    is NULL only for calls that failed before any identity was resolved -- such
    rows are insertable but invisible to every tenant, and are removed by the
    retention purge (`purge_mcp_requests`). `request_summary` never holds raw
    request content: see `core.observability.redaction`.
    """

    __tablename__ = "mcp_requests"
    __table_args__ = (
        Index("ix_mcp_requests_tool_name_occurred_at", "tool_name", "occurred_at"),
        Index("ix_mcp_requests_org_occurred_at", "organization_id", "occurred_at"),
    )

    organization_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("organizations.id", ondelete="CASCADE"),
        nullable=True,
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    tool_name: Mapped[str] = mapped_column(Text, nullable=False)
    identity: Mapped[str] = mapped_column(Text, nullable=False)
    request_summary: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    status_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class OAuthClient(Base):
    """This table stores registered MCP OAuth clients (e.g., Claude) in the database 
    so their credentials survive server restarts.
    - Why persist it? Previously, clients were stored in an in-memory dictionary, 
    which was emptied whenever the server restarted.
    - No organization ID or RLS: A client like Claude is software, not a member of one 
    organization. The organization is chosen later during user authorization.
    - `client_metadata`: Stores client details together in one JSONB object instead 
    of many separate columns.
    - `client_secret_encrypted`: Stores the secret using reversible encryption, 
    not hashing, because the MCP authentication process needs to retrieve the original 
    secret to verify the client.
    """

    __tablename__ = "oauth_clients"

    client_id: Mapped[str] = mapped_column(Text, primary_key=True)
    client_secret_encrypted: Mapped[str | None] = mapped_column(Text, nullable=True)
    client_secret_expires_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    client_id_issued_at: Mapped[int | None] = mapped_column(Integer, nullable=True)
    client_metadata: Mapped[dict] = mapped_column("metadata", JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class McpToolCatalogueEntry(Base):
    """One tool the running MCP server has REGISTERED, as published by that
    server itself at startup (`scripts/run_mcp_server.py` ->
    `core.observability.service.sync_mcp_tool_catalogue`).

    The MCP server's `@mcp_server.tool()` registrations remain the only source of
    truth; this table is how the separately deployed API process learns them
    without importing `app.mcp` (an import-linter contract forbids that). It holds
    tool metadata only -- name, docstring, JSON input schema -- never tenant data,
    so like `permissions` it is platform-wide and not row-level secured. Each sync
    replaces the whole set, so a tool removed from the server disappears here.
    """

    __tablename__ = "mcp_tool_catalogue"

    name: Mapped[str] = mapped_column(Text, primary_key=True)
    description: Mapped[str] = mapped_column(Text, nullable=False, server_default=text("''"))
    input_schema: Mapped[dict] = mapped_column(JSONB, nullable=False)
    synced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

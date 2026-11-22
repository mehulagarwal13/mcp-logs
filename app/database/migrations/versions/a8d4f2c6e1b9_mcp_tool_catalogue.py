"""mcp_tool_catalogue: the MCP server's registered tools, published for the API

Revision ID: a8d4f2c6e1b9
Revises: f1b6d3a8c5e2
Create Date: 2026-11-15

The MCP Tools page showed usage statistics only. The MCP server's tool
registrations are the source of truth for what tools exist, but the API process
cannot import `app.mcp` (import-linter) and runs as a separate deployment. So the
MCP server publishes its own registration here at startup and the API reads it.

Platform-wide tool metadata (name, docstring, JSON input schema) -- no tenant data,
so, like `permissions`, no row-level security. Each sync replaces the whole set.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = 'a8d4f2c6e1b9'
down_revision: str | None = 'f1b6d3a8c5e2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'mcp_tool_catalogue',
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('description', sa.Text(), server_default=sa.text("''"), nullable=False),
        sa.Column('input_schema', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            'synced_at',
            sa.DateTime(timezone=True),
            server_default=sa.text('now()'),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint('name'),
    )


def downgrade() -> None:
    op.drop_table('mcp_tool_catalogue')

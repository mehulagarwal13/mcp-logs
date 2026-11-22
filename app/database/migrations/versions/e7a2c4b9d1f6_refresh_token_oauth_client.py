"""bind refresh tokens to the OAuth client they were issued to

Revision ID: e7a2c4b9d1f6
Revises: b7c1d9e4f2a3
Create Date: 2026-11-15

`refresh_tokens.oauth_client_id` records which MCP OAuth client
(`oauth_clients.client_id`) a refresh token was issued to, so that only that
client can refresh or revoke it. NULL means an ordinary REST session (and every
pre-existing row): such a token can never be presented by an OAuth client, so
already-connected MCP clients re-authorize once after this migration.

Nullable, no default, no backfill, no index (it is never a lookup key -- rows
are found by `token_hash`) -- a metadata-only change on PostgreSQL.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'e7a2c4b9d1f6'
down_revision: str | None = 'b7c1d9e4f2a3'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('refresh_tokens', sa.Column('oauth_client_id', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('refresh_tokens', 'oauth_client_id')

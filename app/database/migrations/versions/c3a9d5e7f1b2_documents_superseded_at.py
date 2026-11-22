"""documents.superseded_at: document version lifecycle marker

Revision ID: c3a9d5e7f1b2
Revises: a4c8e1f3b6d2
Create Date: 2026-11-13 00:00:00.000000

Schema-only. Adds a nullable `superseded_at` timestamp to `documents` plus a
partial index over the *current* rows.

Why: a changed connector document is inserted as a NEW `documents` row with
`version + 1` (`app.ingestion.service._process_one_item`) and the previous
row used to stay fully searchable -- retrieval and the `/knowledge` browse
list had no notion of "current version", and nothing removed the old
version's chunks. `superseded_at IS NULL` now defines "current".

This migration changes no behavior by itself: the column is nullable with no
default and no existing row is modified (so it is a metadata-only change on
PostgreSQL 11+ and takes no long table rewrite). Existing superseded rows are
marked by the separate, tenant-scoped, resumable backfill
`scripts/backfill_superseded_documents.py`, deliberately NOT done here:
`documents` has FORCE ROW LEVEL SECURITY, so a bulk UPDATE issued by a
migration role without tenant context can silently match zero rows.

RLS: `documents` already has a direct `organization_id` policy
(`c7d4e8f19a2b`); a new column needs no policy change.

Not in this migration (pending approval, see the design discussion): the
partial UNIQUE index "one current row per (organization_id, source,
external_id)" and the drop of `uq_documents_org_source_external_hash`, which
is what makes an A -> B -> A content revert insertable. That index can only
be created after the backfill has left exactly one current row per key.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3a9d5e7f1b2'
down_revision: str | None = 'a4c8e1f3b6d2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        'documents',
        sa.Column('superseded_at', sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        'ix_documents_current',
        'documents',
        ['organization_id', 'source', 'external_id'],
        unique=False,
        postgresql_where=sa.text('superseded_at IS NULL'),
    )


def downgrade() -> None:
    op.drop_index(
        'ix_documents_current',
        table_name='documents',
        postgresql_where=sa.text('superseded_at IS NULL'),
    )
    op.drop_column('documents', 'superseded_at')

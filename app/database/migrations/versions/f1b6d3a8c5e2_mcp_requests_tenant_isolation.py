"""mcp_requests: tenant ownership, row-level security, append-only, retention purge

Revision ID: f1b6d3a8c5e2
Revises: e7a2c4b9d1f6
Create Date: 2026-11-15

`mcp_requests` was deliberately left out of the Milestone 10 RLS migration and
carried no `organization_id`: any session as `ekip_app` could read, change and
delete every tenant's rows, and `GET /observability/mcp` aggregated all tenants.

This migration:

1. Adds `organization_id` (nullable FK, ON DELETE CASCADE) and an index on
   `(organization_id, occurred_at)`.
2. Backfills `organization_id` for existing rows -- NON-DESTRUCTIVELY and only
   where it is unambiguous: the row's `identity` is `user:<uuid>` and that user
   belongs to exactly ONE organization today. Everything else (multi-organization
   users, deleted users, `unresolved`, non-user identities) stays NULL. No row is
   deleted or rewritten. (Reading `user_roles`, itself forced-RLS, needs the
   migration role to be a superuser/BYPASSRLS, as for every other migration here;
   otherwise nothing is attributed and every legacy row simply stays NULL.)
3. Enables and FORCES row-level security with per-command policies using
   `NULLIF(current_setting('app.current_organization_id', true), '')::uuid`
   (the NULLIF also makes a connection whose setting was reset to '' by an earlier
   transaction behave as "no context" instead of raising a uuid cast error):
     SELECT / DELETE  -> only rows of the session's own organization
     INSERT           -> own organization, or NULL (a call that failed before any
                         identity existed; such a row is invisible to every tenant)
     UPDATE           -> no policy => denied
4. Revokes UPDATE from `ekip_app`: the log is append-only.
5. Adds `purge_mcp_requests(cutoff)`, a SECURITY DEFINER function (pinned
   search_path, EXECUTE only for `ekip_app`): the single, narrow cross-tenant
   path, used by the operator-run retention job. As with the other definer
   functions, it bypasses RLS only because the owning role is superuser/BYPASSRLS.

Downgrade removes the policies, function and column (attribution is lost) and
restores UPDATE to `ekip_app`; no rows are deleted.
"""
from collections.abc import Sequence

from alembic import op

revision: str = 'f1b6d3a8c5e2'
down_revision: str | None = 'e7a2c4b9d1f6'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_GUC = "NULLIF(current_setting('app.current_organization_id', true), '')::uuid"

BACKFILL_SQL = """
UPDATE mcp_requests AS m
SET organization_id = one.organization_id
FROM (
    SELECT user_id, (array_agg(organization_id))[1] AS organization_id
    FROM user_roles
    GROUP BY user_id
    HAVING count(DISTINCT organization_id) = 1
) AS one
WHERE m.organization_id IS NULL
  AND CASE WHEN m.identity ~ '^user:[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$'
           THEN substring(m.identity from 6)::uuid END = one.user_id
"""

_GUARD_APP_ROLE = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'ekip_app') THEN
        {statement};
    END IF;
END
$$;
"""


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE mcp_requests
            ADD COLUMN organization_id uuid REFERENCES organizations(id) ON DELETE CASCADE
        """
    )
    op.execute(
        "CREATE INDEX ix_mcp_requests_org_occurred_at "
        "ON mcp_requests (organization_id, occurred_at)"
    )
    op.execute(BACKFILL_SQL)

    op.execute("ALTER TABLE mcp_requests ENABLE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE mcp_requests FORCE ROW LEVEL SECURITY")
    op.execute(
        "CREATE POLICY mcp_requests_select ON mcp_requests FOR SELECT "
        f"USING (organization_id = {_GUC})"
    )
    op.execute(
        "CREATE POLICY mcp_requests_insert ON mcp_requests FOR INSERT "
        f"WITH CHECK (organization_id IS NULL OR organization_id = {_GUC})"
    )
    op.execute(
        "CREATE POLICY mcp_requests_delete ON mcp_requests FOR DELETE "
        f"USING (organization_id = {_GUC})"
    )
    op.execute(_GUARD_APP_ROLE.format(statement="REVOKE UPDATE ON mcp_requests FROM ekip_app"))

    op.execute(
        """
        CREATE FUNCTION purge_mcp_requests(cutoff timestamptz)
        RETURNS bigint
        LANGUAGE sql
        SECURITY DEFINER
        SET search_path = public
        AS $$
            WITH deleted AS (
                DELETE FROM mcp_requests WHERE occurred_at < cutoff RETURNING 1
            )
            SELECT count(*) FROM deleted;
        $$
        """
    )
    op.execute("REVOKE ALL ON FUNCTION purge_mcp_requests(timestamptz) FROM PUBLIC")
    op.execute(
        _GUARD_APP_ROLE.format(
            statement="GRANT EXECUTE ON FUNCTION purge_mcp_requests(timestamptz) TO ekip_app"
        )
    )


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS purge_mcp_requests(timestamptz)")
    op.execute("DROP POLICY IF EXISTS mcp_requests_delete ON mcp_requests")
    op.execute("DROP POLICY IF EXISTS mcp_requests_insert ON mcp_requests")
    op.execute("DROP POLICY IF EXISTS mcp_requests_select ON mcp_requests")
    op.execute("ALTER TABLE mcp_requests NO FORCE ROW LEVEL SECURITY")
    op.execute("ALTER TABLE mcp_requests DISABLE ROW LEVEL SECURITY")
    op.execute(_GUARD_APP_ROLE.format(statement="GRANT UPDATE ON mcp_requests TO ekip_app"))
    op.execute("DROP INDEX IF EXISTS ix_mcp_requests_org_occurred_at")
    op.execute("ALTER TABLE mcp_requests DROP COLUMN organization_id")

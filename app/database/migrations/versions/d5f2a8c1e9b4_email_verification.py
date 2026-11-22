"""email verification: users.email_verified_at + email_verifications

Revision ID: d5f2a8c1e9b4
Revises: c3a9d5e7f1b2
Create Date: 2026-11-13 00:00:00.000000

What this migration does, and deliberately does not do:

* ADDs `users.email_verified_at` / `users.email_verified_via`, both NULL.
  **No backfill.** Every existing user keeps `email_verified_at = NULL`
  because nothing that happened before this change is defensible evidence of
  mailbox ownership (signup never checked, SSO email claims were never read,
  invitation tokens were visible to the inviting admin). Those accounts must
  re-verify by email link or be attested by an operator
  (`scripts/attest_legacy_email_verification.py`). See
  docs/security/email-verification-implementation-plan.md section D.
* CREATEs `email_verifications` (pending registrations / invitation
  acceptances / legacy-user re-verification) with a partial unique index so
  there is at most one open challenge per (email, kind), and per-kind CHECK
  constraints so a registration row can never exist without its payload.
* Touches NO existing table other than the two nullable `users` columns,
  no RLS policy, and no GRANT: `ekip_app` already has DML on every table in
  `public` through the default privileges set in `b8f3d6a1c4e7`.
  `email_verifications` is intentionally not RLS-protected (global,
  pre-tenant, like `users`).

Downgrade is LOSSY and exists for emergency rollback only: it drops the
columns and the table, discarding every verification timestamp and every
pending registration. Re-running `upgrade` afterwards leaves all users
unverified again.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd5f2a8c1e9b4'
down_revision: str | None = 'c3a9d5e7f1b2'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('users', sa.Column('email_verified_at', sa.DateTime(timezone=True), nullable=True))
    op.add_column('users', sa.Column('email_verified_via', sa.Text(), nullable=True))

    op.create_table(
        'email_verifications',
        sa.Column('id', postgresql.UUID(as_uuid=True), server_default=sa.text('gen_random_uuid()'), nullable=False),
        sa.Column('kind', sa.Text(), nullable=False),
        sa.Column('email', sa.Text(), nullable=False),
        sa.Column('token_hash', sa.Text(), nullable=False),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('consumed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('last_sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('send_count', sa.Integer(), server_default=sa.text('0'), nullable=False),
        sa.Column('send_window_started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('display_name', sa.Text(), nullable=True),
        sa.Column('password_hash', sa.Text(), nullable=True),
        sa.Column('organization_name', sa.Text(), nullable=True),
        sa.Column('organization_slug', sa.Text(), nullable=True),
        sa.Column('invitation_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.ForeignKeyConstraint(['invitation_id'], ['invitations.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('token_hash'),
        sa.CheckConstraint(
            "kind IN ('registration', 'invitation', 'existing_user')",
            name='ck_email_verifications_kind',
        ),
        sa.CheckConstraint(
            "(kind <> 'registration') OR (display_name IS NOT NULL AND password_hash IS NOT NULL "
            "AND organization_name IS NOT NULL AND organization_slug IS NOT NULL)",
            name='ck_email_verifications_registration_payload',
        ),
        sa.CheckConstraint(
            "(kind <> 'invitation') OR (invitation_id IS NOT NULL AND display_name IS NOT NULL "
            "AND password_hash IS NOT NULL)",
            name='ck_email_verifications_invitation_payload',
        ),
        sa.CheckConstraint(
            "(kind <> 'existing_user') OR (user_id IS NOT NULL)",
            name='ck_email_verifications_existing_user_payload',
        ),
    )
    op.create_index(
        'uq_email_verifications_open_email_kind',
        'email_verifications',
        ['email', 'kind'],
        unique=True,
        postgresql_where=sa.text('consumed_at IS NULL'),
    )
    op.create_index('ix_email_verifications_expires_at', 'email_verifications', ['expires_at'])

    # Narrow RLS-bypass resolver, same shape as `resolve_refresh_token_
    # organization` (d2e5f8a3c1b6): the unauthenticated invitation-accept
    # endpoint has no tenant context yet and `invitations` is FORCE RLS, so
    # without this the invitation can never be found on the real `ekip_app`
    # role. It returns ONLY the owning organization_id for an invitation id;
    # the caller still has to present the single-use token, and every further
    # read runs under that organization's tenant context. EXECUTE reaches
    # `ekip_app` through the default privileges from b8f3d6a1c4e7.
    op.execute(
        """
        CREATE FUNCTION resolve_invitation_organization(invitation_id_arg uuid)
        RETURNS uuid
        LANGUAGE sql
        SECURITY DEFINER
        SET search_path = public
        AS $$
            SELECT organization_id FROM invitations WHERE id = invitation_id_arg;
        $$;
        """
    )


def downgrade() -> None:
    op.execute('DROP FUNCTION IF EXISTS resolve_invitation_organization(uuid)')
    op.drop_index('ix_email_verifications_expires_at', table_name='email_verifications')
    op.drop_index('uq_email_verifications_open_email_kind', table_name='email_verifications')
    op.drop_table('email_verifications')
    op.drop_column('users', 'email_verified_via')
    op.drop_column('users', 'email_verified_at')

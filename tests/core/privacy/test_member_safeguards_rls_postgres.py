"""Member-safeguard (V1/V2) behavior against REAL PostgreSQL, the REAL
migrations, the REAL `list_user_organization_ids` SECURITY DEFINER function and
the production runtime role `ekip_app` (NOSUPERUSER NOBYPASSRLS) with row-level
security FORCED on `user_roles`.

Nothing here recreates a simplified function or schema: the database is the
one `alembic upgrade head` builds. This is the verification the mocked and
`create_all`-based tests cannot give -- in particular that Rule D ("anonymize
the global account only if the person has no membership in any other
organization") really sees memberships in OTHER organizations when called
through a session whose tenant context is scoped to one organization.

Skipped unless BOTH are set and point at a LOCAL, disposable, MIGRATED database
(same contract as `tests/core/auth/test_email_verification_postgres.py`):

    EKIP_TEST_POSTGRES_URL   superuser/owner URL -- used only to seed rows,
                             inspect rows past RLS, and clean up
    EKIP_TEST_APP_URL        the `ekip_app` URL -- every service call under
                             test runs through THIS role

Every row is tagged with a unique token and deleted with targeted DELETEs.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from datetime import UTC, datetime
from urllib.parse import urlsplit

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.exceptions import ConflictError, NotFoundError, ValidationError
from app.core.privacy import service as privacy_service
from app.core.users import service as users_service
from app.database.session import set_tenant_context
from app.shared.schemas import ActorKind, Identity

_SUPER_URL = os.environ.get("EKIP_TEST_POSTGRES_URL", "")
_APP_URL = os.environ.get("EKIP_TEST_APP_URL", "")


def _is_local(url: str) -> bool:
    return (urlsplit(url.replace("+asyncpg", "")).hostname or "") in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


pytestmark = pytest.mark.skipif(
    not (_SUPER_URL and _APP_URL and _is_local(_SUPER_URL) and _is_local(_APP_URL)),
    reason="needs EKIP_TEST_POSTGRES_URL and EKIP_TEST_APP_URL pointing at a LOCAL migrated DB",
)

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _engine(url: str):
    clean = url.replace("?sslmode=disable", "")
    return create_async_engine(clean, poolclass=NullPool, connect_args={"ssl": False})


class _Env:
    pass


@pytest_asyncio.fixture
async def env():
    super_engine, app_engine = _engine(_SUPER_URL), _engine(_APP_URL)
    tag = uuid.uuid4().hex[:10]
    app_sessions = async_sessionmaker(app_engine, expire_on_commit=False)

    async with super_engine.connect() as conn:
        present = (await conn.execute(text("SELECT to_regclass('public.user_roles')"))).scalar()
        fn = (
            await conn.execute(
                text("SELECT 1 FROM pg_proc WHERE proname = 'list_user_organization_ids'")
            )
        ).scalar()
    if present is None or fn is None:
        pytest.skip("migrations not applied: run `alembic upgrade head` on this database")

    e = _Env()
    e.tag = tag
    e.super_engine = super_engine
    e.app_engine = app_engine

    async def sup(sql, **params):
        async with super_engine.begin() as conn:
            res = await conn.execute(text(sql), params)
            return res.fetchall() if res.returns_rows else None

    async def as_app(org_id, fn):
        """One request-shaped unit of work as `ekip_app` with the tenant GUC set,
        like `get_current_identity` does; commit on success, roll back on error."""
        async with app_sessions() as session:
            try:
                await set_tenant_context(session, org_id)
                result = await fn(session)
                await session.commit()
                return result
            except BaseException:
                await session.rollback()
                raise

    e.sup, e.as_app = sup, as_app

    # --- seed (as the superuser; the app role cannot write across tenants) ---
    e.admin_role, e.member_role = uuid.uuid4(), uuid.uuid4()
    perm_id = (
        await sup(
            "INSERT INTO permissions (id, code, description) VALUES (gen_random_uuid(), "
            "'tenancy:manage', 'x') ON CONFLICT (code) DO UPDATE SET code = EXCLUDED.code "
            "RETURNING id"
        )
    )[0][0]
    await sup(
        "INSERT INTO roles (id, name) VALUES (:a, :an), (:m, :mn)",
        a=e.admin_role, an=f"adm-{tag}", m=e.member_role, mn=f"mem-{tag}",
    )
    await sup(
        "INSERT INTO role_permissions (role_id, permission_id) VALUES (:r, :p)",
        r=e.admin_role, p=perm_id,
    )

    async def org(name):
        oid = uuid.uuid4()
        await sup(
            "INSERT INTO organizations (id, name, slug, status) VALUES (:i, :n, :s, 'active')",
            i=oid, n=f"rls-{tag}-{name}", s=f"rls-{tag}-{name}",
        )
        return oid

    async def user(label, roles):
        uid = uuid.uuid4()
        await sup(
            "INSERT INTO users (id, email, display_name, password_hash, is_active, "
            "email_verified_at) VALUES (:i, :e, :n, 'hash', true, :v)",
            i=uid, e=f"rls-{tag}-{label}@example.invalid", n=label, v=_NOW,
        )
        for org_id, role_id in roles.items():
            await sup(
                "INSERT INTO user_roles (user_id, organization_id, role_id) VALUES (:u, :o, :r)",
                u=uid, o=org_id, r=role_id,
            )
        return uid

    e.org, e.user = org, user

    async def user_row(uid):
        rows = await sup(
            "SELECT email, display_name, password_hash, is_active FROM users WHERE id = :i", i=uid
        )
        return tuple(rows[0])

    async def role_orgs(uid):
        rows = await sup("SELECT organization_id FROM user_roles WHERE user_id = :u", u=uid)
        return {r[0] for r in rows}

    e.user_row, e.role_orgs = user_row, role_orgs
    yield e

    # --- targeted cleanup of this test's own rows (never TRUNCATE/DROP) ---
    like = f"rls-{tag}-%"
    for sql in (
        "DELETE FROM audit_logs WHERE organization_id IN "
        "(SELECT id FROM organizations WHERE slug LIKE :l)",
        "DELETE FROM refresh_tokens WHERE user_id IN (SELECT id FROM users WHERE email LIKE :l "
        "OR email LIKE 'deleted-user-%')",
        "DELETE FROM user_roles WHERE organization_id IN "
        "(SELECT id FROM organizations WHERE slug LIKE :l)",
        "DELETE FROM invitations WHERE organization_id IN "
        "(SELECT id FROM organizations WHERE slug LIKE :l)",
        "DELETE FROM projects WHERE organization_id IN "
        "(SELECT id FROM organizations WHERE slug LIKE :l)",
        "DELETE FROM organizations WHERE slug LIKE :l",
        "DELETE FROM users WHERE email LIKE :l",
    ):
        with contextlib.suppress(Exception):  # disposable database; leftovers are harmless
            await sup(sql, l=like)
    await super_engine.dispose()
    await app_engine.dispose()


def _actor(org_id, user_id):
    return Identity(
        kind=ActorKind.USER,
        subject=str(user_id or uuid.uuid4()),
        organization_id=org_id,
        user_id=user_id,
        permissions=frozenset({"tenancy:manage"}),
    )


# --- the function and the role, as actually configured ----------------------


async def test_runtime_role_is_subject_to_rls(env):
    async def check(session):
        return (
            await session.execute(
                text(
                    "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
                )
            )
        ).one()

    superuser, bypass = await env.as_app(uuid.uuid4(), check)
    assert (superuser, bypass) == (False, False)


async def test_function_is_security_definer_owned_by_a_role_that_bypasses_rls(env):
    """The precondition the whole of Rule D depends on: under FORCE RLS a table
    owner is only exempt if it is a superuser or has BYPASSRLS."""
    rows = await env.sup(
        "SELECT p.prosecdef, r.rolsuper OR r.rolbypassrls "
        "FROM pg_proc p JOIN pg_roles r ON r.oid = p.proowner "
        "WHERE p.proname = 'list_user_organization_ids'"
    )
    assert rows and rows[0][0] is True, "function must be SECURITY DEFINER"
    assert rows[0][1] is True, "function owner must bypass RLS or Rule D cannot see other orgs"


async def test_function_sees_every_organization_while_plain_queries_see_only_one(env):
    org_a, org_b = await env.org("a"), await env.org("b")
    shared = await env.user("shared", {org_a: env.member_role, org_b: env.member_role})

    async def probe(session):
        plain = set(
            (
                await session.execute(
                    text("SELECT organization_id FROM user_roles WHERE user_id = :u"),
                    {"u": shared},
                )
            ).scalars()
        )
        via_function = set(await users_service.list_organizations_for_login(session, shared))
        return plain, via_function

    plain, via_function = await env.as_app(org_a, probe)
    assert plain == {org_a}  # RLS is in force: the other organization is invisible
    assert via_function == {org_a, org_b}  # the real function sees past it


# --- the safeguards, end to end under RLS ------------------------------------


async def test_cross_tenant_target_is_refused_and_untouched_under_rls(env):
    org_a, org_b = await env.org("a"), await env.org("b")
    admin_a = await env.user("adminA", {org_a: env.admin_role})
    await env.user("adminA2", {org_a: env.admin_role})
    victim = await env.user("victim", {org_b: env.member_role})
    before = await env.user_row(victim)

    for call in (
        lambda s: privacy_service.execute_user_data_deletion(s, _actor(org_a, admin_a), victim),
        lambda s: privacy_service.plan_user_data_deletion(s, _actor(org_a, admin_a), victim),
    ):
        with pytest.raises(NotFoundError):
            await env.as_app(org_a, call)

    assert await env.user_row(victim) == before
    assert await env.role_orgs(victim) == {org_b}


async def test_multi_org_person_keeps_global_account_under_rls(env):
    org_a, org_b = await env.org("a"), await env.org("b")
    admin_a = await env.user("adminA", {org_a: env.admin_role})
    shared = await env.user("shared", {org_a: env.member_role, org_b: env.member_role})
    before = await env.user_row(shared)

    result = await env.as_app(
        org_a,
        lambda s: privacy_service.execute_user_data_deletion(s, _actor(org_a, admin_a), shared),
    )

    assert result.status == "completed"
    assert next(s for s in result.steps if s.category == "user_record").action == "retain"
    assert await env.role_orgs(shared) == {org_b}
    assert await env.user_row(shared) == before  # email, name, password, is_active untouched


async def test_single_org_person_is_anonymized_under_rls(env):
    org_a = await env.org("a")
    admin_a = await env.user("adminA", {org_a: env.admin_role})
    member = await env.user("member", {org_a: env.member_role})

    result = await env.as_app(
        org_a,
        lambda s: privacy_service.execute_user_data_deletion(s, _actor(org_a, admin_a), member),
    )

    assert result.status == "completed"
    email, name, password_hash, is_active = await env.user_row(member)
    assert email.endswith("@deleted.invalid")
    assert (name, password_hash, is_active) == ("Deleted User", None, False)
    assert await env.role_orgs(member) == set()


async def test_self_and_last_admin_refused_under_rls(env):
    org = await env.org("a")
    sole = await env.user("sole", {org: env.admin_role})

    with pytest.raises(ValidationError) as self_err:
        await env.as_app(
            org,
            lambda s: privacy_service.execute_user_data_deletion(s, _actor(org, sole), sole),
        )
    assert self_err.value.error_code == "privacy.cannot_delete_self"

    with pytest.raises(ConflictError) as last_err:
        await env.as_app(
            org,
            lambda s: privacy_service.execute_user_data_deletion(s, _actor(org, None), sole),
        )
    assert last_err.value.error_code == "org.last_admin"
    assert await env.role_orgs(sole) == {org}


async def test_failed_final_step_is_atomic_and_audited_under_rls(env, monkeypatch):
    """The savepoint behavior as `ekip_app` with FORCED RLS: a real database
    error in the final role deletion rolls back the anonymization too, the
    tenant context survives the rollback (the audit insert is RLS-checked and
    still succeeds), and the same request then completes."""
    org = await env.org("atomic")
    admin = await env.user("admin", {org: env.admin_role})
    member = await env.user("member", {org: env.member_role})
    before = await env.user_row(member)

    real = privacy_service.repository.delete_user_roles
    state = {"fail": True}

    async def delete_roles(session, user_id, organization_id):
        if state["fail"]:
            await session.execute(text("SELECT 1/0"))
        return await real(session, user_id, organization_id)

    monkeypatch.setattr(privacy_service.repository, "delete_user_roles", delete_roles)

    first = await env.as_app(
        org, lambda s: privacy_service.execute_user_data_deletion(s, _actor(org, admin), member)
    )
    assert first.status == "partially_completed"
    assert await env.user_row(member) == before
    assert await env.role_orgs(member) == {org}
    audited = await env.sup(
        "SELECT count(*) FROM audit_logs WHERE organization_id = :o "
        "AND metadata->>'status' = 'partially_completed'",
        o=org,
    )
    assert audited[0][0] == 1

    state["fail"] = False
    second = await env.as_app(
        org, lambda s: privacy_service.execute_user_data_deletion(s, _actor(org, admin), member)
    )
    assert second.status == "completed"
    assert await env.role_orgs(member) == set()

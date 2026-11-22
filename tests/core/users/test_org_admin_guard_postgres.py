"""Real-PostgreSQL tests for the member-management remediation (V1 + V2).

The mocked tests prove the decision logic. These prove what a mock cannot:
the real SQL, the real `SELECT ... FOR UPDATE` / advisory-lock behavior under
concurrent transactions, and that a cross-tenant `users` row is genuinely left
untouched in the database.

SKIPPED by default. Set `EKIP_TEST_POSTGRES_URL` to a LOCAL, disposable
PostgreSQL (pgvector installed), e.g.

    EKIP_TEST_POSTGRES_URL=postgresql://postgres:@127.0.0.1:5432/postgres

Safety: host must be localhost/127.0.0.1/::1; every test builds its own
uniquely named schema and drops it afterwards. Never point this at a shared
or production database.

Scope note: tables come from `Base.metadata.create_all`, so RLS policies are
NOT present (tenant isolation by RLS is covered by `scripts/verify_rls_
isolation.py`). That makes these tests *stricter* for V1: the application-
level membership check is the only thing standing between an admin and another
tenant's user row, exactly the situation for the global `users` table, which
has no RLS even in production. `list_user_organization_ids` (a SECURITY
DEFINER function in production) is recreated in the test schema.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

_RAW_URL = os.environ.get("EKIP_TEST_POSTGRES_URL", "")


def _local_asyncpg_url() -> str | None:
    if not _RAW_URL:
        return None
    url = make_url(_RAW_URL)
    if (url.host or "") not in {"localhost", "127.0.0.1", "::1"}:
        return None
    return url.set(drivername="postgresql+asyncpg").render_as_string(hide_password=False)


_URL = _local_asyncpg_url()

pytestmark = pytest.mark.skipif(
    _URL is None,
    reason="set EKIP_TEST_POSTGRES_URL to a LOCAL disposable PostgreSQL (see module docstring)",
)

from app.core.exceptions import (  # noqa: E402
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from app.core.privacy import service as privacy_service  # noqa: E402
from app.core.users import service as users_service  # noqa: E402
from app.database.models import (  # noqa: E402,F401  (register every table on Base.metadata)
    agent_models,
    auth_models,
    core_models,
    graph_models,
    ingestion_models,
    mcp_models,
    memory_models,
    pattern_models,
    retrieval_models,
    tenancy_models,
)
from app.database.session import Base, set_tenant_context  # noqa: E402
from app.shared.schemas import ActorKind, Identity  # noqa: E402

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


class _Env:
    pass


@pytest_asyncio.fixture
async def env():
    schema = f"adm_{uuid.uuid4().hex[:12]}"
    admin = create_async_engine(_URL)
    async with admin.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    engine = create_async_engine(
        _URL,
        pool_size=8,
        connect_args={"server_settings": {"search_path": f'"{schema}",public'}},
    )
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Production has this as a SECURITY DEFINER function
            # (f6a7b8c9d0e1); same body, created in the test schema.
            await conn.execute(
                text(
                    """
                    CREATE FUNCTION list_user_organization_ids(user_id_arg uuid)
                    RETURNS TABLE(organization_id uuid) LANGUAGE sql AS $$
                        SELECT DISTINCT organization_id FROM user_roles WHERE user_id = user_id_arg;
                    $$;
                    """
                )
            )
        env = _Env()
        env.engine = engine
        env.factory = async_sessionmaker(engine, expire_on_commit=False)

        async with env.factory() as s:
            admin_perm, other_perm = uuid.uuid4(), uuid.uuid4()
            env.admin_role, env.member_role = uuid.uuid4(), uuid.uuid4()
            await s.execute(
                text("INSERT INTO permissions (id, code, description) VALUES (:i, :c, 'x')"),
                {"i": admin_perm, "c": "tenancy:manage"},
            )
            await s.execute(
                text("INSERT INTO permissions (id, code, description) VALUES (:i, :c, 'x')"),
                {"i": other_perm, "c": "incident:read"},
            )
            await s.execute(
                text("INSERT INTO roles (id, name) VALUES (:i, 'admin'), (:j, 'member')"),
                {"i": env.admin_role, "j": env.member_role},
            )
            await s.execute(
                text("INSERT INTO role_permissions (role_id, permission_id) VALUES (:r, :p)"),
                {"r": env.admin_role, "p": admin_perm},
            )
            await s.execute(
                text("INSERT INTO role_permissions (role_id, permission_id) VALUES (:r, :p)"),
                {"r": env.member_role, "p": other_perm},
            )
            await s.commit()
        yield env
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()


async def _org(env, slug: str) -> uuid.UUID:
    org_id = uuid.uuid4()
    async with env.factory() as s:
        await s.execute(
            text(
                "INSERT INTO organizations (id, name, slug, status) VALUES (:i, :n, :n, 'active')"
            ),
            {"i": org_id, "n": f"{slug}-{org_id.hex[:6]}"},
        )
        await s.commit()
    return org_id


async def _user(env, label: str, *, orgs: dict[uuid.UUID, uuid.UUID]) -> uuid.UUID:
    """Create a verified, active user holding `role_id` in each `{org: role}`."""
    user_id = uuid.uuid4()
    async with env.factory() as s:
        await s.execute(
            text(
                "INSERT INTO users (id, email, display_name, password_hash, is_active, "
                "email_verified_at) VALUES (:i, :e, :n, 'hash', true, :v)"
            ),
            {"i": user_id, "e": f"{label}-{user_id.hex[:8]}@example.com", "n": label, "v": _NOW},
        )
        for org_id, role_id in orgs.items():
            await s.execute(
                text(
                    "INSERT INTO user_roles (user_id, organization_id, role_id) "
                    "VALUES (:u, :o, :r)"
                ),
                {"u": user_id, "o": org_id, "r": role_id},
            )
        await s.commit()
    return user_id


def _actor(org_id: uuid.UUID, user_id: uuid.UUID | None) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(user_id or uuid.uuid4()),
        organization_id=org_id,
        user_id=user_id,
        permissions=frozenset({"tenancy:manage"}),
    )


async def _delete_as(env, actor: Identity, target: uuid.UUID):
    """Run the real execute path in its own session and COMMIT, like a request."""
    async with env.factory() as s:
        await set_tenant_context(s, actor.organization_id)
        try:
            result = await privacy_service.execute_user_data_deletion(s, actor, target)
        except BaseException:
            await s.rollback()
            raise
        await s.commit()
        return result


async def _user_row(env, user_id: uuid.UUID) -> dict:
    async with env.factory() as s:
        row = (
            await s.execute(
                text(
                    "SELECT email, display_name, password_hash, is_active "
                    "FROM users WHERE id = :i"
                ),
                {"i": user_id},
            )
        ).one()
        return dict(row._mapping)


async def _role_orgs(env, user_id: uuid.UUID) -> set[uuid.UUID]:
    async with env.factory() as s:
        rows = await s.execute(
            text("SELECT organization_id FROM user_roles WHERE user_id = :u"), {"u": user_id}
        )
        return set(rows.scalars().all())


# --- V1: cross-tenant ------------------------------------------------------


async def test_admin_of_one_org_cannot_touch_a_user_of_another_org(env):
    org_a, org_b = await _org(env, "a"), await _org(env, "b")
    admin_a = await _user(env, "adminA", orgs={org_a: env.admin_role})
    await _user(env, "adminA2", orgs={org_a: env.admin_role})
    victim = await _user(env, "victim", orgs={org_b: env.member_role})
    before = await _user_row(env, victim)

    with pytest.raises(NotFoundError):
        await _delete_as(env, _actor(org_a, admin_a), victim)

    assert await _user_row(env, victim) == before  # NOT anonymized, still active, password intact
    assert await _role_orgs(env, victim) == {org_b}


async def test_unknown_user_gives_the_same_error_as_another_orgs_user(env):
    org_a, org_b = await _org(env, "a"), await _org(env, "b")
    admin_a = await _user(env, "adminA", orgs={org_a: env.admin_role})
    victim = await _user(env, "victim", orgs={org_b: env.member_role})
    with pytest.raises(NotFoundError) as other_org:
        await _delete_as(env, _actor(org_a, admin_a), victim)
    with pytest.raises(NotFoundError) as unknown:
        await _delete_as(env, _actor(org_a, admin_a), uuid.uuid4())
    assert other_org.value.error_code == unknown.value.error_code


async def test_multi_org_user_loses_this_org_but_keeps_the_global_account(env):
    org_a, org_b = await _org(env, "a"), await _org(env, "b")
    admin_a = await _user(env, "adminA", orgs={org_a: env.admin_role})
    shared = await _user(env, "shared", orgs={org_a: env.member_role, org_b: env.member_role})
    before = await _user_row(env, shared)

    result = await _delete_as(env, _actor(org_a, admin_a), shared)

    assert result.status == "completed"
    assert await _role_orgs(env, shared) == {org_b}  # removed from A only
    assert await _user_row(env, shared) == before  # global account untouched
    step = next(s for s in result.steps if s.category == "user_record")
    assert step.action == "retain"


async def test_single_org_member_is_fully_anonymized(env):
    org_a = await _org(env, "a")
    admin_a = await _user(env, "adminA", orgs={org_a: env.admin_role})
    member = await _user(env, "member", orgs={org_a: env.member_role})

    result = await _delete_as(env, _actor(org_a, admin_a), member)

    assert result.status == "completed"
    row = await _user_row(env, member)
    assert row["is_active"] is False
    assert row["password_hash"] is None
    assert row["email"].endswith("@deleted.invalid")
    assert await _role_orgs(env, member) == set()


# --- V2: self and last admin -----------------------------------------------


async def test_admin_cannot_delete_themselves_even_with_another_admin(env):
    org = await _org(env, "a")
    me = await _user(env, "me", orgs={org: env.admin_role})
    await _user(env, "other", orgs={org: env.admin_role})
    with pytest.raises(ValidationError) as excinfo:
        await _delete_as(env, _actor(org, me), me)
    assert excinfo.value.error_code == "privacy.cannot_delete_self"
    assert await _role_orgs(env, me) == {org}


async def test_sole_admin_cannot_be_deleted_by_a_non_user_actor(env):
    """A service/agent identity (no user_id) holding tenancy:manage is not
    itself an admin member, so only the last-admin rule protects the org."""
    org = await _org(env, "a")
    sole = await _user(env, "sole", orgs={org: env.admin_role})
    with pytest.raises(ConflictError) as excinfo:
        await _delete_as(env, _actor(org, None), sole)
    assert excinfo.value.error_code == "org.last_admin"
    assert await _role_orgs(env, sole) == {org}
    assert (await _user_row(env, sole))["is_active"] is True


async def test_an_admin_can_be_deleted_when_another_admin_remains(env):
    org = await _org(env, "a")
    keeper = await _user(env, "keeper", orgs={org: env.admin_role})
    leaver = await _user(env, "leaver", orgs={org: env.admin_role})
    result = await _delete_as(env, _actor(org, keeper), leaver)
    assert result.status == "completed"
    assert await _role_orgs(env, keeper) == {org}


async def test_inactive_admin_does_not_count_as_a_remaining_admin(env):
    org = await _org(env, "a")
    sole_active = await _user(env, "active", orgs={org: env.admin_role})
    dormant = await _user(env, "dormant", orgs={org: env.admin_role})
    async with env.factory() as s:
        await s.execute(text("UPDATE users SET is_active = false WHERE id = :i"), {"i": dormant})
        await s.commit()
    with pytest.raises(ConflictError):
        await _delete_as(env, _actor(org, None), sole_active)


# --- V2: concurrency -------------------------------------------------------


async def test_two_admins_deleting_each_other_concurrently_leave_exactly_one(env):
    """Repeat the race several times, on fresh organizations each time."""
    for _ in range(6):
        org = await _org(env, "race")
        p = await _user(env, "p", orgs={org: env.admin_role})
        q = await _user(env, "q", orgs={org: env.admin_role})

        outcomes = await asyncio.gather(
            _delete_as(env, _actor(org, p), q),
            _delete_as(env, _actor(org, q), p),
            return_exceptions=True,
        )

        successes = [o for o in outcomes if not isinstance(o, BaseException)]
        refusals = [o for o in outcomes if isinstance(o, ConflictError)]
        assert len(successes) == 1, outcomes
        assert len(refusals) == 1, outcomes
        # The invariant that matters: the organization still has an admin.
        async with env.factory() as s:
            admins = (
                await s.execute(
                    text(
                        "SELECT count(*) FROM user_roles WHERE organization_id = :o "
                        "AND role_id = :r"
                    ),
                    {"o": org, "r": env.admin_role},
                )
            ).scalar_one()
        assert admins == 1


async def test_second_deleter_blocks_until_the_first_commits_then_re_evaluates(env):
    """Deterministic interleaving (no reliance on timing luck): transaction 1
    holds the admin lock mid-deletion; transaction 2 must WAIT, and once
    transaction 1 commits, transaction 2 sees the new state and is refused."""
    org = await _org(env, "interleave")
    p = await _user(env, "p", orgs={org: env.admin_role})
    q = await _user(env, "q", orgs={org: env.admin_role})

    s1 = env.factory()
    await set_tenant_context(s1, org)
    # T1 takes and holds the admin lock.
    await users_service.assert_organization_keeps_an_admin(s1, org, q, lock=True)

    t2 = asyncio.create_task(_delete_as(env, _actor(org, q), p))  # T2: q deletes p
    await asyncio.sleep(1.0)
    assert not t2.done(), "second deleter did not wait for the admin lock"

    # T1 finishes: removes q's admin role and commits.
    await s1.execute(
        text("DELETE FROM user_roles WHERE user_id = :u AND organization_id = :o"),
        {"u": q, "o": org},
    )
    await s1.commit()
    await s1.close()

    with pytest.raises(ConflictError):  # p is now the last admin
        await asyncio.wait_for(t2, timeout=10)
    assert await _role_orgs(env, p) == {org}


async def test_a_concurrent_demotion_of_a_locked_admin_row_waits(env):
    """A writer that does NOT use our helper (e.g. a future naive role
    change) still cannot slip in mid-check: the FOR UPDATE row lock makes its
    UPDATE wait. (It still needs to call the helper itself to make a *correct*
    decision afterwards -- see `assert_organization_keeps_an_admin`.)"""
    org = await _org(env, "demote")
    p = await _user(env, "p", orgs={org: env.admin_role})
    await _user(env, "q", orgs={org: env.admin_role})

    s1 = env.factory()
    await set_tenant_context(s1, org)
    await users_service.assert_organization_keeps_an_admin(s1, org, p, lock=True)

    async def naive_demote():
        async with env.factory() as s2:
            await s2.execute(
                text(
                    "UPDATE user_roles SET role_id = :m "
                    "WHERE user_id = :u AND organization_id = :o"
                ),
                {"m": env.member_role, "u": p, "o": org},
            )
            await s2.commit()

    t2 = asyncio.create_task(naive_demote())
    await asyncio.sleep(1.0)
    assert not t2.done(), "naive demotion was not blocked by the row lock"
    await s1.commit()
    await s1.close()
    await asyncio.wait_for(t2, timeout=10)


# --- Partial-deletion recovery (membership removed last) --------------------


async def test_failed_cleanup_keeps_membership_and_a_retry_completes(env, monkeypatch):
    """A non-database failure in a middle step does not roll the request back,
    so the partial state IS committed. The person must still be a member (and
    not anonymized) so the same request can be repeated and finish."""
    org = await _org(env, "recover")
    admin = await _user(env, "admin", orgs={org: env.admin_role})
    member = await _user(env, "member", orgs={org: env.member_role})
    before = await _user_row(env, member)

    real = privacy_service.repository.delete_project_memberships
    state = {"fail": True}

    async def flaky(*args, **kwargs):
        if state["fail"]:
            raise RuntimeError("simulated transient cleanup failure")
        return await real(*args, **kwargs)

    monkeypatch.setattr(privacy_service.repository, "delete_project_memberships", flaky)

    first = await _delete_as(env, _actor(org, admin), member)
    assert first.status == "partially_completed"
    assert await _role_orgs(env, member) == {org}  # still a member
    assert await _user_row(env, member) == before  # NOT anonymized

    state["fail"] = False
    second = await _delete_as(env, _actor(org, admin), member)
    assert second.status == "completed"
    assert await _role_orgs(env, member) == set()
    assert (await _user_row(env, member))["is_active"] is False

    # Cross-tenant protection is intact: now a non-member, the same call is a 404.
    with pytest.raises(NotFoundError):
        await _delete_as(env, _actor(org, admin), member)


# --- Multi-organization concurrency (global user-row lock) -----------------


async def test_second_org_deleting_the_same_person_waits_and_sees_the_first_result(env):
    """Deterministic: transaction 1 (org A's deletion) holds the person's row
    lock; org B's deletion must wait; after A commits (A's role gone), B
    re-reads the person's memberships, finds none left, and anonymizes. Without
    the lock both could decide 'the other org still has them' and nobody would."""
    org_a, org_b = await _org(env, "a"), await _org(env, "b")
    await _user(env, "adminA", orgs={org_a: env.admin_role})
    admin_b = await _user(env, "adminB", orgs={org_b: env.admin_role})
    shared = await _user(env, "shared", orgs={org_a: env.member_role, org_b: env.member_role})

    s1 = env.factory()
    await set_tenant_context(s1, org_a)
    assert await users_service.lock_user_for_deletion(s1, shared)  # A is "in progress"

    t2 = asyncio.create_task(_delete_as(env, _actor(org_b, admin_b), shared))
    await asyncio.sleep(1.0)
    assert not t2.done(), "org B's deletion did not wait for the person's row lock"

    # A finishes: keeps the account (B still has them), removes A's role, commits.
    await s1.execute(
        text("DELETE FROM user_roles WHERE user_id = :u AND organization_id = :o"),
        {"u": shared, "o": org_a},
    )
    await s1.commit()
    await s1.close()

    result = await asyncio.wait_for(t2, timeout=10)
    assert result.status == "completed"
    assert await _role_orgs(env, shared) == set()
    row = await _user_row(env, shared)
    assert row["is_active"] is False  # nobody left: now correctly anonymized
    assert row["email"].endswith("@deleted.invalid")


async def test_concurrent_deletions_of_the_same_person_by_two_orgs_end_consistent(env):
    for _ in range(5):
        org_a, org_b = await _org(env, "a"), await _org(env, "b")
        admin_a = await _user(env, "adminA", orgs={org_a: env.admin_role})
        admin_b = await _user(env, "adminB", orgs={org_b: env.admin_role})
        shared = await _user(env, "shared", orgs={org_a: env.member_role, org_b: env.member_role})

        outcomes = await asyncio.gather(
            _delete_as(env, _actor(org_a, admin_a), shared),
            _delete_as(env, _actor(org_b, admin_b), shared),
            return_exceptions=True,
        )
        assert not any(isinstance(o, BaseException) for o in outcomes), outcomes
        assert await _role_orgs(env, shared) == set()
        row = await _user_row(env, shared)
        # The invariant: no memberships left => the account is anonymized.
        assert row["is_active"] is False, "orphaned, un-anonymized account left behind"
        assert row["email"].endswith("@deleted.invalid")


# --- Race with membership creation (assign_role) ---------------------------


async def test_role_grant_waits_for_an_in_progress_deletion_then_is_refused(env):
    """Deletion decided to anonymize and holds the person's row lock; an
    invitation-accept / SSO `assign_role` for another org must wait, and once
    the account is anonymized it is refused (no membership on a dead account)."""
    org_a, org_c = await _org(env, "a"), await _org(env, "c")
    person = await _user(env, "person", orgs={org_a: env.member_role})

    s1 = env.factory()
    assert await users_service.lock_user_for_deletion(s1, person)  # deletion in progress

    async def grant():
        async with env.factory() as s2:
            await users_service.assign_role(
                s2, user_id=person, organization_id=org_c, role_id=env.member_role
            )
            await s2.commit()

    t2 = asyncio.create_task(grant())
    await asyncio.sleep(1.0)
    assert not t2.done(), "role grant did not wait for the person's row lock"

    await s1.execute(text("UPDATE users SET is_active = false WHERE id = :u"), {"u": person})
    await s1.commit()
    await s1.close()

    with pytest.raises(PermissionDeniedError) as excinfo:
        await asyncio.wait_for(t2, timeout=10)
    assert excinfo.value.error_code == "user.inactive"
    assert await _role_orgs(env, person) == {org_a}  # nothing granted in org C


async def test_deletion_waits_for_an_in_flight_role_grant_and_keeps_the_account(env):
    """Opposite order: the grant for org C is in flight (uncommitted) when org
    B's deletion starts. The deletion must wait, then see the new membership
    and RETAIN the global account instead of anonymizing it."""
    org_b, org_c = await _org(env, "b"), await _org(env, "c")
    admin_b = await _user(env, "adminB", orgs={org_b: env.admin_role})
    person = await _user(env, "person", orgs={org_b: env.member_role})
    before = await _user_row(env, person)

    s1 = env.factory()
    await users_service.assign_role(
        s1, user_id=person, organization_id=org_c, role_id=env.member_role
    )  # holds the shared lock, uncommitted

    t2 = asyncio.create_task(_delete_as(env, _actor(org_b, admin_b), person))
    await asyncio.sleep(1.0)
    assert not t2.done(), "deletion did not wait for the in-flight role grant"

    await s1.commit()
    await s1.close()

    result = await asyncio.wait_for(t2, timeout=10)
    assert result.status == "completed"
    assert next(s for s in result.steps if s.category == "user_record").action == "retain"
    assert await _role_orgs(env, person) == {org_c}  # B removed, C kept
    assert await _user_row(env, person) == before  # global account untouched


# --- Final two steps are atomic: a failure rolls BOTH back ------------------


async def _seed_refresh_token(env, org: uuid.UUID, user_id: uuid.UUID) -> None:
    async with env.factory() as s:
        await s.execute(
            text(
                "INSERT INTO refresh_tokens (id, user_id, organization_id, family_id, "
                "token_hash, expires_at) VALUES (:i, :u, :o, :f, :h, :e)"
            ),
            {
                "i": uuid.uuid4(), "u": user_id, "o": org, "f": uuid.uuid4(),
                "h": uuid.uuid4().hex, "e": datetime(2099, 1, 1, tzinfo=UTC),
            },
        )
        await s.commit()


async def _count(env, sql: str, **params) -> int:
    async with env.factory() as s:
        return int((await s.execute(text(sql), params)).scalar_one())


async def _failed_final_step_scenario(env, monkeypatch, failing_delete):
    org = await _org(env, "atomic")
    admin = await _user(env, "admin", orgs={org: env.admin_role})
    member = await _user(env, "member", orgs={org: env.member_role})
    await _seed_refresh_token(env, org, member)
    before = await _user_row(env, member)

    real = privacy_service.repository.delete_user_roles
    state = {"fail": True}

    async def delete_roles(session, user_id, organization_id):
        if state["fail"]:
            return await failing_delete(session, user_id, organization_id)
        return await real(session, user_id, organization_id)

    monkeypatch.setattr(privacy_service.repository, "delete_user_roles", delete_roles)

    # --- the failing run: returns a result (NOT an exception), request commits ---
    first = await _delete_as(env, _actor(org, admin), member)
    assert first.status == "partially_completed"
    steps = {s.category: s for s in first.steps}
    assert steps["user_record"].succeeded is False
    assert steps["organization_roles"].succeeded is False

    # Both final operations were rolled back together.
    assert await _user_row(env, member) == before  # account NOT anonymized
    assert await _role_orgs(env, member) == {org}  # membership intact
    # Earlier cleanup stayed committed.
    assert await _count(
        env, "SELECT count(*) FROM refresh_tokens WHERE user_id = :u", u=member
    ) == 0
    # The outer transaction stayed usable: the audit record of the partial failure exists.
    assert await _count(
        env,
        "SELECT count(*) FROM audit_logs WHERE organization_id = :o "
        "AND action = 'privacy.user_data_deleted' "
        "AND metadata->>'status' = 'partially_completed'",
        o=org,
    ) == 1

    # --- the retry: the same request now completes ---
    state["fail"] = False
    second = await _delete_as(env, _actor(org, admin), member)
    assert second.status == "completed"
    assert await _role_orgs(env, member) == set()
    row = await _user_row(env, member)
    assert row["is_active"] is False
    assert row["email"].endswith("@deleted.invalid")
    assert await _count(
        env,
        "SELECT count(*) FROM audit_logs WHERE organization_id = :o "
        "AND action = 'privacy.user_data_deleted'",
        o=org,
    ) == 2


async def test_final_role_deletion_python_error_rolls_back_the_anonymization(env, monkeypatch):
    async def python_error(session, user_id, organization_id):
        raise RuntimeError("simulated non-database failure in the final step")

    await _failed_final_step_scenario(env, monkeypatch, python_error)


async def test_final_role_deletion_database_error_rolls_back_the_anonymization(env, monkeypatch):
    async def database_error(session, user_id, organization_id):
        await session.execute(text("SELECT 1/0"))  # a genuine PostgreSQL error

    await _failed_final_step_scenario(env, monkeypatch, database_error)


async def test_failed_final_step_does_not_disturb_the_rest_of_the_request(env, monkeypatch):
    """After the savepoint rolled back a REAL database error, the same
    transaction is still healthy: further statements run and the commit is real
    (contrast with an un-nested failure, which poisons the transaction)."""
    org = await _org(env, "usable")
    admin = await _user(env, "admin", orgs={org: env.admin_role})
    member = await _user(env, "member", orgs={org: env.member_role})

    async def database_error(session, user_id, organization_id):
        await session.execute(text("SELECT 1/0"))

    monkeypatch.setattr(privacy_service.repository, "delete_user_roles", database_error)
    result = await _delete_as(env, _actor(org, admin), member)  # must not raise
    assert result.status == "partially_completed"


# --- Multi-org: role-grant interleaving with a real deletion ---------------


async def test_multi_org_person_grant_waits_for_deletion_then_succeeds(env):
    """Person in A and B; A's deletion holds the person's row lock; a grant for
    org C must WAIT. The deletion keeps the account (B remains) and commits; the
    grant then succeeds on the intact, active account (roles: B and C)."""
    org_a, org_b, org_c = await _org(env, "a"), await _org(env, "b"), await _org(env, "c")
    await _user(env, "adminA", orgs={org_a: env.admin_role})
    person = await _user(env, "person", orgs={org_a: env.member_role, org_b: env.member_role})
    before = await _user_row(env, person)

    s1 = env.factory()
    assert await users_service.lock_user_for_deletion(s1, person)  # A's deletion holds the lock

    async def grant():
        async with env.factory() as s2:
            await users_service.assign_role(
                s2, user_id=person, organization_id=org_c, role_id=env.member_role
            )
            await s2.commit()

    t2 = asyncio.create_task(grant())
    await asyncio.sleep(1.0)
    assert not t2.done(), "grant did not wait for the multi-org deletion's row lock"

    # A's deletion decision: B remains -> retain; removes A's role; commits.
    await s1.execute(
        text("DELETE FROM user_roles WHERE user_id = :u AND organization_id = :o"),
        {"u": person, "o": org_a},
    )
    await s1.commit()
    await s1.close()

    await asyncio.wait_for(t2, timeout=10)
    assert await _role_orgs(env, person) == {org_b, org_c}
    assert await _user_row(env, person) == before  # account active and untouched


async def test_real_multi_org_deletion_waits_for_an_in_flight_grant_and_retains(env):
    """Other order, through the REAL deletion path: a grant for org C is in
    flight (uncommitted) when org A's deletion of a person in A and B runs. The
    deletion waits, then retains the account; the person ends in B and C."""
    org_a, org_b, org_c = await _org(env, "a"), await _org(env, "b"), await _org(env, "c")
    admin_a = await _user(env, "adminA", orgs={org_a: env.admin_role})
    person = await _user(env, "person", orgs={org_a: env.member_role, org_b: env.member_role})
    before = await _user_row(env, person)

    s1 = env.factory()
    await users_service.assign_role(
        s1, user_id=person, organization_id=org_c, role_id=env.member_role
    )  # in flight: holds the shared lock

    t2 = asyncio.create_task(_delete_as(env, _actor(org_a, admin_a), person))
    await asyncio.sleep(1.0)
    assert not t2.done(), "deletion did not wait for the in-flight grant"
    await s1.commit()
    await s1.close()

    result = await asyncio.wait_for(t2, timeout=10)
    assert result.status == "completed"
    assert next(s for s in result.steps if s.category == "user_record").action == "retain"
    assert await _role_orgs(env, person) == {org_b, org_c}
    assert await _user_row(env, person) == before

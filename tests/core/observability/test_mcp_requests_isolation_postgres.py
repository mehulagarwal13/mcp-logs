"""`mcp_requests` tenant isolation against REAL PostgreSQL, as the real `ekip_app`
role (NOSUPERUSER NOBYPASSRLS, forced RLS) -- the proof that matters, and the
one a mocked session cannot give.

Skipped unless BOTH point at a LOCAL, disposable database migrated to head
(including `f1b6d3a8c5e2`):

    EKIP_TEST_POSTGRES_URL   owner/superuser URL (seeding, inspection, cleanup)
    EKIP_TEST_APP_URL        the `ekip_app` URL

Every row is tagged with a per-test token and removed with targeted DELETEs.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import uuid
from datetime import UTC, datetime
from urllib.parse import urlsplit

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, ProgrammingError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.observability import service as observability_service
from app.core.privacy import repository as privacy_repository
from app.database.session import set_tenant_context
from app.shared.schemas import ActorKind, Identity

_SUPER_URL = os.environ.get("EKIP_TEST_POSTGRES_URL", "")
_APP_URL = os.environ.get("EKIP_TEST_APP_URL", "")


def _is_local(url: str) -> bool:
    host = urlsplit(url.replace("+asyncpg", "")).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"}


pytestmark = pytest.mark.skipif(
    not (_SUPER_URL and _APP_URL and _is_local(_SUPER_URL) and _is_local(_APP_URL)),
    reason="needs EKIP_TEST_POSTGRES_URL and EKIP_TEST_APP_URL (LOCAL, migrated database)",
)

_MIGRATION = (
    pathlib.Path(__file__).resolve().parents[3]
    / "app/database/migrations/versions/f1b6d3a8c5e2_mcp_requests_tenant_isolation.py"
)


def _engine(url: str):
    return create_async_engine(
        url.replace("?sslmode=disable", ""), poolclass=NullPool, connect_args={"ssl": False}
    )


@pytest_asyncio.fixture
async def env():
    super_engine, app_engine = _engine(_SUPER_URL), _engine(_APP_URL)
    sessions = async_sessionmaker(app_engine, expire_on_commit=False)
    tag = uuid.uuid4().hex[:10]

    async with super_engine.connect() as conn:
        has_column = (
            await conn.execute(
                text(
                    "SELECT 1 FROM information_schema.columns "
                    "WHERE table_name='mcp_requests' AND column_name='organization_id'"
                )
            )
        ).scalar()
    if not has_column:
        pytest.skip("mcp_requests.organization_id missing: run `alembic upgrade head`")

    class Env:
        pass

    e = Env()
    e.tag = tag

    async def sup(sql, **params):
        async with super_engine.begin() as conn:
            res = await conn.execute(text(sql), params)
            return res.fetchall() if res.returns_rows else None

    async def as_app(org, fn):
        """One request-shaped transaction as `ekip_app`: tenant context `org`
        (or none), commit on success, roll back on any exception."""
        async with sessions() as session:
            try:
                if org is not None:
                    await set_tenant_context(session, org)
                result = await fn(session)
                await session.commit()
                return result
            except BaseException:
                await session.rollback()
                raise

    async def make_org(name):
        return (
            await sup(
                "INSERT INTO organizations(name, slug, status) "
                "VALUES (:n, :n, 'active') RETURNING id",
                n=f"mrq-{tag}-{name}",
            )
        )[0][0]

    e.sup, e.as_app = sup, as_app
    e.org_a, e.org_b = await make_org("a"), await make_org("b")
    e.user_a, e.user_b = uuid.uuid4(), uuid.uuid4()

    async def record(org, user, tool="ask_question", status=200, summary=None):
        return await as_app(
            org,
            lambda s: observability_service.record_mcp_request(
                s,
                organization_id=org,
                tool_name=f"{tool}-{tag}",
                identity=f"user:{user}",
                request_summary=summary if summary is not None else {"query": "synthetic text"},
                status_code=status,
                latency_ms=10,
            ),
        )

    e.record = record

    async def count_as_owner(where="TRUE", **params):
        return (
            await sup(
                f"SELECT count(*) FROM mcp_requests WHERE tool_name LIKE :t AND ({where})",
                t=f"%-{tag}",
                **params,
            )
        )[0][0]

    e.count_as_owner = count_as_owner
    yield e

    await sup("DELETE FROM mcp_requests WHERE tool_name LIKE :t", t=f"%-{tag}")
    await sup("DELETE FROM user_roles WHERE organization_id IN (:a, :b)", a=e.org_a, b=e.org_b)
    await sup("DELETE FROM organizations WHERE id IN (:a, :b)", a=e.org_a, b=e.org_b)
    await super_engine.dispose()
    await app_engine.dispose()


async def _visible(env, org, where="TRUE"):
    async def go(s):
        return (
            await s.execute(
                text(f"SELECT count(*) FROM mcp_requests WHERE tool_name LIKE :t AND ({where})"),
                {"t": f"%-{env.tag}"},
            )
        ).scalar_one()

    return await env.as_app(org, go)


# ------------------------------------------------------------------ the role


async def test_the_connection_is_the_rls_enforced_app_role_and_the_table_is_forced(env):
    row = (
        await env.as_app(
            None,
            lambda s: s.execute(
                text(
                    "SELECT current_user, rolsuper, rolbypassrls FROM pg_roles "
                    "WHERE rolname = current_user"
                )
            ),
        )
    ).one()
    assert tuple(row) == ("ekip_app", False, False)
    flags = (
        await env.sup(
            "SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname='mcp_requests'"
        )
    )[0]
    assert tuple(flags) == (True, True)


# ------------------------------------------------- reads (cross-tenant SELECT)


async def test_each_organization_sees_only_its_own_rows(env):
    await env.record(env.org_a, env.user_a)
    await env.record(env.org_b, env.user_b)
    assert await _visible(env, env.org_a) == 1
    assert await _visible(env, env.org_b) == 1
    assert await _visible(env, env.org_a, f"identity = 'user:{env.user_b}'") == 0
    assert await _visible(env, env.org_b, f"identity = 'user:{env.user_a}'") == 0
    assert await env.count_as_owner() == 2  # the owner role still sees both


async def test_missing_tenant_context_sees_nothing(env):
    await env.record(env.org_a, env.user_a)
    assert await _visible(env, None) == 0


async def test_a_reused_connection_with_a_reset_context_is_treated_as_no_context(env):
    """After a transaction-local `set_config`, the setting reads '' (not NULL) on the
    same connection. The policies use NULLIF, so that is 'no context', not a uuid
    cast error."""
    await env.record(env.org_a, env.user_a)
    engine = create_async_engine(
        _APP_URL.replace("?sslmode=disable", ""), pool_size=1, max_overflow=0,
        connect_args={"ssl": False},
    )
    try:
        async with async_sessionmaker(engine, expire_on_commit=False)() as s1:
            await set_tenant_context(s1, env.org_a)
            await s1.commit()
        async with async_sessionmaker(engine, expire_on_commit=False)() as s2:  # same connection
            n = (
                await s2.execute(
                    text("SELECT count(*) FROM mcp_requests WHERE tool_name LIKE :t"),
                    {"t": f"%-{env.tag}"},
                )
            ).scalar_one()
            assert n == 0
    finally:
        await engine.dispose()


# ----------------------------------------- writes (INSERT / UPDATE / DELETE)


async def test_cannot_insert_a_row_for_another_organization(env):
    with pytest.raises(DBAPIError) as excinfo:
        await env.record(env.org_a, env.user_a)  # fine
        await env.as_app(
            env.org_b,
            lambda s: observability_service.record_mcp_request(
                s,
                organization_id=env.org_a,  # B's session claiming A's ownership
                tool_name=f"forged-{env.tag}",
                identity=f"user:{env.user_b}",
                request_summary=None,
                status_code=200,
                latency_ms=1,
            ),
        )
    assert "row-level security" in str(excinfo.value).lower()
    assert await env.count_as_owner("tool_name LIKE 'forged-%'") == 0


async def test_cannot_insert_with_an_organization_but_no_context(env):
    with pytest.raises(DBAPIError):
        await env.as_app(
            None,
            lambda s: observability_service.record_mcp_request(
                s,
                organization_id=env.org_a,
                tool_name=f"nocontext-{env.tag}",
                identity="user:x",
                request_summary=None,
                status_code=200,
                latency_ms=1,
            ),
        )
    assert await env.count_as_owner("tool_name LIKE 'nocontext-%'") == 0


@pytest.mark.parametrize("with_context", [True, False])
async def test_an_unattributed_row_can_be_written_but_no_tenant_can_see_it(env, with_context):
    """A call that failed before any identity existed: insertable (so failed-auth
    logging keeps working), visible to nobody but the owner."""
    org = env.org_a if with_context else None
    await env.as_app(
        org,
        lambda s: observability_service.record_mcp_request(
            s,
            organization_id=None,
            tool_name=f"unresolved-{env.tag}",
            identity="unresolved",
            request_summary=None,
            status_code=401,
            latency_ms=1,
        ),
    )
    assert await env.count_as_owner("organization_id IS NULL") == 1
    assert await _visible(env, env.org_a) == 0
    assert await _visible(env, env.org_b) == 0
    assert await _visible(env, None) == 0


async def test_update_is_denied_for_every_tenant(env):
    await env.record(env.org_a, env.user_a)
    for org in (env.org_a, env.org_b, None):
        with pytest.raises((ProgrammingError, DBAPIError)) as excinfo:
            await env.as_app(
                org,
                lambda s: s.execute(
                    text("UPDATE mcp_requests SET latency_ms = 999 WHERE tool_name LIKE :t"),
                    {"t": f"%-{env.tag}"},
                ),
            )
        assert "permission denied" in str(excinfo.value).lower()
    assert await env.count_as_owner("latency_ms = 10") == 1


async def test_another_organization_cannot_delete_a_row(env):
    await env.record(env.org_a, env.user_a)

    async def delete(s):
        return (
            await s.execute(
                text("DELETE FROM mcp_requests WHERE tool_name LIKE :t"), {"t": f"%-{env.tag}"}
            )
        ).rowcount

    assert await env.as_app(env.org_b, delete) == 0
    assert await env.as_app(None, delete) == 0
    assert await env.count_as_owner() == 1
    assert await env.as_app(env.org_a, delete) == 1  # the owning organization can


# --------------------------------------------------------------- statistics


async def test_dashboard_statistics_are_tenant_scoped(env):
    await env.record(env.org_a, env.user_a, status=200)
    await env.record(env.org_a, env.user_a, status=500)
    await env.record(env.org_b, env.user_b, status=200)
    admin = lambda org: Identity(  # noqa: E731
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=org,
        user_id=uuid.uuid4(),
        permissions=frozenset({"observability:read"}),
    )

    async def stats(org):
        rows = await env.as_app(
            org, lambda s: observability_service.get_mcp_dashboard(s, admin(org))
        )
        return {
            r.tool_name: (r.request_count, r.error_count)
            for r in rows
            if r.tool_name.endswith(env.tag)
        }

    assert await stats(env.org_a) == {f"ask_question-{env.tag}": (2, 1)}
    assert await stats(env.org_b) == {f"ask_question-{env.tag}": (1, 0)}


async def test_application_filter_alone_also_scopes_the_statistics(env):
    """Defence in depth: even a session whose context is the OTHER tenant gets only the
    requested organization's rows from the query's own WHERE (the owner connection has no
    RLS context at all, so this isolates the explicit filter)."""
    await env.record(env.org_a, env.user_a)
    await env.record(env.org_b, env.user_b)
    async with _engine(_SUPER_URL).connect() as conn:
        from app.core.observability import repository

        rows = await repository.get_mcp_tool_stats(
            type("S", (), {"execute": staticmethod(conn.execute)})(), env.org_a
        )
    names = {r.tool_name: r.request_count for r in rows if r.tool_name.endswith(env.tag)}
    assert names == {f"ask_question-{env.tag}": 1}


# ------------------------------------------------------- privacy and retention


async def test_user_erasure_deletes_only_that_users_rows_in_that_organization(env):
    await env.record(env.org_a, env.user_a)
    await env.record(env.org_a, env.user_b)  # a different user, same org
    await env.record(env.org_b, env.user_a)  # same user tag, a different org

    deleted = await env.as_app(
        env.org_a, lambda s: privacy_repository.delete_mcp_requests(s, env.user_a, env.org_a)
    )
    assert deleted == 1
    assert await env.count_as_owner() == 2
    other_org_same_tag = f"identity = 'user:{env.user_a}' AND organization_id = :o"
    assert await env.count_as_owner(other_org_same_tag, o=env.org_b) == 1
    assert await env.as_app(
        env.org_a, lambda s: privacy_repository.count_mcp_requests(s, env.user_b, env.org_a)
    ) == 1


async def test_retention_purge_removes_old_rows_across_tenants_and_keeps_recent_ones(env):
    for org, user in ((env.org_a, env.user_a), (env.org_b, env.user_b)):
        await env.record(org, user)
    await env.as_app(
        None,
        lambda s: observability_service.record_mcp_request(
            s,
            organization_id=None,
            tool_name=f"unresolved-{env.tag}",
            identity="unresolved",
            request_summary=None,
            status_code=401,
            latency_ms=1,
        ),
    )
    await env.sup(
        "UPDATE mcp_requests SET occurred_at = now() - interval '200 days' "
        "WHERE tool_name LIKE :t AND (organization_id = :a OR organization_id IS NULL)",
        t=f"%-{env.tag}",
        a=env.org_a,
    )
    assert await env.count_as_owner() == 3

    # The purge function is the ONLY cross-tenant deletion path, callable by ekip_app.
    deleted = await env.as_app(
        None,
        lambda s: observability_service.purge_expired_mcp_requests(
            s, now=datetime.now(UTC), retention_days=90
        ),
    )
    assert deleted >= 2  # (other tests' leftovers older than 90 days would add to this)
    assert await env.count_as_owner() == 1
    assert await env.count_as_owner("organization_id = :b", b=env.org_b) == 1


async def test_purge_function_is_not_executable_by_public(env):
    acl = (
        await env.sup("SELECT proacl::text FROM pg_proc WHERE proname = 'purge_mcp_requests'")
    )[0][0]
    grantees = [entry.split("=")[0] for entry in acl.strip("{}").split(",")]
    assert "" not in grantees, f"PUBLIC must not have EXECUTE: {acl}"  # '' = PUBLIC
    assert "ekip_app" in grantees


# ------------------------------------------------------------ legacy backfill


def _backfill_sql() -> str:
    spec = importlib.util.spec_from_file_location("mcp_requests_tenant_migration", _MIGRATION)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module.BACKFILL_SQL


async def test_legacy_rows_are_attributed_only_when_unambiguous_and_nothing_is_deleted(env):
    role_id = (
        await env.sup(
            "INSERT INTO roles(name) VALUES (:n) RETURNING id", n=f"mrq-role-{env.tag}"
        )
    )[0][0]
    user_single, user_multi = uuid.uuid4(), uuid.uuid4()
    for uid, orgs in ((user_single, [env.org_a]), (user_multi, [env.org_a, env.org_b])):
        await env.sup(
            "INSERT INTO users(id, email, display_name, is_active) VALUES (:i, :e, 'x', true)",
            i=uid,
            e=f"{uid}@example.invalid",
        )
        for org in orgs:
            await env.sup(
                "INSERT INTO user_roles(user_id, organization_id, role_id) VALUES (:u, :o, :r)",
                u=uid,
                o=org,
                r=role_id,
            )
    legacy = {
        "single": f"user:{user_single}",
        "multi": f"user:{user_multi}",
        "unresolved": "unresolved",
        "agent": "agent:postmortem_agent",
        "garbage": "user:not-a-uuid",
        "unknown_user": f"user:{uuid.uuid4()}",
    }
    for label, identity in legacy.items():
        await env.sup(
            "INSERT INTO mcp_requests(tool_name, identity, request_summary) "
            "VALUES (:t, :i, CAST(:s AS jsonb))",
            t=f"{label}-{env.tag}",
            i=identity,
            s='{"query": "legacy plaintext"}',
        )
    try:
        await env.sup(_backfill_sql())
        owners = {
            r[0].split("-")[0]: r[1]
            for r in await env.sup(
                "SELECT tool_name, organization_id FROM mcp_requests WHERE tool_name LIKE :t",
                t=f"%-{env.tag}",
            )
        }
        assert owners["single"] == env.org_a
        for ambiguous in ("multi", "unresolved", "agent", "garbage", "unknown_user"):
            assert owners[ambiguous] is None, ambiguous
        assert await env.count_as_owner() == len(legacy)  # nothing deleted
        assert await env.count_as_owner("request_summary->>'query' = 'legacy plaintext'") == len(
            legacy
        )  # nothing rewritten
    finally:
        await env.sup(
            "DELETE FROM user_roles WHERE user_id IN (:a, :b)", a=user_single, b=user_multi
        )
        await env.sup("DELETE FROM users WHERE id IN (:a, :b)", a=user_single, b=user_multi)
        await env.sup("DELETE FROM roles WHERE id = :r", r=role_id)


# ----------------------------------------- the real dispatch -> real RLS path


@pytest.fixture
def real_dispatch(env, monkeypatch):
    """`run_mcp_tool` wired exactly like `scripts/run_mcp_server.py` wires it (real
    session factory as `ekip_app`, real `set_tenant_context`), with only the bearer-token
    -> Identity step replaced so no user rows are needed."""
    from contextlib import asynccontextmanager

    from app.mcp import dispatch as dispatch_module
    from app.mcp.servers import server as server_module

    engine = _engine(_APP_URL)
    sessions = async_sessionmaker(engine, expire_on_commit=False)

    @asynccontextmanager
    async def session_factory():
        async with sessions() as session:
            yield session
            await session.commit()

    monkeypatch.setattr(server_module, "session_factory", session_factory)
    monkeypatch.setattr(server_module, "set_tenant_context", set_tenant_context)
    return dispatch_module


async def test_a_real_tool_call_logs_an_attributed_minimized_row_through_rls(
    env, real_dispatch, monkeypatch
):
    from app.mcp.dispatch import run_mcp_tool

    identity = Identity(
        kind=ActorKind.USER,
        subject=str(env.user_a),
        organization_id=env.org_a,
        user_id=env.user_a,
        permissions=frozenset(),
    )

    async def resolve(session, raw_token):
        return identity

    monkeypatch.setattr(real_dispatch, "resolve_mcp_identity", resolve)

    async def handler(session, who):
        return "ok"

    await run_mcp_tool(
        tool_name=f"ask_question-{env.tag}",
        raw_token="t",
        request_summary={"query": "private question text", "incident_id": None},
        handler=handler,
    )
    rows = await env.sup(
        "SELECT organization_id, identity, request_summary::text, status_code FROM mcp_requests "
        "WHERE tool_name = :t",
        t=f"ask_question-{env.tag}",
    )
    assert len(rows) == 1
    org, who, summary, status = rows[0]
    assert (org, who, status) == (env.org_a, f"user:{env.user_a}", 200)
    assert "private question text" not in summary and '"query_chars": 21' in summary
    assert await _visible(env, env.org_b, f"tool_name = 'ask_question-{env.tag}'") == 0


async def test_a_real_call_with_a_bad_token_is_still_logged_unattributed(
    env, real_dispatch, monkeypatch
):
    from app.core.exceptions import PermissionDeniedError
    from app.mcp.dispatch import run_mcp_tool

    async def reject(session, raw_token):
        raise PermissionDeniedError("bad", error_code="auth.invalid_token")

    monkeypatch.setattr(real_dispatch, "resolve_mcp_identity", reject)

    async def handler(session, who):
        return "unreachable"

    with pytest.raises(PermissionDeniedError):
        await run_mcp_tool(
            tool_name=f"ask_question-{env.tag}",
            raw_token="bad",
            request_summary={},
            handler=handler,
        )
    rows = await env.sup(
        "SELECT organization_id, identity, status_code FROM mcp_requests WHERE tool_name = :t",
        t=f"ask_question-{env.tag}",
    )
    assert [tuple(r) for r in rows] == [(None, "unresolved", 403)]

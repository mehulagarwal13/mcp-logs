"""F5 against REAL PostgreSQL as the real `ekip_app` role (forced RLS):
refresh tokens issued to an MCP OAuth client are bound to it.

Skipped unless BOTH are set and point at a LOCAL, migrated, disposable database
(`alembic upgrade head`, including `e7a2c4b9d1f6`):

    EKIP_TEST_POSTGRES_URL   owner/superuser URL (seeding, inspection, cleanup)
    EKIP_TEST_APP_URL        the `ekip_app` URL (NOSUPERUSER NOBYPASSRLS)

Every row is tagged with a unique token and removed with targeted DELETEs.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlsplit

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.auth import service as auth_service
from app.core.auth.schemas import RefreshRequest
from app.core.exceptions import PermissionDeniedError

_SUPER_URL = os.environ.get("EKIP_TEST_POSTGRES_URL", "")
_APP_URL = os.environ.get("EKIP_TEST_APP_URL", "")


def _is_local(url: str) -> bool:
    host = urlsplit(url.replace("+asyncpg", "")).hostname or ""
    return host in {"localhost", "127.0.0.1", "::1"}


pytestmark = pytest.mark.skipif(
    not (_SUPER_URL and _APP_URL and _is_local(_SUPER_URL) and _is_local(_APP_URL)),
    reason="needs EKIP_TEST_POSTGRES_URL and EKIP_TEST_APP_URL (LOCAL, migrated database)",
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
                    "WHERE table_name='refresh_tokens' AND column_name='oauth_client_id'"
                )
            )
        ).scalar()
    if not has_column:
        pytest.skip("refresh_tokens.oauth_client_id missing: run `alembic upgrade head`")

    class Env:
        pass

    e = Env()

    async def sup(sql, **params):
        async with super_engine.begin() as conn:
            res = await conn.execute(text(sql), params)
            return res.fetchall() if res.returns_rows else None

    async def run(fn):
        async with sessions() as session:
            try:
                result = await fn(session)
                await session.commit()
                return result
            except BaseException:
                await session.rollback()
                raise

    org = (
        await sup(
            "INSERT INTO organizations(name, slug, status) VALUES (:n, :n, 'active') RETURNING id",
            n=f"rtb-{tag}",
        )
    )[0][0]
    user = (
        await sup(
            "INSERT INTO users(email, display_name, is_active, email_verified_at) "
            "VALUES (:e, 'RTB', true, now()) RETURNING id",
            e=f"rtb-{tag}@example.invalid",
        )
    )[0][0]
    e.org, e.user, e.run, e.sup = org, user, run, sup
    yield e

    await sup("DELETE FROM refresh_tokens WHERE user_id = :u", u=user)
    await sup("DELETE FROM users WHERE id = :u", u=user)
    await sup("DELETE FROM organizations WHERE id = :o", o=org)
    await super_engine.dispose()
    await app_engine.dispose()


async def _issue(env, *, client: str | None):
    extra = {"oauth_client_id": client, "scope": "ekip"} if client else {}
    return await env.run(
        lambda s: auth_service._issue_session(
            s, user_id=env.user, organization_id=env.org, family_id=uuid.uuid4(), **extra
        )
    )


async def _row(env, raw: str):
    rows = await env.sup(
        "SELECT oauth_client_id, revoked_at FROM refresh_tokens WHERE token_hash = :h",
        h=auth_service._hash_token(raw),
    )
    return rows[0] if rows else None


async def _refresh(env, raw: str, *, client: str | None):
    return await env.run(
        lambda s: auth_service.refresh(s, RefreshRequest(refresh_token=raw), oauth_client_id=client)
    )


async def test_the_connection_is_the_rls_enforced_app_role(env):
    row = (
        await env.run(
            lambda s: s.execute(
                text(
                    "SELECT current_user, rolsuper, rolbypassrls "
                    "FROM pg_roles WHERE rolname = current_user"
                )
            )
        )
    ).one()
    assert tuple(row) == ("ekip_app", False, False)


async def test_issue_persists_the_client_binding(env):
    bound = await _issue(env, client="client-a")
    plain = await _issue(env, client=None)
    assert (await _row(env, bound.refresh_token))[0] == "client-a"
    assert (await _row(env, plain.refresh_token))[0] is None


async def test_a_different_client_cannot_refresh_and_nothing_is_consumed(env):
    tokens = await _issue(env, client="client-a")
    for presenter in ("client-b", None):
        with pytest.raises(PermissionDeniedError):
            await _refresh(env, tokens.refresh_token, client=presenter)
    assert (await _row(env, tokens.refresh_token))[1] is None


async def test_the_owning_client_refreshes_and_rotation_keeps_the_binding(env):
    tokens = await _issue(env, client="client-a")
    rotated = await _refresh(env, tokens.refresh_token, client="client-a")
    assert (await _row(env, tokens.refresh_token))[1] is not None  # old one rotated away
    assert (await _row(env, rotated.refresh_token))[0] == "client-a"
    with pytest.raises(PermissionDeniedError):
        await _refresh(env, rotated.refresh_token, client="client-b")


async def test_reuse_of_a_rotated_token_is_still_rejected_as_reuse(env):
    """Unchanged behaviour: replaying a rotated-away token is refused with the
    reuse error code (by its own client), exactly as for a REST session."""
    tokens = await _issue(env, client="client-a")
    await _refresh(env, tokens.refresh_token, client="client-a")
    with pytest.raises(PermissionDeniedError) as excinfo:
        await _refresh(env, tokens.refresh_token, client="client-a")
    assert excinfo.value.error_code == "auth.refresh_token_reused"


async def test_an_unbound_rest_session_cannot_be_refreshed_through_a_client(env):
    tokens = await _issue(env, client=None)
    with pytest.raises(PermissionDeniedError):
        await _refresh(env, tokens.refresh_token, client="client-a")
    rotated = await _refresh(env, tokens.refresh_token, client=None)  # REST path unchanged
    assert (await _row(env, rotated.refresh_token))[0] is None


async def test_revocation_only_works_for_the_owning_client(env):
    tokens = await _issue(env, client="client-a")
    request = RefreshRequest(refresh_token=tokens.refresh_token)
    await env.run(lambda s: auth_service.logout(s, request, oauth_client_id="client-b"))
    await env.run(lambda s: auth_service.logout(s, request))
    assert (await _row(env, tokens.refresh_token))[1] is None
    await env.run(lambda s: auth_service.logout(s, request, oauth_client_id="client-a"))
    assert (await _row(env, tokens.refresh_token))[1] is not None


async def test_peek_exposes_the_binding(env):
    tokens = await _issue(env, client="client-a")
    row = await env.run(lambda s: auth_service.peek_refresh_token(s, tokens.refresh_token))
    assert row.oauth_client_id == "client-a"

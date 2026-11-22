"""The OAuth client registry against REAL PostgreSQL as the `ekip_app` role, with
the real (local-provider) KMS envelope encryption of client secrets:

* F6: a legacy row stored with `client_secret_expires_at = 0` is held to the
  expiry window; the secret still round-trips through the KMS decrypted.
* F2: `count_recent_registrations` counts only rows inside the window.

Skipped unless `EKIP_TEST_APP_URL` / `EKIP_TEST_POSTGRES_URL` point at a LOCAL,
migrated, disposable database, and `CONNECTOR_SECRET_MASTER_KEY` is set (the
local KMS provider). Rows are tagged and removed with targeted DELETEs.
"""

from __future__ import annotations

import os
import time
import uuid
from urllib.parse import urlsplit

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.mcp_oauth import service as mcp_oauth_service
from app.shared.config.settings import get_settings

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
    tag = f"reg-{uuid.uuid4().hex[:10]}"

    class Env:
        pass

    e = Env()
    e.tag = tag

    async def run(fn):
        async with sessions() as session:
            try:
                result = await fn(session)
                await session.commit()
                return result
            except BaseException:
                await session.rollback()
                raise

    async def sup(sql, **params):
        async with super_engine.begin() as conn:
            await conn.execute(text(sql), params)

    e.run, e.sup = run, sup
    yield e
    await sup("DELETE FROM oauth_clients WHERE client_id LIKE :p", p=f"{tag}%")
    await super_engine.dispose()
    await app_engine.dispose()


def _info(client_id: str, *, secret: str | None, expires_at: int | None, issued_at: int):
    return {
        "client_id": client_id,
        "client_secret": secret,
        "client_id_issued_at": issued_at,
        "client_secret_expires_at": expires_at,
        "redirect_uris": ["http://localhost:3334/cb"],
        "token_endpoint_auth_method": "client_secret_post",
    }


async def test_secret_and_expiry_round_trip_through_the_real_kms(env):
    cid, now = f"{env.tag}-a", int(time.time())
    expires = now + 1000
    await env.run(
        lambda s: mcp_oauth_service.register_oauth_client(
            s, _info(cid, secret="s3cret-value", expires_at=expires, issued_at=now)
        )
    )
    stored = await env.run(lambda s: mcp_oauth_service.get_registered_client(s, cid))
    assert stored["client_secret"] == "s3cret-value"
    assert stored["client_secret_expires_at"] == expires

    raw = await env.run(
        lambda s: s.execute(
            text("SELECT client_secret_encrypted FROM oauth_clients WHERE client_id = :c"),
            {"c": cid},
        )
    )
    assert "s3cret-value" not in raw.scalar_one(), "the secret must be stored encrypted"


async def test_a_legacy_zero_expiry_row_is_held_to_the_window(env):
    window = get_settings().mcp_oauth_client_secret_expiry_days * 86400
    now = int(time.time())
    fresh, stale = f"{env.tag}-fresh", f"{env.tag}-stale"
    for cid, issued in ((fresh, now - 86400), (stale, now - window - 86400)):
        await env.run(
            lambda s, cid=cid, issued=issued: mcp_oauth_service.register_oauth_client(
                s, _info(cid, secret="legacy-secret", expires_at=0, issued_at=issued)
            )
        )
    fresh_row = await env.run(lambda s: mcp_oauth_service.get_registered_client(s, fresh))
    stale_row = await env.run(lambda s: mcp_oauth_service.get_registered_client(s, stale))
    assert fresh_row["client_secret_expires_at"] > now
    assert stale_row["client_secret_expires_at"] < now


async def test_recent_registration_count_respects_the_window(env):
    now = int(time.time())
    for suffix in ("one", "two"):
        await env.run(
            lambda s, suffix=suffix: mcp_oauth_service.register_oauth_client(
                s, _info(f"{env.tag}-{suffix}", secret=None, expires_at=None, issued_at=now)
            )
        )
    before = await env.run(
        lambda s: mcp_oauth_service.count_recent_registrations(s, within_seconds=3600)
    )
    # Age one of them out of the window (owner connection; created_at is server-set).
    await env.sup(
        "UPDATE oauth_clients SET created_at = now() - interval '2 hours' WHERE client_id = :c",
        c=f"{env.tag}-one",
    )
    after = await env.run(
        lambda s: mcp_oauth_service.count_recent_registrations(s, within_seconds=3600)
    )
    assert before - after == 1

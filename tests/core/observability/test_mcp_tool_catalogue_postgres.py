"""The MCP tool catalogue against REAL PostgreSQL as the `ekip_app` role: the MCP
server's publish (sync) and the API's read, including replacement semantics.

Skipped unless EKIP_TEST_POSTGRES_URL / EKIP_TEST_APP_URL point at a LOCAL database
migrated to head (including `a8d4f2c6e1b9`).
"""

from __future__ import annotations

import json
import os
import uuid
from urllib.parse import urlsplit

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.core.observability import service as observability_service
from app.core.observability.schemas import McpToolSpec
from app.mcp.catalogue import describe_registered_tools
from app.mcp.servers import main as mcp_assembly
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


def _engine(url: str):
    return create_async_engine(
        url.replace("?sslmode=disable", ""), poolclass=NullPool, connect_args={"ssl": False}
    )


def _reader(org: uuid.UUID) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=org,
        user_id=uuid.uuid4(),
        permissions=frozenset({"observability:read"}),
    )


@pytest_asyncio.fixture
async def env():
    sup, app = _engine(_SUPER_URL), _engine(_APP_URL)
    async with sup.connect() as conn:
        table = await conn.execute(text("SELECT to_regclass('public.mcp_tool_catalogue')"))
        if table.scalar() is None:
            pytest.skip("mcp_tool_catalogue missing: run `alembic upgrade head`")
        saved = (await conn.execute(text("SELECT * FROM mcp_tool_catalogue"))).fetchall()
    sessions = async_sessionmaker(app, expire_on_commit=False)

    async def run(fn):
        async with sessions() as session:
            try:
                result = await fn(session)
                await session.commit()
                return result
            except BaseException:
                await session.rollback()
                raise

    yield run
    # restore whatever was there before (the table is shared platform metadata)
    async with sup.begin() as conn:
        await conn.execute(text("DELETE FROM mcp_tool_catalogue"))
        for row in saved:
            await conn.execute(
                text(
                    "INSERT INTO mcp_tool_catalogue(name, description, input_schema, synced_at) "
                    "VALUES (:n, :d, CAST(:s AS jsonb), :t)"
                ),
                {"n": row[0], "d": row[1], "s": json.dumps(row[2]), "t": row[3]},
            )
    await sup.dispose()
    await app.dispose()


async def test_publish_then_read_returns_exactly_the_registered_tools(env):
    specs = await describe_registered_tools(mcp_assembly.mcp_server)
    await env(lambda s: observability_service.sync_mcp_tool_catalogue(s, specs))
    catalogue = await env(
        lambda s: observability_service.list_mcp_tool_catalogue(s, _reader(uuid.uuid4()))
    )
    assert [t.name for t in catalogue] == [s.name for s in specs]
    by_name = {t.name: t for t in catalogue}
    assert {p.name: p.required for p in by_name["create_project"].parameters} == {
        "name": True,
        "is_default": False,
    }
    assert all(t.status == "available" and t.description for t in catalogue)


async def test_a_new_sync_replaces_the_set_updates_and_removes(env):
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    await env(
        lambda s: observability_service.sync_mcp_tool_catalogue(
            s,
            [
                McpToolSpec(name="keep", description="v1", input_schema=schema),
                McpToolSpec(name="drop", description="old", input_schema=schema),
            ],
        )
    )
    await env(
        lambda s: observability_service.sync_mcp_tool_catalogue(
            s,
            [
                McpToolSpec(name="keep", description="v2", input_schema=schema),
                McpToolSpec(name="added", description="new", input_schema=schema),
            ],
        )
    )
    catalogue = await env(
        lambda s: observability_service.list_mcp_tool_catalogue(s, _reader(uuid.uuid4()))
    )
    assert {t.name: t.description for t in catalogue} == {"added": "new", "keep": "v2"}


async def test_an_empty_registration_empties_the_catalogue(env):
    await env(lambda s: observability_service.sync_mcp_tool_catalogue(s, []))
    assert await env(
        lambda s: observability_service.list_mcp_tool_catalogue(s, _reader(uuid.uuid4()))
    ) == []

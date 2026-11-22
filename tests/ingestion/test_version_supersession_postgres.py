"""Real-PostgreSQL tests for document version supersession.

`test_version_supersession.py` proves the service LOGIC against an in-memory
model. These tests run the same scenarios against a real PostgreSQL + pgvector
so the parts a fake cannot prove are exercised for real: the advisory lock,
savepoint rollback, the `uq_documents_org_source_external_hash` constraint,
the retrieval SQL, and the migration DDL.

SKIPPED by default. Set `EKIP_TEST_POSTGRES_URL` to a LOCAL, disposable
PostgreSQL (pgvector extension installed), e.g.

    EKIP_TEST_POSTGRES_URL=postgresql://postgres:@127.0.0.1:5432/postgres

Safety: the URL's host must be localhost/127.0.0.1/::1 (anything else is
refused), and every test creates its own uniquely named schema and drops it
afterwards -- no existing table or row is read, changed or dropped. Never point
this at a shared or production database.

Scope note: tables are built with `Base.metadata.create_all`, so row-level
security policies (added by separate migrations) are NOT present here. These
tests verify version-lifecycle behavior, not tenant isolation; RLS is covered
by `scripts/verify_rls_isolation.py` in CI. Nothing in the feature touches RLS.
The embedding model is replaced by a deterministic stub (no model download).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import uuid
from pathlib import Path
from types import SimpleNamespace

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

# Imported after the skip marker is defined; importing models is side-effect free.
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
from app.ingestion import repository  # noqa: E402
from app.ingestion import service as ingestion_service  # noqa: E402
from app.ingestion.connectors.incidents import IncidentsConnector  # noqa: E402
from app.ingestion.schemas import RawDocument  # noqa: E402
from app.retrieval import embedding  # noqa: E402
from app.retrieval.pgvector.store import PgVectorStore  # noqa: E402
from app.retrieval.schemas import SearchFilters, UpsertChunk  # noqa: E402

A = "Runbook A: restart the connection pool before failing over the primary."
B = "Runbook B: drain traffic first, then restart the connection pool carefully."
C = "Runbook C: page the on-call database engineer immediately."

_CHUNK_TABLES = ["documentation_chunks", "code_chunks", "conversations_chunks", "incidents_chunks"]
_ALL_COLLECTIONS = ["documentation", "code", "conversations", "incidents"]


class _Env(SimpleNamespace):
    pass


@pytest_asyncio.fixture
async def env(monkeypatch):
    schema = f"sup_{uuid.uuid4().hex[:12]}"
    admin = create_async_engine(_URL)
    async with admin.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))

    engine = create_async_engine(
        _URL,
        pool_size=6,
        connect_args={"server_settings": {"search_path": f'"{schema}",public'}},
    )
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)

        org_id, other_org_id, project_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        other_project_id = uuid.uuid4()
        async with factory() as session:
            for oid, slug in ((org_id, "org-a"), (other_org_id, "org-b")):
                await session.execute(
                    text(
                        "INSERT INTO organizations (id, name, slug, status) "
                        "VALUES (:i, :n, :s, 'active')"
                    ),
                    {"i": oid, "n": slug, "s": slug},
                )
            for pid, oid, name in (
                (project_id, org_id, "General"),
                (other_project_id, other_org_id, "General"),
            ):
                await session.execute(
                    text(
                        "INSERT INTO projects (id, organization_id, name, is_default) "
                        "VALUES (:i, :o, :n, true)"
                    ),
                    {"i": pid, "o": oid, "n": name},
                )
            await session.commit()

        state = _Env(embed_delay=0.0, fail_embedding=False)

        async def fake_embed_texts(texts):
            if state.embed_delay:
                await asyncio.sleep(state.embed_delay)
            if state.fail_embedding:
                raise RuntimeError("embedding service unavailable")
            return [[0.01] * 384 for _ in texts]

        monkeypatch.setattr(embedding, "embed_texts", fake_embed_texts)

        yield _Env(
            engine=engine,
            factory=factory,
            org_id=org_id,
            other_org_id=other_org_id,
            project_id=project_id,
            state=state,
        )
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await admin.dispose()


class _Connector:
    def __init__(self, source: str) -> None:
        self.source_name = source

    def normalize(self, raw_item):
        return RawDocument(
            source=self.source_name,
            external_id=raw_item["id"],
            content=raw_item["content"],
            title="Title",
            source_url=f"https://example.test/{raw_item['id']}",
            metadata=raw_item.get("metadata", {}),
        )


async def ingest(env, content, *, source="confluence", external_id="doc-1", metadata=None):
    """One item through the REAL `_process_one_item_with_retry` on its own
    session/connection, then COMMIT (the unchanged path leaves its read
    transaction open until the page checkpoint, which is what this models)."""
    async with env.factory() as session:
        await set_tenant_context(session, env.org_id)
        result = await ingestion_service._process_one_item_with_retry(
            session,
            organization_id=env.org_id,
            connector=_Connector(source),
            raw_item={"id": external_id, "content": content, "metadata": metadata or {}},
            resolved_config=SimpleNamespace(organization_id=env.org_id),
            project_id=env.project_id,
        )
        await session.commit()
        return result


async def doc_rows(env, external_id="doc-1"):
    async with env.factory() as session:
        result = await session.execute(
            text(
                "SELECT id, version, superseded_at IS NULL AS is_current, content_hash "
                "FROM documents WHERE external_id = :e ORDER BY version"
            ),
            {"e": external_id},
        )
        return result.all()


async def chunk_counts(env, document_id):
    async with env.factory() as session:
        counts = {}
        for table in _CHUNK_TABLES:
            counts[table] = (
                await session.execute(
                    text(f"SELECT count(*) FROM {table} WHERE document_id = :d"),
                    {"d": document_id},
                )
            ).scalar_one()
        return counts


def _filters(env) -> SearchFilters:
    return SearchFilters(organization_id=env.org_id, permission_codes=frozenset())


# --- normal update ------------------------------------------------------------


async def test_update_supersedes_old_version_and_purges_its_chunks(env):
    await ingest(env, A)
    [v1] = await doc_rows(env)
    assert (await chunk_counts(env, v1.id))["documentation_chunks"] >= 1

    await ingest(env, B)

    v1_after, v2 = await doc_rows(env)
    assert v1_after.id == v1.id and v1_after.is_current is False  # history kept
    assert v2.version == 2 and v2.is_current is True
    assert sum((await chunk_counts(env, v1.id)).values()) == 0
    assert (await chunk_counts(env, v2.id))["documentation_chunks"] >= 1


async def test_unchanged_content_creates_no_new_row(env):
    await ingest(env, A)
    result = await ingest(env, A)

    assert result.documents_processed == 0
    assert len(await doc_rows(env)) == 1


async def test_incident_collection_chunks_are_purged_on_supersession(env):
    source = IncidentsConnector.source_name
    await ingest(env, "Incident: outage. Root cause: pool exhaustion.", source=source)
    [v1] = await doc_rows(env)
    assert (await chunk_counts(env, v1.id))["incidents_chunks"] >= 1

    await ingest(env, "Incident: outage. Root cause: failover misconfiguration.", source=source)

    assert (await chunk_counts(env, v1.id))["incidents_chunks"] == 0


# --- retrieval / browse exclude superseded rows (real SQL) -------------------


async def test_search_never_returns_chunks_of_a_superseded_version(env):
    await ingest(env, A)
    [v1] = await doc_rows(env)
    await ingest(env, B)
    [_, v2] = await doc_rows(env)

    # Simulate a stale chunk surviving (e.g. history from before cleanup existed).
    async with env.factory() as session:
        await set_tenant_context(session, env.org_id)
        await PgVectorStore().upsert(
            session,
            "documentation",
            [
                UpsertChunk(
                    document_id=v1.id,
                    organization_id=env.org_id,
                    project_id=env.project_id,
                    collection="documentation",
                    chunk_index=99,
                    content="stale connection pool content from the superseded version",
                    source_offset_start=0,
                    source_offset_end=10,
                )
            ],
            [[0.01] * 384],
        )
        await session.commit()

    async with env.factory() as session:
        await set_tenant_context(session, env.org_id)
        store = PgVectorStore()
        dense = await store.search(session, "documentation", [0.01] * 384, _filters(env), 20)
        lexical = await store.lexical_search(
            session, "documentation", "connection pool", _filters(env), 20
        )
        dense_all = await store.search_all(session, [0.01] * 384, _filters(env), 20)
        lexical_all = await store.lexical_search_all(session, "connection pool", _filters(env), 20)

    for name, results in (
        ("dense", dense),
        ("lexical", lexical),
        ("dense_all", dense_all),
        ("lexical_all", lexical_all),
    ):
        ids = {r.document_id for r in results}
        assert v1.id not in ids, f"{name} search returned a superseded version's chunk"
        assert v2.id in ids, f"{name} search lost the current version"


async def test_knowledge_browse_lists_only_current_versions(env):
    from app.core.knowledge import repository as knowledge_repository

    await ingest(env, A)
    await ingest(env, B)
    await ingest(env, C)
    rows = await doc_rows(env)

    async with env.factory() as session:
        await set_tenant_context(session, env.org_id)
        listed = await knowledge_repository.list_published_documents(session, env.org_id)
        by_id = await knowledge_repository.get_document_by_id(session, rows[0].id)

    assert [d.id for d in listed] == [rows[-1].id]
    assert by_id is not None  # an old version's id still resolves (citation integrity)


# --- A -> B -> A reverts ------------------------------------------------------


async def test_revert_a_b_a_succeeds_against_the_real_unique_constraint(env):
    await ingest(env, A, metadata={"author": "alice"})
    [a_row] = await doc_rows(env)
    await ingest(env, B)

    result = await ingest(env, A, metadata={"author": "bob"})  # must not raise

    assert result.documents_processed == 1
    rows = await doc_rows(env)
    assert len(rows) == 2
    revived = next(r for r in rows if r.id == a_row.id)
    assert (revived.version, revived.is_current) == (3, True)
    assert [r.is_current for r in rows if r.id != a_row.id] == [False]
    async with env.factory() as session:
        authors = (
            await session.execute(
                text(
                    "SELECT value FROM document_metadata "
                    "WHERE document_id = :d AND key = 'author'"
                ),
                {"d": a_row.id},
            )
        ).scalars().all()
    assert authors == ["bob"]


async def test_repeated_reverts_never_duplicate_a_hash(env):
    for content in (A, B, A, B, A, C, A):
        await ingest(env, content)

    rows = await doc_rows(env)
    assert len(rows) == 3
    assert len({r.content_hash for r in rows}) == 3
    assert [r.is_current for r in rows].count(True) == 1
    assert len({r.version for r in rows}) == 3


# --- atomicity ----------------------------------------------------------------


async def test_failed_embedding_rolls_back_supersession_and_purge(env):
    await ingest(env, A)
    [v1] = await doc_rows(env)
    before_chunks = await chunk_counts(env, v1.id)

    env.state.fail_embedding = True
    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        await ingest(env, B)
    env.state.fail_embedding = False

    rows = await doc_rows(env)
    assert [(r.id, r.is_current) for r in rows] == [(v1.id, True)]  # no v2, v1 still current
    assert await chunk_counts(env, v1.id) == before_chunks  # old chunks intact


async def test_failed_revert_rolls_back_revival(env):
    await ingest(env, A)
    await ingest(env, B)
    before = [(r.id, r.version, r.is_current) for r in await doc_rows(env)]

    env.state.fail_embedding = True
    with pytest.raises(RuntimeError):
        await ingest(env, A)
    env.state.fail_embedding = False

    assert [(r.id, r.version, r.is_current) for r in await doc_rows(env)] == before


async def test_retry_after_failure_succeeds_and_no_lock_is_leaked(env):
    await ingest(env, A)
    env.state.fail_embedding = True
    with pytest.raises(RuntimeError):
        await ingest(env, B)
    env.state.fail_embedding = False

    result = await asyncio.wait_for(ingest(env, B), timeout=10)  # hangs if the lock leaked

    assert result.documents_processed == 1
    assert [r.version for r in await doc_rows(env)] == [1, 2]


# --- concurrency (the Redis lock is not involved at all here) ------------------


async def test_concurrent_identical_new_content_creates_exactly_one_row(env):
    env.state.embed_delay = 0.3  # widen the race window
    results = await asyncio.wait_for(asyncio.gather(ingest(env, A), ingest(env, A)), timeout=30)

    assert sorted(r.documents_processed for r in results) == [0, 1]
    assert len(await doc_rows(env)) == 1


async def test_concurrent_different_content_builds_a_clean_version_chain(env):
    await ingest(env, A)
    env.state.embed_delay = 0.3

    await asyncio.wait_for(asyncio.gather(ingest(env, B), ingest(env, C)), timeout=30)

    rows = await doc_rows(env)
    assert [r.version for r in rows] == [1, 2, 3]
    assert [r.is_current for r in rows].count(True) == 1
    assert rows[-1].is_current is True
    for old in rows[:-1]:
        assert sum((await chunk_counts(env, old.id)).values()) == 0


async def test_concurrent_revert_and_update_stay_consistent(env):
    await ingest(env, A)
    await ingest(env, B)
    env.state.embed_delay = 0.2

    await asyncio.wait_for(asyncio.gather(ingest(env, A), ingest(env, C)), timeout=30)

    rows = await doc_rows(env)
    assert [r.is_current for r in rows].count(True) == 1
    assert len({r.version for r in rows}) == len(rows)
    assert len({r.content_hash for r in rows}) == len(rows)


async def test_advisory_lock_is_per_key_transaction_scoped_and_tenant_scoped(env):
    async with env.factory() as holder, env.factory() as probe:
        await repository.lock_document_key(holder, env.org_id, "github", "x")

        async def try_lock(org, source, external_id) -> bool:
            key = f"{org}:{source}:{external_id}"
            return (
                await probe.execute(
                    text("SELECT pg_try_advisory_xact_lock(hashtextextended(:k, 0))"), {"k": key}
                )
            ).scalar_one()

        assert await try_lock(env.org_id, "github", "x") is False  # same key blocked
        await probe.rollback()
        assert await try_lock(env.org_id, "github", "y") is True  # other item free
        await probe.rollback()
        assert await try_lock(env.other_org_id, "github", "x") is True  # other tenant free
        await probe.rollback()

        await holder.commit()  # COMMIT releases the transaction-scoped lock
        assert await try_lock(env.org_id, "github", "x") is True
        await probe.rollback()


async def test_unique_constraint_still_rejects_a_second_row_with_the_same_hash(env):
    """The revive path exists precisely because this constraint is real."""
    from sqlalchemy.exc import IntegrityError

    await ingest(env, A)
    [row] = await doc_rows(env)
    async with env.factory() as session:
        with pytest.raises(IntegrityError) as excinfo:
            await session.execute(
                text(
                    "INSERT INTO documents (organization_id, project_id, source, external_id, "
                    "content_hash, status, version) "
                    "SELECT organization_id, project_id, source, external_id, content_hash, "
                    "status, 99 FROM documents WHERE id = :i"
                ),
                {"i": row.id},
            )
        # Not just any integrity error: specifically the idempotency key.
        assert "uq_documents_org_source_external_hash" in str(excinfo.value)


# --- migration DDL -------------------------------------------------------------


def _load_migration():
    path = (
        Path(__file__).resolve().parents[2]
        / "app/database/migrations/versions/c3a9d5e7f1b2_documents_superseded_at.py"
    )
    spec = importlib.util.spec_from_file_location("migration_c3a9d5e7f1b2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_migration_upgrade_downgrade_upgrade_on_real_postgres(env):
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    migration = _load_migration()

    def apply(sync_conn, fn):
        ctx = MigrationContext.configure(sync_conn)
        with Operations.context(ctx):
            fn()

    async def column_and_index_exist() -> tuple[bool, bool]:
        async with env.factory() as session:
            col = (
                await session.execute(
                    text(
                        "SELECT count(*) FROM information_schema.columns "
                        "WHERE table_schema = current_schema() AND table_name = 'documents' "
                        "AND column_name = 'superseded_at'"
                    )
                )
            ).scalar_one()
            idx = (
                await session.execute(
                    text(
                        "SELECT count(*) FROM pg_indexes WHERE schemaname = current_schema() "
                        "AND indexname = 'ix_documents_current'"
                    )
                )
            ).scalar_one()
        return bool(col), bool(idx)

    # Start from the pre-migration shape.
    async with env.engine.begin() as conn:
        await conn.execute(text("DROP INDEX ix_documents_current"))
        await conn.execute(text("ALTER TABLE documents DROP COLUMN superseded_at"))
    assert await column_and_index_exist() == (False, False)

    # Existing rows must survive and read as "current" (NULL) after upgrade.
    async with env.engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO documents (organization_id, project_id, source, external_id, "
                "content_hash, status, version) "
                "VALUES (:o, :p, 'github', 'pre-existing', 'h', 'published', 1)"
            ),
            {"o": env.org_id, "p": env.project_id},
        )
        await conn.run_sync(lambda c: apply(c, migration.upgrade))
    assert await column_and_index_exist() == (True, True)
    async with env.factory() as session:
        superseded = (
            await session.execute(
                text("SELECT superseded_at FROM documents WHERE external_id = 'pre-existing'")
            )
        ).scalar_one()
    assert superseded is None

    async with env.engine.begin() as conn:
        await conn.run_sync(lambda c: apply(c, migration.downgrade))
    assert await column_and_index_exist() == (False, False)

    async with env.engine.begin() as conn:
        await conn.run_sync(lambda c: apply(c, migration.upgrade))
    assert await column_and_index_exist() == (True, True)


# --- backfill (app.ingestion.lifecycle) on real PostgreSQL ---------------------

from contextlib import asynccontextmanager  # noqa: E402
from datetime import UTC, datetime, timedelta  # noqa: E402

from app.ingestion import lifecycle  # noqa: E402

_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def _scope(env):
    """A `session_scope` equivalent bound to the test schema (commit per batch)."""

    @asynccontextmanager
    async def scope():
        async with env.factory() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    return scope


async def seed_history(
    env,
    versions: int,
    *,
    external_id="hist-1",
    source="github",
    org=None,
    project=None,
    collections=("documentation",),
):
    """Reproduce PRE-MIGRATION state for one logical document: `versions` rows,
    all `superseded_at IS NULL`, each with metadata and chunks in the given
    collections (so every old version is still 'current' and searchable)."""
    org = org or env.org_id
    project = project or env.project_id
    ids = []
    async with env.factory() as session:
        for n in range(1, versions + 1):
            doc_id = uuid.uuid4()
            await session.execute(
                text(
                    "INSERT INTO documents (id, organization_id, project_id, source, external_id, "
                    "content_hash, status, version, created_at, updated_at) VALUES "
                    "(:id, :o, :p, :s, :e, :h, 'published', :v, :t, :t)"
                ),
                {
                    "id": doc_id, "o": org, "p": project, "s": source, "e": external_id,
                    "h": f"hash-{external_id}-{n}", "v": n, "t": _BASE_TIME + timedelta(days=n),
                },
            )
            await session.execute(
                text(
                    "INSERT INTO document_metadata (document_id, key, value) "
                    "VALUES (:d, 'author', :v)"
                ),
                {"d": doc_id, "v": f"author-{n}"},
            )
            for collection in collections:
                await PgVectorStore().upsert(
                    session,
                    collection,
                    [
                        UpsertChunk(
                            document_id=doc_id, organization_id=org, project_id=project,
                            collection=collection, chunk_index=0,
                            content=f"version {n} connection pool content", source_offset_start=0,
                            source_offset_end=10,
                        )
                    ],
                    [[0.01] * 384],
                )
            ids.append(doc_id)
        await session.commit()
    return ids


async def states(env, ids):
    async with env.factory() as session:
        rows = (
            await session.execute(
                text("SELECT id, superseded_at, deleted_at FROM documents WHERE id = ANY(:ids)"),
                {"ids": ids},
            )
        ).all()
    return {r.id: r for r in rows}


async def all_chunk_counts(env, ids):
    return {i: await chunk_counts(env, i) for i in ids}


async def run(env, **kwargs):
    kwargs.setdefault("organization_ids", [env.org_id])
    kwargs.setdefault("sleep_seconds", 0)
    kwargs.setdefault("log", lambda _m: None)
    return await lifecycle.run_backfill(_scope(env), **kwargs)


async def test_backfill_dry_run_reports_but_changes_nothing(env):
    ids = await seed_history(env, 3)
    chunks_before = await all_chunk_counts(env, ids)

    report = await run(env, apply=False)

    assert report.organizations[0].versions_to_mark == 2
    assert all(r.superseded_at is None for r in (await states(env, ids)).values())
    assert await all_chunk_counts(env, ids) == chunks_before


async def test_mark_phase_supersedes_all_but_latest_without_deleting_anything(env):
    ids = await seed_history(env, 3)
    chunks_before = await all_chunk_counts(env, ids)

    report = await run(env, apply=True, phase="mark")

    state = await states(env, ids)
    assert state[ids[2]].superseded_at is None  # latest stays current
    # superseded_at = when the replacement happened (the NEXT version's created_at)
    assert state[ids[0]].superseded_at == _BASE_TIME + timedelta(days=2)
    assert state[ids[1]].superseded_at == _BASE_TIME + timedelta(days=3)
    assert report.organizations[0].versions_marked == 2
    assert report.organizations[0].keys_still_with_multiple_current == 0
    # mark phase deletes nothing
    assert await all_chunk_counts(env, ids) == chunks_before
    async with env.factory() as session:
        docs = (await session.execute(text("SELECT count(*) FROM documents"))).scalar_one()
        meta = (await session.execute(text("SELECT count(*) FROM document_metadata"))).scalar_one()
    assert (docs, meta) == (3, 3)


async def test_after_marking_retrieval_hides_old_versions_even_before_any_purge(env):
    ids = await seed_history(env, 3)
    await run(env, apply=True, phase="mark")

    async with env.factory() as session:
        await set_tenant_context(session, env.org_id)
        results = await PgVectorStore().lexical_search(
            session, "documentation", "connection pool", _filters(env), 20
        )

    assert {r.document_id for r in results} == {ids[2]}


async def test_purge_phase_deletes_chunks_of_superseded_docs_in_all_four_tables(env):
    ids = await seed_history(env, 3, collections=tuple(_ALL_COLLECTIONS))
    await run(env, apply=True, phase="mark")

    report = await run(env, apply=True, phase="purge")

    after = await all_chunk_counts(env, ids)
    assert all(sum(after[i].values()) == 0 for i in ids[:2])  # old versions: every table empty
    assert all(after[ids[2]][t] == 1 for t in _CHUNK_TABLES)  # current version untouched
    assert report.organizations[0].chunks_purged == {t: 2 for t in _CHUNK_TABLES}
    # History preserved: all three rows and their metadata still exist.
    state = await states(env, ids)
    assert len(state) == 3 and all(r.deleted_at is None for r in state.values())
    async with env.factory() as session:
        meta = (await session.execute(text("SELECT count(*) FROM document_metadata"))).scalar_one()
    assert meta == 3


async def test_backfill_is_idempotent(env):
    ids = await seed_history(env, 3, collections=tuple(_ALL_COLLECTIONS))
    await run(env, apply=True)
    snapshot = (await states(env, ids), await all_chunk_counts(env, ids))

    second = await run(env, apply=True)

    assert second.organizations[0].versions_marked == 0
    assert sum(second.organizations[0].chunks_purged.values()) == 0
    assert (await states(env, ids), await all_chunk_counts(env, ids)) == snapshot


async def test_backfill_resumes_after_an_interruption(env):
    ids_a = await seed_history(env, 3, external_id="k-a")
    ids_b = await seed_history(env, 3, external_id="k-b")
    ids_c = await seed_history(env, 3, external_id="k-c")

    class Boom(Exception):
        pass

    entries = 0
    real_scope = _scope(env)

    @asynccontextmanager
    async def flaky_scope():
        nonlocal entries
        entries += 1
        # Entry 1 is the backlog count, entry 2 the first mark batch (one key,
        # committed); die on entry 3 so the run stops mid-marking.
        if entries == 3:
            raise Boom
        async with real_scope() as session:
            yield session

    with pytest.raises(Boom):
        await lifecycle.run_backfill(
            flaky_scope, organization_ids=[env.org_id], apply=True, batch_size=1,
            sleep_seconds=0, log=lambda _m: None,
        )
    partial = await states(env, ids_a + ids_b + ids_c)
    marked_so_far = sum(1 for r in partial.values() if r.superseded_at is not None)
    assert 0 < marked_so_far < 6, "interruption should leave partial, committed progress"

    await run(env, apply=True, batch_size=1)  # resume: just run it again

    final = await states(env, ids_a + ids_b + ids_c)
    for ids in (ids_a, ids_b, ids_c):
        assert [final[i].superseded_at is None for i in ids] == [False, False, True]
        assert sum((await chunk_counts(env, ids[0])).values()) == 0
        assert sum((await chunk_counts(env, ids[1])).values()) == 0
        assert sum((await chunk_counts(env, ids[2])).values()) >= 1


async def test_backfill_only_touches_the_requested_organization(env):
    mine = await seed_history(env, 3, external_id="shared-key")
    async with env.factory() as session:
        other_project = (
            await session.execute(
                text("SELECT id FROM projects WHERE organization_id = :o"), {"o": env.other_org_id}
            )
        ).scalar_one()
    theirs = await seed_history(
        env, 3, external_id="shared-key", org=env.other_org_id, project=other_project
    )
    theirs_before = (await states(env, theirs), await all_chunk_counts(env, theirs))

    await run(env, apply=True, organization_ids=[env.org_id])

    assert (await states(env, theirs), await all_chunk_counts(env, theirs)) == theirs_before
    assert (await states(env, mine))[mine[0]].superseded_at is not None

    await run(env, apply=True, organization_ids=[env.other_org_id])
    assert (await states(env, theirs))[theirs[0]].superseded_at is not None


async def test_manual_documents_are_never_marked(env):
    ids = await seed_history(env, 3, external_id="manual-doc", source="manual")

    report = await run(env, apply=True)

    assert report.organizations[0].versions_to_mark == 0
    assert all(r.superseded_at is None for r in (await states(env, ids)).values())


async def test_soft_deleted_documents_keep_their_chunks_unless_asked(env):
    ids = await seed_history(env, 1, external_id="pruned")
    async with env.factory() as session:
        await session.execute(
            text("UPDATE documents SET deleted_at = now() WHERE id = :i"), {"i": ids[0]}
        )
        await session.commit()

    await run(env, apply=True)  # default: leave soft-deleted documents' chunks alone
    assert sum((await chunk_counts(env, ids[0])).values()) == 1

    await run(env, apply=True, include_soft_deleted=True)
    assert sum((await chunk_counts(env, ids[0])).values()) == 0
    assert ids[0] in await states(env, ids)  # the documents row itself is never deleted


async def test_documents_ingested_after_backfill_stay_consistent(env):
    ids = await seed_history(env, 2, external_id="live-doc", source="confluence")
    await run(env, apply=True)

    # A later real sync of changed content must continue the chain cleanly.
    await ingest(env, B, external_id="live-doc")

    rows = await doc_rows(env, external_id="live-doc")
    assert [r.is_current for r in rows].count(True) == 1
    assert rows[-1].version == 3 and rows[-1].is_current
    assert {r.id for r in rows} >= set(ids)  # nothing from history was deleted


async def test_backfill_concurrent_with_live_ingestion_converges(env):
    await seed_history(env, 3, external_id="busy-doc", source="confluence")
    env.state.embed_delay = 0.2

    await asyncio.wait_for(
        asyncio.gather(
            run(env, apply=True, batch_size=1),
            ingest(env, C, external_id="busy-doc"),
        ),
        timeout=60,
    )
    await run(env, apply=True)  # a final pass picks up anything the race left

    rows = await doc_rows(env, external_id="busy-doc")
    assert [r.is_current for r in rows].count(True) == 1
    assert len({r.version for r in rows}) == len(rows)
    for old in [r for r in rows if not r.is_current]:
        assert sum((await chunk_counts(env, old.id)).values()) == 0

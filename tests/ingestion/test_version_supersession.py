"""Tests for document version supersession in
`app.ingestion.service._process_one_item` and the repository functions it uses.

What these tests are, and are not
---------------------------------
No Postgres is available to this suite (same disclosed limitation as every
other repository test here). The service logic is therefore exercised against
`_FakeDb`, an in-memory model of exactly the behaviors this feature relies on:

* the `uq_documents_org_source_external_hash` unique constraint,
* a per-key mutual-exclusion lock standing in for `pg_advisory_xact_lock`
  (held until the item's COMMIT / ROLLBACK),
* savepoint semantics -- `begin_nested()` snapshots state and restores it if
  the block raises,
* a retrieval view that returns only chunks of current, non-deleted documents.

`_process_one_item_with_retry` (the real wrapper around the real
`_process_one_item`) drives every scenario, so the real call order, real
savepoint use and real commit/lock-release points are what is under test.
The fake proves the LOGIC (ordering, double-checked locking, atomic scope,
revive-instead-of-duplicate); it cannot prove Postgres' own lock/savepoint
implementation. The statement-shape tests at the bottom compile the real SQL
instead.
"""

from __future__ import annotations

import asyncio
import copy
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.ingestion import repository
from app.ingestion import service as ingestion_service
from app.ingestion.connectors.incidents import IncidentsConnector
from app.ingestion.schemas import DocumentMetadataEntry, RawDocument

ORG = uuid.uuid4()
OTHER_ORG = uuid.uuid4()
PROJECT = uuid.uuid4()

_ALL_COLLECTIONS = ["documentation", "code", "conversations", "incidents"]


class _UniqueViolation(Exception):
    """Stands in for the `uq_documents_org_source_external_hash` violation."""


class _FakeDb:
    def __init__(self) -> None:
        self.rows: dict[uuid.UUID, SimpleNamespace] = {}
        self.chunks: dict[tuple[str, uuid.UUID], list[str]] = {}
        self.locks: dict[tuple, asyncio.Lock] = {}
        self.ops: list[str] = []
        self.use_lock = True
        self.fail_upsert = False
        self.on_upsert_failure = None

    # -- savepoint emulation ------------------------------------------------
    def snapshot(self):
        return copy.deepcopy((self.rows, self.chunks))

    def restore(self, snap) -> None:
        self.rows, self.chunks = copy.deepcopy(snap)

    # -- helpers for assertions --------------------------------------------
    def rows_for(self, external_id: str = "doc-1") -> list[SimpleNamespace]:
        return sorted(
            (r for r in self.rows.values() if r.external_id == external_id),
            key=lambda r: r.version,
        )

    def current(self, external_id: str = "doc-1") -> list[SimpleNamespace]:
        return [r for r in self.rows_for(external_id) if r.superseded_at is None]

    def searchable_chunks(self) -> list[tuple[str, uuid.UUID, str]]:
        """What retrieval would return: chunks of current, non-deleted docs."""
        out = []
        for (collection, doc_id), contents in self.chunks.items():
            row = self.rows.get(doc_id)
            if row is None or row.superseded_at is not None or row.deleted_at is not None:
                continue
            out.extend((collection, doc_id, c) for c in contents)
        return out

    def chunk_collections_of(self, doc_id: uuid.UUID) -> set[str]:
        return {c for (c, d), v in self.chunks.items() if d == doc_id and v}

    # -- install fakes over the real repository / retrieval functions -------
    def install(self, monkeypatch) -> None:
        db = self
        repo = ingestion_service.repository
        retrieval = ingestion_service.retrieval_service

        async def get_latest_document(session, org, source, external_id):
            await asyncio.sleep(0)  # let a concurrent task interleave here
            rows = [
                r
                for r in db.rows.values()
                if (r.organization_id, r.source, r.external_id) == (org, source, external_id)
            ]
            return max(rows, key=lambda r: r.version) if rows else None

        async def lock_document_key(session, org, source, external_id):
            db.ops.append("lock")
            if not db.use_lock:
                return
            lock = db.locks.setdefault((org, source, external_id), asyncio.Lock())
            await lock.acquire()
            session.held.append(lock)

        async def find_document_by_content_hash(session, org, source, external_id, content_hash):
            for r in db.rows.values():
                if (r.organization_id, r.source, r.external_id, r.content_hash) == (
                    org,
                    source,
                    external_id,
                    content_hash,
                ):
                    return r
            return None

        async def insert_document(
            session,
            *,
            organization_id,
            project_id,
            source,
            external_id,
            content_hash,
            title,
            source_url,
            version,
            acl_permission_code=None,
        ):
            await asyncio.sleep(0)
            if await find_document_by_content_hash(
                session, organization_id, source, external_id, content_hash
            ):
                raise _UniqueViolation("uq_documents_org_source_external_hash")
            row = SimpleNamespace(
                id=uuid.uuid4(),
                organization_id=organization_id,
                project_id=project_id,
                source=source,
                external_id=external_id,
                content_hash=content_hash,
                title=title,
                source_url=source_url,
                version=version,
                acl_permission_code=acl_permission_code,
                superseded_at=None,
                deleted_at=None,
                metadata=[],
            )
            db.rows[row.id] = row
            db.ops.append("insert")
            return row

        async def insert_document_metadata(session, *, document_id, entries):
            db.rows[document_id].metadata = [(e.key, e.value) for e in entries]

        async def revive_document(session, row, *, version, project_id, title, source_url):
            row.superseded_at = None
            row.version = version
            row.project_id = project_id
            row.title = title
            row.source_url = source_url
            db.ops.append("revive")
            return row

        async def replace_document_metadata(session, *, document_id, entries):
            db.rows[document_id].metadata = [(e.key, e.value) for e in entries]
            db.ops.append("replace_metadata")

        async def supersede_other_versions(
            session, *, organization_id, source, external_id, keep_document_id
        ):
            ids = []
            for r in db.rows.values():
                if (
                    (r.organization_id, r.source, r.external_id)
                    == (organization_id, source, external_id)
                    and r.superseded_at is None
                    and r.id != keep_document_id
                ):
                    r.superseded_at = datetime.now(UTC)
                    ids.append(r.id)
            db.ops.append("supersede")
            return ids

        async def delete(session, collection, document_id):
            db.ops.append(f"purge:{collection}")
            db.chunks.pop((collection, document_id), None)

        async def upsert(session, chunks):
            await asyncio.sleep(0)
            db.ops.append("upsert")
            if db.fail_upsert:
                if db.on_upsert_failure is not None:
                    db.on_upsert_failure()
                raise RuntimeError("embedding service unavailable")
            grouped: dict[tuple[str, uuid.UUID], list[str]] = {}
            for chunk in chunks:
                grouped.setdefault((chunk.collection, chunk.document_id), []).append(chunk.content)
            db.chunks.update(grouped)

        async def no_tenant_context(session, organization_id):
            return None

        for name, fn in {
            "get_latest_document": get_latest_document,
            "lock_document_key": lock_document_key,
            "find_document_by_content_hash": find_document_by_content_hash,
            "insert_document": insert_document,
            "insert_document_metadata": insert_document_metadata,
            "revive_document": revive_document,
            "replace_document_metadata": replace_document_metadata,
            "supersede_other_versions": supersede_other_versions,
        }.items():
            monkeypatch.setattr(repo, name, fn)
        monkeypatch.setattr(retrieval, "delete", delete)
        monkeypatch.setattr(retrieval, "upsert", upsert)
        monkeypatch.setattr(ingestion_service, "set_tenant_context", no_tenant_context)


class _Savepoint:
    def __init__(self, session: _FakeSession) -> None:
        self._session = session

    async def __aenter__(self):
        self._snapshot = self._session.db.snapshot()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self._session.db.restore(self._snapshot)  # ROLLBACK TO SAVEPOINT
        return False


class _FakeSession:
    def __init__(self, db: _FakeDb) -> None:
        self.db = db
        self.held: list[asyncio.Lock] = []
        self.commits = 0

    def begin_nested(self) -> _Savepoint:
        return _Savepoint(self)

    def _release(self) -> None:
        for lock in self.held:
            lock.release()
        self.held.clear()

    async def commit(self) -> None:  # COMMIT releases transaction-scoped locks
        self.commits += 1
        self._release()

    async def rollback(self) -> None:
        self._release()


class _Connector:
    def __init__(self, source: str) -> None:
        self.source_name = source

    def normalize(self, raw_item):
        return RawDocument(
            source=self.source_name,
            external_id=raw_item["id"],
            content=raw_item["content"],
            title=raw_item.get("title", "Title"),
            source_url=f"https://example.test/{raw_item['id']}",
            metadata=raw_item.get("metadata", {}),
        )


@pytest.fixture
def db(monkeypatch) -> _FakeDb:
    database = _FakeDb()
    database.install(monkeypatch)
    return database


async def _ingest(
    db: _FakeDb,
    content: str,
    *,
    source: str = "confluence",
    external_id: str = "doc-1",
    org: uuid.UUID = ORG,
    metadata: dict[str, str] | None = None,
):
    session = _FakeSession(db)
    try:
        result = await ingestion_service._process_one_item_with_retry(
            session,
            organization_id=org,
            connector=_Connector(source),
            raw_item={"id": external_id, "content": content, "metadata": metadata or {}},
            resolved_config=SimpleNamespace(organization_id=org),
            project_id=PROJECT,
        )
    except BaseException:
        await session.rollback()
        raise
    # End of the item's transaction: releases a lock still held by an item
    # that took the lock, found nothing to do, and so never committed.
    await session.rollback()
    return result, session


A = "Runbook A: restart the connection pool before failing over the primary."
B = "Runbook B: drain traffic first, then restart the connection pool carefully."
C = "Runbook C: page the on-call database engineer immediately."


# --- normal updates ---------------------------------------------------------


@pytest.mark.asyncio
async def test_first_ingest_creates_version_one_with_chunks(db):
    result, session = await _ingest(db, A)

    assert result.documents_processed == 1
    assert session.commits == 1
    [row] = db.rows_for()
    assert (row.version, row.superseded_at) == (1, None)
    assert db.searchable_chunks()


@pytest.mark.asyncio
async def test_unchanged_content_is_a_noop_and_takes_no_lock(db):
    await _ingest(db, A)
    db.ops.clear()

    result, session = await _ingest(db, A)

    assert (result.documents_processed, result.chunks_embedded) == (0, 0)
    assert session.commits == 0
    assert db.ops == []  # fast path: no lock, no writes
    assert len(db.rows_for()) == 1


@pytest.mark.asyncio
async def test_update_supersedes_old_version_keeps_its_row_and_purges_its_chunks(db):
    await _ingest(db, A)
    [v1] = db.rows_for()
    v1_id = v1.id
    assert db.chunk_collections_of(v1_id) == {"documentation"}
    db.ops.clear()

    await _ingest(db, B)

    v1_after, v2 = db.rows_for()
    # History preserved: same id, same row, now superseded -- a citation that
    # names v1's document_id still resolves.
    assert v1_after.id == v1_id and v1_after.superseded_at is not None
    assert (v2.version, v2.superseded_at) == (2, None)
    # Old chunks gone from every collection; only the new version is searchable.
    assert db.chunk_collections_of(v1_id) == set()
    assert {doc_id for _, doc_id, _ in db.searchable_chunks()} == {v2.id}
    # Order matters for atomicity/consistency: serialize, write the new
    # version, then supersede + purge, then embed/store the new chunks.
    ops = db.ops
    assert ops.index("lock") < ops.index("insert") < ops.index("supersede") < ops.index("upsert")
    assert ops.index("supersede") < min(i for i, o in enumerate(ops) if o.startswith("purge:"))


@pytest.mark.asyncio
async def test_supersession_purges_all_four_collections_including_incidents(db):
    await _ingest(db, A)
    [v1] = db.rows_for()
    # Simulate chunks left in every collection (e.g. the doc was re-classified
    # between versions, or an older purge missed one).
    for collection in _ALL_COLLECTIONS:
        db.chunks[(collection, v1.id)] = ["stale"]

    await _ingest(db, B)

    assert db.chunk_collections_of(v1.id) == set()
    purged = {o.split(":", 1)[1] for o in db.ops if o.startswith("purge:")}
    assert purged == set(_ALL_COLLECTIONS)


@pytest.mark.asyncio
async def test_incident_documents_lose_old_chunks_from_incidents_collection(db):
    source = IncidentsConnector.source_name
    await _ingest(db, "Incident: database outage. Root cause: pool exhaustion.", source=source)
    [v1] = db.rows_for()
    assert db.chunk_collections_of(v1.id) == {"incidents"}

    await _ingest(db, "Incident: database outage. Root cause: failover misconfig.", source=source)

    assert db.chunk_collections_of(v1.id) == set()
    [_, v2] = db.rows_for()
    assert db.chunk_collections_of(v2.id) == {"incidents"}


@pytest.mark.asyncio
async def test_ingestion_self_heals_history_that_predates_superseded_at(db):
    """Rows written before the column existed are all `superseded_at IS NULL`.
    Touching the key must leave exactly one current row and purge the rest."""
    for content in (A, B, C):
        await _ingest(db, content)
    # Undo the bookkeeping to reproduce the pre-migration state.
    for row in db.rows.values():
        row.superseded_at = None
    for row in db.rows_for():
        db.chunks[("documentation", row.id)] = ["old chunk"]

    await _ingest(db, A + " (edited)")

    assert len(db.current()) == 1
    assert db.current()[0].version == 4
    assert {doc_id for _, doc_id, _ in db.searchable_chunks()} == {db.current()[0].id}


@pytest.mark.asyncio
async def test_other_documents_and_tenants_are_untouched(db):
    await _ingest(db, A, external_id="doc-1")
    await _ingest(db, A, external_id="doc-2")
    await _ingest(db, A, external_id="doc-1", org=OTHER_ORG)
    before = {
        r.id
        for r in db.rows.values()
        if not (r.external_id == "doc-1" and r.organization_id == ORG)
    }

    await _ingest(db, B, external_id="doc-1", org=ORG)

    for row in db.rows.values():
        if row.id in before:
            assert row.superseded_at is None
            assert db.chunk_collections_of(row.id) == {"documentation"}


@pytest.mark.asyncio
async def test_soft_deleted_latest_with_same_content_stays_deleted(db):
    """Ops soft-delete (e.g. `scripts/prune_vendored_documents.py`) must not be
    undone by the next sync of unchanged content."""
    await _ingest(db, A)
    [v1] = db.rows_for()
    v1.deleted_at = datetime.now(UTC)

    result, _ = await _ingest(db, A)

    assert result.documents_processed == 0
    assert len(db.rows_for()) == 1 and v1.deleted_at is not None


# --- A -> B -> A reverts ----------------------------------------------------


@pytest.mark.asyncio
async def test_revert_a_b_a_revives_the_old_row_instead_of_inserting_a_duplicate(db):
    await _ingest(db, A, metadata={"author": "alice"})
    [a_row] = db.rows_for()
    a_id = a_row.id
    await _ingest(db, B)

    result, _ = await _ingest(db, A, metadata={"author": "bob"})  # must not raise

    assert result.documents_processed == 1
    rows = db.rows_for()
    assert len(rows) == 2, "a revert must reuse the A row, not add a third"
    revived = next(r for r in rows if r.id == a_id)
    b_row = next(r for r in rows if r.id != a_id)
    assert revived.superseded_at is None and revived.version == 3
    assert b_row.superseded_at is not None and b_row.version == 2
    assert dict(revived.metadata)["author"] == "bob"  # metadata reflects this fetch
    assert "replace_metadata" in db.ops
    # Exactly one current version, and it is the only searchable one.
    assert [r.id for r in db.current()] == [a_id]
    assert {doc_id for _, doc_id, _ in db.searchable_chunks()} == {a_id}
    assert db.chunk_collections_of(b_row.id) == set()


@pytest.mark.asyncio
async def test_repeated_reverts_keep_one_row_per_distinct_content(db):
    for content in (A, B, A, B, A, C, A):
        await _ingest(db, content)

    assert len(db.rows_for()) == 3  # A, B, C -- never a duplicate hash
    assert len(db.current()) == 1
    versions = [r.version for r in db.rows_for()]
    assert len(set(versions)) == len(versions)
    current = db.current()[0]
    # Seven ingests, each a change: the last one (A again) holds version 7.
    assert current.version == 7 == max(versions)


@pytest.mark.asyncio
async def test_revert_clears_stale_chunks_of_the_revived_row_before_upserting(db):
    await _ingest(db, A)
    [a_row] = db.rows_for()
    await _ingest(db, B)
    db.chunks[("documentation", a_row.id)] = ["leftover from before cleanup existed"]
    db.ops.clear()

    await _ingest(db, A)

    assert db.chunks[("documentation", a_row.id)] != ["leftover from before cleanup existed"]
    # purge of the revived row happens before the new chunks are stored
    first_purge = min(i for i, o in enumerate(db.ops) if o.startswith("purge:"))
    assert first_purge < db.ops.index("upsert")


@pytest.mark.asyncio
async def test_revert_never_revives_a_soft_deleted_row(db):
    await _ingest(db, A)
    [a_row] = db.rows_for()
    await _ingest(db, B)
    a_row.deleted_at = datetime.now(UTC)

    def state():
        return sorted((str(r.id), r.version, r.superseded_at is None) for r in db.rows_for())

    before = state()

    with pytest.raises(_UniqueViolation):  # same outcome as before this change
        await _ingest(db, A)

    assert state() == before  # rolled back: nothing revived, nothing added
    assert db.rows[a_row.id].deleted_at is not None


# --- atomicity / rollback ---------------------------------------------------


@pytest.mark.asyncio
async def test_failed_embedding_rolls_back_supersession_and_chunk_purge(db):
    await _ingest(db, A)
    [v1] = db.rows_for()
    observed_in_flight: dict[str, object] = {}

    def capture_state_at_failure() -> None:
        # At the moment embedding fails, the supersession and purge have
        # ALREADY happened inside this item's transaction -- so they are only
        # undone if the savepoint rollback covers them.
        observed_in_flight["v1_superseded"] = v1.superseded_at is not None
        observed_in_flight["v1_chunks_gone"] = not db.chunk_collections_of(v1.id)

    db.fail_upsert = True
    db.on_upsert_failure = capture_state_at_failure
    snapshot_before = db.snapshot()

    session = _FakeSession(db)
    with pytest.raises(RuntimeError, match="embedding service unavailable"):
        await ingestion_service._process_one_item_with_retry(
            session,
            organization_id=ORG,
            connector=_Connector("confluence"),
            raw_item={"id": "doc-1", "content": B},
            resolved_config=SimpleNamespace(organization_id=ORG),
            project_id=PROJECT,
        )
    await session.rollback()

    assert observed_in_flight == {"v1_superseded": True, "v1_chunks_gone": True}
    # Rolled back: v1 is current and searchable again, no v2 row exists.
    assert session.commits == 0
    rows = db.rows_for()
    assert len(rows) == 1
    assert rows[0].id == v1.id and rows[0].superseded_at is None
    assert db.chunk_collections_of(v1.id) == {"documentation"}
    assert db.snapshot()[1] == snapshot_before[1]


@pytest.mark.asyncio
async def test_failed_revert_rolls_back_revival_and_leaves_state_unchanged(db):
    await _ingest(db, A)
    await _ingest(db, B)
    state_before = db.snapshot()
    db.fail_upsert = True

    with pytest.raises(RuntimeError):
        await _ingest(db, A)

    rows_after, chunks_after = db.snapshot()
    assert chunks_after == state_before[1]
    assert {(r.id, r.version, r.superseded_at is None) for r in rows_after.values()} == {
        (r.id, r.version, r.superseded_at is None) for r in state_before[0].values()
    }


@pytest.mark.asyncio
async def test_failed_item_releases_its_lock_so_the_retry_can_proceed(db):
    await _ingest(db, A)
    db.fail_upsert = True
    with pytest.raises(RuntimeError):
        await _ingest(db, B)

    db.fail_upsert = False
    result, _ = await asyncio.wait_for(_ingest(db, B), timeout=2)  # would hang if the lock leaked

    assert result.documents_processed == 1
    assert [r.version for r in db.rows_for()] == [1, 2]


# --- concurrency (Redis lock failing open) ----------------------------------


@pytest.mark.asyncio
async def test_concurrent_ingest_of_identical_new_content_creates_one_version(db):
    results = await asyncio.gather(_ingest(db, A), _ingest(db, A))

    processed = sorted(r.documents_processed for r, _ in results)
    assert processed == [0, 1], "exactly one writer persists; the other sees it and no-ops"
    assert len(db.rows_for()) == 1
    assert len(db.current()) == 1


@pytest.mark.asyncio
async def test_concurrent_ingest_of_different_content_serializes_into_a_clean_chain(db):
    await _ingest(db, A)

    await asyncio.gather(_ingest(db, B), _ingest(db, C))

    rows = db.rows_for()
    versions = [r.version for r in rows]
    assert versions == [1, 2, 3], "no duplicate or skipped version numbers"
    assert len(db.current()) == 1 and db.current()[0].version == 3
    assert len({doc_id for _, doc_id, _ in db.searchable_chunks()}) == 1


@pytest.mark.asyncio
async def test_concurrent_revert_and_update_stay_consistent(db):
    await _ingest(db, A)
    await _ingest(db, B)

    await asyncio.gather(_ingest(db, A), _ingest(db, C))

    assert len(db.current()) == 1
    versions = [r.version for r in db.rows_for()]
    assert len(set(versions)) == len(versions)
    assert len({doc_id for _, doc_id, _ in db.searchable_chunks()}) == 1


@pytest.mark.asyncio
async def test_harness_detects_the_race_when_the_lock_is_removed(db):
    """Mutation check: proves the concurrency tests above are sensitive --
    with the per-key lock disabled the same scenario corrupts the chain."""
    await _ingest(db, A)
    db.use_lock = False

    await asyncio.gather(_ingest(db, B), _ingest(db, C))

    versions = [r.version for r in db.rows_for()]
    assert len(set(versions)) != len(versions), "expected duplicate versions without the lock"


@pytest.mark.asyncio
async def test_different_keys_do_not_block_each_other(db):
    results = await asyncio.wait_for(
        asyncio.gather(
            _ingest(db, A, external_id="doc-1"),
            _ingest(db, B, external_id="doc-2"),
            _ingest(db, C, external_id="doc-3", org=OTHER_ORG),
        ),
        timeout=2,
    )
    assert all(r.documents_processed == 1 for r, _ in results)


# --- real repository functions: statement shape -----------------------------


class _Result:
    def __init__(self, rows=None):
        self._rows = rows or []

    def scalars(self):
        return self

    def all(self):
        return self._rows

    def scalar_one_or_none(self):
        return self._rows[0] if self._rows else None


class _CapturingSession:
    def __init__(self):
        self.executed: list[tuple[object, dict]] = []
        self.added: list[object] = []

    async def execute(self, statement, params=None):
        self.executed.append((statement, params or {}))
        return _Result()

    def add(self, row):
        self.added.append(row)

    async def flush(self):
        return None


def _sql(statement) -> str:
    return str(statement.compile(compile_kwargs={"literal_binds": True})).lower()


@pytest.mark.asyncio
async def test_lock_document_key_uses_a_transaction_scoped_advisory_lock_keyed_per_item():
    session = _CapturingSession()

    await repository.lock_document_key(session, ORG, "github", "owner/repo:README.md")
    await repository.lock_document_key(session, OTHER_ORG, "github", "owner/repo:README.md")

    (stmt1, params1), (stmt2, params2) = session.executed
    sql = str(stmt1).lower()
    assert "pg_advisory_xact_lock" in sql, "must be transaction-scoped, never session-scoped"
    assert "pg_advisory_lock(" not in sql
    assert str(ORG) in params1["key"] and "github" in params1["key"]
    assert params1["key"] != params2["key"], "tenants must not share a lock key"


@pytest.mark.asyncio
async def test_supersede_other_versions_is_tenant_scoped_idempotent_and_returns_ids():
    session = _CapturingSession()
    keep = uuid.uuid4()

    await repository.supersede_other_versions(
        session, organization_id=ORG, source="github", external_id="x", keep_document_id=keep
    )

    sql = _sql(session.executed[0][0])
    assert sql.startswith("update documents")
    assert "organization_id" in sql and ORG.hex in sql
    assert "documents.source = 'github'" in sql and "documents.external_id = 'x'" in sql
    assert "superseded_at is null" in sql  # idempotent: already-superseded rows skipped
    assert "documents.id !=" in sql and keep.hex in sql
    assert "returning documents.id" in sql
    assert "deleted_at" not in sql.split("where", 1)[1], "must not depend on deleted_at"


@pytest.mark.asyncio
async def test_find_document_by_content_hash_matches_the_full_idempotency_key():
    session = _CapturingSession()

    await repository.find_document_by_content_hash(session, ORG, "github", "x", "abc123")

    sql = _sql(session.executed[0][0])
    for fragment in (
        f"documents.organization_id = '{ORG.hex}'",
        "documents.source = 'github'",
        "documents.external_id = 'x'",
        "documents.content_hash = 'abc123'",
    ):
        assert fragment in sql


@pytest.mark.asyncio
async def test_replace_document_metadata_only_touches_the_given_document():
    session = _CapturingSession()
    doc_id = uuid.uuid4()

    await repository.replace_document_metadata(
        session, document_id=doc_id, entries=[DocumentMetadataEntry(key="k", value="v")]
    )

    delete_sql = _sql(session.executed[0][0])
    assert delete_sql.startswith("delete from document_metadata")
    assert f"document_metadata.document_id = '{doc_id.hex}'" in delete_sql
    assert len(session.added) == 1 and session.added[0].document_id == doc_id


@pytest.mark.asyncio
async def test_revive_document_clears_superseded_at_and_sets_the_new_version():
    session = _CapturingSession()
    row = SimpleNamespace(
        superseded_at=datetime.now(UTC), version=1, project_id=None, title="old", source_url=None
    )
    new_project = uuid.uuid4()

    result = await repository.revive_document(
        session, row, version=5, project_id=new_project, title="new", source_url="https://x"
    )

    assert result is row
    assert (row.superseded_at, row.version, row.project_id, row.title) == (
        None,
        5,
        new_project,
        "new",
    )

"""Shared helpers for the real-PostgreSQL HNSW tests (not a test module).

Contract (same as `tests/core/privacy/test_member_safeguards_rls_postgres.py`):
the tests are skipped unless BOTH of these point at a LOCAL, disposable,
MIGRATED (`alembic upgrade head`) database with pgvector >= 0.8.0:

    EKIP_TEST_POSTGRES_URL   superuser/owner URL -- seeds data, inspects past
                             RLS, runs the migration, cleans up
    EKIP_TEST_APP_URL        the `ekip_app` URL (NOSUPERUSER NOBYPASSRLS, RLS
                             FORCED on the tenant tables) -- every search under
                             test runs through THIS role

Embeddings are CLUSTERED (topic centres + noise), not uniform random: HNSW
behaves very differently on random vectors, where every neighbour is almost
equidistant. The exact baseline is an independent numpy brute force over the
in-memory copy of what was seeded, not another SQL query.
"""

from __future__ import annotations

import asyncio
import importlib.util
import io
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import asyncpg
import numpy as np
import pytest
from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

SUPER_URL = os.environ.get("EKIP_TEST_POSTGRES_URL", "")
APP_URL = os.environ.get("EKIP_TEST_APP_URL", "")
DIM = 384
TABLES = ("documentation_chunks", "code_chunks", "conversations_chunks", "incidents_chunks")
COLLECTION_TABLE = {
    "documentation": "documentation_chunks",
    "code": "code_chunks",
    "conversations": "conversations_chunks",
    "incidents": "incidents_chunks",
}
RESTRICTED = "docs:restricted"


def _is_local(url: str) -> bool:
    return (urlsplit(url.replace("+asyncpg", "")).hostname or "") in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


requires_pg = pytest.mark.skipif(
    not (SUPER_URL and APP_URL and _is_local(SUPER_URL) and _is_local(APP_URL)),
    reason="needs EKIP_TEST_POSTGRES_URL and EKIP_TEST_APP_URL pointing at a LOCAL migrated DB",
)


def asyncpg_dsn(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql://").split("?")[0]


def engine(url: str, **connect_args):
    return create_async_engine(
        url.replace("?sslmode=disable", ""),
        poolclass=NullPool,
        connect_args={"ssl": False, **connect_args},
    )


# --- the migration under test -----------------------------------------------

_MIGRATION_FILE = (
    Path(__file__).resolve().parents[2]
    / "app/database/migrations/versions/b7c1d9e4f2a3_hnsw_chunk_embedding_indexes.py"
)


def load_migration():
    spec = importlib.util.spec_from_file_location("hnsw_migration_under_test", _MIGRATION_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_migration(direction: str, *, command_timeout: float | None = None) -> None:
    """Run the migration's `upgrade`/`downgrade` through Alembic's own
    operations/context on a real connection (so `autocommit_block` behaves as
    it does under `alembic upgrade`). `command_timeout` mimics the engine's
    client-side asyncpg timeout (the request engines use 30 s)."""
    module = load_migration()
    eng = engine(SUPER_URL, command_timeout=command_timeout)

    def _run(sync_conn):
        ctx = MigrationContext.configure(sync_conn)
        with Operations.context(ctx):
            getattr(module, direction)()

    try:
        async with eng.connect() as conn:
            await conn.run_sync(_run)
            await conn.commit()
    finally:
        await eng.dispose()


# --- clustered dataset -------------------------------------------------------


@dataclass
class Table:
    ids: list = field(default_factory=list)
    org: list = field(default_factory=list)
    project: list = field(default_factory=list)
    acl: list = field(default_factory=list)  # None or RESTRICTED
    repo: list = field(default_factory=list)
    hidden: list = field(default_factory=list)  # document deleted OR superseded
    vecs: list = field(default_factory=list)
    matrix: np.ndarray | None = None


@dataclass
class Dataset:
    tag: str
    centres: np.ndarray
    orgs: dict = field(default_factory=dict)  # slug -> uuid str
    projects: dict = field(default_factory=dict)  # slug -> [uuid str]
    tables: dict = field(default_factory=dict)  # table -> Table

    def org_id(self, slug: str) -> uuid.UUID:
        return uuid.UUID(self.orgs[slug])

    def project_id(self, slug: str, i: int) -> uuid.UUID:
        return uuid.UUID(self.projects[slug][i])

    def queries(self, n: int = 20, seed: int = 5) -> list[list[float]]:
        """Topical queries: a topic centre plus noise, normalized."""
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            c = self.centres[rng.integers(0, len(self.centres))]
            v = c + 0.04 * rng.normal(size=DIM).astype(np.float32)
            out.append((v / np.linalg.norm(v)).astype(float).tolist())
        return out

    def exact_top(
        self, collections, query, *, org, k, projects=None, perms=(), repo=None
    ) -> list[str]:
        """Brute-force exact top-k ids over everything seeded, applying the same
        filters the store applies (tenant, project, ACL, deleted/superseded)."""
        scored: list[tuple[float, str]] = []
        q = np.asarray(query, dtype=np.float32)
        for name in collections:
            t = self.tables[COLLECTION_TABLE[name]]
            if t.matrix is None:
                t.matrix = np.asarray(t.vecs, dtype=np.float32).reshape(-1, DIM)
            org_a, proj_a = np.asarray(t.org), np.asarray(t.project)
            mask = (org_a == org) & ~np.asarray(t.hidden, dtype=bool)
            if projects is not None:
                mask &= np.isin(proj_a, [str(p) for p in projects])
            acl_ok = np.asarray([a is None or a in perms for a in t.acl], dtype=bool)
            mask &= acl_ok
            if repo is not None:
                mask &= np.asarray([r == repo for r in t.repo], dtype=bool)
            idx = np.nonzero(mask)[0]
            if len(idx) == 0:
                continue
            sims = t.matrix[idx] @ q
            scored.extend((float(s), t.ids[i]) for s, i in zip(sims, idx, strict=True))
        scored.sort(key=lambda x: -x[0])
        return [i for _, i in scored[:k]]

    def eligible_count(self, collections, *, org, projects=None, perms=(), repo=None) -> int:
        return len(
            self.exact_top(
                collections,
                [1.0] + [0.0] * (DIM - 1),
                org=org,
                k=10**9,
                projects=projects,
                perms=perms,
                repo=repo,
            )
        )


# slug -> rows per table. Skewed on purpose: one dominant tenant (the planner
# should use HNSW for it), a mid tenant, and tenants far too small for an
# approximate scan to be the right plan.
_SIZES = {
    "documentation_chunks": {"big": 20000, "mid": 2500, "small": 600, "needle": 60, "tiny": 20},
    "code_chunks": {"big": 4000, "mid": 600, "needle": 40},
    "conversations_chunks": {"big": 4000, "mid": 500, "tiny": 20},
    "incidents_chunks": {"big": 300, "mid": 50},
}
_REPOS = ["acme/api", "acme/web", "acme/infra"]


def _vec_text(v) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


async def seed(tag: str, scale: float = 1.0) -> Dataset:
    rng = np.random.default_rng(11)
    centres = rng.normal(size=(120, DIM)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    ds = Dataset(tag=tag, centres=centres, tables={t: Table() for t in TABLES})
    conn = await asyncpg.connect(asyncpg_dsn(SUPER_URL), ssl=False)
    try:
        slugs = sorted({s for per in _SIZES.values() for s in per})
        for slug in slugs:
            full = f"hnsw-{tag}-{slug}"
            ds.orgs[slug] = str(
                await conn.fetchval(
                    "INSERT INTO organizations(name, slug, status) VALUES($1,$1,'active') RETURNING id",  # noqa: E501
                    full,
                )
            )
            n_proj = 4 if slug == "big" else 2
            ds.projects[slug] = [
                str(
                    await conn.fetchval(
                        "INSERT INTO projects(organization_id,name,is_default) VALUES($1,$2,$3) RETURNING id",  # noqa: E501
                        uuid.UUID(ds.orgs[slug]),
                        f"p{i}",
                        i == 0,
                    )
                )
                for i in range(n_proj)
            ]
        for table, per_org in _SIZES.items():
            t = ds.tables[table]
            for slug, total in per_org.items():
                org = uuid.UUID(ds.orgs[slug])
                plist = [uuid.UUID(p) for p in ds.projects[slug]]
                w = np.array(
                    [0.55, 0.30, 0.145, 0.005][: len(plist)] if slug == "big" else [0.7, 0.3]
                )
                w /= w.sum()
                total = max(10, int(total * scale))
                n_docs = max(1, total // 10)
                docs, buf = [], io.StringIO()
                for d in range(n_docs):
                    did, proj = uuid.uuid4(), plist[int(rng.choice(len(plist), p=w))]
                    deleted = bool(rng.random() < 0.04)
                    superseded = (not deleted) and bool(rng.random() < 0.04)
                    docs.append(
                        (
                            did,
                            org,
                            proj,
                            f"{table}-{slug}-{d}",
                            f"h-{table}-{slug}-{d}",
                            deleted,
                            superseded,
                            f"Doc {slug} {d}",
                        )
                    )
                    topic = int(rng.integers(0, len(centres)))
                    k = total // n_docs + (1 if d < total % n_docs else 0)
                    topics = np.where(rng.random(k) < 0.8, topic, rng.integers(0, len(centres), k))
                    v = centres[topics] + 0.04 * rng.normal(size=(k, DIM)).astype(np.float32)
                    v /= np.linalg.norm(v, axis=1, keepdims=True)
                    restricted = bool(rng.random() < 0.15)
                    has_repo = table in ("documentation_chunks", "code_chunks")
                    repo = _REPOS[int(rng.integers(0, 3))] if has_repo else None
                    for i in range(k):
                        cid = str(uuid.uuid4())
                        acl = RESTRICTED if restricted and rng.random() < 0.9 else None
                        cols = [
                            cid,
                            str(org),
                            str(proj),
                            str(did),
                            str(i),
                            f"chunk {i} {slug} {d}",
                            _vec_text(v[i]),
                            "0",
                            "10",
                            acl or "\\N",
                        ]
                        if has_repo:
                            cols.append(repo or "\\N")
                        buf.write("\t".join(cols) + "\n")
                        t.ids.append(cid)
                        t.org.append(str(org))
                        t.project.append(str(proj))
                        t.acl.append(acl)
                        t.repo.append(repo)
                        t.hidden.append(deleted or superseded)
                        t.vecs.append(v[i])
                await conn.executemany(
                    "INSERT INTO documents(id,organization_id,project_id,source,external_id,content_hash,"  # noqa: E501
                    "status,version,deleted_at,superseded_at,title) VALUES($1,$2,$3,'seed',$4,$5,"
                    "'published',1, CASE WHEN $6 THEN now() END, CASE WHEN $7 THEN now() END, $8)",
                    docs,
                )
                cols = (
                    "id,organization_id,project_id,document_id,chunk_index,content,embedding,"
                    "source_offset_start,source_offset_end,acl_permission_code"
                )
                if table in ("documentation_chunks", "code_chunks"):
                    cols += ",repo_full_name"
                await conn.copy_to_table(
                    table,
                    source=io.BytesIO(buf.getvalue().encode()),
                    columns=cols.split(","),
                    format="text",
                )
        await conn.execute("ANALYZE")
    except BaseException:
        await cleanup(ds)
        raise
    finally:
        await conn.close()
    return ds


async def cleanup(ds: Dataset) -> None:
    conn = await asyncpg.connect(asyncpg_dsn(SUPER_URL), ssl=False)
    try:
        org_ids = [uuid.UUID(v) for v in ds.orgs.values()]
        if not org_ids:
            return
        for table in (*TABLES, "documents", "projects"):
            await conn.execute(
                f"DELETE FROM {table} WHERE organization_id = ANY($1::uuid[])", org_ids
            )
        await conn.execute("DELETE FROM organizations WHERE id = ANY($1::uuid[])", org_ids)
    finally:
        await conn.close()


def seeded_dataset_fixture_body():
    """(setup, teardown) used by the module-scoped fixtures of the test files:
    drop the HNSW indexes, seed, then run the REAL migration on the populated
    tables -- so every test below runs against migration-built indexes."""

    async def setup() -> Dataset:
        await run_migration("downgrade")
        ds = await seed(uuid.uuid4().hex[:8])
        await run_migration("upgrade")
        conn = await asyncpg.connect(asyncpg_dsn(SUPER_URL), ssl=False)
        try:
            await conn.execute("ANALYZE")
        finally:
            await conn.close()
        return ds

    return setup, cleanup


def run(coro):
    return asyncio.run(coro)

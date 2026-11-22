"""Dense retrieval against REAL PostgreSQL with the REAL migration-built HNSW
indexes, as the production runtime role `ekip_app` (NOSUPERUSER NOBYPASSRLS,
row-level security FORCED on the chunk tables).

What this proves (and the mocked tests cannot):
  * index existence, validity, operator class and parameters;
  * `EXPLAIN` of `search()` and `search_all()` -- HNSW is used for a large
    tenant in EVERY branch, and NOT used for tiny tenants;
  * the transaction-scoped settings really are transaction-scoped;
  * correctness after many repeated executions of one prepared statement (the
    generic-plan trap);
  * strict distance ordering, `dense[0]`, recall against an independent numpy
    exact search over CLUSTERED embeddings, every filter, completeness
    (`top_k` rows whenever enough permitted rows exist), tenant isolation.

Skipped unless EKIP_TEST_POSTGRES_URL / EKIP_TEST_APP_URL point at a LOCAL,
disposable, migrated database with pgvector >= 0.8.0 (see `hnsw_support`).
"""

from __future__ import annotations

import json

import asyncpg
import pytest
from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import AsyncAdaptedQueuePool

from app.database.session import set_tenant_context
from app.retrieval.pgvector import store as store_module
from app.retrieval.pgvector.store import PgVectorStore
from app.retrieval.schemas import SearchFilters
from tests.retrieval import hnsw_support as H

pytestmark = H.requires_pg

K = 10
ALL3 = ("documentation", "code", "conversations")


@pytest.fixture(scope="module")
def ds():
    async def _version():
        conn = await asyncpg.connect(H.asyncpg_dsn(H.SUPER_URL), ssl=False)
        try:
            return await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname='vector'")
        finally:
            await conn.close()

    version = H.run(_version())
    if tuple(int(p) for p in version.split(".")[:3]) < (0, 8, 0):
        pytest.skip(f"pgvector {version} < 0.8.0")
    setup, teardown = H.seeded_dataset_fixture_body()
    dataset = H.run(setup())
    yield dataset
    H.run(teardown(dataset))


def _filters(ds, slug, *, projects=None, perms=(), repo=None) -> SearchFilters:
    return SearchFilters(
        organization_id=ds.org_id(slug),
        project_ids=None if projects is None else [ds.project_id(slug, i) for i in projects],
        repository=repo,
        permission_codes=frozenset(perms),
    )


class _App:
    """An `ekip_app` engine/session factory; `capture` records vector SQL."""

    def __init__(self, pooled: bool = False):
        url = H.APP_URL.replace("?sslmode=disable", "")
        if pooled:  # one reused connection => asyncpg's prepared-statement cache is exercised
            self.engine = create_async_engine(
                url,
                poolclass=AsyncAdaptedQueuePool,
                pool_size=1,
                max_overflow=0,
                connect_args={"ssl": False},
            )
        else:
            self.engine = H.engine(H.APP_URL)
        self.sessions = async_sessionmaker(self.engine, expire_on_commit=False)
        self.captured: list[tuple] = []

        @event.listens_for(self.engine.sync_engine, "before_cursor_execute")
        def _grab(conn, cur, statement, params, ctx, many):
            if "<#>" in statement and not statement.lstrip().upper().startswith("EXPLAIN"):
                self.captured.append((statement, params))

    async def close(self):
        await self.engine.dispose()


async def _run(app, ds, slug, kind, query, *, tenant=None, projects=None, perms=(), repo=None, k=K):
    async with app.sessions() as s:
        await set_tenant_context(s, ds.org_id(tenant or slug))
        f = _filters(ds, slug, projects=projects, perms=perms, repo=repo)
        store = PgVectorStore()
        if kind == "all":
            out = await store.search_all(s, query, f, k)
        else:
            out = await store.search(s, "documentation", query, f, k)
        await s.rollback()
    return out


def _nodes(plan, acc=None):
    acc = [] if acc is None else acc
    acc.append(plan)
    for child in plan.get("Plans", []):
        _nodes(child, acc)
    return acc


async def _plan(ds, slug, kind, *, projects=None):
    """EXPLAIN of the first vector statement the store sends, run in the SAME
    transaction (so the store's SET LOCAL settings are in force)."""
    app = _App()
    try:
        async with app.sessions() as s:
            await set_tenant_context(s, ds.org_id(slug))
            f = _filters(ds, slug, projects=projects)
            q = ds.queries(1)[0]
            store = PgVectorStore()
            if kind == "all":
                await store.search_all(s, q, f, K)
            else:
                await store.search(s, "documentation", q, f, K)
            statement, params = app.captured[0]
            conn = await s.connection()
            row = (
                await conn.exec_driver_sql("EXPLAIN (FORMAT JSON) " + statement, params)
            ).scalar()
            await s.rollback()
        doc = json.loads(row) if isinstance(row, str) else row
        return _nodes(doc[0]["Plan"])
    finally:
        await app.close()


def _hnsw_relations(nodes) -> set[str]:
    return {
        n["Relation Name"]
        for n in nodes
        if "hnsw" in n.get("Index Name", "") and "Relation Name" in n
    }


def _seq_scans(nodes) -> set[str]:
    return {
        n["Relation Name"]
        for n in nodes
        if n.get("Node Type") == "Seq Scan" and n.get("Relation Name", "").endswith("_chunks")
    }


# --- 1. index existence, validity, operator class, parameters ----------------


async def test_hnsw_indexes_exist_valid_with_ip_ops_and_parameters(ds):
    conn = await asyncpg.connect(H.asyncpg_dsn(H.SUPER_URL), ssl=False)
    try:
        for table in H.TABLES:
            row = await conn.fetchrow(
                "SELECT i.indisvalid AS valid, i.indisready AS ready, pg_get_indexdef(i.indexrelid) AS d "  # noqa: E501
                "FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname = $1",
                f"ix_{table}_embedding_hnsw",
            )
            assert row is not None, table
            assert row["valid"] and row["ready"], table
            assert (
                f"ON public.{table} USING hnsw (embedding vector_ip_ops) "
                "WITH (m='16', ef_construction='128')"
            ) in row["d"], row["d"]
        # no OTHER approximate index anywhere -- in particular none on agent_memories
        others = await conn.fetch(
            "SELECT indexname FROM pg_indexes WHERE schemaname='public' AND "
            "(indexdef ILIKE '%hnsw%' OR indexdef ILIKE '%ivfflat%') AND indexname NOT LIKE 'ix\\_%\\_embedding\\_hnsw'"  # noqa: E501
        )
        assert others == []
        assert (
            await conn.fetchval(
                "SELECT count(*) FROM pg_indexes WHERE tablename='agent_memories' "
                "AND (indexdef ILIKE '%hnsw%' OR indexdef ILIKE '%ivfflat%')"
            )
            == 0
        )
    finally:
        await conn.close()


# --- 2/3. plans: large vs small tenants --------------------------------------


async def test_search_all_uses_hnsw_in_every_branch_for_a_large_tenant(ds):
    nodes = await _plan(ds, "big", "all")
    assert _hnsw_relations(nodes) == {"documentation_chunks", "code_chunks", "conversations_chunks"}
    assert _seq_scans(nodes) == set(), "a full scan of a chunk table: HNSW not serving the query"


async def test_search_uses_hnsw_for_a_large_tenant(ds):
    nodes = await _plan(ds, "big", "single")
    assert _hnsw_relations(nodes) == {"documentation_chunks"}


@pytest.mark.parametrize("slug", ["needle", "tiny"])
@pytest.mark.parametrize("kind", ["all", "single"])
async def test_small_tenants_do_not_use_hnsw(ds, slug, kind):
    """The planner must keep tiny tenants on the exact (btree + sort) path: an
    HNSW scan there would filter a global candidate list down to almost nothing."""
    nodes = await _plan(ds, slug, kind)
    assert _hnsw_relations(nodes) == set()


# --- settings are transaction-scoped ------------------------------------------


async def test_settings_apply_in_the_callers_transaction_and_do_not_leak(ds):
    app = _App(pooled=True)
    try:
        async with app.sessions() as s:
            await set_tenant_context(s, ds.org_id("big"))
            await PgVectorStore().search_all(s, ds.queries(1)[0], _filters(ds, "big"), K)
            inside = {
                g: (await s.execute(text(f"SHOW {g}"))).scalar()
                for g in ("plan_cache_mode", "hnsw.ef_search", "hnsw.iterative_scan")
            }
            await s.rollback()
        async with app.sessions() as s:  # same pooled connection, next transaction
            after = {
                g: (await s.execute(text(f"SHOW {g}"))).scalar()
                for g in ("plan_cache_mode", "hnsw.ef_search", "hnsw.iterative_scan")
            }
            await s.rollback()
    finally:
        await app.close()
    assert inside == {
        "plan_cache_mode": "force_custom_plan",
        "hnsw.ef_search": "100",
        "hnsw.iterative_scan": "strict_order",
    }
    assert after == {
        "plan_cache_mode": "auto",
        "hnsw.ef_search": "40",
        "hnsw.iterative_scan": "off",
    }


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.5.1", False),
        ("0.7.4", False),
        ("0.8.0", True),
        ("0.8.7", True),
        ("0.10.2", True),
        ("1.0.0", True),
    ],
)
async def test_version_predicate_matches_0_8_0_and_later(version, expected):
    conn = await asyncpg.connect(H.asyncpg_dsn(H.SUPER_URL), ssl=False)
    try:
        got = await conn.fetchval(
            f"SELECT {store_module._PGVECTOR_AT_LEAST_0_8} FROM (SELECT $1::text AS extversion) t",
            version,
        )
    finally:
        await conn.close()
    assert got is expected


# --- 4. repeated executions (generic-plan trap) -------------------------------


@pytest.mark.parametrize(
    ("slug", "kind"), [("needle", "all"), ("tiny", "single"), ("small", "all"), ("big", "all")]
)
async def test_results_stay_exact_and_complete_across_repeated_executions(ds, slug, kind):
    """One pooled connection reuses asyncpg's prepared statement for the same SQL
    text; PostgreSQL may switch it to a generic plan after ~5 executions.
    `force_custom_plan` must keep every execution correct and complete."""
    app = _App(pooled=True)
    colls = ALL3 if kind == "all" else ("documentation",)
    try:
        # Make the generic-plan switch deterministic: put the pooled connection
        # itself into `force_generic_plan` (session level, survives the pool's
        # rollback). The store's transaction-local `force_custom_plan` must
        # override that for every vector query. A mixed workload (large tenant
        # first, then the target) is what triggers the same switch on its own
        # in production once a statement has been executed a few times.
        async with app.sessions() as s:
            await s.execute(text("SET plan_cache_mode = force_generic_plan"))
            await s.commit()
        for q in ds.queries(8, seed=77):
            await _run(app, ds, "big", kind, q)
        for i, q in enumerate(ds.queries(12, seed=21)):
            got = await _run(app, ds, slug, kind, q)
            want = ds.exact_top(colls, q, org=str(ds.org_id(slug)), k=K)
            assert len(got) == len(want), f"execution {i}: short result"
            overlap = len({str(c.chunk_id) for c in got} & set(want)) / max(1, len(want))
            floor = 1.0 if slug != "big" else 0.8
            assert overlap >= floor, f"execution {i}: recall {overlap}"
        # The exact fallback would mask a generic-plan regression in the result
        # checks above, so also read the plan-cache state of the SAME pooled
        # connection: no vector statement may ever have used a generic plan.
        async with app.sessions() as s:
            stats = (
                await s.execute(
                    text(
                        "SELECT coalesce(sum(generic_plans), 0) AS generic, "
                        "coalesce(sum(custom_plans), 0) AS custom "
                        "FROM pg_prepared_statements WHERE statement LIKE '%<#>%' "
                        "AND statement NOT LIKE '%pg_prepared_statements%'"
                    )
                )
            ).one()
            await s.rollback()
        assert stats.custom >= 20, "the prepared vector statement was not reused"
        assert stats.generic == 0, "PostgreSQL switched the vector query to a generic plan"
    finally:
        await app.close()


# --- 5. strict ordering and dense[0] -------------------------------------------


@pytest.mark.parametrize("slug", ["big", "mid", "small"])
@pytest.mark.parametrize("kind", ["all", "single"])
async def test_results_are_in_strict_distance_order_and_dense0_is_the_best(ds, slug, kind):
    app = _App()
    colls = ALL3 if kind == "all" else ("documentation",)
    top1_hits = 0
    queries = ds.queries(20)
    try:
        for q in queries:
            got = await _run(app, ds, slug, kind, q)
            scores = [c.score for c in got]
            assert scores == sorted(scores, reverse=True), "rows out of distance order"
            assert got[0].score == max(scores)  # what `top_dense_similarity` relies on
            best = ds.exact_top(colls, q, org=str(ds.org_id(slug)), k=1)
            top1_hits += str(got[0].chunk_id) == best[0]
    finally:
        await app.close()
    assert top1_hits / len(queries) >= 0.85


# --- 6. recall against exact search --------------------------------------------


@pytest.mark.parametrize("slug", ["big", "mid"])
@pytest.mark.parametrize("kind", ["all", "single"])
async def test_recall_against_exact_search_on_clustered_embeddings(ds, slug, kind):
    app = _App()
    colls = ALL3 if kind == "all" else ("documentation",)
    recalls = []
    try:
        for q in ds.queries(20, seed=33):
            got = {str(c.chunk_id) for c in await _run(app, ds, slug, kind, q)}
            want = set(ds.exact_top(colls, q, org=str(ds.org_id(slug)), k=K))
            recalls.append(len(got & want) / K)
    finally:
        await app.close()
    assert sum(recalls) / len(recalls) >= 0.95, recalls


# --- 7/8. every filter, and top_k whenever enough permitted rows exist ---------

_CASES = [
    # (label, slug, kind, projects, perms, repo)
    ("big, restricted ACL hidden", "big", "all", None, (), None),
    ("big, restricted ACL visible", "big", "all", None, (H.RESTRICTED,), None),
    ("big, project p0", "big", "all", [0], (H.RESTRICTED,), None),
    ("big, tiny project p3", "big", "all", [3], (H.RESTRICTED,), None),
    ("big, projects p1+p2", "big", "all", [1, 2], (), None),
    ("mid", "mid", "all", None, (), None),
    ("small", "small", "all", None, (), None),
    ("needle", "needle", "all", None, (), None),
    ("single: big", "big", "single", None, (), None),
    ("single: big + repo", "big", "single", None, (H.RESTRICTED,), "acme/api"),
    ("single: big p3 + repo", "big", "single", [3], (H.RESTRICTED,), "acme/web"),
    ("single: mid + repo", "mid", "single", None, (), "acme/infra"),
]


@pytest.mark.parametrize("case", _CASES, ids=[c[0] for c in _CASES])
async def test_every_filter_is_honored_and_top_k_is_returned_when_enough_rows_exist(ds, case):
    _, slug, kind, projects, perms, repo = case
    colls = ALL3 if kind == "all" else ("documentation",)
    org = str(ds.org_id(slug))
    proj_ids = None if projects is None else [ds.project_id(slug, i) for i in projects]
    eligible = set(
        ds.exact_top(
            colls, ds.queries(1)[0], org=org, k=10**9, projects=proj_ids, perms=perms, repo=repo
        )
    )
    assert eligible, "the scenario must have permitted rows"
    app = _App()
    try:
        for q in ds.queries(8, seed=44):
            got = await _run(app, ds, slug, kind, q, projects=projects, perms=perms, repo=repo)
            ids = [str(c.chunk_id) for c in got]
            # nothing outside tenant / project / ACL / repo / deleted / superseded
            assert set(ids) <= eligible, "a filtered-out (or other-tenant) row was returned"
            assert len(ids) == len(set(ids)), "duplicate rows"
            # completeness: exactly min(k, permitted) rows
            assert len(ids) == min(K, len(eligible)), (len(ids), len(eligible))
    finally:
        await app.close()


async def test_deleted_and_superseded_documents_are_never_returned(ds):
    t = ds.tables["documentation_chunks"]
    hidden = {i for i, h in zip(t.ids, t.hidden, strict=True) if h}
    assert hidden, "the dataset must contain deleted/superseded documents"
    app = _App()
    try:
        for kind in ("all", "single"):
            for q in ds.queries(15, seed=55):
                got = await _run(app, ds, "big", kind, q, perms=(H.RESTRICTED,))
                assert not ({str(c.chunk_id) for c in got} & hidden)
    finally:
        await app.close()


# --- 9. tenant isolation as ekip_app with FORCED RLS ---------------------------


async def test_runtime_role_is_not_privileged_and_rls_is_forced(ds):
    app = _App()
    try:
        async with app.sessions() as s:
            role = (
                await s.execute(
                    text("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user")
                )
            ).one()
            forced = (
                await s.execute(
                    text(
                        "SELECT bool_and(relrowsecurity AND relforcerowsecurity) FROM pg_class "
                        "WHERE relname = ANY(:t)"
                    ),
                    {"t": list(H.TABLES)},
                )
            ).scalar()
            await s.rollback()
    finally:
        await app.close()
    assert tuple(role) == (False, False)
    assert forced is True


@pytest.mark.parametrize("kind", ["all", "single"])
async def test_wrong_or_missing_tenant_context_returns_nothing(ds, kind):
    """The filters ask for `big`, but the transaction's tenant context is `mid`
    (or absent): RLS must return nothing even though the SQL filter matches."""
    app = _App()
    q = ds.queries(1)[0]
    try:
        wrong = await _run(app, ds, "big", kind, q, tenant="mid")
        async with app.sessions() as s:  # no set_tenant_context at all
            f = _filters(ds, "big")
            store = PgVectorStore()
            none = (
                await store.search_all(s, q, f, K)
                if kind == "all"
                else await store.search(s, "documentation", q, f, K)
            )
            await s.rollback()
    finally:
        await app.close()
    assert wrong == [] and none == []


async def test_every_result_belongs_to_the_requesting_tenant(ds):
    app = _App()
    try:
        for slug in ("big", "mid", "small", "needle", "tiny"):
            owned = {
                i
                for t in ds.tables.values()
                for i, o in zip(t.ids, t.org, strict=True)
                if o == str(ds.org_id(slug))
            }
            for q in ds.queries(5, seed=66):
                got = await _run(app, ds, slug, "all", q)
                assert {str(c.chunk_id) for c in got} <= owned
    finally:
        await app.close()


# --- the exact fallback, against a genuinely short HNSW result -----------------


async def test_exact_fallback_recovers_a_short_hnsw_result(ds, monkeypatch):
    """Induce the failure the fallback exists for: an HNSW scan whose candidate
    list is mostly filtered out (tiny `ef_search`, iterative scan off) with the
    planner steered onto the index. Prove the approximate statement really is
    short, then that `search_all` still returns `top_k` exact results."""
    weak = text(
        "SELECT set_config('plan_cache_mode', 'force_custom_plan', true), "
        "set_config('hnsw.ef_search', "
        "least(CAST(CAST(:ef_search AS text) AS int), 10)::text, true), "
        "set_config('hnsw.iterative_scan', 'off', true)"
    )
    monkeypatch.setattr(store_module, "_VECTOR_SEARCH_SETTINGS_SQL", weak)
    q = ds.queries(1, seed=88)[0]
    filters = _filters(ds, "small")
    want = ds.exact_top(ALL3, q, org=str(ds.org_id("small")), k=K)
    assert len(want) == K
    app = _App()
    try:
        async with app.sessions() as s:
            await set_tenant_context(s, ds.org_id("small"))
            await s.execute(text("SET LOCAL enable_sort = off"))  # steer the planner to the index
            await store_module._apply_vector_search_settings(s)
            approximate = await s.execute(
                PgVectorStore._dense_all_statement(q, filters, K, exact=False)
            )
            assert len(approximate.all()) < K, "precondition: the HNSW scan must be short here"

            got = await PgVectorStore().search_all(s, q, filters, K)
            await s.rollback()
    finally:
        await app.close()
    assert {str(c.chunk_id) for c in got} == set(want)

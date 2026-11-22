"""`PgVectorStore` HNSW-related behaviour that needs no database:

* the per-branch `ORDER BY distance LIMIT top_k` shape of `search_all`
  (and that every pre-existing filter is still repeated in every branch),
* the transaction-scoped search settings (and their version gate),
* the exact-search fallback when fewer than `top_k` rows come back.

The real-PostgreSQL counterpart (query plans, recall, RLS) is
`test_pgvector_hnsw_postgres.py`.
"""

from __future__ import annotations

import re
import uuid

import pytest
from sqlalchemy.dialects import postgresql

from app.retrieval.pgvector import store as store_module
from app.retrieval.pgvector.store import PgVectorStore
from app.retrieval.schemas import SearchFilters

_Q = [0.1] * 384


def _filters(**kwargs) -> SearchFilters:
    base = {"organization_id": uuid.uuid4(), "permission_codes": frozenset({"docs:restricted"})}
    base.update(kwargs)
    return SearchFilters(**base)


def _sql(stmt) -> str:
    compiled = stmt.compile(dialect=postgresql.dialect())
    return re.sub(r"\s+", " ", str(compiled))


class _Rows:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return list(self._rows)


class _Row:
    """A row shaped like the store's SELECT (works for search and search_all)."""

    def __init__(self, i: int, distance: float):
        self.id = uuid.uuid4()
        self.document_id = uuid.uuid4()
        self.collection = "documentation"
        self.content = f"chunk {i}"
        self.source_offset_start = 0
        self.source_offset_end = 1
        self.distance = distance
        self.title = "t"
        self.source_url = None


class _ScriptedSession:
    """Records every statement; vector SELECTs get scripted row batches."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.statements: list[object] = []
        self.params: list[object] = []

    async def execute(self, statement, params=None, *args, **kwargs):
        self.statements.append(statement)
        self.params.append(params)
        if "plan_cache_mode" in str(statement):
            return _Rows([])
        return _Rows(self.batches.pop(0) if self.batches else [])

    def vector_statements(self):
        return [s for s in self.statements if "plan_cache_mode" not in str(s)]


def _rows(n: int):
    return [_Row(i, -float(n - i)) for i in range(n)]


# --- settings ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_settings_statement_sets_all_three_values_transaction_locally():
    sql = str(store_module._VECTOR_SEARCH_SETTINGS_SQL)
    assert "set_config('plan_cache_mode', 'force_custom_plan', true)" in sql
    assert "set_config('hnsw.ef_search', :ef_search, true)" in sql
    assert "set_config('hnsw.iterative_scan', 'strict_order', true)" in sql
    # every set_config is transaction-local (third argument true) -- none may
    # outlive the caller's transaction on a pooled connection
    assert sql.count("set_config(") == 3
    assert len(re.findall(r", true\)", sql)) == 3


@pytest.mark.asyncio
async def test_iterative_scan_is_gated_on_pgvector_version_inside_the_sql():
    sql = str(store_module._VECTOR_SEARCH_SETTINGS_SQL)
    # the iterative_scan set_config only runs inside the CASE branch whose
    # condition is the >= 0.8.0 version check (CASE short-circuits), so an older
    # pgvector never sees the unknown/reserved parameter
    case = sql[sql.index("CASE WHEN") :]
    assert "ARRAY[0, 8, 0]" in case.split("THEN")[0]
    assert "hnsw.iterative_scan" in case.split("THEN")[1]
    assert "hnsw.iterative_scan" not in sql[: sql.index("CASE WHEN")]


@pytest.mark.asyncio
async def test_ef_search_is_100():
    session = _ScriptedSession(_rows(10))
    await PgVectorStore().search_all(session, _Q, _filters(), top_k=10)
    assert session.params[0] == {"ef_search": "100"}


@pytest.mark.parametrize("method", ["search", "search_all"])
@pytest.mark.asyncio
async def test_settings_are_applied_first_in_the_same_session(method):
    session = _ScriptedSession(_rows(5))
    store = PgVectorStore()
    if method == "search":
        await store.search(session, "documentation", _Q, _filters(), top_k=5)
    else:
        await store.search_all(session, _Q, _filters(), top_k=5)
    assert "plan_cache_mode" in str(session.statements[0])
    assert len(session.vector_statements()) == 1  # full result: no fallback


@pytest.mark.asyncio
async def test_lexical_search_does_not_issue_vector_settings():
    session = _ScriptedSession()
    await PgVectorStore().lexical_search_all(session, "pool", _filters(), top_k=5)
    assert all("plan_cache_mode" not in str(s) for s in session.statements)


# --- search_all statement shape ----------------------------------------------


def test_every_branch_has_its_own_order_by_and_limit():
    stmt = PgVectorStore._dense_all_statement(_Q, _filters(), 7, exact=False)
    sql = _sql(stmt)
    branch_orders = re.findall(
        r"ORDER BY \(\w+\.embedding <#> %\(\w+\)s\) ASC LIMIT %\(\w+\)s", sql
    )
    assert len(branch_orders) == 3, sql
    # ... and the global merge, ordered by the selected distance column
    assert re.search(r"ORDER BY anon_1\.distance ASC\s+LIMIT %\(param_\d+\)s\s*$", sql), sql
    assert sql.count("UNION ALL") == 2


def test_all_filters_are_repeated_in_every_branch():
    filters = _filters(project_ids=[uuid.uuid4()])
    sql = _sql(PgVectorStore._dense_all_statement(_Q, filters, 5, exact=False))
    for predicate in (
        r"_chunks\.organization_id = ",
        r"documents\.deleted_at IS NULL",
        r"documents\.superseded_at IS NULL",
        r"acl_permission_code IS NULL OR",
        r"_chunks\.project_id IN ",
    ):
        assert len(re.findall(predicate, sql)) == 3, predicate


def test_repository_filter_still_unsupported_for_all_collections():
    """Existing behaviour, preserved: `conversations` has no repo column."""
    with pytest.raises(ValueError):
        PgVectorStore._dense_all_statement(_Q, _filters(repository="a/b"), 5, exact=False)


@pytest.mark.asyncio
async def test_unsupported_repository_filter_fails_before_any_statement():
    session = _ScriptedSession()
    with pytest.raises(ValueError):
        await PgVectorStore().search_all(session, _Q, _filters(repository="a/b"), top_k=5)
    assert session.statements == []


def test_exact_variant_cannot_use_the_hnsw_ordering_expression():
    approx = _sql(PgVectorStore._dense_all_statement(_Q, _filters(), 5, exact=False))
    exact = _sql(PgVectorStore._dense_all_statement(_Q, _filters(), 5, exact=True))
    assert "+ 0" not in approx
    assert exact.count("+ 0") == 3  # one per branch
    # the selected `distance` column itself is unchanged in both
    assert approx.count("AS distance") == exact.count("AS distance") == 3


# --- exact fallback -----------------------------------------------------------


@pytest.mark.asyncio
async def test_search_all_falls_back_to_exact_when_short():
    short, full = _rows(3), _rows(10)
    session = _ScriptedSession(short, full)
    result = await PgVectorStore().search_all(session, _Q, _filters(), top_k=10)

    vector = session.vector_statements()
    assert len(vector) == 2
    assert "+ 0" not in _sql(vector[0])
    assert "+ 0" in _sql(vector[1])
    assert len(result) == 10  # the exact result replaced the short one


@pytest.mark.asyncio
async def test_search_all_does_not_fall_back_when_full():
    session = _ScriptedSession(_rows(10))
    await PgVectorStore().search_all(session, _Q, _filters(), top_k=10)
    assert len(session.vector_statements()) == 1


@pytest.mark.asyncio
async def test_search_falls_back_to_exact_when_short():
    session = _ScriptedSession(_rows(2), _rows(5))
    result = await PgVectorStore().search(session, "documentation", _Q, _filters(), top_k=5)
    vector = session.vector_statements()
    assert len(vector) == 2
    assert "+ 0" not in _sql(vector[0])
    assert "+ 0" in _sql(vector[1])
    assert len(result) == 5


@pytest.mark.asyncio
async def test_fallback_runs_the_same_filters():
    """The exact retry must not drop (or add) any predicate."""
    filters = _filters(project_ids=[uuid.uuid4()])
    session = _ScriptedSession([], [])
    await PgVectorStore().search(session, "code", _Q, filters, top_k=5)
    first, second = (_sql(s).replace(" + 0", "") for s in session.vector_statements())
    assert first == second


@pytest.mark.asyncio
async def test_search_result_structure_is_preserved():
    session = _ScriptedSession(_rows(3))
    result = await PgVectorStore().search_all(session, _Q, _filters(), top_k=3)
    first = result[0]
    assert (first.collection, first.title, first.source_url) == ("documentation", "t", None)
    assert first.score == pytest.approx(3.0)  # -(-3.0): distance negated back to a score
    assert [r.score for r in result] == sorted((r.score for r in result), reverse=True)

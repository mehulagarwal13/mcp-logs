"""`top_k` bounds for dense and lexical retrieval (review finding F5).

`top_k` is user-controlled on two REST endpoints and reaches SQL `LIMIT`
directly; an unbounded value meant an HNSW scan up to its tuple cap followed by
an exact retry sorting a whole tenant. These tests pin the bounds at every
internal entry point (the REST layer is covered in
`tests/api/test_search_top_k_bounds.py`).
"""

from __future__ import annotations

import uuid

import pytest

from app.retrieval import service as retrieval_service
from app.retrieval.pgvector.store import PgVectorStore
from app.retrieval.schemas import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    MIN_TOP_K,
    SearchFilters,
    validate_top_k,
)
from tests.retrieval.test_pgvector_hnsw_search import _rows, _ScriptedSession

_Q = [0.1] * 384
_BAD = [-(10**9), -1, 0, MAX_TOP_K + 1, 1000, 10**9]
_GOOD = [MIN_TOP_K, 2, DEFAULT_TOP_K, 24, MAX_TOP_K]


def _filters() -> SearchFilters:
    return SearchFilters(organization_id=uuid.uuid4(), permission_codes=frozenset())


def test_bounds_are_sane():
    assert (MIN_TOP_K, DEFAULT_TOP_K, MAX_TOP_K) == (1, 10, 100)
    assert MIN_TOP_K <= DEFAULT_TOP_K <= MAX_TOP_K


@pytest.mark.parametrize("value", _GOOD)
def test_valid_values_pass_through_unchanged(value):
    assert validate_top_k(value) == value


@pytest.mark.parametrize("value", _BAD)
def test_out_of_range_values_are_rejected(value):
    with pytest.raises(ValueError, match="between 1 and 100"):
        validate_top_k(value)


@pytest.mark.parametrize("value", [True, False, 5.0, "10", None, [10]])
def test_non_integers_are_rejected_including_bools(value):
    """`True` is an `int` in Python; it must not slip through as `top_k=1`."""
    with pytest.raises(ValueError, match="integer"):
        validate_top_k(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", _BAD)
@pytest.mark.parametrize("method", ["search", "search_all", "lexical_search", "lexical_search_all"])
async def test_every_store_query_method_rejects_a_bad_top_k_before_touching_the_database(
    method, value
):
    session = _ScriptedSession(_rows(3))
    store = PgVectorStore()
    calls = {
        "search": lambda: store.search(session, "documentation", _Q, _filters(), value),
        "search_all": lambda: store.search_all(session, _Q, _filters(), value),
        "lexical_search": lambda: store.lexical_search(
            session, "documentation", "pool", _filters(), value
        ),
        "lexical_search_all": lambda: store.lexical_search_all(session, "pool", _filters(), value),
    }
    with pytest.raises(ValueError):
        await calls[method]()
    assert session.statements == [], "a rejected top_k must not reach the database"


@pytest.mark.parametrize("value", [MIN_TOP_K, MAX_TOP_K])
async def test_the_boundary_values_are_accepted_by_the_store(value):
    session = _ScriptedSession(_rows(min(value, 3)), _rows(min(value, 3)))
    await PgVectorStore().search_all(session, _Q, _filters(), value)  # must not raise
    assert session.statements


@pytest.mark.parametrize("value", _BAD)
async def test_service_rejects_a_bad_top_k_before_embedding_the_query(monkeypatch, value):
    async def must_not_embed(*_a, **_k):
        raise AssertionError("embedding is expensive; a bad top_k must be rejected first")

    monkeypatch.setattr(retrieval_service.embedding, "embed_query", must_not_embed)
    with pytest.raises(ValueError):
        await retrieval_service.search(None, "q", _filters(), value)
    with pytest.raises(ValueError):
        await retrieval_service.search_with_signals(None, "q", _filters(), value)


def test_every_internal_top_k_is_within_the_bound():
    """A future internal caller (or a raised constant) that exceeds the cap would
    start failing at runtime; catch it here instead."""
    from app.agents.investigation import evidence
    from app.agents.retrieval import node
    from app.evaluation import runner

    internal = {
        "retrieval node candidate pool": node._CANDIDATE_POOL_SIZE,
        "retrieval node historical similarity": node._HISTORICAL_SIMILARITY_TOP_K,
        "investigation per-source": evidence._PER_SOURCE_TOP_K,
        "evaluation max k": max(runner.DEFAULT_K_VALUES),
        "evaluation context k": runner.DEFAULT_CONTEXT_TOP_K,
    }
    for name, value in internal.items():
        assert MIN_TOP_K <= value <= MAX_TOP_K, f"{name}={value} is outside the allowed top_k range"

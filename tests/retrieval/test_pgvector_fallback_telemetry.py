"""Structured logging and metrics for the exact-search fallback (review F4).

Counters are created from a PRIVATE `MeterProvider` + `InMemoryMetricReader`
(no global provider is installed, so nothing leaks into other tests). The module
logger is replaced by a recorder so DEBUG records are observable regardless of
how logging is configured.
"""

from __future__ import annotations

import json
import uuid

import pytest
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from app.retrieval.pgvector import telemetry
from app.retrieval.pgvector.store import PgVectorStore
from app.retrieval.schemas import SearchFilters
from tests.retrieval.test_pgvector_hnsw_search import _Row, _rows, _ScriptedSession

_SECRET_CONTENT = "TOP-SECRET-CHUNK-CONTENT-7f3a"
_SECRET_TITLE = "TOP-SECRET-TITLE-91bc"
_Q = [0.123456] * 384


class _Recorder:
    def __init__(self):
        self.records: list[tuple[str, str, dict]] = []

    def warning(self, event, **kw):
        self.records.append(("warning", event, kw))

    def debug(self, event, **kw):
        self.records.append(("debug", event, kw))

    def info(self, event, **kw):  # pragma: no cover - the module must not use it
        self.records.append(("info", event, kw))


@pytest.fixture()
def obs(monkeypatch):
    reader = InMemoryMetricReader()
    meter = MeterProvider(metric_readers=[reader]).get_meter("test")
    monkeypatch.setattr(
        telemetry, "dense_searches", meter.create_counter("ekip.retrieval.dense_searches")
    )
    monkeypatch.setattr(
        telemetry, "exact_fallbacks", meter.create_counter("ekip.retrieval.exact_fallbacks")
    )
    recorder = _Recorder()
    monkeypatch.setattr(telemetry, "logger", recorder)

    def points(name: str) -> list[tuple[dict, int]]:
        data = reader.get_metrics_data()
        if data is None:
            return []
        return [
            (dict(p.attributes), p.value)
            for rm in data.resource_metrics
            for sm in rm.scope_metrics
            for m in sm.metrics
            if m.name == name
            for p in m.data.data_points
        ]

    class Obs:
        logs = recorder

        @staticmethod
        def searches():
            return points("ekip.retrieval.dense_searches")

        @staticmethod
        def fallbacks():
            return points("ekip.retrieval.exact_fallbacks")

    return Obs


def _filters(**kw) -> SearchFilters:
    return SearchFilters(
        organization_id=kw.pop("organization_id", uuid.uuid4()),
        permission_codes=frozenset(),
        **kw,
    )


def _secret_rows(n: int):
    rows = _rows(n)
    for r in rows:
        r.content = _SECRET_CONTENT
        r.title = _SECRET_TITLE
    return rows


# --- no fallback -------------------------------------------------------------


@pytest.mark.parametrize("path", ["search", "search_all"])
async def test_a_full_result_counts_a_search_but_no_fallback(obs, path):
    session = _ScriptedSession(_rows(10))
    store = PgVectorStore()
    if path == "search":
        await store.search(session, "code", _Q, _filters(), 10)
        expected = {"path": "search", "collection": "code"}
    else:
        await store.search_all(session, _Q, _filters(), 10)
        expected = {"path": "search_all", "collection": "all"}
    assert obs.searches() == [(expected, 1)]
    assert obs.fallbacks() == []
    assert obs.logs.records == []


# --- fallback: recovered (the approximate pass under-retrieved) ---------------


@pytest.mark.parametrize("path", ["search", "search_all"])
async def test_a_recovering_fallback_is_counted_and_warned(obs, path):
    org = uuid.uuid4()
    session = _ScriptedSession(_rows(4), _rows(10))  # approximate: 4, exact: 10
    filters = _filters(organization_id=org, project_ids=[uuid.uuid4()])
    if path == "search":
        await PgVectorStore().search(session, "documentation", _Q, filters, 10)
        collection = "documentation"
    else:
        await PgVectorStore().search_all(session, _Q, filters, 10)
        collection = "all"

    assert obs.fallbacks() == [
        ({"path": path, "collection": collection, "outcome": "recovered"}, 1)
    ]
    assert len(obs.logs.records) == 1
    level, event, fields = obs.logs.records[0]
    assert (level, event) == ("warning", "dense_search_exact_fallback")
    assert fields == {
        "path": path,
        "collection": collection,
        "top_k": 10,
        "approximate_rows": 4,
        "exact_rows": 10,
        "outcome": "recovered",
        "organization_id": str(org),
        "has_project_filter": True,
        "has_repository_filter": False,
    }


# --- fallback: complete (a small tenant, nothing was missed) ------------------


@pytest.mark.parametrize("path", ["search", "search_all"])
async def test_a_small_tenant_fallback_is_counted_but_only_logged_at_debug(obs, path):
    session = _ScriptedSession(_rows(3), _rows(3))  # exact finds nothing more
    if path == "search":
        await PgVectorStore().search(session, "code", _Q, _filters(), 10)
    else:
        await PgVectorStore().search_all(session, _Q, _filters(), 10)

    ((attrs, value),) = obs.fallbacks()
    assert (attrs["outcome"], value) == ("complete", 1)
    assert [(lvl, evt) for lvl, evt, _ in obs.logs.records] == [
        ("debug", "dense_search_exact_fallback")
    ]
    assert obs.logs.records[0][2]["approximate_rows"] == obs.logs.records[0][2]["exact_rows"] == 3


async def test_an_empty_tenant_falls_back_with_complete(obs):
    session = _ScriptedSession([], [])
    await PgVectorStore().search_all(session, _Q, _filters(), 10)
    ((attrs, _),) = obs.fallbacks()
    assert attrs["outcome"] == "complete"


async def test_repository_filter_is_reported_as_a_flag_not_a_value(obs):
    session = _ScriptedSession([], [])
    await PgVectorStore().search(session, "code", _Q, _filters(repository="acme/private-repo"), 10)
    fields = obs.logs.records[0][2]
    assert fields["has_repository_filter"] is True
    assert "acme/private-repo" not in json.dumps(fields)


# --- counters accumulate -------------------------------------------------------


async def test_counters_accumulate_across_calls_and_split_by_outcome(obs):
    store = PgVectorStore()
    for batch in ((_rows(4), _rows(10)), (_rows(3), _rows(3)), (_rows(3), _rows(3))):
        await store.search_all(_ScriptedSession(*batch), _Q, _filters(), 10)
    await store.search_all(_ScriptedSession(_rows(10)), _Q, _filters(), 10)

    assert obs.searches() == [({"path": "search_all", "collection": "all"}, 4)]
    by_outcome = {a["outcome"]: v for a, v in obs.fallbacks()}
    assert by_outcome == {"recovered": 1, "complete": 2}


# --- nothing sensitive ----------------------------------------------------------


@pytest.mark.parametrize("path", ["search", "search_all"])
async def test_no_query_embedding_or_chunk_content_reaches_logs_or_metrics(obs, path):
    session = _ScriptedSession(_secret_rows(2), _secret_rows(10))
    store = PgVectorStore()
    if path == "search":
        await store.search(session, "documentation", _Q, _filters(), 10)
    else:
        await store.search_all(session, _Q, _filters(), 10)

    everything = json.dumps(
        {"logs": obs.logs.records, "searches": obs.searches(), "fallbacks": obs.fallbacks()},
        default=str,
    )
    assert _SECRET_CONTENT not in everything
    assert _SECRET_TITLE not in everything
    assert "0.123456" not in everything, "the query embedding must never be logged"
    allowed_log_keys = {
        "path", "collection", "top_k", "approximate_rows", "exact_rows", "outcome",
        "organization_id", "has_project_filter", "has_repository_filter",
    }  # fmt: skip
    for _, _, fields in obs.logs.records:
        assert set(fields) <= allowed_log_keys
    for attrs, _ in obs.searches() + obs.fallbacks():
        assert set(attrs) <= {"path", "collection", "outcome"}, (
            "metric attributes must stay low-cardinality"
        )


async def test_telemetry_never_logs_at_info_or_leaks_the_organization_into_metrics(obs):
    org = uuid.uuid4()
    await PgVectorStore().search_all(
        _ScriptedSession(_rows(1), _rows(10)), _Q, _filters(organization_id=org), 10
    )
    assert all(level != "info" for level, _, _ in obs.logs.records)
    assert str(org) not in json.dumps([obs.searches(), obs.fallbacks()], default=str)
    assert any(str(org) == f["organization_id"] for _, _, f in obs.logs.records)


# --- a rejected request records nothing -----------------------------------------


async def test_a_rejected_top_k_records_no_metrics(obs):
    with pytest.raises(ValueError):
        await PgVectorStore().search_all(_ScriptedSession(), _Q, _filters(), 0)
    assert obs.searches() == [] and obs.fallbacks() == [] and obs.logs.records == []


def test_row_helper_is_the_expected_shape():
    assert isinstance(_rows(1)[0], _Row)

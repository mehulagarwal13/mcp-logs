"""Unit tests (no database) for `app.ingestion.lifecycle` and the
`scripts/backfill_superseded_documents.py` safety guards.

The behavior of the backfill's SQL against a real PostgreSQL (marking order,
`superseded_at` value, purge across all four tables, idempotency, resume,
tenant scoping) is covered in `test_version_supersession_postgres.py`, which
needs `EKIP_TEST_POSTGRES_URL`. Here: that a dry run cannot write, that every
batch is scoped to exactly one organization, and that the script refuses unsafe
invocations.
"""

from __future__ import annotations

import importlib.util
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from app.ingestion import lifecycle

ORG_1 = uuid.uuid4()
ORG_2 = uuid.uuid4()


class _Result:
    def __init__(self, scalar=0, rows=None, rowcount=0):
        self._scalar = scalar
        self._rows = rows or []
        self.rowcount = rowcount

    def scalar_one(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _RecordingSession:
    def __init__(self, log: list) -> None:
        self._log = log

    async def execute(self, statement, params=None):
        self._log.append(("sql", str(statement), params or {}))
        return _Result()


def _make_scope(log: list):
    @asynccontextmanager
    async def scope():
        log.append(("begin",))
        yield _RecordingSession(log)
        log.append(("commit",))

    return scope


@pytest.fixture
def tenant_calls(monkeypatch):
    calls: list[uuid.UUID] = []

    async def fake_set_tenant_context(session, organization_id):
        calls.append(organization_id)

    monkeypatch.setattr(lifecycle, "set_tenant_context", fake_set_tenant_context)
    return calls


def _statements(log):
    return [entry[1].upper() for entry in log if entry[0] == "sql"]


@pytest.mark.asyncio
async def test_dry_run_issues_no_writes(tenant_calls):
    log: list = []

    report = await lifecycle.run_backfill(
        _make_scope(log),
        organization_ids=[ORG_1, ORG_2],
        apply=False,
        sleep_seconds=0,
        log=lambda _m: None,
    )

    assert report.applied is False
    assert len(report.organizations) == 2
    for sql in _statements(log):
        assert not sql.lstrip().startswith(("UPDATE", "DELETE", "INSERT")), sql
        assert "UPDATE DOCUMENTS" not in sql and "DELETE FROM" not in sql
    assert ("commit",) in log  # transactions still close cleanly


@pytest.mark.asyncio
async def test_every_transaction_is_scoped_to_exactly_one_organization(tenant_calls):
    log: list = []

    await lifecycle.run_backfill(
        _make_scope(log),
        organization_ids=[ORG_1, ORG_2],
        apply=True,
        sleep_seconds=0,
        log=lambda _m: None,
    )

    # Walk the log transaction by transaction: the first thing each one does is
    # set tenant context, and every SQL parameter naming an org matches it.
    begins = sum(1 for e in log if e == ("begin",))
    assert begins == len(tenant_calls), "a transaction ran without setting tenant context"
    current_org = None
    calls = iter(tenant_calls)
    for entry in log:
        if entry == ("begin",):
            current_org = next(calls)
        elif entry[0] == "sql":
            org_param = entry[2].get("org")
            if org_param is not None:
                assert org_param == current_org, "SQL targeted a different organization"
    assert set(tenant_calls) == {ORG_1, ORG_2}


@pytest.mark.asyncio
async def test_apply_sets_a_lock_timeout_in_each_batch(tenant_calls):
    log: list = []

    await lifecycle.run_backfill(
        _make_scope(log),
        organization_ids=[ORG_1],
        apply=True,
        sleep_seconds=0,
        log=lambda _m: None,
    )

    assert sum("LOCK_TIMEOUT" in s for s in _statements(log)) == sum(
        1 for e in log if e == ("begin",)
    )


@pytest.mark.asyncio
async def test_phase_mark_never_deletes_and_phase_purge_never_marks(tenant_calls):
    mark_log: list = []
    purge_log: list = []

    await lifecycle.run_backfill(
        _make_scope(mark_log), organization_ids=[ORG_1], apply=True, phase="mark",
        sleep_seconds=0, log=lambda _m: None,
    )
    await lifecycle.run_backfill(
        _make_scope(purge_log), organization_ids=[ORG_1], apply=True, phase="purge",
        sleep_seconds=0, log=lambda _m: None,
    )

    assert not any("DELETE FROM" in s for s in _statements(mark_log))
    assert not any(s.lstrip().startswith("WITH RANKED") or "UPDATE DOCUMENTS" in s
                   for s in _statements(purge_log))
    assert any("DELETE FROM" in s for s in _statements(purge_log))


@pytest.mark.asyncio
async def test_never_deletes_documents_or_metadata_rows(tenant_calls):
    log: list = []
    await lifecycle.run_backfill(
        _make_scope(log), organization_ids=[ORG_1], apply=True, sleep_seconds=0,
        include_soft_deleted=True, log=lambda _m: None,
    )

    for sql in _statements(log):
        assert "DELETE FROM DOCUMENTS" not in sql
        assert "DELETE FROM DOCUMENT_METADATA" not in sql


@pytest.mark.asyncio
async def test_invalid_arguments_are_rejected():
    with pytest.raises(ValueError):
        await lifecycle.run_backfill(
            _make_scope([]), organization_ids=[], apply=False, phase="bogus"
        )
    with pytest.raises(ValueError):
        await lifecycle.run_backfill(
            _make_scope([]), organization_ids=[], apply=False, batch_size=0
        )


@pytest.mark.asyncio
async def test_chunk_table_names_are_whitelisted():
    session = _RecordingSession([])
    with pytest.raises(AssertionError):
        await lifecycle.purge_chunk_batch(
            session, ORG_1, "documents; DROP TABLE x", batch_size=1, include_soft_deleted=False
        )
    with pytest.raises(AssertionError):
        await lifecycle.count_chunks_to_purge(
            session, ORG_1, "users", include_soft_deleted=False
        )
    assert set(lifecycle.CHUNK_TABLES) == {
        "documentation_chunks",
        "code_chunks",
        "conversations_chunks",
        "incidents_chunks",
    }


@pytest.mark.asyncio
async def test_purge_sql_is_org_scoped_on_both_tables_and_skips_soft_deleted_by_default():
    log: list = []
    session = _RecordingSession(log)

    await lifecycle.purge_chunk_batch(
        session, ORG_1, "incidents_chunks", batch_size=10, include_soft_deleted=False
    )
    await lifecycle.purge_chunk_batch(
        session, ORG_1, "incidents_chunks", batch_size=10, include_soft_deleted=True
    )

    (_, sql_default, params), (_, sql_with_deleted, _) = log
    assert "d.organization_id = :org" in sql_default and "c.organization_id = :org" in sql_default
    assert params["org"] == ORG_1
    assert "d.superseded_at IS NOT NULL" in sql_default
    assert "deleted_at" not in sql_default
    assert "d.deleted_at IS NOT NULL" in sql_with_deleted


def _load_script():
    path = Path(__file__).resolve().parents[2] / "scripts" / "backfill_superseded_documents.py"
    spec = importlib.util.spec_from_file_location("backfill_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_requires_an_explicit_scope():
    script = _load_script()
    with pytest.raises(SystemExit):
        script._parse_args([])  # neither --org nor --all-orgs
    with pytest.raises(SystemExit):
        script._parse_args(["--org", str(ORG_1), "--all-orgs"])  # mutually exclusive


def test_script_defaults_to_dry_run():
    script = _load_script()
    args = script._parse_args(["--all-orgs"])
    assert args.apply is False and args.yes is False
    assert args.phase == "all" and args.include_soft_deleted is False


@pytest.mark.asyncio
async def test_script_refuses_apply_without_yes(capsys):
    script = _load_script()

    code = await script.main(["--all-orgs", "--apply"])

    assert code == 2
    assert "requires --yes" in capsys.readouterr().err

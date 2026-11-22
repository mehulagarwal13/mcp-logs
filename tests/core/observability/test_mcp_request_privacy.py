"""Issue 1 unit tests: request-content minimization, tenant-scoped statistics,
the insert statement shape, and retention. (Database/RLS behaviour is proved
against real PostgreSQL in `test_mcp_requests_isolation_postgres.py`.)"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.dialects import postgresql

from app.core.observability import repository
from app.core.observability import service as observability_service
from app.core.observability.redaction import minimize_request_summary

_SECRET_TEXT = "the checkout service leaks card numbers"  # synthetic marker text


# --------------------------------------------------------------- minimization


def test_free_text_is_reduced_to_its_length():
    out = minimize_request_summary({"query": _SECRET_TEXT, "incident_id": None})
    assert out == {"query_chars": len(_SECRET_TEXT), "incident_id": None}
    assert _SECRET_TEXT not in repr(out)


@pytest.mark.parametrize(
    "tool_summary",
    [
        {"description": _SECRET_TEXT},
        {"name": _SECRET_TEXT, "is_default": False},
        {"title": _SECRET_TEXT, "source_incident_id": None},
        {"value": _SECRET_TEXT, "rule_type": "email_domain", "grants_role": "admin"},
        {
            "query": _SECRET_TEXT,
            "since": "2026-01-01",
            "repository": "acme/api",
            "collection": "code",
        },
    ],
)
def test_no_free_text_survives_for_any_tool_shape(tool_summary):
    assert _SECRET_TEXT not in repr(minimize_request_summary(tool_summary))


def test_email_keeps_only_the_domain():
    out = minimize_request_summary({"email": "Jane.Doe@Example.COM", "grants_role": "member"})
    assert out["email_domain"] == "example.com"
    assert "jane" not in repr(out).lower()
    assert out["grants_role"] == "member"


def test_an_email_without_a_domain_leaks_nothing():
    out = minimize_request_summary({"email": "not-an-email"})
    assert "email_domain" not in out and "not-an-email" not in repr(out)


def test_sso_issuer_keeps_only_the_host():
    out = minimize_request_summary(
        {"provider": "oidc", "issuer_url": "https://login.example.com/tenant/secret-path?x=1"}
    )
    assert out == {"provider": "oidc", "issuer_host": "login.example.com"}


def test_identifiers_enums_and_dates_are_kept():
    summary = {
        "incident_id": str(uuid.uuid4()),
        "document_id": str(uuid.uuid4()),
        "collection": "code",
        "since": "2026-01-01T00:00:00Z",
        "is_default": True,
    }
    assert minimize_request_summary(summary) == summary


def test_kept_strings_are_truncated():
    out = minimize_request_summary({"repository": "x" * 1000})
    assert len(out["repository"]) == 128


def test_unknown_keys_are_default_denied_and_collections_become_sizes():
    out = minimize_request_summary(
        {"mystery": _SECRET_TEXT, "items": ["a", "b"], "blob": {"k": "v"}, "n": 3, "obj": object()}
    )
    assert out == {"mystery_chars": len(_SECRET_TEXT), "items_items": 2, "blob_items": 1, "n": 3}


def test_none_summary_stays_none():
    assert minimize_request_summary(None) is None


# --------------------------------------------------------------- service layer


@pytest.mark.asyncio
async def test_record_mcp_request_persists_only_the_minimized_summary(monkeypatch):
    captured = {}

    async def fake_insert(session, **kwargs):
        captured.update(kwargs)
        return type(
            "Row",
            (),
            {
                "id": uuid.uuid4(),
                "organization_id": kwargs["organization_id"],
                "tool_name": kwargs["tool_name"],
                "identity": kwargs["identity"],
                "request_summary": kwargs["request_summary"],
                "status_code": kwargs["status_code"],
                "latency_ms": kwargs["latency_ms"],
                "occurred_at": datetime.now(UTC),
            },
        )()

    monkeypatch.setattr(observability_service.repository, "insert_mcp_request", fake_insert)
    org = uuid.uuid4()
    log = await observability_service.record_mcp_request(
        None,
        organization_id=org,
        tool_name="ask_question",
        identity="user:abc",
        request_summary={"query": _SECRET_TEXT, "incident_id": None},
        status_code=200,
        latency_ms=5,
    )
    assert captured["organization_id"] == org
    assert _SECRET_TEXT not in repr(captured["request_summary"])
    assert _SECRET_TEXT not in repr(log)
    assert log.request_summary == {"query_chars": len(_SECRET_TEXT), "incident_id": None}


@pytest.mark.asyncio
async def test_purge_uses_the_configured_retention_window(monkeypatch):
    seen = {}

    async def fake_purge(session, cutoff):
        seen["cutoff"] = cutoff
        return 7

    monkeypatch.setattr(observability_service.repository, "purge_expired", fake_purge)
    now = datetime(2026, 6, 1, tzinfo=UTC)
    deleted = await observability_service.purge_expired_mcp_requests(
        None, now=now, retention_days=30
    )
    assert deleted == 7
    assert seen["cutoff"] == now - timedelta(days=30)


# ------------------------------------------------------------ repository SQL


class _CapturingSession:
    def __init__(self):
        self.statements = []

    async def execute(self, statement, params=None):
        self.statements.append(statement)

        class _R:
            def all(self_inner):
                return []

            def scalar_one(self_inner):
                return 0

        return _R()


def _sql(statement) -> str:
    return str(
        statement.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )


@pytest.mark.asyncio
async def test_stats_query_is_filtered_by_organization():
    session = _CapturingSession()
    org = uuid.uuid4()
    await repository.get_mcp_tool_stats(session, org)
    sql = _sql(session.statements[0])
    assert "organization_id" in sql and str(org) in sql


@pytest.mark.asyncio
async def test_insert_has_no_returning_clause():
    """`RETURNING` would require the new row to pass the SELECT policy, which an
    unattributed (NULL-organization) row never does."""
    session = _CapturingSession()
    row = await repository.insert_mcp_request(
        session,
        organization_id=None,
        tool_name="ask_question",
        identity="unresolved",
        request_summary=None,
        status_code=401,
        latency_ms=1,
    )
    assert "RETURNING" not in _sql(session.statements[0]).upper()
    assert row.organization_id is None and row.id is not None and row.occurred_at is not None

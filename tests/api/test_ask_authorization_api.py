"""HTTP-level authorization for `POST /ask` and `POST /incidents/{id}/investigate`
with an `incident_id`: 404 for a foreign/unknown incident, 403 for an
unreadable one, and no graph/memory/LLM work (or execution row) for either.

Runs the REAL router and the REAL `answer_question`/`triage_incident` +
`incidents_service.get_incident`; only the repository and the steps after
authorization are faked.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.agents import service as agents_service
from app.api import main as api_main
from app.api.deps import get_current_identity
from app.core.incidents import service as incidents_service
from app.database.session import get_db_session
from app.shared.schemas import ActorKind, AskResponse, Identity


def _user(org, permissions=(), project_permissions=None) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=org,
        user_id=uuid.uuid4(),
        permissions=frozenset(permissions),
        project_permissions={k: frozenset(v) for k, v in (project_permissions or {}).items()},
    )


def _incident(org, project) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=org,
        project_id=project,
        title="t",
        description="d",
        status="open",
        severity="high",
        owner_team=None,
        reported_by=uuid.uuid4(),
        resolved_at=None,
        created_at=now,
        updated_at=now,
    )


@pytest.fixture()
def harness(monkeypatch):
    """Yields (client_factory, calls). `client_factory(actor, incident)`."""
    calls = {"memory": 0, "llm": 0, "graph_build": 0, "graph_run": 0, "execution_rows": 0}

    async def fake_recall(session, actor, query):
        calls["memory"] += 1
        return []

    def fake_get_llm():
        calls["llm"] += 1
        return object()

    def fake_build_graph(session, llm):
        calls["graph_build"] += 1
        return object()

    def fake_build_investigation_graph(session, llm):
        calls["graph_build"] += 1
        return object()

    async def fake_run_graph_and_record(session, **kwargs):
        calls["graph_run"] += 1
        calls["execution_rows"] += 1
        return AskResponse(confidence=0.9, route_taken="answer", answer="authorized answer")

    async def fail_insert_execution(*args, **kwargs):
        calls["execution_rows"] += 1
        raise AssertionError("no execution row may be created")

    monkeypatch.setattr(agents_service.memory_service, "recall_relevant", fake_recall)
    monkeypatch.setattr(agents_service, "get_llm", fake_get_llm)
    monkeypatch.setattr(agents_service, "build_graph", fake_build_graph)
    monkeypatch.setattr(agents_service, "build_investigation_graph", fake_build_investigation_graph)
    monkeypatch.setattr(agents_service, "_run_graph_and_record", fake_run_graph_and_record)
    monkeypatch.setattr(agents_service.repository, "insert_agent_execution", fail_insert_execution)

    def make_client(actor: Identity, incident: SimpleNamespace | None) -> TestClient:
        async def fake_get_incident_by_id(session, incident_id):
            return incident

        async def _fake_session():
            yield None

        monkeypatch.setattr(
            incidents_service.repository, "get_incident_by_id", fake_get_incident_by_id
        )
        api_main.app.dependency_overrides[get_current_identity] = lambda: actor
        api_main.app.dependency_overrides[get_db_session] = _fake_session
        return TestClient(api_main.app)

    yield make_client, calls

    api_main.app.dependency_overrides.clear()


def _nothing_started(calls: dict[str, int]) -> None:
    assert calls == {"memory": 0, "llm": 0, "graph_build": 0, "graph_run": 0, "execution_rows": 0}


def test_ask_with_a_readable_incident_succeeds(harness) -> None:
    make_client, calls = harness
    org, project = uuid.uuid4(), uuid.uuid4()
    incident = _incident(org, project)
    client = make_client(_user(org, {"incident:read"}), incident)

    response = client.post("/ask", json={"query": "why?", "incident_id": str(incident.id)})

    assert response.status_code == 200
    assert response.json()["answer"] == "authorized answer"
    assert calls["graph_run"] == 1


def test_ask_with_another_organizations_incident_is_404(harness) -> None:
    make_client, calls = harness
    my_org, other_org = uuid.uuid4(), uuid.uuid4()
    foreign = _incident(other_org, uuid.uuid4())
    client = make_client(_user(my_org, {"incident:read"}), foreign)

    response = client.post("/ask", json={"query": "why?", "incident_id": str(foreign.id)})

    assert response.status_code == 404
    _nothing_started(calls)


def test_ask_with_an_unreadable_incident_is_403(harness) -> None:
    make_client, calls = harness
    org = uuid.uuid4()
    incident = _incident(org, uuid.uuid4())
    client = make_client(_user(org, {"incident:write"}), incident)  # no incident:read

    response = client.post("/ask", json={"query": "why?", "incident_id": str(incident.id)})

    assert response.status_code == 403
    _nothing_started(calls)


def test_ask_with_another_projects_incident_is_403(harness) -> None:
    make_client, calls = harness
    org, project_a, project_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    incident_in_b = _incident(org, project_b)
    actor = _user(org, set(), project_permissions={project_a: {"incident:read"}})
    client = make_client(actor, incident_in_b)

    response = client.post("/ask", json={"query": "why?", "incident_id": str(incident_in_b.id)})

    assert response.status_code == 403
    _nothing_started(calls)


def test_ask_without_an_incident_id_is_unchanged(harness) -> None:
    make_client, calls = harness
    client = make_client(_user(uuid.uuid4()), None)  # no permissions needed for plain /ask

    response = client.post("/ask", json={"query": "how do we deploy?"})

    assert response.status_code == 200
    assert calls["graph_run"] == 1


def test_investigate_endpoint_has_the_same_403_and_404_behavior(harness) -> None:
    make_client, calls = harness
    org = uuid.uuid4()
    unreadable = _incident(org, uuid.uuid4())
    client = make_client(_user(org, {"incident:write"}), unreadable)
    assert client.post(f"/incidents/{unreadable.id}/investigate").status_code == 403

    foreign = _incident(uuid.uuid4(), uuid.uuid4())
    client = make_client(_user(org, {"incident:read"}), foreign)
    assert client.post(f"/incidents/{foreign.id}/investigate").status_code == 404

    _nothing_started(calls)

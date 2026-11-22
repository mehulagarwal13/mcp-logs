"""`agents.service.answer_question`: explicit incident authorization.

When `incident_id` is supplied the caller must be able to read that incident
(same organization + `incident:read` on its project -- the rule
`triage_incident` applies), checked BEFORE memory recall, LLM setup, graph
construction, or any execution row. These tests run the REAL
`incidents_service.get_incident` over a faked repository and replace every
downstream step with a tripwire that fails the test if it is reached.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.agents import service as agents_service
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.incidents import service as incidents_service
from app.shared.schemas import ActorKind, AskResponse, Identity


def _user(
    organization_id: uuid.UUID,
    permissions: set[str] | None = None,
    project_permissions: dict[uuid.UUID, set[str]] | None = None,
) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=uuid.uuid4(),
        permissions=frozenset(permissions or set()),
        project_permissions={k: frozenset(v) for k, v in (project_permissions or {}).items()},
    )


def _incident_row(organization_id: uuid.UUID, project_id: uuid.UUID) -> SimpleNamespace:
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=organization_id,
        project_id=project_id,
        title="Checkout 500s",
        description="Checkout returns 500 for card payments",
        status="open",
        severity="high",
        owner_team=None,
        reported_by=uuid.uuid4(),
        resolved_at=None,
        created_at=now,
        updated_at=now,
    )


class _Tripwires:
    """Records which downstream steps ran; any of them running for an
    unauthorized caller is a failure.
    """

    def __init__(self) -> None:
        self.memory_recalls = 0
        self.llm_setups = 0
        self.graph_builds = 0
        self.graph_runs = 0
        self.execution_rows = 0


@pytest.fixture()
def tripwires(monkeypatch) -> _Tripwires:
    wires = _Tripwires()

    async def fake_recall(session, actor, query):
        wires.memory_recalls += 1
        return []

    def fake_get_llm():
        wires.llm_setups += 1
        return object()

    def fake_build_graph(session, llm):
        wires.graph_builds += 1
        return object()

    async def fake_run_graph_and_record(session, **kwargs):
        wires.graph_runs += 1
        wires.execution_rows += 1  # the real function inserts the execution row
        return AskResponse(confidence=0.9, route_taken="answer", answer="ok")

    async def fail_insert_execution(*args, **kwargs):
        wires.execution_rows += 1
        raise AssertionError("an execution row must not be created")

    monkeypatch.setattr(agents_service.memory_service, "recall_relevant", fake_recall)
    monkeypatch.setattr(agents_service, "get_llm", fake_get_llm)
    monkeypatch.setattr(agents_service, "build_graph", fake_build_graph)
    monkeypatch.setattr(agents_service, "_run_graph_and_record", fake_run_graph_and_record)
    monkeypatch.setattr(
        agents_service.repository, "insert_agent_execution", fail_insert_execution
    )
    return wires


def _serve_incident(monkeypatch, row: SimpleNamespace | None) -> None:
    async def fake_get_incident_by_id(session, incident_id):
        return row

    monkeypatch.setattr(
        incidents_service.repository, "get_incident_by_id", fake_get_incident_by_id
    )


def _assert_nothing_started(wires: _Tripwires) -> None:
    assert wires.memory_recalls == 0
    assert wires.llm_setups == 0
    assert wires.graph_builds == 0
    assert wires.graph_runs == 0
    assert wires.execution_rows == 0


# --- authorized -------------------------------------------------------------


@pytest.mark.asyncio
async def test_authorized_user_with_valid_incident_runs_the_graph(monkeypatch, tripwires) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident = _incident_row(org, project)
    _serve_incident(monkeypatch, incident)
    actor = _user(org, {"incident:read"})

    result = await agents_service.answer_question(None, "why 500s?", incident.id, actor)

    assert result.answer == "ok"
    assert tripwires.memory_recalls == 1
    assert tripwires.llm_setups == 1
    assert tripwires.graph_builds == 1
    assert tripwires.graph_runs == 1


@pytest.mark.asyncio
async def test_project_scoped_read_grant_is_enough(monkeypatch, tripwires) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident = _incident_row(org, project)
    _serve_incident(monkeypatch, incident)
    actor = _user(org, set(), project_permissions={project: {"incident:read"}})

    result = await agents_service.answer_question(None, "why 500s?", incident.id, actor)

    assert result.answer == "ok"
    assert tripwires.graph_runs == 1


# --- unauthorized: nothing starts, no execution row -------------------------


@pytest.mark.asyncio
async def test_incident_from_another_project_is_rejected_before_any_work(
    monkeypatch, tripwires
) -> None:
    org, project_a, project_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    incident_in_b = _incident_row(org, project_b)
    _serve_incident(monkeypatch, incident_in_b)
    actor = _user(org, set(), project_permissions={project_a: {"incident:read"}})

    with pytest.raises(PermissionDeniedError):
        await agents_service.answer_question(None, "q", incident_in_b.id, actor)

    _assert_nothing_started(tripwires)


@pytest.mark.asyncio
async def test_incident_from_another_organization_is_rejected_as_not_found(
    monkeypatch, tripwires
) -> None:
    my_org, other_org = uuid.uuid4(), uuid.uuid4()
    foreign = _incident_row(other_org, uuid.uuid4())
    _serve_incident(monkeypatch, foreign)
    actor = _user(my_org, {"incident:read"})

    with pytest.raises(NotFoundError):  # not PermissionDenied: do not reveal it exists
        await agents_service.answer_question(None, "q", foreign.id, actor)

    _assert_nothing_started(tripwires)


@pytest.mark.asyncio
async def test_unknown_incident_is_rejected_as_not_found(monkeypatch, tripwires) -> None:
    _serve_incident(monkeypatch, None)
    actor = _user(uuid.uuid4(), {"incident:read"})

    with pytest.raises(NotFoundError):
        await agents_service.answer_question(None, "q", uuid.uuid4(), actor)

    _assert_nothing_started(tripwires)


@pytest.mark.asyncio
async def test_user_without_incident_read_is_rejected_before_any_work(
    monkeypatch, tripwires
) -> None:
    org = uuid.uuid4()
    incident = _incident_row(org, uuid.uuid4())
    _serve_incident(monkeypatch, incident)
    actor = _user(org, {"tenancy:manage", "incident:write"})  # no incident:read

    with pytest.raises(PermissionDeniedError):
        await agents_service.answer_question(None, "q", incident.id, actor)

    _assert_nothing_started(tripwires)


@pytest.mark.asyncio
async def test_project_override_that_drops_read_is_rejected(monkeypatch, tripwires) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident = _incident_row(org, project)
    _serve_incident(monkeypatch, incident)
    # Org-wide read, but this project's override is more restrictive.
    actor = _user(org, {"incident:read"}, project_permissions={project: {"knowledge:review"}})

    with pytest.raises(PermissionDeniedError):
        await agents_service.answer_question(None, "q", incident.id, actor)

    _assert_nothing_started(tripwires)


# --- no incident id: unchanged ---------------------------------------------


@pytest.mark.asyncio
async def test_question_without_incident_id_is_unchanged_and_never_looks_up_an_incident(
    monkeypatch, tripwires
) -> None:
    async def boom(*args, **kwargs):
        raise AssertionError("no incident lookup is expected without an incident_id")

    monkeypatch.setattr(incidents_service, "get_incident", boom)
    actor = _user(uuid.uuid4())  # no permissions at all: /ask itself needs none

    result = await agents_service.answer_question(None, "how do we deploy?", None, actor)

    assert result.answer == "ok"
    assert tripwires.memory_recalls == 1
    assert tripwires.graph_runs == 1

"""Authorization of `core.incidents.service.record_investigation_result` and
the investigation node's `_attach_to_timeline` wrapper.

Automated investigation results may be recorded for any actor who can READ
the incident (`incident:read` on its project) -- the same rule as
`get_incident`/`triage_incident` -- and for nobody else. Manual notes
(`add_timeline_note`) must keep requiring `incident:write`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.agents.graph import GraphState
from app.agents.investigation import node as node_module
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.incidents import service as incidents_service
from app.core.incidents.schemas import TimelineNoteCreate
from app.shared.schemas import ActorKind, AskResponse, Identity, InvestigationResult


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


def _investigation() -> InvestigationResult:
    return InvestigationResult(
        evidence=[], hypotheses=[], suggested_owner_team=None, suggested_next_steps=["look"]
    )


class _Writes:
    def __init__(self) -> None:
        self.inserted: list[dict] = []


@pytest.fixture()
def writes(monkeypatch) -> _Writes:
    recorded = _Writes()

    async def fake_insert_timeline_entry(session, **kwargs):
        recorded.inserted.append(kwargs)
        return SimpleNamespace(id=uuid.uuid4(), occurred_at=datetime.now(UTC), **kwargs)

    async def fake_record_audit_event(*args, **kwargs):
        return None

    monkeypatch.setattr(
        incidents_service.repository, "insert_timeline_entry", fake_insert_timeline_entry
    )
    monkeypatch.setattr(incidents_service, "record_audit_event", fake_record_audit_event)
    return recorded


def _serve_incident(monkeypatch, organization_id: uuid.UUID, project_id: uuid.UUID) -> uuid.UUID:
    incident_id = uuid.uuid4()

    async def fake_get_incident_by_id(session, requested_id):
        return SimpleNamespace(
            id=incident_id, organization_id=organization_id, project_id=project_id
        )

    monkeypatch.setattr(
        incidents_service.repository, "get_incident_by_id", fake_get_incident_by_id
    )
    return incident_id


# --- record_investigation_result -------------------------------------------


@pytest.mark.asyncio
async def test_read_only_investigator_can_record_an_investigation_result(
    monkeypatch, writes
) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident_id = _serve_incident(monkeypatch, org, project)
    read_only = _user(org, {"incident:read"})  # no incident:write, no tenancy:manage

    entry = await incidents_service.record_investigation_result(
        None, read_only, org, incident_id, _investigation()
    )

    assert entry.event_type == "investigation"
    assert len(writes.inserted) == 1
    assert writes.inserted[0]["incident_id"] == incident_id
    assert writes.inserted[0]["organization_id"] == org
    assert writes.inserted[0]["actor"] == read_only.audit_tag


@pytest.mark.asyncio
async def test_project_scoped_read_grant_can_record(monkeypatch, writes) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident_id = _serve_incident(monkeypatch, org, project)
    actor = _user(org, set(), project_permissions={project: {"incident:read"}})

    await incidents_service.record_investigation_result(
        None, actor, org, incident_id, _investigation()
    )

    assert len(writes.inserted) == 1


@pytest.mark.asyncio
async def test_actor_without_incident_read_cannot_record(monkeypatch, writes) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident_id = _serve_incident(monkeypatch, org, project)
    # Write access and admin rights are NOT read access to the incident.
    actor = _user(org, {"incident:write", "tenancy:manage"})

    with pytest.raises(PermissionDeniedError):
        await incidents_service.record_investigation_result(
            None, actor, org, incident_id, _investigation()
        )

    assert writes.inserted == []


@pytest.mark.asyncio
async def test_read_on_a_different_project_cannot_record(monkeypatch, writes) -> None:
    org, project_a, project_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    incident_id = _serve_incident(monkeypatch, org, project_b)
    actor = _user(org, set(), project_permissions={project_a: {"incident:read"}})

    with pytest.raises(PermissionDeniedError):
        await incidents_service.record_investigation_result(
            None, actor, org, incident_id, _investigation()
        )

    assert writes.inserted == []


@pytest.mark.asyncio
async def test_cross_organization_request_is_rejected(monkeypatch, writes) -> None:
    my_org, other_org, project = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    incident_id = _serve_incident(monkeypatch, other_org, project)
    actor = _user(my_org, {"incident:read"})

    # Naming the other organization explicitly...
    with pytest.raises(PermissionDeniedError):
        await incidents_service.record_investigation_result(
            None, actor, other_org, incident_id, _investigation()
        )
    # ...or naming one's own organization with the other tenant's incident id.
    with pytest.raises(NotFoundError):
        await incidents_service.record_investigation_result(
            None, actor, my_org, incident_id, _investigation()
        )

    assert writes.inserted == []


@pytest.mark.asyncio
async def test_unknown_incident_cannot_be_recorded_against(monkeypatch, writes) -> None:
    async def none_found(session, requested_id):
        return None

    monkeypatch.setattr(incidents_service.repository, "get_incident_by_id", none_found)
    org = uuid.uuid4()

    with pytest.raises(NotFoundError):
        await incidents_service.record_investigation_result(
            None, _user(org, {"incident:read"}), org, uuid.uuid4(), _investigation()
        )

    assert writes.inserted == []


# --- manual notes keep requiring incident:write ----------------------------


@pytest.mark.asyncio
async def test_manual_note_still_requires_incident_write(monkeypatch, writes) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident_id = _serve_incident(monkeypatch, org, project)
    read_only = _user(org, {"incident:read"})

    with pytest.raises(PermissionDeniedError):
        await incidents_service.add_timeline_note(
            None, read_only, org, incident_id, TimelineNoteCreate(note="hello")
        )
    assert writes.inserted == []

    writer = _user(org, {"incident:read", "incident:write"})
    await incidents_service.add_timeline_note(
        None, writer, org, incident_id, TimelineNoteCreate(note="hello")
    )
    assert [w["event_type"] for w in writes.inserted] == ["note"]


# --- the investigation node's best-effort wrapper --------------------------


def _state(actor: Identity, incident_id: uuid.UUID | None) -> GraphState:
    return GraphState(query="q", incident_id=incident_id, actor=actor)


def _response() -> AskResponse:
    return AskResponse(confidence=0.1, route_taken="investigation", investigation=_investigation())


@pytest.mark.asyncio
async def test_attach_to_timeline_swallows_an_unauthorized_actor_and_writes_nothing(
    monkeypatch, writes
) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident_id = _serve_incident(monkeypatch, org, project)
    unauthorized = _user(org, {"incident:write"})  # cannot read the incident

    # Must not raise: the investigation response is still returned to the caller...
    await node_module._attach_to_timeline(
        None, _state(unauthorized, incident_id), _response()
    )

    # ...but nothing is written to the incident's timeline.
    assert writes.inserted == []


@pytest.mark.asyncio
async def test_attach_to_timeline_records_for_a_read_only_investigator(
    monkeypatch, writes
) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident_id = _serve_incident(monkeypatch, org, project)
    read_only = _user(org, {"incident:read"})

    await node_module._attach_to_timeline(None, _state(read_only, incident_id), _response())

    assert len(writes.inserted) == 1


@pytest.mark.asyncio
async def test_attach_to_timeline_is_a_noop_without_an_incident(monkeypatch, writes) -> None:
    actor = _user(uuid.uuid4(), {"incident:read"})

    await node_module._attach_to_timeline(None, _state(actor, None), _response())

    assert writes.inserted == []

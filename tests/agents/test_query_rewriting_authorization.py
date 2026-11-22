"""`rewrite_query` is the second layer protecting `/ask` with an `incident_id`
(`answer_question` checks first). It must keep rejecting foreign or
unreadable incidents on its own, and must never send such an incident's
description to the LLM or let the rewrite fall back silently.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.agents.retrieval.rewriting import rewrite_query
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.incidents import service as incidents_service
from app.shared.schemas import ActorKind, Identity

_SECRET_DESCRIPTION = "internal: payment gateway credentials rotated after breach"


class _RecordingLLM:
    def __init__(self) -> None:
        self.prompts: list[object] = []

    async def ainvoke(self, messages):
        self.prompts.append(messages)
        return SimpleNamespace(content="rewritten checkout query")


def _user(organization_id: uuid.UUID, permissions=(), project_permissions=None) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=uuid.uuid4(),
        permissions=frozenset(permissions),
        project_permissions={k: frozenset(v) for k, v in (project_permissions or {}).items()},
    )


def _serve(monkeypatch, organization_id: uuid.UUID, project_id: uuid.UUID) -> uuid.UUID:
    now = datetime.now(UTC)
    row = SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=organization_id,
        project_id=project_id,
        title="t",
        description=_SECRET_DESCRIPTION,
        status="open",
        severity="high",
        owner_team=None,
        reported_by=uuid.uuid4(),
        resolved_at=None,
        created_at=now,
        updated_at=now,
    )

    async def fake_get_incident_by_id(session, incident_id):
        return row

    monkeypatch.setattr(incidents_service.repository, "get_incident_by_id", fake_get_incident_by_id)
    return row.id


async def _rewrite(actor, incident_id, llm):
    return await rewrite_query(
        None, query="why is it failing?", incident_id=incident_id, actor=actor, llm=llm,
        retry_count={},
    )


@pytest.mark.asyncio
async def test_authorized_actor_gets_the_incident_context_rewrite(monkeypatch) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident_id = _serve(monkeypatch, org, project)
    llm = _RecordingLLM()

    result = await _rewrite(_user(org, {"incident:read"}), incident_id, llm)

    assert result == "rewritten checkout query"
    assert _SECRET_DESCRIPTION in str(llm.prompts)  # legitimate use of the context


@pytest.mark.asyncio
async def test_foreign_organization_incident_is_rejected_and_never_reaches_the_llm(
    monkeypatch,
) -> None:
    my_org, other_org = uuid.uuid4(), uuid.uuid4()
    incident_id = _serve(monkeypatch, other_org, uuid.uuid4())
    llm = _RecordingLLM()

    with pytest.raises(NotFoundError):
        await _rewrite(_user(my_org, {"incident:read"}), incident_id, llm)

    assert llm.prompts == []


@pytest.mark.asyncio
async def test_unreadable_incident_is_rejected_and_never_reaches_the_llm(monkeypatch) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    incident_id = _serve(monkeypatch, org, project)
    llm = _RecordingLLM()

    with pytest.raises(PermissionDeniedError):
        await _rewrite(_user(org, {"incident:write"}), incident_id, llm)

    assert llm.prompts == []


@pytest.mark.asyncio
async def test_incident_in_another_project_is_rejected(monkeypatch) -> None:
    org, project_a, project_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    incident_id = _serve(monkeypatch, org, project_b)
    llm = _RecordingLLM()
    actor = _user(org, set(), project_permissions={project_a: {"incident:read"}})

    with pytest.raises(PermissionDeniedError):
        await _rewrite(actor, incident_id, llm)

    assert llm.prompts == []

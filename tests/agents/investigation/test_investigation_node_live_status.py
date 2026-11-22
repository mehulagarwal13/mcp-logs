"""Investigation node: live-evidence status reaches the result.

Evidence gathering, hypothesis generation and critique are faked at the node
module boundary -- this file only checks the node's own wiring of
`LiveEvidenceStatus` into `InvestigationResult` and the "incomplete" note.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.agents.graph import GraphState
from app.agents.investigation import node as node_module
from app.shared.schemas import (
    EvidenceItem,
    Identity,
    LiveEvidenceStatus,
    LiveSourceOutcome,
)


def _state() -> GraphState:
    return GraphState(
        query="checkout failing",
        incident_id=uuid.uuid4(),
        actor=Identity.for_agent("test_agent", uuid.uuid4()),
    )


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source="commit", reference="r1", summary="s", retrieved_at=datetime.now(UTC)
        )
    ]


def _patch(monkeypatch, *, evidence, live_status) -> None:
    async def fake_gather(session, **kwargs):
        return evidence, live_status

    async def fake_hypotheses(llm, query, ev):
        return [], None, ["look at the commit"]

    async def fake_review(llm, query, ev, hypotheses, owner, next_steps, retry_count):
        return SimpleNamespace(
            hypotheses=hypotheses,
            suggested_owner_team=owner,
            suggested_next_steps=next_steps,
            review_status="not_reviewed",
            critique_verdict=None,
            revision_count=0,
            critique_issues=[],
        )

    async def no_timeline(session, state, result):
        return None

    monkeypatch.setattr(node_module, "gather_evidence_with_status", fake_gather)
    monkeypatch.setattr(node_module, "generate_hypotheses", fake_hypotheses)
    monkeypatch.setattr(node_module.critique_module, "review_investigation", fake_review)
    monkeypatch.setattr(node_module, "_attach_to_timeline", no_timeline)


async def _run(state: GraphState):
    node = node_module.make_investigation_agent_node(None, None)
    update = await node(state)
    return update["result"].investigation


@pytest.mark.asyncio
async def test_live_status_is_returned_with_the_investigation(monkeypatch) -> None:
    status = LiveEvidenceStatus(
        status="ok", sources=[LiveSourceOutcome(source="github", outcome="ok", items_returned=2)]
    )
    _patch(monkeypatch, evidence=_evidence(), live_status=status)

    investigation = await _run(_state())

    assert investigation.live_evidence == status
    assert investigation.suggested_next_steps == ["look at the commit"]  # no note when all ok


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        LiveEvidenceStatus(status="failed", sources=[]),
        LiveEvidenceStatus(status="partial", sources=[]),
        LiveEvidenceStatus(status="unavailable", reason="not_authorized"),
        LiveEvidenceStatus(status="unavailable", reason="incident_not_authorized"),
    ],
)
async def test_incomplete_live_evidence_adds_a_next_step_note(monkeypatch, status) -> None:
    _patch(monkeypatch, evidence=_evidence(), live_status=status)

    investigation = await _run(_state())

    assert investigation.live_evidence == status
    assert investigation.suggested_next_steps[-1] == node_module._LIVE_EVIDENCE_INCOMPLETE_NOTE


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        LiveEvidenceStatus(status="not_attempted", reason="disabled"),
        LiveEvidenceStatus(status="not_attempted", reason="no_incident_context"),
        LiveEvidenceStatus(status="unavailable", reason="no_live_connectors"),
        LiveEvidenceStatus(status="unavailable", reason="no_implemented_live_sources"),
    ],
)
async def test_deliberate_or_configuration_states_do_not_add_a_warning(monkeypatch, status) -> None:
    _patch(monkeypatch, evidence=_evidence(), live_status=status)

    investigation = await _run(_state())

    assert investigation.live_evidence == status
    assert node_module._LIVE_EVIDENCE_INCOMPLETE_NOTE not in investigation.suggested_next_steps


@pytest.mark.asyncio
async def test_status_is_kept_even_when_no_evidence_was_found(monkeypatch) -> None:
    status = LiveEvidenceStatus(status="failed", sources=[])
    _patch(monkeypatch, evidence=[], live_status=status)

    investigation = await _run(_state())

    assert investigation.evidence == []
    assert investigation.live_evidence == status
    assert investigation.suggested_next_steps[-1] == node_module._LIVE_EVIDENCE_INCOMPLETE_NOTE

"""`LiveEvidenceStatus` / `InvestigationResult.live_evidence` contract tests."""

from __future__ import annotations

import uuid

import pytest
from pydantic import ValidationError

from app.shared.schemas import InvestigationResult, LiveEvidenceStatus, LiveSourceOutcome


def _result(**kwargs) -> InvestigationResult:
    return InvestigationResult(
        evidence=[], hypotheses=[], suggested_owner_team=None, suggested_next_steps=[], **kwargs
    )


def test_live_evidence_is_optional_so_existing_payloads_still_validate() -> None:
    # Old timeline entries / callers never had this field.
    assert _result().live_evidence is None


def test_status_round_trips_through_json() -> None:
    status = LiveEvidenceStatus(
        status="partial",
        sources=[
            LiveSourceOutcome(
                source="github", connector_config_id=uuid.uuid4(), outcome="ok", items_returned=3
            ),
            LiveSourceOutcome(source="slack", outcome="auth_failed", reason="auth_failed"),
        ],
    )
    result = _result(live_evidence=status)

    restored = InvestigationResult.model_validate(result.model_dump(mode="json"))

    assert restored.live_evidence == status


def test_status_rejects_unknown_vocabulary() -> None:
    with pytest.raises(ValidationError):
        LiveEvidenceStatus(status="great")
    with pytest.raises(ValidationError):
        LiveSourceOutcome(source="github", outcome="exploded")


def test_not_implemented_is_a_valid_outcome() -> None:
    outcome = LiveSourceOutcome(
        source="monitoring", outcome="not_implemented", reason="not_implemented"
    )
    assert outcome.items_returned == 0

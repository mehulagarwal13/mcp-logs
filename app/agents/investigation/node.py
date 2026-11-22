"""The Investigation Agent node (PROJECT_PLAN.md section 6.4 /
AGENT_WORKFLOWS.md section 2.4): orchestrates sub-stage A (evidence
gathering, `investigation.evidence`) -> sub-stage B (hypothesis generation,
`investigation.hypothesis`) -> sub-stage C (bounded critique/reflection,
`investigation.critique`, Priority 7), producing `result.investigation`.

Reached two ways:
  - Via the main `answer_question` graph's `route == "investigation"`
    branch, after Retrieval Agent + Confidence Evaluation already ran --
    this replaces task #21's `_investigation_not_implemented_node`
    placeholder in `agents.graph.build_graph`.
  - Via `agents.graph.build_investigation_graph`, where this is the *only*
    node: `triage_incident` always investigates, bypassing Retrieval
    Agent/Confidence Evaluation entirely (API_DESIGN.md section 2 /
    AGENT_WORKFLOWS.md section 11.3's request-flow diagram).

Built as a factory (`make_investigation_agent_node`), matching
`agents.retrieval.node`'s rationale: this node needs a request-scoped
`session` (evidence gathering reads the database) as well as `llm`
(hypothesis generation) bound for the lifetime of one invocation.

Failure handling: sub-stage A already treats each individual evidence
source's failure as non-fatal (logged and skipped -- see
`investigation.evidence`'s own docstring), and a genuinely empty evidence
list short-circuits straight to a "no automated evidence found" result
without spending an LLM call to discover that (mirroring
`agents.answer.node`'s zero-chunks guard). Sub-stage B's LLM call is
retried via `agents.retry.call_with_retry`; if retries are exhausted, this
degrades to zero hypotheses plus a generic "review the evidence manually"
next-steps list, never a raised exception -- consistent with
AGENT_WORKFLOWS.md section 2.4's own "no automated evidence found"
degradation, generalized to "no automated hypothesis produced."

When `state.incident_id` is set (always true for `triage_incident`;
optionally true for an incident-scoped `answer_question` call), this node
also best-effort attaches its result to that incident's timeline via
`core.incidents.service.record_investigation_result` -- task #24's
Postmortem Agent depends on being able to find "any Investigation Agent
evidence attached to the incident" (AGENT_WORKFLOWS.md section 2.5) there.
That write's own failure is logged and swallowed, never allowed to turn an
otherwise-successful investigation into a failed one: the `AskResponse`
this node produces is already complete and correct on its own; the
timeline write is additional bookkeeping for a *different* future agent
run, not part of this node's own contract.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.language_models import BaseChatModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.graph import GraphState
from app.agents.investigation import critique as critique_module
from app.agents.investigation.evidence import gather_evidence_with_status
from app.agents.investigation.hypothesis import generate_hypotheses
from app.agents.retry import call_with_retry
from app.core.incidents import service as incidents_service
from app.shared.config.logging import get_logger
from app.shared.schemas import AskResponse, InvestigationResult, LiveEvidenceStatus

logger = get_logger(__name__)

_NO_EVIDENCE_NEXT_STEPS = [
    "No automated evidence was found across code, chat, or existing "
    "postmortems. Recommend manual investigation starting with recent "
    "deploys and on-call escalation."
]
# Appended to `suggested_next_steps` when live GitHub/Slack evidence was
# expected to be available but was not (fully) collected, so a reader knows
# the evidence set may be missing very recent activity. Not added for a
# deliberate non-attempt (kill switch, no incident) or for an org that simply
# has no live connectors -- those are configuration, not a problem to flag.
_LIVE_EVIDENCE_INCOMPLETE_NOTE = (
    "Live GitHub/Slack evidence could not be fully collected for this "
    "investigation (see `live_evidence`); recent activity may be missing "
    "from the evidence below. Check the affected connectors."
)
_NO_HYPOTHESIS_NEXT_STEPS = [
    "Evidence was gathered but no well-supported root-cause hypothesis "
    "could be generated automatically. Recommend manual review of the "
    "evidence listed below."
]


def make_investigation_agent_node(
    session: AsyncSession, llm: BaseChatModel
) -> Callable[[GraphState], Awaitable[dict[str, Any]]]:
    """Build the LangGraph-callable Investigation Agent node, bound to
    `session`/`llm`.
    """

    async def node(state: GraphState) -> dict[str, Any]:
        evidence, live_status = await gather_evidence_with_status(
            session,
            query=state.query,
            actor=state.actor,
            retry_count=state.retry_count,
            incident_id=state.incident_id,
        )

        if not evidence:
            result = AskResponse(
                confidence=state.confidence_score or 0.0,
                route_taken="investigation",
                investigation=InvestigationResult(
                    evidence=[],
                    hypotheses=[],
                    suggested_owner_team=None,
                    suggested_next_steps=_with_live_note(_NO_EVIDENCE_NEXT_STEPS, live_status),
                    live_evidence=live_status,
                ),
            )
            await _attach_to_timeline(session, state, result)
            return {"evidence": [], "result": result, "retry_count": state.retry_count}

        try:
            hypotheses, suggested_owner_team, suggested_next_steps = await call_with_retry(
                "investigation_agent.hypothesis",
                lambda: generate_hypotheses(llm, state.query, evidence),
                retry_count=state.retry_count,
            )
        except Exception as exc:
            logger.warning(
                "investigation_agent_hypothesis_exhausted", query=state.query, error=str(exc)
            )
            hypotheses, suggested_owner_team, suggested_next_steps = (
                [],
                None,
                _NO_HYPOTHESIS_NEXT_STEPS,
            )

        # Priority 7: bounded critique/reflection over the generated
        # hypotheses. A no-op (review_status stays "not_reviewed") when
        # hypothesis generation already produced nothing above -- see
        # `critique.review_investigation`'s own docstring.
        review = await critique_module.review_investigation(
            llm,
            state.query,
            evidence,
            hypotheses,
            suggested_owner_team,
            suggested_next_steps,
            state.retry_count,
        )

        result = AskResponse(
            confidence=state.confidence_score or 0.0,
            route_taken="investigation",
            investigation=InvestigationResult(
                evidence=evidence,
                hypotheses=review.hypotheses,
                suggested_owner_team=review.suggested_owner_team,
                suggested_next_steps=_with_live_note(review.suggested_next_steps, live_status),
                review_status=review.review_status,
                critique_verdict=review.critique_verdict,
                revision_count=review.revision_count,
                critique_issues=review.critique_issues,
                live_evidence=live_status,
            ),
        )
        await _attach_to_timeline(session, state, result)
        return {
            "evidence": evidence,
            "hypotheses": review.hypotheses,
            "result": result,
            "retry_count": state.retry_count,
        }

    return node


def _with_live_note(next_steps: list[str], live_status: LiveEvidenceStatus) -> list[str]:
    """`next_steps` plus `_LIVE_EVIDENCE_INCOMPLETE_NOTE` when live evidence
    was attempted and failed in whole or part, or was unavailable for a
    reason other than configuration ("this org has no live connectors" or
    "only not-yet-implemented live sources").
    """
    incomplete = live_status.status in ("failed", "partial") or (
        live_status.status == "unavailable"
        and live_status.reason not in ("no_live_connectors", "no_implemented_live_sources")
    )
    return [*next_steps, _LIVE_EVIDENCE_INCOMPLETE_NOTE] if incomplete else list(next_steps)


async def _attach_to_timeline(session: AsyncSession, state: GraphState, result: AskResponse) -> None:
    """Best-effort: attach `result.investigation` to `state.incident_id`'s
    timeline (`core.incidents.service.record_investigation_result`) -- see
    module docstring. A no-op when there is no incident to attach to
    (`answer_question` called with no `incident_id`).
    """
    if state.incident_id is None or result.investigation is None:
        return
    try:
        await incidents_service.record_investigation_result(
            session,
            state.actor,
            state.actor.organization_id,
            state.incident_id,
            result.investigation,
        )
    except Exception as exc:
        logger.warning(
            "investigation_agent_timeline_attach_failed",
            incident_id=str(state.incident_id),
            error=str(exc),
        )

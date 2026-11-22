"""Tests for `app.agents.investigation.evidence`:
  - `_chunk_to_evidence`'s kind -> source mapping -- the piece that lets the
    Investigation Agent tell a GitHub file chunk apart from a commit/
    pull-request/issue chunk, now that `retrieval.search(...,
    include_metadata=True)` can surface each chunk's `document_metadata`
    (see `ingestion.connectors.github`'s module docstring for the `"kind"`
    metadata key convention this relies on).
  - `_should_augment_with_live_evidence`'s hybrid trigger logic and
    `_gather_live_evidence`'s authorization, credential handling, dispatch
    and status reporting -- the live-evidence extension
    (`agents.investigation.live/`).
  - `_gather_postmortem_evidence`'s P4 deduplication against runbook
    evidence, and `_is_duplicate_of_runbook_evidence`'s correlation-key/
    content-identity-fallback logic it relies on.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from app.agents.investigation import evidence as evidence_module
from app.agents.investigation.evidence import (
    _LIVE_SOURCES,
    _chunk_to_evidence,
    _gather_knowledge_evidence,
    _gather_live_evidence,
    _gather_postmortem_evidence,
    _is_duplicate_of_runbook_evidence,
    _parse_source_timestamp,
    _should_augment_with_live_evidence,
)
from app.agents.investigation.live.base import LiveSourceAuthError
from app.agents.investigation.live.monitoring_live import MonitoringLiveSource
from app.core.exceptions import NotFoundError, PermissionDeniedError
from app.core.incidents.schemas import Postmortem
from app.core.tenancy.live_credentials import (
    LiveConnectorCredential,
    LiveConnectorResolution,
    LiveConnectorSkip,
)
from app.retrieval.schemas import ScoredChunk, SearchFilters
from app.shared.schemas import ActorKind, EvidenceItem, Identity


def _chunk(metadata: dict[str, str], content: str = "some content") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        collection="code",
        content=content,
        score=0.9,
        source_offset_start=0,
        source_offset_end=len(content),
        title="a title",
        source_url="https://github.com/acme/widgets/commit/abc123",
        metadata=metadata,
    )


def test_file_chunk_with_no_kind_defaults_to_github() -> None:
    chunk = _chunk({"repo": "acme/widgets", "path": "src/app.py", "ref": "main"})

    evidence = _chunk_to_evidence(chunk)

    assert evidence.source == "github"
    assert evidence.metadata == chunk.metadata
    assert evidence.source_timestamp is None  # no "timestamp" key on a file chunk


def test_commit_chunk_maps_to_commit_source() -> None:
    chunk = _chunk(
        {
            "repo": "acme/widgets",
            "kind": "commit",
            "sha": "abc123",
            "author": "Ada Lovelace",
            "timestamp": "2026-07-01T10:00:00Z",
            "changed_files": "src/checkout.py",
        }
    )

    evidence = _chunk_to_evidence(chunk)

    assert evidence.source == "commit"
    assert evidence.metadata["sha"] == "abc123"
    assert evidence.source_timestamp == datetime(2026, 7, 1, 10, 0, 0, tzinfo=timezone.utc)


def test_pull_request_chunk_maps_to_pull_request_source() -> None:
    chunk = _chunk({"repo": "acme/widgets", "kind": "pull_request", "number": "42"})

    evidence = _chunk_to_evidence(chunk)

    assert evidence.source == "pull_request"


def test_issue_chunk_maps_to_issue_source() -> None:
    chunk = _chunk({"repo": "acme/widgets", "kind": "issue", "number": "7", "labels": "bug"})

    evidence = _chunk_to_evidence(chunk)

    assert evidence.source == "issue"
    assert evidence.metadata["labels"] == "bug"


def test_explicit_source_override_wins_over_kind_metadata() -> None:
    """`_gather_slack_evidence` always passes `source="slack"` explicitly --
    confirms that override takes priority even if `metadata["kind"]` were
    somehow present (it never is for real Slack chunks).
    """
    chunk = _chunk({"kind": "commit"})

    evidence = _chunk_to_evidence(chunk, source="slack")

    assert evidence.source == "slack"


def test_reference_falls_back_to_chunk_id_without_source_url() -> None:
    chunk = _chunk({})
    chunk = chunk.model_copy(update={"source_url": None})

    evidence = _chunk_to_evidence(chunk)

    assert evidence.reference == f"chunk:{chunk.chunk_id}"


def test_summary_is_truncated_and_marked_with_ellipsis() -> None:
    long_content = "x" * 500
    chunk = _chunk({}, content=long_content)

    evidence = _chunk_to_evidence(chunk)

    assert len(evidence.summary) == 303  # 300 chars + "..."
    assert evidence.summary.endswith("...")


def test_parse_source_timestamp_handles_missing_and_malformed_values() -> None:
    assert _parse_source_timestamp(None) is None
    assert _parse_source_timestamp("") is None
    assert _parse_source_timestamp("not-a-timestamp") is None
    assert _parse_source_timestamp("2026-07-01T10:00:00Z") == datetime(
        2026, 7, 1, 10, 0, 0, tzinfo=timezone.utc
    )


# --- _gather_knowledge_evidence ---------------------------------------------


def _knowledge_chunk(metadata: dict[str, str] | None = None, content: str = "runbook content") -> ScoredChunk:
    return ScoredChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        collection="documentation",
        content=content,
        score=0.8,
        source_offset_start=0,
        source_offset_end=len(content),
        title="Runbook title",
        source_url="https://example.com/knowledge/doc-1",
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_gather_knowledge_evidence_excludes_github_tagged_chunks(monkeypatch) -> None:
    """The complementary filter to `_gather_code_evidence`'s: a chunk with
    `repo` metadata is GitHub-sourced and belongs to that source instead,
    not this one -- otherwise the same "documentation" collection chunk
    could be double-counted across both sources.
    """
    github_chunk = _knowledge_chunk({"repo": "acme/widgets"})
    runbook_chunk = _knowledge_chunk({})

    async def fake_search(session, query, filters, top_k, collection=None, *, include_metadata=False):
        assert collection == "documentation"
        assert include_metadata is True
        return [github_chunk, runbook_chunk]

    monkeypatch.setattr(evidence_module.retrieval_service, "search", fake_search)

    result = await _gather_knowledge_evidence(
        session=None,
        query="checkout failing",
        filters=SearchFilters(organization_id=uuid.uuid4()),
        retry_count={},
    )

    assert len(result) == 1
    assert result[0].reference == runbook_chunk.source_url


@pytest.mark.asyncio
async def test_gather_knowledge_evidence_returns_non_github_documentation_chunks_as_runbook_source(
    monkeypatch,
) -> None:
    manual_chunk = _knowledge_chunk({}, content="Restart the payment worker.")

    async def fake_search(session, query, filters, top_k, collection=None, *, include_metadata=False):
        return [manual_chunk]

    monkeypatch.setattr(evidence_module.retrieval_service, "search", fake_search)

    result = await _gather_knowledge_evidence(
        session=None,
        query="payment worker stuck",
        filters=SearchFilters(organization_id=uuid.uuid4()),
        retry_count={},
    )

    assert len(result) == 1
    assert result[0].source == "runbook"
    assert result[0].summary.startswith("Restart the payment worker.")


@pytest.mark.asyncio
async def test_gather_knowledge_evidence_logs_and_skips_on_failure(monkeypatch) -> None:
    import app.agents.retry as retry_module

    async def instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retry_module.asyncio, "sleep", instant_sleep)

    async def failing_search(session, query, filters, top_k, collection=None, *, include_metadata=False):
        raise RuntimeError("retrieval unavailable")

    monkeypatch.setattr(evidence_module.retrieval_service, "search", failing_search)

    result = await _gather_knowledge_evidence(
        session=None,
        query="q",
        filters=SearchFilters(organization_id=uuid.uuid4()),
        retry_count={},
    )

    assert result == []


@pytest.mark.asyncio
async def test_gather_evidence_includes_knowledge_source_in_priority_order(monkeypatch) -> None:
    """End-to-end: `gather_evidence` calls `_gather_knowledge_evidence`
    between code and Slack evidence (module docstring's priority order,
    item 2), and its results (source="runbook") reach the final list.
    """
    from app.agents.investigation.evidence import gather_evidence

    async def fake_code(session, query, filters, retry_count):
        return []

    async def fake_slack(session, query, filters, retry_count):
        return []

    knowledge_item = EvidenceItem(
        source="runbook", reference="chunk:1", summary="restart the worker", retrieved_at=datetime.now(timezone.utc)
    )

    async def fake_knowledge(session, query, filters, retry_count):
        return [knowledge_item]

    async def fake_postmortem(session, actor, retry_count, existing_evidence):
        return []

    monkeypatch.setattr(evidence_module, "_gather_code_evidence", fake_code)
    monkeypatch.setattr(evidence_module, "_gather_knowledge_evidence", fake_knowledge)
    monkeypatch.setattr(evidence_module, "_gather_slack_evidence", fake_slack)
    monkeypatch.setattr(evidence_module, "_gather_postmortem_evidence", fake_postmortem)
    monkeypatch.setattr(evidence_module, "_gather_jira_evidence", lambda: [])
    monkeypatch.setattr(evidence_module, "_gather_monitoring_evidence", lambda: [])
    monkeypatch.setattr(evidence_module.get_settings(), "investigation_live_evidence_enabled", False)

    actor = Identity.for_agent("test_agent", uuid.uuid4())
    result = await gather_evidence(session=None, query="worker stuck", actor=actor, retry_count={})

    assert knowledge_item in result


# --- _gather_postmortem_evidence / P4 deduplication -------------------------


def _postmortem(
    incident_id: uuid.UUID | None = None, root_cause: str = "Cache stampede on checkout"
) -> Postmortem:
    now = datetime.now(timezone.utc)
    return Postmortem(
        id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        incident_id=incident_id or uuid.uuid4(),
        status="approved",
        root_cause=root_cause,
        action_items=[],
        generated_by="agent",
        reviewed_by=None,
        created_at=now,
        updated_at=now,
    )


def _runbook_evidence(
    metadata: dict[str, str] | None = None, summary: str = "Restart the payment worker."
) -> EvidenceItem:
    return EvidenceItem(
        source="runbook",
        reference="chunk:1",
        summary=summary,
        retrieved_at=datetime.now(timezone.utc),
        metadata=metadata or {},
    )


@pytest.mark.asyncio
async def test_gather_postmortem_evidence_deduplicates_by_incident_id(monkeypatch) -> None:
    """Req 1: the same postmortem represented as both postmortem evidence
    and runbook evidence (correlated via `incident_id`) appears only once
    -- the runbook copy is kept, the redundant postmortem item is dropped.
    """
    incident_id = uuid.uuid4()
    postmortem = _postmortem(incident_id=incident_id)

    async def fake_list_recent(session, actor, organization_id, limit):
        return [postmortem]

    monkeypatch.setattr(evidence_module.incidents_service, "list_recent_postmortems", fake_list_recent)

    runbook_item = _runbook_evidence(metadata={"incident_id": str(incident_id)})

    result = await _gather_postmortem_evidence(
        session=None,
        actor=Identity.for_agent("test_agent", uuid.uuid4()),
        retry_count={},
        existing_evidence=[runbook_item],
    )

    assert result == []


@pytest.mark.asyncio
async def test_gather_postmortem_evidence_preserves_distinct_postmortems(monkeypatch) -> None:
    """Req 2: postmortems whose incident is not represented in runbook
    evidence at all are unaffected by dedup -- both are kept.
    """
    postmortem_a = _postmortem()
    postmortem_b = _postmortem()

    async def fake_list_recent(session, actor, organization_id, limit):
        return [postmortem_a, postmortem_b]

    monkeypatch.setattr(evidence_module.incidents_service, "list_recent_postmortems", fake_list_recent)

    result = await _gather_postmortem_evidence(
        session=None,
        actor=Identity.for_agent("test_agent", uuid.uuid4()),
        retry_count={},
        existing_evidence=[_runbook_evidence(metadata={"incident_id": str(uuid.uuid4())})],
    )

    assert {item.reference for item in result} == {
        f"postmortem:{postmortem_a.id}",
        f"postmortem:{postmortem_b.id}",
    }


@pytest.mark.asyncio
async def test_gather_evidence_preserves_unrelated_runbook_evidence(monkeypatch) -> None:
    """Req 3: a runbook evidence item unrelated to any gathered postmortem
    stays in the final evidence list untouched -- dedup only ever removes
    postmortem items, never runbook ones.
    """
    from app.agents.investigation.evidence import gather_evidence

    unrelated_runbook = _runbook_evidence(
        metadata={"incident_id": str(uuid.uuid4())}, summary="Restart the payment worker."
    )
    postmortem = _postmortem()  # unrelated incident_id, no overlap

    async def fake_code(session, query, filters, retry_count):
        return []

    async def fake_knowledge(session, query, filters, retry_count):
        return [unrelated_runbook]

    async def fake_slack(session, query, filters, retry_count):
        return []

    async def fake_list_recent(session, actor, organization_id, limit):
        return [postmortem]

    monkeypatch.setattr(evidence_module, "_gather_code_evidence", fake_code)
    monkeypatch.setattr(evidence_module, "_gather_knowledge_evidence", fake_knowledge)
    monkeypatch.setattr(evidence_module, "_gather_slack_evidence", fake_slack)
    monkeypatch.setattr(evidence_module, "_gather_jira_evidence", lambda: [])
    monkeypatch.setattr(evidence_module, "_gather_monitoring_evidence", lambda: [])
    monkeypatch.setattr(evidence_module.incidents_service, "list_recent_postmortems", fake_list_recent)
    monkeypatch.setattr(evidence_module.get_settings(), "investigation_live_evidence_enabled", False)

    actor = Identity.for_agent("test_agent", uuid.uuid4())
    result = await gather_evidence(session=None, query="worker stuck", actor=actor, retry_count={})

    assert unrelated_runbook in result
    assert any(item.reference == f"postmortem:{postmortem.id}" for item in result)


@pytest.mark.asyncio
async def test_gather_evidence_dedup_does_not_remove_github_or_slack_evidence(monkeypatch) -> None:
    """Req 4: deduplication is scoped to postmortem-vs-runbook only -- a
    duplicate postmortem being dropped never touches GitHub/Slack (or any
    other source's) evidence already gathered.
    """
    from app.agents.investigation.evidence import gather_evidence

    incident_id = uuid.uuid4()
    postmortem = _postmortem(incident_id=incident_id)
    runbook_item = _runbook_evidence(metadata={"incident_id": str(incident_id)})
    github_item = EvidenceItem(
        source="github", reference="ref-1", summary="a file", retrieved_at=datetime.now(timezone.utc)
    )
    slack_item = EvidenceItem(
        source="slack", reference="ref-2", summary="a message", retrieved_at=datetime.now(timezone.utc)
    )

    async def fake_code(session, query, filters, retry_count):
        return [github_item]

    async def fake_knowledge(session, query, filters, retry_count):
        return [runbook_item]

    async def fake_slack(session, query, filters, retry_count):
        return [slack_item]

    async def fake_list_recent(session, actor, organization_id, limit):
        return [postmortem]

    monkeypatch.setattr(evidence_module, "_gather_code_evidence", fake_code)
    monkeypatch.setattr(evidence_module, "_gather_knowledge_evidence", fake_knowledge)
    monkeypatch.setattr(evidence_module, "_gather_slack_evidence", fake_slack)
    monkeypatch.setattr(evidence_module, "_gather_jira_evidence", lambda: [])
    monkeypatch.setattr(evidence_module, "_gather_monitoring_evidence", lambda: [])
    monkeypatch.setattr(evidence_module.incidents_service, "list_recent_postmortems", fake_list_recent)
    monkeypatch.setattr(evidence_module.get_settings(), "investigation_live_evidence_enabled", False)

    actor = Identity.for_agent("test_agent", uuid.uuid4())
    result = await gather_evidence(session=None, query="checkout failing", actor=actor, retry_count={})

    assert github_item in result
    assert slack_item in result
    assert runbook_item in result
    assert not any(item.reference == f"postmortem:{postmortem.id}" for item in result)


@pytest.mark.asyncio
async def test_gather_evidence_priority_order_unchanged_when_no_duplicate(monkeypatch) -> None:
    """Req 5: when no duplicate exists, the final evidence list's order is
    exactly the pre-P4 priority order (code, knowledge/runbook, slack,
    postmortem) -- dedup only ever removes items, never reorders survivors.
    """
    from app.agents.investigation.evidence import gather_evidence

    github_item = EvidenceItem(
        source="github", reference="ref-1", summary="a file", retrieved_at=datetime.now(timezone.utc)
    )
    runbook_item = _runbook_evidence(metadata={"incident_id": str(uuid.uuid4())})
    slack_item = EvidenceItem(
        source="slack", reference="ref-2", summary="a message", retrieved_at=datetime.now(timezone.utc)
    )
    postmortem = _postmortem()  # distinct incident_id, no overlap

    async def fake_code(session, query, filters, retry_count):
        return [github_item]

    async def fake_knowledge(session, query, filters, retry_count):
        return [runbook_item]

    async def fake_slack(session, query, filters, retry_count):
        return [slack_item]

    async def fake_list_recent(session, actor, organization_id, limit):
        return [postmortem]

    monkeypatch.setattr(evidence_module, "_gather_code_evidence", fake_code)
    monkeypatch.setattr(evidence_module, "_gather_knowledge_evidence", fake_knowledge)
    monkeypatch.setattr(evidence_module, "_gather_slack_evidence", fake_slack)
    monkeypatch.setattr(evidence_module, "_gather_jira_evidence", lambda: [])
    monkeypatch.setattr(evidence_module, "_gather_monitoring_evidence", lambda: [])
    monkeypatch.setattr(evidence_module.incidents_service, "list_recent_postmortems", fake_list_recent)
    monkeypatch.setattr(evidence_module.get_settings(), "investigation_live_evidence_enabled", False)

    actor = Identity.for_agent("test_agent", uuid.uuid4())
    result = await gather_evidence(session=None, query="checkout failing", actor=actor, retry_count={})

    assert [item.source for item in result] == ["github", "runbook", "slack", "postmortem"]


def test_is_duplicate_of_runbook_evidence_fallback_matches_on_content_identity() -> None:
    """Positive fallback case: no runbook evidence item carries an
    `incident_id` key at all (e.g. a P3 manually-published document), so
    correlation falls back to content identity -- the postmortem's own
    root-cause text appearing verbatim in the runbook item's summary is
    treated as the same underlying knowledge.
    """
    postmortem = _postmortem(root_cause="Cache stampede on checkout")
    runbook_item = _runbook_evidence(metadata={}, summary="Root cause: Cache stampede on checkout")

    assert _is_duplicate_of_runbook_evidence(
        postmortem, postmortem.root_cause, [runbook_item], correlated_incident_ids=set()
    )


def test_is_duplicate_of_runbook_evidence_missing_correlation_metadata_does_not_remove_unrelated() -> None:
    """Req 6: a runbook evidence item with no `incident_id` key (so the
    correlation key is unavailable) must not cause an *unrelated*
    postmortem to be incorrectly matched -- the content-identity fallback
    requires the postmortem's own summary text to actually appear in that
    item, not just "some runbook evidence exists with no key."
    """
    postmortem = _postmortem(root_cause="Cache stampede on checkout")
    unrelated_runbook_item = _runbook_evidence(metadata={}, summary="Restart the payment worker.")

    assert not _is_duplicate_of_runbook_evidence(
        postmortem, postmortem.root_cause, [unrelated_runbook_item], correlated_incident_ids=set()
    )


# --- _should_augment_with_live_evidence -------------------------------------


def _evidence_item(source_timestamp: datetime | None = None) -> EvidenceItem:
    return EvidenceItem(
        source="github",
        reference="ref",
        summary="summary",
        retrieved_at=datetime.now(timezone.utc),
        source_timestamp=source_timestamp,
    )


def test_should_augment_when_incident_id_present() -> None:
    now = datetime.now(timezone.utc)
    evidence = [_evidence_item(source_timestamp=now) for _ in range(10)]  # plenty, fresh

    assert _should_augment_with_live_evidence(evidence, uuid.uuid4()) is True


def test_should_augment_when_evidence_is_thin() -> None:
    now = datetime.now(timezone.utc)
    evidence = [_evidence_item(source_timestamp=now)]  # below _LIVE_EVIDENCE_MIN_COUNT

    assert _should_augment_with_live_evidence(evidence, None) is True


def test_should_augment_when_evidence_is_stale() -> None:
    stale = datetime.now(timezone.utc) - timedelta(hours=5)
    evidence = [_evidence_item(source_timestamp=stale) for _ in range(5)]

    assert _should_augment_with_live_evidence(evidence, None) is True


def test_should_augment_when_no_evidence_is_timestamped() -> None:
    evidence = [_evidence_item(source_timestamp=None) for _ in range(5)]

    assert _should_augment_with_live_evidence(evidence, None) is True


def test_should_not_augment_when_plenty_of_fresh_evidence_and_no_incident() -> None:
    now = datetime.now(timezone.utc)
    evidence = [_evidence_item(source_timestamp=now) for _ in range(5)]

    assert _should_augment_with_live_evidence(evidence, None) is False


# --- live evidence: authorization, credentials, status ----------------------
#
# `_gather_live_evidence` / `gather_evidence_with_status`. The incident read
# and the connector resolver are faked at the `evidence_module` boundary, plus
# a few tests that run the REAL `core.tenancy` resolver over fake rows to
# prove a non-admin investigator's real (decrypted) token reaches the source.

_REAL_TOKEN = "ghp_REAL_decrypted_token_SECRET"


def _credential(source: str, token: str = _REAL_TOKEN, organization_id=None, project_id=None):
    return LiveConnectorCredential(
        connector_config_id=uuid.uuid4(),
        organization_id=organization_id or uuid.uuid4(),
        project_id=project_id,
        source=source,
        config={},
        token=token,
    )


def _item(source: str, reference: str) -> EvidenceItem:
    return EvidenceItem(
        source=source, reference=reference, summary="s", retrieved_at=datetime.now(timezone.utc)
    )


class _FakeLiveSource:
    def __init__(self, items: list[EvidenceItem] | None = None, error: Exception | None = None):
        self._items = items or []
        self._error = error
        self.calls = 0
        self.tokens_seen: list[str] = []

    async def fetch_live_evidence(self, *, connector, query, since, limit):
        self.calls += 1
        self.tokens_seen.append(connector.reveal_token())
        if self._error is not None:
            raise self._error
        return self._items


def _incident(organization_id: uuid.UUID, project_id: uuid.UUID | None = None):
    return SimpleNamespace(organization_id=organization_id, project_id=project_id or uuid.uuid4())


def _stub_incident(monkeypatch, incident=None, error: Exception | None = None) -> None:
    async def fake_get_incident(session, actor, organization_id, incident_id):
        if error is not None:
            raise error
        return incident or _incident(organization_id)

    monkeypatch.setattr(evidence_module.incidents_service, "get_incident", fake_get_incident)


def _stub_resolver(monkeypatch, credentials=(), skipped=(), error: Exception | None = None):
    calls = {"count": 0}

    async def fake_resolve(session, actor, organization_id, **kwargs):
        calls["count"] += 1
        if error is not None:
            raise error
        return LiveConnectorResolution(credentials=list(credentials), skipped=list(skipped))

    monkeypatch.setattr(
        evidence_module.tenancy_service, "resolve_live_evidence_connectors", fake_resolve
    )
    return calls


def _no_sleep(monkeypatch) -> None:
    import app.agents.retry as retry_module

    async def instant_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr(retry_module.asyncio, "sleep", instant_sleep)


def _actor() -> Identity:
    return Identity.for_agent("test_agent", uuid.uuid4())


async def _live(monkeypatch, actor=None):
    actor = actor or _actor()
    return await _gather_live_evidence(
        session=None, query="checkout failing", actor=actor, retry_count={}, incident_id=uuid.uuid4()
    )


def test_monitoring_connector_source_resolves_to_monitoring_live_source() -> None:
    """A `source="monitoring"` connector must dispatch to `MonitoringLiveSource`
    (real, un-monkeypatched `_LIVE_SOURCES`, so this fails if the registration
    is ever removed).
    """
    assert "monitoring" in _LIVE_SOURCES
    assert isinstance(_LIVE_SOURCES["monitoring"], MonitoringLiveSource)


@pytest.mark.asyncio
async def test_live_evidence_dispatches_by_connector_source_and_reports_ok(monkeypatch) -> None:
    github_item, slack_item = _item("commit", "r1"), _item("slack", "r2")
    fake_github, fake_slack = _FakeLiveSource([github_item]), _FakeLiveSource([slack_item])
    monkeypatch.setattr(
        evidence_module, "_LIVE_SOURCES", {"github": fake_github, "slack": fake_slack}
    )
    _stub_incident(monkeypatch)
    _stub_resolver(monkeypatch, credentials=[_credential("github"), _credential("slack")])

    items, status = await _live(monkeypatch)

    assert items == [github_item, slack_item]
    assert status.status == "ok"
    assert [(o.source, o.outcome, o.items_returned) for o in status.sources] == [
        ("github", "ok", 1),
        ("slack", "ok", 1),
    ]


@pytest.mark.asyncio
async def test_live_evidence_dispatches_monitoring_connector(monkeypatch) -> None:
    monitoring_item = _item("monitoring", "r3")
    fake_monitoring = _FakeLiveSource([monitoring_item])
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"monitoring": fake_monitoring})
    _stub_incident(monkeypatch)
    _stub_resolver(monkeypatch, credentials=[_credential("monitoring")])

    items, status = await _live(monkeypatch)

    assert items == [monitoring_item]
    assert fake_monitoring.calls == 1
    assert status.status == "ok"


# -- the original bugs: non-admin investigator + real (not redacted) token --


@pytest.mark.asyncio
async def test_non_admin_investigator_gets_live_evidence_with_the_real_token(monkeypatch) -> None:
    """End to end through the REAL `core.tenancy` resolver: an actor with only
    `incident:read` (no `tenancy:manage`) gets live evidence, and the source
    receives the decrypted token -- not the redaction placeholder.
    """
    from app.core.tenancy import service as real_tenancy
    from app.shared.security import encrypt_secret, get_kms

    organization_id, project_id = uuid.uuid4(), uuid.uuid4()
    investigator = Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=uuid.uuid4(),
        permissions=frozenset({"incident:read"}),
    )
    row = SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=organization_id,
        project_id=None,
        source="github",
        status="active",
        credential_ref=await encrypt_secret(get_kms(), _REAL_TOKEN),
        config={"repos": [{"repo": "acme/widgets"}]},
    )

    async def fake_rows(session, org_id):
        return [row]

    monkeypatch.setattr(real_tenancy.repository, "list_connector_configs", fake_rows)
    fake_github = _FakeLiveSource([_item("commit", "r1")])
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"github": fake_github})
    _stub_incident(monkeypatch, incident=_incident(organization_id, project_id))

    items, status = await _live(monkeypatch, actor=investigator)

    assert status.status == "ok"
    assert len(items) == 1
    assert fake_github.tokens_seen == [_REAL_TOKEN]
    assert fake_github.tokens_seen[0] != real_tenancy._REDACTED_CREDENTIAL


@pytest.mark.asyncio
async def test_actor_without_incident_read_gets_no_live_evidence_and_no_token_access(
    monkeypatch,
) -> None:
    """Real resolver, real permission check: no `incident:read` -> the
    resolver refuses before reading connectors; status says why.
    """
    organization_id = uuid.uuid4()
    nobody = Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=uuid.uuid4(),
        permissions=frozenset({"tenancy:manage"}),  # admin-ish, but cannot read incidents
    )
    fake_github = _FakeLiveSource([_item("commit", "r1")])
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"github": fake_github})
    _stub_incident(monkeypatch)  # the incident lookup itself succeeds in this fake

    items, status = await _live(monkeypatch, actor=nobody)

    assert items == []
    assert (status.status, status.reason) == ("unavailable", "not_authorized")
    assert fake_github.calls == 0


@pytest.mark.asyncio
async def test_incident_the_actor_cannot_read_yields_unavailable_and_touches_no_connectors(
    monkeypatch,
) -> None:
    fake_github = _FakeLiveSource([_item("commit", "r1")])
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"github": fake_github})
    _stub_incident(monkeypatch, error=PermissionDeniedError("no"))
    resolver_calls = _stub_resolver(monkeypatch, credentials=[_credential("github")])

    items, status = await _live(monkeypatch)

    assert items == []
    assert (status.status, status.reason) == ("unavailable", "incident_not_authorized")
    assert resolver_calls["count"] == 0
    assert fake_github.calls == 0


@pytest.mark.asyncio
async def test_unknown_or_cross_tenant_incident_looks_the_same_as_unauthorized(monkeypatch) -> None:
    _stub_incident(monkeypatch, error=NotFoundError("missing"))
    resolver_calls = _stub_resolver(monkeypatch, credentials=[_credential("github")])

    items, status = await _live(monkeypatch)

    assert items == []
    assert (status.status, status.reason) == ("unavailable", "incident_not_authorized")
    assert resolver_calls["count"] == 0


@pytest.mark.asyncio
async def test_incident_from_another_organization_is_refused_by_the_real_resolver(
    monkeypatch,
) -> None:
    my_org, other_org = uuid.uuid4(), uuid.uuid4()
    actor = Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=my_org,
        user_id=uuid.uuid4(),
        permissions=frozenset({"incident:read"}),
    )
    fake_github = _FakeLiveSource([_item("commit", "r1")])
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"github": fake_github})
    _stub_incident(monkeypatch, incident=_incident(other_org))  # forged: wrong org

    items, status = await _live(monkeypatch, actor=actor)

    assert items == []
    assert (status.status, status.reason) == ("unavailable", "not_authorized")
    assert fake_github.calls == 0


@pytest.mark.asyncio
async def test_resolver_permission_error_is_reported_not_swallowed(monkeypatch) -> None:
    _stub_incident(monkeypatch)
    _stub_resolver(monkeypatch, error=PermissionDeniedError("no"))

    items, status = await _live(monkeypatch)

    assert items == []
    assert (status.status, status.reason) == ("unavailable", "not_authorized")


@pytest.mark.asyncio
async def test_unexpected_resolver_failure_is_reported_as_unavailable(monkeypatch) -> None:
    _stub_incident(monkeypatch)
    _stub_resolver(monkeypatch, error=RuntimeError("db unavailable"))

    items, status = await _live(monkeypatch)

    assert items == []
    assert (status.status, status.reason) == ("unavailable", "connector_lookup_failed")
    assert "db unavailable" not in status.model_dump_json()


@pytest.mark.asyncio
async def test_no_live_connectors_is_reported(monkeypatch) -> None:
    _stub_incident(monkeypatch)
    _stub_resolver(monkeypatch)

    items, status = await _live(monkeypatch)

    assert items == []
    assert (status.status, status.reason) == ("unavailable", "no_live_connectors")


# -- invalid credentials and failures are visible --


@pytest.mark.asyncio
async def test_invalid_credential_is_reported_not_retried_and_isolated(monkeypatch) -> None:
    slack_item = _item("slack", "r2")
    rejected = _FakeLiveSource(error=LiveSourceAuthError("rejected"))
    healthy = _FakeLiveSource([slack_item])
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"github": rejected, "slack": healthy})
    _stub_incident(monkeypatch)
    _stub_resolver(monkeypatch, credentials=[_credential("github"), _credential("slack")])

    items, status = await _live(monkeypatch)

    assert items == [slack_item]  # one connector's bad token does not affect another
    assert status.status == "partial"
    assert [(o.source, o.outcome) for o in status.sources] == [
        ("github", "auth_failed"),
        ("slack", "ok"),
    ]
    assert rejected.calls == 1  # an auth failure is not retried


@pytest.mark.asyncio
async def test_all_sources_failing_is_reported_as_failed_without_leaking_details(
    monkeypatch,
) -> None:
    _no_sleep(monkeypatch)
    failing = _FakeLiveSource(error=RuntimeError("github search failed: token=" + _REAL_TOKEN))
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"github": failing})
    _stub_incident(monkeypatch)
    _stub_resolver(monkeypatch, credentials=[_credential("github")])

    items, status = await _live(monkeypatch)

    assert items == []
    assert status.status == "failed"
    assert [(o.outcome, o.reason) for o in status.sources] == [("error", "lookup_failed")]
    dumped = status.model_dump_json()
    assert _REAL_TOKEN not in dumped and "github search failed" not in dumped


@pytest.mark.asyncio
async def test_credential_that_could_not_be_decrypted_is_reported_as_skipped(monkeypatch) -> None:
    healthy = _FakeLiveSource([_item("slack", "r2")])
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"github": healthy, "slack": healthy})
    _stub_incident(monkeypatch)
    skipped = LiveConnectorSkip(
        connector_config_id=uuid.uuid4(), source="github", reason="credential_unavailable"
    )
    _stub_resolver(monkeypatch, credentials=[_credential("slack")], skipped=[skipped])

    _items, status = await _live(monkeypatch)

    assert status.status == "partial"
    assert {(o.source, o.outcome, o.reason) for o in status.sources} == {
        ("github", "skipped", "credential_unavailable"),
        ("slack", "ok", None),
    }


# -- when live evidence is (not) attempted, via the public entry point --


async def _gather_with_status(monkeypatch, *, incident_id, indexed=()):
    from app.agents.investigation.evidence import gather_evidence_with_status

    async def fake_code(session, query, filters, retry_count):
        return list(indexed)

    async def nothing(session, query, filters, retry_count):
        return []

    async def no_postmortems(session, actor, retry_count, existing_evidence):
        return []

    monkeypatch.setattr(evidence_module, "_gather_code_evidence", fake_code)
    monkeypatch.setattr(evidence_module, "_gather_knowledge_evidence", nothing)
    monkeypatch.setattr(evidence_module, "_gather_slack_evidence", nothing)
    monkeypatch.setattr(evidence_module, "_gather_postmortem_evidence", no_postmortems)
    monkeypatch.setattr(evidence_module, "_gather_jira_evidence", lambda: [])
    monkeypatch.setattr(evidence_module, "_gather_monitoring_evidence", lambda: [])
    return await gather_evidence_with_status(
        session=None, query="q", actor=_actor(), retry_count={}, incident_id=incident_id
    )


@pytest.mark.asyncio
async def test_question_without_an_incident_never_triggers_live_lookups(monkeypatch) -> None:
    """`POST /ask` with no `incident_id`: thin indexed evidence would normally
    trigger live augmentation, but an org-wide token must not be used for it.
    """
    monkeypatch.setattr(evidence_module.get_settings(), "investigation_live_evidence_enabled", True)
    fake_github = _FakeLiveSource([_item("commit", "r1")])
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"github": fake_github})
    _stub_incident(monkeypatch)
    resolver_calls = _stub_resolver(monkeypatch, credentials=[_credential("github")])

    items, status = await _gather_with_status(monkeypatch, incident_id=None)

    assert items == []
    assert (status.status, status.reason) == ("not_attempted", "no_incident_context")
    assert resolver_calls["count"] == 0
    assert fake_github.calls == 0


@pytest.mark.asyncio
async def test_kill_switch_reports_disabled(monkeypatch) -> None:
    monkeypatch.setattr(evidence_module.get_settings(), "investigation_live_evidence_enabled", False)
    resolver_calls = _stub_resolver(monkeypatch, credentials=[_credential("github")])

    _items, status = await _gather_with_status(monkeypatch, incident_id=uuid.uuid4())

    assert (status.status, status.reason) == ("not_attempted", "disabled")
    assert resolver_calls["count"] == 0


@pytest.mark.asyncio
async def test_full_indexed_evidence_skips_live_lookup(monkeypatch) -> None:
    monkeypatch.setattr(evidence_module.get_settings(), "investigation_live_evidence_enabled", True)
    resolver_calls = _stub_resolver(monkeypatch, credentials=[_credential("github")])
    full = [_item("commit", f"r{i}") for i in range(evidence_module._EVIDENCE_CAP)]

    items, status = await _gather_with_status(monkeypatch, incident_id=uuid.uuid4(), indexed=full)

    assert len(items) == evidence_module._EVIDENCE_CAP
    assert (status.status, status.reason) == ("not_attempted", "indexed_evidence_sufficient")
    assert resolver_calls["count"] == 0


@pytest.mark.asyncio
async def test_authorized_incident_investigation_appends_live_evidence(monkeypatch) -> None:
    monkeypatch.setattr(evidence_module.get_settings(), "investigation_live_evidence_enabled", True)
    live_item = _item("commit", "live-1")
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"github": _FakeLiveSource([live_item])})
    _stub_incident(monkeypatch)
    _stub_resolver(monkeypatch, credentials=[_credential("github")])

    items, status = await _gather_with_status(monkeypatch, incident_id=uuid.uuid4())

    assert items == [live_item]
    assert status.status == "ok"


# -- the monitoring stub must never be reported as a successful source --


def test_monitoring_source_is_flagged_as_not_implemented() -> None:
    assert _LIVE_SOURCES["monitoring"].implemented is False
    assert getattr(_LIVE_SOURCES["github"], "implemented", True) is True


def _investigator(organization_id: uuid.UUID) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=uuid.uuid4(),
        permissions=frozenset({"incident:read"}),
    )


def _row(organization_id: uuid.UUID, source: str, credential_ref: str) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        organization_id=organization_id,
        project_id=None,
        source=source,
        status="active",
        credential_ref=credential_ref,
        config={},
    )


@pytest.mark.asyncio
async def test_monitoring_only_reports_no_implemented_live_source_and_decrypts_nothing(
    monkeypatch,
) -> None:
    """REAL resolver, REAL `MonitoringLiveSource`: an org whose only live
    connector is the monitoring stub gets `unavailable` /
    `no_implemented_live_sources`, the source is never called, and no
    credential is decrypted.
    """
    from app.core.tenancy import service as real_tenancy

    organization_id = uuid.uuid4()
    monitoring_row = _row(organization_id, "monitoring", "would-be-decrypted-if-the-bug-existed")

    async def fake_rows(session, org_id):
        return [monitoring_row]

    async def decrypt_must_not_run(kms, encrypted):
        raise AssertionError("a stub source must never have its credential decrypted")

    called = {"n": 0}

    async def spy_fetch(self, **kwargs):
        called["n"] += 1
        return []

    monkeypatch.setattr(real_tenancy.repository, "list_connector_configs", fake_rows)
    monkeypatch.setattr(real_tenancy, "decrypt_secret", decrypt_must_not_run)
    monkeypatch.setattr(MonitoringLiveSource, "fetch_live_evidence", spy_fetch)
    _stub_incident(monkeypatch, incident=_incident(organization_id))

    items, status = await _live(monkeypatch, actor=_investigator(organization_id))

    assert items == []
    assert (status.status, status.reason) == ("unavailable", "no_implemented_live_sources")
    assert [(o.source, o.outcome) for o in status.sources] == [("monitoring", "not_implemented")]
    assert called["n"] == 0


@pytest.mark.asyncio
async def test_github_ok_plus_monitoring_stub_reports_github_ok_and_monitoring_not_implemented(
    monkeypatch,
) -> None:
    from app.core.tenancy import service as real_tenancy
    from app.shared.security import encrypt_secret, get_kms

    organization_id = uuid.uuid4()
    github_row = _row(organization_id, "github", await encrypt_secret(get_kms(), _REAL_TOKEN))
    monitoring_row = _row(organization_id, "monitoring", "not-decryptable-and-must-not-be-touched")

    async def fake_rows(session, org_id):
        return [github_row, monitoring_row]

    monkeypatch.setattr(real_tenancy.repository, "list_connector_configs", fake_rows)
    github_item = _item("commit", "r1")
    fake_github = _FakeLiveSource([github_item])
    monkeypatch.setattr(
        evidence_module,
        "_LIVE_SOURCES",
        {"github": fake_github, "monitoring": MonitoringLiveSource()},
    )
    _stub_incident(monkeypatch, incident=_incident(organization_id))

    items, status = await _live(monkeypatch, actor=_investigator(organization_id))

    assert items == [github_item]
    assert status.status == "ok"  # the stub neither inflates nor degrades it
    assert {(o.source, o.outcome) for o in status.sources} == {
        ("github", "ok"),
        ("monitoring", "not_implemented"),
    }
    assert fake_github.tokens_seen == [_REAL_TOKEN]


@pytest.mark.asyncio
async def test_failure_is_not_diluted_by_a_monitoring_stub(monkeypatch) -> None:
    _no_sleep(monkeypatch)
    failing = _FakeLiveSource(error=RuntimeError("boom"))
    monkeypatch.setattr(
        evidence_module,
        "_LIVE_SOURCES",
        {"github": failing, "monitoring": MonitoringLiveSource()},
    )
    _stub_incident(monkeypatch)
    skip = LiveConnectorSkip(
        connector_config_id=uuid.uuid4(), source="monitoring", reason="not_implemented"
    )
    _stub_resolver(monkeypatch, credentials=[_credential("github")], skipped=[skip])

    _items, status = await _live(monkeypatch)

    assert status.status == "failed"  # monitoring is not a success that makes this "partial"

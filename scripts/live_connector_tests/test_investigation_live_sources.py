"""Live integration tests for the Investigation Agent's live evidence sources
(`GitHubLiveSource`, `SlackLiveSource`) and the credential path they depend on.

OPT-IN, NEVER AUTO-COLLECTED
    Lives under `scripts/` (see `conftest.py` for why): run explicitly by path.

        pytest scripts/live_connector_tests/test_investigation_live_sources.py -v -s

TWO GROUPS
    1. NO CREDENTIALS NEEDED. Deliberately invalid tokens are sent to the real
       GitHub / Slack APIs (one or two requests each) to prove a rejected
       credential is reported as `LiveSourceAuthError` / `auth_failed` rather
       than silently becoming "no evidence". These run whenever this file is
       run, with no secrets involved.
    2. REAL TOKEN REQUIRED. Skipped (never failed) unless the existing
       `tests/ingestion_retrieval/.env` supplies:
           GitHub: EKIP_TEST_GITHUB_TOKEN, EKIP_TEST_GITHUB_REPO
           Slack:  EKIP_TEST_SLACK_BOT_TOKEN, EKIP_TEST_SLACK_CHANNEL_IDS
       Use dedicated, read-only, throwaway credentials scoped to one test repo /
       channel (fine-grained PAT with read-only contents/metadata/issues/pull
       requests; a Slack bot with `channels:history` on one test channel).

WHAT IS EXERCISED END TO END (no database needed)
    real `tenancy_service.register_connector` (real envelope encryption; only
    the row insert and audit write are replaced by in-memory stand-ins)
    -> the stored `credential_ref` is ciphertext, never the token
    -> real `resolve_live_evidence_connectors` (real decryption, org/project
       scoping, incident:read check)
    -> the real live source authenticating against the real API
    -> for the status tests, the real `_gather_live_evidence` reporting it.
    Real database persistence and RLS are covered separately by
    `scripts/verify_rls_isolation.py` against a disposable Postgres.

SAFETY
    * Tokens are never printed or logged; each test asserts the token does not
      appear in captured stdout/stderr or in any raised exception text.
    * Rate limits: at most one repo / one channel per test, and a pause after
      every test (GitHub's Search API allows 30 requests/minute; each GitHub
      lookup makes 2 requests).
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from app.agents.investigation import evidence as evidence_module
from app.agents.investigation.live.base import LiveSourceAuthError
from app.agents.investigation.live.github_live import GitHubLiveSource
from app.agents.investigation.live.slack_live import SlackLiveSource
from app.core.tenancy import service as tenancy_service
from app.core.tenancy.live_credentials import LiveConnectorCredential
from app.core.tenancy.schemas import ConnectorConfigCreate
from app.shared.schemas import ActorKind, Identity

_PAUSE_BETWEEN_TESTS_SECONDS = 3.0

_INVALID_GITHUB_TOKEN = "ghp_deliberately_invalid_investigation_live_token"  # noqa: S105
_INVALID_SLACK_TOKEN = "xoxb-deliberately-invalid-investigation-live-token"  # noqa: S105
# A public repo, used only so the (rejected) request is well-formed.
_PUBLIC_REPO = {"repo": "octocat/Hello-World", "ref": "master"}


@pytest.fixture(autouse=True)
async def _respect_rate_limits():
    yield
    await asyncio.sleep(_PAUSE_BETWEEN_TESTS_SECONDS)


class _StoredRow:
    """In-memory stand-in for a `connector_configs` row."""

    def __init__(self, **kwargs) -> None:
        now = datetime.now(UTC)
        self.id = uuid.uuid4()
        self.organization_id = kwargs["organization_id"]
        self.project_id = kwargs.get("project_id")
        self.source = kwargs["source"]
        self.credential_ref = kwargs["credential_ref"]
        self.config = kwargs.get("config") or {}
        self.status = "active"
        self.last_synced_at = None
        self.created_at = now
        self.updated_at = now


def _identity(organization_id: uuid.UUID, permissions: set[str]) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=uuid.uuid4(),
        permissions=frozenset(permissions),
    )


async def _register_and_resolve(
    monkeypatch, *, source: str, token: str, config: dict
) -> tuple[LiveConnectorCredential, str, Identity, uuid.UUID]:
    """Register through the REAL `register_connector` (admin), then resolve
    through the REAL `resolve_live_evidence_connectors` as a read-only
    investigator -- exactly the two actors the production fix separates.
    Returns (credential, stored_credential_ref, investigator, organization_id).
    """
    organization_id, project_id = uuid.uuid4(), uuid.uuid4()
    stored: list[_StoredRow] = []

    async def fake_insert_connector_config(session, **kwargs):
        row = _StoredRow(**kwargs)
        stored.append(row)
        return row

    async def fake_record_audit_event(*args, **kwargs):
        return None

    async def fake_list_connector_configs(session, org_id):
        return stored

    monkeypatch.setattr(
        tenancy_service.repository, "insert_connector_config", fake_insert_connector_config
    )
    monkeypatch.setattr(tenancy_service, "record_audit_event", fake_record_audit_event)
    monkeypatch.setattr(
        tenancy_service.repository, "list_connector_configs", fake_list_connector_configs
    )

    admin = _identity(organization_id, {"tenancy:manage"})
    await tenancy_service.register_connector(
        None,
        admin,
        organization_id,
        ConnectorConfigCreate(source=source, credential_ref=token, config=config),
    )
    stored_credential_ref = stored[0].credential_ref

    investigator = _identity(organization_id, {"incident:read"})  # NOT tenancy:manage
    resolution = await tenancy_service.resolve_live_evidence_connectors(
        None,
        investigator,
        organization_id,
        incident_organization_id=organization_id,
        incident_project_id=project_id,
        sources={source},
    )
    assert len(resolution.credentials) == 1 and not resolution.skipped
    return resolution.credentials[0], stored_credential_ref, investigator, organization_id


def _assert_no_secret_leak(capsys, token: str, *texts: str) -> None:
    captured = capsys.readouterr()
    assert token not in captured.out and token not in captured.err
    for text in texts:
        assert token not in text


_SINCE = datetime.now(UTC) - timedelta(days=30)


# --- group 1: no real credentials needed ------------------------------------


@pytest.mark.asyncio
async def test_registration_stores_ciphertext_and_resolution_round_trips(monkeypatch, capsys):
    credential, stored_ref, _investigator, _org = await _register_and_resolve(
        monkeypatch,
        source="github",
        token=_INVALID_GITHUB_TOKEN,
        config={"repos": [_PUBLIC_REPO]},
    )

    assert stored_ref != _INVALID_GITHUB_TOKEN
    assert _INVALID_GITHUB_TOKEN not in stored_ref  # encrypted, not merely wrapped
    assert credential.reveal_token() == _INVALID_GITHUB_TOKEN  # real decrypt round trip
    assert credential.reveal_token() != tenancy_service._REDACTED_CREDENTIAL
    assert _INVALID_GITHUB_TOKEN not in repr(credential)
    _assert_no_secret_leak(capsys, _INVALID_GITHUB_TOKEN, repr(credential))


@pytest.mark.asyncio
async def test_github_live_source_rejects_an_invalid_token(monkeypatch, capsys):
    credential, _ref, _inv, _org = await _register_and_resolve(
        monkeypatch,
        source="github",
        token=_INVALID_GITHUB_TOKEN,
        config={"repos": [_PUBLIC_REPO]},
    )

    with pytest.raises(LiveSourceAuthError) as excinfo:
        await GitHubLiveSource().fetch_live_evidence(
            connector=credential, query="outage", since=_SINCE, limit=3
        )

    _assert_no_secret_leak(capsys, _INVALID_GITHUB_TOKEN, str(excinfo.value), repr(excinfo.value))
    print("PASS: GitHubLiveSource reported the invalid token as LiveSourceAuthError")


@pytest.mark.asyncio
async def test_slack_live_source_rejects_an_invalid_token(monkeypatch, capsys):
    credential, _ref, _inv, _org = await _register_and_resolve(
        monkeypatch,
        source="slack",
        token=_INVALID_SLACK_TOKEN,
        config={"channels": ["C0000000000"]},
    )

    with pytest.raises(LiveSourceAuthError) as excinfo:
        await SlackLiveSource().fetch_live_evidence(
            connector=credential, query="outage", since=_SINCE, limit=3
        )

    _assert_no_secret_leak(capsys, _INVALID_SLACK_TOKEN, str(excinfo.value), repr(excinfo.value))
    print("PASS: SlackLiveSource reported the invalid token as LiveSourceAuthError")


@pytest.mark.asyncio
async def test_investigation_status_reports_auth_failed_for_an_invalid_github_token(
    monkeypatch, capsys
):
    """The whole chain through the REAL `_gather_live_evidence`: an invalid
    token must surface as `auth_failed` in the status, not as "ok, no items".
    """
    _cred, _ref, investigator, organization_id = await _register_and_resolve(
        monkeypatch,
        source="github",
        token=_INVALID_GITHUB_TOKEN,
        config={"repos": [_PUBLIC_REPO]},
    )

    async def fake_get_incident(session, actor, org_id, incident_id):
        # The incident read itself is covered by the unit tests; here only the
        # live path is under test. Same org, a project the actor can read.
        return SimpleNamespace(organization_id=organization_id, project_id=uuid.uuid4())

    monkeypatch.setattr(evidence_module.incidents_service, "get_incident", fake_get_incident)
    monkeypatch.setattr(
        evidence_module,
        "_LIVE_SOURCES",
        {"github": GitHubLiveSource(), "slack": SlackLiveSource()},
    )

    items, status = await evidence_module._gather_live_evidence(
        None, "outage", investigator, {}, uuid.uuid4()
    )

    assert items == []
    assert status.status == "failed"
    assert [(o.source, o.outcome) for o in status.sources] == [("github", "auth_failed")]
    _assert_no_secret_leak(capsys, _INVALID_GITHUB_TOKEN, status.model_dump_json())


# --- group 2: a real, dedicated read-only token is required -----------------


@pytest.mark.asyncio
async def test_github_real_token_end_to_end(github_spec, monkeypatch, capsys):
    token = github_spec.credential_ref
    repos = github_spec.config["repos"][:1]  # one repo: 2 Search API requests
    credential, stored_ref, _inv, _org = await _register_and_resolve(
        monkeypatch, source="github", token=token, config={"repos": repos}
    )

    assert token not in stored_ref
    assert credential.reveal_token() == token

    items = await GitHubLiveSource().fetch_live_evidence(
        connector=credential, query="error", since=_SINCE, limit=3
    )

    assert isinstance(items, list)  # may legitimately be empty
    assert all(item.metadata.get("retrieval_mode") == "live" for item in items)
    _assert_no_secret_leak(capsys, token)
    print(f"PASS: real GitHub token authenticated end to end; {len(items)} live item(s)")


@pytest.mark.asyncio
async def test_slack_real_token_end_to_end(slack_spec, monkeypatch, capsys):
    token = slack_spec.credential_ref
    channels = slack_spec.config["channels"][:1]  # one channel: 1 request
    credential, stored_ref, _inv, _org = await _register_and_resolve(
        monkeypatch, source="slack", token=token, config={"channels": channels}
    )

    assert token not in stored_ref
    assert credential.reveal_token() == token

    items = await SlackLiveSource().fetch_live_evidence(
        connector=credential, query="error", since=_SINCE, limit=3
    )

    assert isinstance(items, list)
    assert all(item.metadata.get("retrieval_mode") == "live" for item in items)
    _assert_no_secret_leak(capsys, token)
    print(f"PASS: real Slack token authenticated end to end; {len(items)} live item(s)")


@pytest.mark.asyncio
async def test_investigation_status_reports_ok_for_a_real_github_token(
    github_spec, monkeypatch, capsys
):
    token = github_spec.credential_ref
    _cred, _ref, investigator, organization_id = await _register_and_resolve(
        monkeypatch, source="github", token=token, config={"repos": github_spec.config["repos"][:1]}
    )

    async def fake_get_incident(session, actor, org_id, incident_id):
        return SimpleNamespace(organization_id=organization_id, project_id=uuid.uuid4())

    monkeypatch.setattr(evidence_module.incidents_service, "get_incident", fake_get_incident)
    monkeypatch.setattr(evidence_module, "_LIVE_SOURCES", {"github": GitHubLiveSource()})

    _items, status = await evidence_module._gather_live_evidence(
        None, "error", investigator, {}, uuid.uuid4()
    )

    assert status.status == "ok"
    assert [(o.source, o.outcome) for o in status.sources] == [("github", "ok")]
    _assert_no_secret_leak(capsys, token, status.model_dump_json())

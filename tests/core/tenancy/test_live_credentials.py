"""Tests for `core.tenancy.service.resolve_live_evidence_connectors` and the
`LiveConnectorCredential` carrier.

Covers: admin vs non-admin investigators, permission failure, project
scoping, cross-tenant denial, credential failures, and that the decrypted
secret cannot leak through repr / serialization / pickling / logs -- plus a
regression that the public `list_connectors` is still gated and redacted.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import pickle
import uuid
from datetime import UTC, datetime

import pytest
from fastapi.encoders import jsonable_encoder

from app.core.exceptions import PermissionDeniedError
from app.core.tenancy import service as tenancy_service
from app.core.tenancy.live_credentials import LiveConnectorCredential
from app.shared.schemas import ActorKind, Identity
from app.shared.security import KmsUnavailableError, encrypt_secret, get_kms

_SOURCES = {"github", "slack", "monitoring"}
_GITHUB_TOKEN = "ghp_livetoken_SUPERSECRET_1234567890"
_SLACK_TOKEN = "xoxb-livetoken-SUPERSECRET-0987654321"


def _identity(
    organization_id: uuid.UUID,
    permissions: set[str],
    project_permissions: dict[uuid.UUID, set[str]] | None = None,
) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=uuid.uuid4(),
        permissions=frozenset(permissions),
        project_permissions={k: frozenset(v) for k, v in (project_permissions or {}).items()},
    )


class _Row:
    def __init__(
        self,
        *,
        organization_id: uuid.UUID,
        source: str,
        credential_ref: str,
        project_id: uuid.UUID | None = None,
        status: str = "active",
        config: dict | None = None,
    ) -> None:
        now = datetime.now(UTC)
        self.id = uuid.uuid4()
        self.organization_id = organization_id
        self.source = source
        self.credential_ref = credential_ref
        self.project_id = project_id
        self.status = status
        self.config = config or {}
        self.last_synced_at = None
        self.created_at = now
        self.updated_at = now


async def _row(organization_id: uuid.UUID, source: str, token: str, **kwargs) -> _Row:
    return _Row(
        organization_id=organization_id,
        source=source,
        credential_ref=await encrypt_secret(get_kms(), token),
        **kwargs,
    )


def _patch_rows(monkeypatch, rows: list[_Row]) -> dict[str, int]:
    calls = {"count": 0}

    async def fake_list_connector_configs(session, organization_id):
        calls["count"] += 1
        return rows

    monkeypatch.setattr(
        tenancy_service.repository, "list_connector_configs", fake_list_connector_configs
    )
    return calls


async def _resolve(actor, organization_id, project_id, *, incident_org=None, sources=_SOURCES):
    return await tenancy_service.resolve_live_evidence_connectors(
        None,
        actor,
        organization_id,
        incident_organization_id=incident_org or organization_id,
        incident_project_id=project_id,
        sources=sources,
    )


# --- who may resolve ---------------------------------------------------------


@pytest.mark.asyncio
async def test_non_admin_investigator_gets_decrypted_credentials_without_tenancy_manage(
    monkeypatch,
) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    investigator = _identity(org, {"incident:read"})  # NOT tenancy:manage
    _patch_rows(
        monkeypatch,
        [
            await _row(org, "github", _GITHUB_TOKEN, config={"repos": [{"repo": "o/r"}]}),
            await _row(org, "slack", _SLACK_TOKEN),
        ],
    )

    result = await _resolve(investigator, org, project)

    tokens = {c.source: c.reveal_token() for c in result.credentials}
    assert tokens == {"github": _GITHUB_TOKEN, "slack": _SLACK_TOKEN}
    assert result.skipped == []


@pytest.mark.asyncio
async def test_admin_also_gets_decrypted_credentials_not_redacted_text(monkeypatch) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    admin = _identity(org, {"tenancy:manage", "incident:read"})
    _patch_rows(monkeypatch, [await _row(org, "github", _GITHUB_TOKEN)])

    result = await _resolve(admin, org, project)

    token = result.credentials[0].reveal_token()
    assert token == _GITHUB_TOKEN
    assert token != tenancy_service._REDACTED_CREDENTIAL


@pytest.mark.asyncio
async def test_tenancy_manage_alone_is_not_enough_without_incident_read(monkeypatch) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    admin_without_incident_access = _identity(org, {"tenancy:manage"})
    calls = _patch_rows(monkeypatch, [await _row(org, "github", _GITHUB_TOKEN)])

    with pytest.raises(PermissionDeniedError):
        await _resolve(admin_without_incident_access, org, project)

    assert calls["count"] == 0  # denied before touching connectors


@pytest.mark.asyncio
async def test_no_permissions_is_denied_before_reading_connectors(monkeypatch) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    calls = _patch_rows(monkeypatch, [await _row(org, "github", _GITHUB_TOKEN)])

    with pytest.raises(PermissionDeniedError):
        await _resolve(_identity(org, set()), org, project)

    assert calls["count"] == 0


@pytest.mark.asyncio
async def test_project_override_without_incident_read_denies_even_with_org_level_read(
    monkeypatch,
) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    actor = _identity(org, {"incident:read"}, project_permissions={project: {"tenancy:manage"}})
    calls = _patch_rows(monkeypatch, [await _row(org, "github", _GITHUB_TOKEN)])

    with pytest.raises(PermissionDeniedError):
        await _resolve(actor, org, project)

    assert calls["count"] == 0


@pytest.mark.asyncio
async def test_incident_read_on_a_different_project_does_not_authorize(monkeypatch) -> None:
    org, project_a, project_b = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    actor = _identity(org, set(), project_permissions={project_a: {"incident:read"}})
    calls = _patch_rows(monkeypatch, [await _row(org, "github", _GITHUB_TOKEN)])

    with pytest.raises(PermissionDeniedError):
        await _resolve(actor, org, project_b)  # incident lives in project B

    assert calls["count"] == 0


# --- tenant isolation --------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_organization_request_is_denied(monkeypatch) -> None:
    my_org, other_org, project = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    actor = _identity(my_org, {"incident:read"})
    calls = _patch_rows(monkeypatch, [await _row(other_org, "github", _GITHUB_TOKEN)])

    with pytest.raises(PermissionDeniedError):
        await _resolve(actor, other_org, project)

    assert calls["count"] == 0


@pytest.mark.asyncio
async def test_incident_from_another_organization_is_denied(monkeypatch) -> None:
    my_org, other_org, project = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    actor = _identity(my_org, {"incident:read"})
    calls = _patch_rows(monkeypatch, [await _row(my_org, "github", _GITHUB_TOKEN)])

    with pytest.raises(PermissionDeniedError):
        await _resolve(actor, my_org, project, incident_org=other_org)

    assert calls["count"] == 0


@pytest.mark.asyncio
async def test_a_row_from_another_org_is_never_returned_even_if_the_repository_leaks_it(
    monkeypatch,
) -> None:
    my_org, other_org, project = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    actor = _identity(my_org, {"incident:read"})
    _patch_rows(
        monkeypatch,
        [
            await _row(other_org, "github", "ghp_OTHER_TENANT_TOKEN"),
            await _row(my_org, "slack", _SLACK_TOKEN),
        ],
    )

    result = await _resolve(actor, my_org, project)

    assert [c.source for c in result.credentials] == ["slack"]
    assert all(c.organization_id == my_org for c in result.credentials)


# --- connector filtering -----------------------------------------------------


@pytest.mark.asyncio
async def test_project_scoping_uses_org_wide_and_same_project_only(monkeypatch) -> None:
    org, project, other_project = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    actor = _identity(org, {"incident:read"})
    _patch_rows(
        monkeypatch,
        [
            await _row(org, "github", "tok-org-wide", project_id=None),
            await _row(org, "github", "tok-same-project", project_id=project),
            await _row(org, "github", "tok-other-project", project_id=other_project),
        ],
    )

    result = await _resolve(actor, org, project)

    assert {c.reveal_token() for c in result.credentials} == {"tok-org-wide", "tok-same-project"}


@pytest.mark.asyncio
async def test_inactive_and_unsupported_sources_are_excluded(monkeypatch) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    actor = _identity(org, {"incident:read"})
    _patch_rows(
        monkeypatch,
        [
            await _row(org, "github", "tok-active"),
            await _row(org, "github", "tok-disconnected", status="disconnected"),
            await _row(org, "github", "tok-error", status="error"),
            await _row(org, "jira", "tok-unsupported-source"),
        ],
    )

    result = await _resolve(actor, org, project)

    assert [c.reveal_token() for c in result.credentials] == ["tok-active"]


@pytest.mark.asyncio
async def test_underscore_config_keys_are_stripped(monkeypatch) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    actor = _identity(org, {"incident:read"})
    _patch_rows(
        monkeypatch,
        [
            await _row(
                org,
                "github",
                _GITHUB_TOKEN,
                config={"repos": [{"repo": "o/r"}], "_cursor": "bearer-ish-secret"},
            )
        ],
    )

    result = await _resolve(actor, org, project)

    assert result.credentials[0].config == {"repos": [{"repo": "o/r"}]}


# --- credential failures -----------------------------------------------------


@pytest.mark.asyncio
async def test_corrupt_credential_is_skipped_without_aborting_others_or_leaking(
    monkeypatch, caplog
) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    actor = _identity(org, {"incident:read"})
    corrupt = _Row(organization_id=org, source="github", credential_ref="not-a-valid-envelope")
    _patch_rows(monkeypatch, [corrupt, await _row(org, "slack", _SLACK_TOKEN)])

    with caplog.at_level(logging.DEBUG):
        result = await _resolve(actor, org, project)

    assert [c.source for c in result.credentials] == ["slack"]
    assert [(s.source, s.reason) for s in result.skipped] == [("github", "credential_unavailable")]
    assert "not-a-valid-envelope" not in caplog.text
    assert _SLACK_TOKEN not in caplog.text


@pytest.mark.asyncio
async def test_kms_outage_is_reported_distinctly(monkeypatch) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    actor = _identity(org, {"incident:read"})
    _patch_rows(monkeypatch, [await _row(org, "github", _GITHUB_TOKEN)])

    async def failing_decrypt(kms, encrypted):
        raise KmsUnavailableError("vault unreachable")

    monkeypatch.setattr(tenancy_service, "decrypt_secret", failing_decrypt)

    result = await _resolve(actor, org, project)

    assert result.credentials == []
    assert [s.reason for s in result.skipped] == ["kms_unavailable"]


# --- the secret cannot leak through the carrier ------------------------------


def _credential() -> LiveConnectorCredential:
    return LiveConnectorCredential(
        connector_config_id=uuid.uuid4(),
        organization_id=uuid.uuid4(),
        project_id=None,
        source="github",
        config={"repos": []},
        token=_GITHUB_TOKEN,
    )


def test_repr_and_str_never_include_the_token() -> None:
    credential = _credential()
    assert _GITHUB_TOKEN not in repr(credential)
    assert _GITHUB_TOKEN not in str(credential)
    assert _GITHUB_TOKEN not in f"{credential}"
    assert _GITHUB_TOKEN not in repr([credential])
    assert "<redacted>" in repr(credential)


def test_generic_serializers_cannot_reach_the_token() -> None:
    credential = _credential()

    with pytest.raises(TypeError):
        dataclasses.asdict(credential)  # not a dataclass
    with pytest.raises(Exception) as excinfo:
        jsonable_encoder(credential)  # what FastAPI would do on a response
    assert _GITHUB_TOKEN not in str(excinfo.value)
    assert not hasattr(credential, "__dict__")
    assert not hasattr(credential, "model_dump")


def test_credential_cannot_be_pickled_or_copied() -> None:
    credential = _credential()
    with pytest.raises(TypeError):
        pickle.dumps(credential)
    with pytest.raises(TypeError):
        copy.copy(credential)
    with pytest.raises(TypeError):
        copy.deepcopy(credential)


def test_token_not_in_a_traceback_message() -> None:
    credential = _credential()
    try:
        raise RuntimeError(f"live lookup failed for {credential!r}")
    except RuntimeError as exc:
        assert _GITHUB_TOKEN not in str(exc)


# --- regression: the public management API is unchanged ---------------------


@pytest.mark.asyncio
async def test_public_list_connectors_still_requires_tenancy_manage_and_redacts(
    monkeypatch,
) -> None:
    org = uuid.uuid4()
    investigator = _identity(org, {"incident:read"})

    with pytest.raises(PermissionDeniedError):
        await tenancy_service.list_connectors(None, investigator, org)

    admin = _identity(org, {"tenancy:manage"})
    _patch_rows(monkeypatch, [await _row(org, "github", _GITHUB_TOKEN)])

    listed = await tenancy_service.list_connectors(None, admin, org)

    assert [c.credential_ref for c in listed] == [tenancy_service._REDACTED_CREDENTIAL]
    assert _GITHUB_TOKEN not in listed[0].model_dump_json()


# --- not-implemented (stub) sources: reported, never decrypted --------------


@pytest.mark.asyncio
async def test_not_implemented_source_is_reported_without_decrypting(monkeypatch) -> None:
    org, project = uuid.uuid4(), uuid.uuid4()
    actor = _identity(org, {"incident:read"})
    stub_row = _Row(organization_id=org, source="monitoring", credential_ref="never-decrypted")
    github_row = await _row(org, "github", _GITHUB_TOKEN)
    _patch_rows(monkeypatch, [stub_row, github_row])
    decrypted: list[str] = []
    real_decrypt = tenancy_service.decrypt_secret

    async def tracking_decrypt(kms, encrypted):
        decrypted.append(encrypted)
        return await real_decrypt(kms, encrypted)

    monkeypatch.setattr(tenancy_service, "decrypt_secret", tracking_decrypt)

    result = await tenancy_service.resolve_live_evidence_connectors(
        None,
        actor,
        org,
        incident_organization_id=org,
        incident_project_id=project,
        sources={"github"},
        not_implemented_sources={"monitoring"},
    )

    assert [c.source for c in result.credentials] == ["github"]
    assert [(s.source, s.reason) for s in result.skipped] == [("monitoring", "not_implemented")]
    assert decrypted == [github_row.credential_ref]  # the stub's credential was never touched


@pytest.mark.asyncio
async def test_not_implemented_source_respects_org_status_and_project_scoping(monkeypatch) -> None:
    org, other_org, project, other_project = (uuid.uuid4() for _ in range(4))
    actor = _identity(org, {"incident:read"})
    _patch_rows(
        monkeypatch,
        [
            _Row(organization_id=other_org, source="monitoring", credential_ref="x"),
            _Row(organization_id=org, source="monitoring", credential_ref="x", status="error"),
            _Row(
                organization_id=org,
                source="monitoring",
                credential_ref="x",
                project_id=other_project,
            ),
        ],
    )

    result = await tenancy_service.resolve_live_evidence_connectors(
        None,
        actor,
        org,
        incident_organization_id=org,
        incident_project_id=project,
        sources=set(),
        not_implemented_sources={"monitoring"},
    )

    assert result.skipped == [] and result.credentials == []

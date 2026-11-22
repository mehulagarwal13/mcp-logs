"""Tests for `app.core.privacy.service` -- data-subject deletion.

`repository.py`'s functions are monkeypatched with in-memory fakes (the same
style as `tests/core/knowledge/test_service.py`), so these exercise the
service's own ownership decisions, sequencing, idempotency, partial-failure
accounting and authorization -- not ORM/SQL behavior. The repository's SQL
scoping is covered separately by `test_repository_scoping.py`.

These are deliberately thorough. Deletion code that misunderstands ownership
destroys data that cannot be recovered, and the whole point of this module is
that the *retain* decisions are as load-bearing as the delete ones.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager

import pytest

from app.core.exceptions import (
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ValidationError,
)
from app.core.privacy import service as privacy_service
from app.shared.schemas import ActorKind, Identity


def _admin(organization_id: uuid.UUID) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=uuid.uuid4(),
        permissions=frozenset({"tenancy:manage"}),
    )


def _member_no_permissions(organization_id: uuid.UUID) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=uuid.uuid4(),
    )


@pytest.fixture()
def repo(monkeypatch):
    """Installs counting/mutating fakes over every repository function and
    returns a mutable state dict the tests can assert against."""
    state: dict[str, object] = {
        "user_exists": True,
        "is_anonymized": False,
        "email": "person@example.com",
        "counts": {
            "refresh_tokens": 3,
            "user_roles": 1,
            "project_memberships": 2,
            "external_identity_mappings": 1,
            "agent_executions": 40,
            "mcp_requests": 6,
            "invitations": 1,
            "incidents_reported": 7,
            "user_private_memory": 4,
        },
        "calls": [],
        "revoked_sessions": 0,
        # Membership / admin-guard / multi-org fakes (member-management audit V1+V2).
        "is_member": True,
        "last_admin": False,
        "other_orgs": [],
        "admin_checks": [],
        "savepoints": 0,
        "savepoint_rollbacks": 0,
    }

    async def _count(key):
        return state["counts"][key]  # type: ignore[index]

    monkeypatch.setattr(
        privacy_service.repository, "user_exists",
        lambda s, u: _async(state["user_exists"]),
    )
    monkeypatch.setattr(
        privacy_service.users_service, "is_organization_member",
        lambda s, u, o: _async(state["is_member"]),
    )

    async def fake_assert_keeps_admin(session, organization_id, removing_user_id, *, lock):
        state["admin_checks"].append((organization_id, removing_user_id, lock))  # type: ignore[union-attr]
        if lock:
            state["calls"].append("admin_lock")  # type: ignore[union-attr]
        if state["last_admin"]:
            raise ConflictError("last admin", error_code="org.last_admin")

    monkeypatch.setattr(
        privacy_service.users_service,
        "assert_organization_keeps_an_admin",
        fake_assert_keeps_admin,
    )

    @asynccontextmanager
    async def fake_atomic(session):
        """Stand-in for the SAVEPOINT: on an exception, forget the repository
        calls made inside it (what ROLLBACK TO SAVEPOINT does to the data)."""
        mark = len(state["calls"])  # type: ignore[arg-type]
        state["savepoints"] += 1  # type: ignore[operator]
        try:
            yield
        except BaseException:
            del state["calls"][mark:]  # type: ignore[union-attr]
            state["savepoint_rollbacks"] += 1  # type: ignore[operator]
            raise

    monkeypatch.setattr(privacy_service, "_atomic", fake_atomic)

    async def fake_lock_user(session, user_id):
        state["calls"].append("user_lock")  # type: ignore[union-attr]
        return state["user_exists"]

    monkeypatch.setattr(privacy_service.users_service, "lock_user_for_deletion", fake_lock_user)
    monkeypatch.setattr(
        privacy_service.users_service, "list_organizations_for_login",
        lambda s, u: _async(list(state["other_orgs"])),
    )
    monkeypatch.setattr(
        privacy_service.repository, "is_user_anonymized",
        lambda s, u: _async(state["is_anonymized"]),
    )
    monkeypatch.setattr(
        privacy_service.repository, "get_user_email",
        lambda s, u: _async(state["email"]),
    )
    monkeypatch.setattr(
        privacy_service.repository, "count_refresh_tokens",
        lambda s, u, o: _count("refresh_tokens"),
    )
    monkeypatch.setattr(
        privacy_service.repository, "count_user_roles",
        lambda s, u, o: _count("user_roles"),
    )
    monkeypatch.setattr(
        privacy_service.repository, "count_project_memberships",
        lambda s, u, o: _count("project_memberships"),
    )
    monkeypatch.setattr(
        privacy_service.repository, "count_external_identity_mappings",
        lambda s, u, o: _count("external_identity_mappings"),
    )
    monkeypatch.setattr(
        privacy_service.repository, "count_agent_executions",
        lambda s, u, o: _count("agent_executions"),
    )
    monkeypatch.setattr(
        privacy_service.repository, "count_mcp_requests",
        lambda s, u, o: _count("mcp_requests"),
    )
    monkeypatch.setattr(
        privacy_service.repository, "count_invitations_for_email",
        lambda s, e, o: _count("invitations"),
    )
    monkeypatch.setattr(
        privacy_service.repository, "count_incidents_reported",
        lambda s, u, o: _count("incidents_reported"),
    )
    monkeypatch.setattr(
        privacy_service.repository, "anonymized_email_for",
        lambda u: f"deleted-user-{u}@deleted.invalid",
    )
    # Priority 4 integration: user-private agent memory is hard-deleted as
    # part of the same plan/execution.
    monkeypatch.setattr(
        privacy_service.memory_repository, "count_user_scoped",
        lambda s, u, o: _count("user_private_memory"),
    )

    def _mutation(name, rows):
        async def _fn(*args, **kwargs):
            state["calls"].append(name)  # type: ignore[union-attr]
            return rows
        return _fn

    monkeypatch.setattr(
        privacy_service.repository, "delete_refresh_tokens", _mutation("delete_refresh_tokens", 3)
    )
    monkeypatch.setattr(
        privacy_service.repository, "delete_user_roles", _mutation("delete_user_roles", 1)
    )
    monkeypatch.setattr(
        privacy_service.repository,
        "delete_project_memberships",
        _mutation("delete_project_memberships", 2),
    )
    monkeypatch.setattr(
        privacy_service.repository,
        "delete_external_identity_mappings",
        _mutation("delete_external_identity_mappings", 1),
    )
    monkeypatch.setattr(
        privacy_service.repository,
        "anonymize_agent_executions",
        _mutation("anonymize_agent_executions", 40),
    )
    monkeypatch.setattr(
        privacy_service.repository, "delete_mcp_requests", _mutation("delete_mcp_requests", 6)
    )
    monkeypatch.setattr(
        privacy_service.repository,
        "anonymize_invitations_for_email",
        _mutation("anonymize_invitations_for_email", 1),
    )
    monkeypatch.setattr(
        privacy_service.repository, "anonymize_user_record", _mutation("anonymize_user_record", 1)
    )
    monkeypatch.setattr(
        privacy_service.memory_repository,
        "hard_delete_user_scoped",
        _mutation("hard_delete_user_scoped_memory", 4),
    )

    async def fake_revoke_all_sessions(session, user_id, organization_id):
        state["revoked_sessions"] = int(state["revoked_sessions"]) + 1  # type: ignore[arg-type]
        return 3

    monkeypatch.setattr(
        privacy_service.auth_service, "revoke_all_sessions", fake_revoke_all_sessions
    )

    audit_events: list[dict] = []

    async def fake_record_audit_event(session, actor, **kwargs):
        audit_events.append(kwargs)

    monkeypatch.setattr(privacy_service, "record_audit_event", fake_record_audit_event)
    state["audit_events"] = audit_events
    return state


async def _async(value):
    return value


# --- authorization ---------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_requires_tenancy_manage(repo):
    organization_id = uuid.uuid4()
    actor = _member_no_permissions(organization_id)
    with pytest.raises(PermissionDeniedError):
        await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())


@pytest.mark.asyncio
async def test_execute_requires_tenancy_manage(repo):
    """The permission check must reject before ANY mutation runs."""
    organization_id = uuid.uuid4()
    actor = _member_no_permissions(organization_id)
    with pytest.raises(PermissionDeniedError):
        await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert repo["calls"] == []


@pytest.mark.asyncio
async def test_deletion_always_uses_the_actors_own_organization(repo):
    """Cross-tenant deletion prevention: the organization is taken from the
    actor, never from a parameter, so there is no caller-supplied org to
    tamper with."""
    organization_id = uuid.uuid4()
    actor = _admin(organization_id)
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert result.organization_id == organization_id


# --- unsupported scopes ----------------------------------------------------


@pytest.mark.parametrize("scope", ["organization", "user_account"])
@pytest.mark.asyncio
async def test_unimplemented_scopes_are_rejected_loudly(repo, scope):
    """Silently downgrading an org-deletion request to user-scope would
    report success while leaving every document and embedding in place."""
    actor = _admin(uuid.uuid4())
    with pytest.raises(ValidationError, match="not implemented"):
        await privacy_service.execute_user_data_deletion(
            None, actor, uuid.uuid4(), scope=scope
        )
    assert repo["calls"] == []


# --- planning (dry run) ----------------------------------------------------


@pytest.mark.asyncio
async def test_plan_mutates_nothing(repo):
    actor = _admin(uuid.uuid4())
    await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())
    assert repo["calls"] == []
    assert repo["revoked_sessions"] == 0
    assert repo["audit_events"] == []


@pytest.mark.asyncio
async def test_plan_missing_user_raises_not_found(repo):
    repo["user_exists"] = False
    actor = _admin(uuid.uuid4())
    with pytest.raises(NotFoundError):
        await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())


@pytest.mark.asyncio
async def test_plan_classifies_organization_knowledge_as_retained(repo):
    """The single most important assertion in this file: a user deletion must
    never propose touching organization-owned knowledge."""
    actor = _admin(uuid.uuid4())
    plan = await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())
    retained = {s.category for s in plan.retain_steps}
    assert "organization_knowledge" in retained
    assert "incidents_reported" in retained
    assert "audit_trail" in retained
    # And none of those may appear in a mutating section.
    mutating = {s.category for s in plan.hard_delete_steps + plan.anonymize_steps}
    assert mutating.isdisjoint({"organization_knowledge", "incidents_reported", "audit_trail"})


@pytest.mark.asyncio
async def test_plan_classifies_access_artifacts_as_hard_delete(repo):
    actor = _admin(uuid.uuid4())
    plan = await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())
    hard = {s.category for s in plan.hard_delete_steps}
    assert hard == {
        "sessions",
        "organization_roles",
        "project_memberships",
        "sso_identity_links",
        "mcp_request_history",
        "user_private_memory",
    }


@pytest.mark.asyncio
async def test_plan_classifies_user_record_and_telemetry_as_anonymize(repo):
    actor = _admin(uuid.uuid4())
    plan = await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())
    anon = {s.category for s in plan.anonymize_steps}
    assert anon == {"agent_execution_history", "invitations_received", "user_record"}


@pytest.mark.asyncio
async def test_plan_counts_exclude_retained_rows_from_blast_radius(repo):
    """`incidents_reported` is 7 in the fixture but is retained, so it must
    not inflate the count of rows the operation will actually change."""
    actor = _admin(uuid.uuid4())
    plan = await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())
    # 3 tokens + 1 role + 2 memberships + 1 mapping + 40 executions
    # + 4 private memories + 1 invite + 1 user
    assert plan.total_rows_affected == 59


@pytest.mark.asyncio
async def test_every_planned_step_carries_a_rationale(repo):
    """A plan whose steps cannot explain themselves is not reviewable."""
    actor = _admin(uuid.uuid4())
    plan = await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())
    assert all(step.rationale.strip() for step in plan.steps)


# --- execution -------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_completes_and_runs_every_mutation(repo):
    actor = _admin(uuid.uuid4())
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())

    assert result.status == "completed"
    assert result.failed_steps == []
    # Membership (`delete_user_roles`) is deliberately LAST.
    assert repo["calls"] == [
        "admin_lock",
        "user_lock",
        "delete_refresh_tokens",
        "delete_project_memberships",
        "delete_external_identity_mappings",
        "anonymize_agent_executions",
        "delete_mcp_requests",
        "hard_delete_user_scoped_memory",
        "anonymize_invitations_for_email",
        "anonymize_user_record",
        "delete_user_roles",
    ]


@pytest.mark.asyncio
async def test_execute_revokes_sessions_through_the_existing_auth_service(repo):
    """Reuse, not reimplementation: revocation goes through
    `core.auth.service.revoke_all_sessions` before token rows are deleted."""
    actor = _admin(uuid.uuid4())
    await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert repo["revoked_sessions"] == 1


@pytest.mark.asyncio
async def test_execute_reports_row_counts_split_by_action(repo):
    actor = _admin(uuid.uuid4())
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert result.hard_deleted_row_count == 3 + 1 + 2 + 1 + 6 + 4
    assert result.anonymized_row_count == 40 + 1 + 1


# --- audit -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_event_records_the_operation_without_personal_data(repo):
    """The audit row must not preserve what the deletion just removed."""
    actor = _admin(uuid.uuid4())
    target = uuid.uuid4()
    await privacy_service.execute_user_data_deletion(None, actor, target)

    assert len(repo["audit_events"]) == 1
    event = repo["audit_events"][0]
    assert event["action"] == "privacy.user_data_deleted"
    assert event["resource_id"] == target
    serialized = str(event)
    assert "person@example.com" not in serialized
    assert "Deleted User" not in serialized
    assert event["metadata"]["status"] == "completed"
    assert event["metadata"]["retained_categories"]


# --- idempotency -----------------------------------------------------------


@pytest.mark.asyncio
async def test_second_run_is_a_safe_noop(repo):
    """Re-running after a completed deletion must succeed, not error."""
    actor = _admin(uuid.uuid4())
    target = uuid.uuid4()
    first = await privacy_service.execute_user_data_deletion(None, actor, target)
    assert first.was_noop is False

    # Simulate post-deletion state: user anonymized, no resolvable email,
    # every mutation now matching zero rows.
    repo["is_anonymized"] = True
    repo["email"] = None
    for key in repo["counts"]:
        repo["counts"][key] = 0  # type: ignore[index]

    second = await privacy_service.execute_user_data_deletion(None, actor, target)
    assert second.status == "completed"
    assert second.was_noop is True
    assert second.failed_steps == []


@pytest.mark.asyncio
async def test_third_run_is_still_safe(repo):
    actor = _admin(uuid.uuid4())
    target = uuid.uuid4()
    repo["is_anonymized"] = True
    repo["email"] = None
    for _ in range(3):
        result = await privacy_service.execute_user_data_deletion(None, actor, target)
        assert result.status == "completed"
        assert result.was_noop is True


@pytest.mark.asyncio
async def test_already_anonymized_user_skips_invitation_lookup_cleanly(repo):
    """With no resolvable email there is nothing to search invitations by --
    recorded as a real succeeded zero-row step, so step lists stay
    comparable across runs."""
    repo["is_anonymized"] = True
    repo["email"] = None
    actor = _admin(uuid.uuid4())
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())

    invitation_step = next(s for s in result.steps if s.category == "invitations_received")
    assert invitation_step.succeeded
    assert invitation_step.rows_affected == 0
    assert "anonymize_invitations_for_email" not in repo["calls"]
    # The step list still covers every category.
    assert len(result.steps) == 9


# --- partial failure -------------------------------------------------------


@pytest.mark.asyncio
async def test_one_failing_step_does_not_prevent_the_others(repo, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("simulated derived-cleanup failure")

    monkeypatch.setattr(privacy_service.repository, "anonymize_agent_executions", boom)

    actor = _admin(uuid.uuid4())
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())

    assert result.status == "partially_completed"
    # The failing step, plus the two steps deliberately held back because of it.
    assert [s.category for s in result.failed_steps] == [
        "agent_execution_history",
        "user_record",
        "organization_roles",
    ]
    # Independent cleanup steps still ran despite the failure ...
    assert "hard_delete_user_scoped_memory" in repo["calls"]
    assert "anonymize_invitations_for_email" in repo["calls"]
    # ... but the global account and the membership were NOT touched.
    assert "anonymize_user_record" not in repo["calls"]
    assert "delete_user_roles" not in repo["calls"]


@pytest.mark.asyncio
async def test_partial_failure_is_never_reported_as_completed(repo, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("nope")

    monkeypatch.setattr(privacy_service.repository, "anonymize_user_record", boom)
    actor = _admin(uuid.uuid4())
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert result.status != "completed"
    assert result.status == "partially_completed"


@pytest.mark.asyncio
async def test_all_steps_failing_reports_failed_not_partial(repo, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("everything is broken")

    for name in (
        "delete_refresh_tokens",
        "delete_user_roles",
        "delete_project_memberships",
        "delete_external_identity_mappings",
        "anonymize_agent_executions",
        "delete_mcp_requests",
        "anonymize_invitations_for_email",
        "anonymize_user_record",
    ):
        monkeypatch.setattr(privacy_service.repository, name, boom)
    # The memory step lives in a different repository module -- it has to
    # fail too for "every step failed" to actually hold.
    monkeypatch.setattr(privacy_service.memory_repository, "hard_delete_user_scoped", boom)

    async def boom_revoke(*args, **kwargs):
        raise RuntimeError("revocation failed")

    monkeypatch.setattr(privacy_service.auth_service, "revoke_all_sessions", boom_revoke)

    actor = _admin(uuid.uuid4())
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert result.status == "failed"
    assert len(result.failed_steps) == 9


@pytest.mark.asyncio
async def test_failure_reason_is_observable_per_step(repo, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(privacy_service.repository, "delete_user_roles", boom)
    actor = _admin(uuid.uuid4())
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())

    failed = next(s for s in result.failed_steps if s.category == "organization_roles")
    assert "disk on fire" in (failed.error or "")


@pytest.mark.asyncio
async def test_audit_event_records_failed_step_names(repo, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("x")

    monkeypatch.setattr(privacy_service.repository, "delete_user_roles", boom)
    actor = _admin(uuid.uuid4())
    await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())

    event = repo["audit_events"][0]
    # The final two steps are atomic, so both are reported when either fails.
    assert event["metadata"]["failed_steps"] == ["user_record", "organization_roles"]
    assert event["metadata"]["status"] == "partially_completed"


# --- retry after partial failure ------------------------------------------


@pytest.mark.asyncio
async def test_retry_after_partial_failure_completes(repo, monkeypatch):
    """The supported recovery path: fix the cause, run it again."""
    calls = {"n": 0}

    async def fails_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return 0

    monkeypatch.setattr(privacy_service.repository, "anonymize_agent_executions", fails_once)

    actor = _admin(uuid.uuid4())
    target = uuid.uuid4()
    first = await privacy_service.execute_user_data_deletion(None, actor, target)
    assert first.status == "partially_completed"

    second = await privacy_service.execute_user_data_deletion(None, actor, target)
    assert second.status == "completed"


# --- V1: cross-tenant protection -------------------------------------------


@pytest.mark.asyncio
async def test_execute_on_a_non_member_is_refused_and_mutates_nothing(repo):
    """V1: a user who is not a current member of the actor's organization must
    never be touched -- above all the global `users` row."""
    repo["is_member"] = False
    actor = _admin(uuid.uuid4())
    with pytest.raises(NotFoundError) as excinfo:
        await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert excinfo.value.error_code == "privacy.user_not_found"
    assert repo["calls"] == []
    assert "anonymize_user_record" not in repo["calls"]
    assert repo["revoked_sessions"] == 0
    assert repo["audit_events"] == []


@pytest.mark.asyncio
async def test_plan_on_a_non_member_is_refused(repo):
    repo["is_member"] = False
    actor = _admin(uuid.uuid4())
    with pytest.raises(NotFoundError):
        await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())


@pytest.mark.asyncio
async def test_non_member_and_nonexistent_user_are_indistinguishable(repo):
    """No existence oracle across tenants: same error code, same message."""
    actor = _admin(uuid.uuid4())
    repo["user_exists"] = False
    with pytest.raises(NotFoundError) as missing:
        await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    repo["user_exists"] = True
    repo["is_member"] = False
    with pytest.raises(NotFoundError) as other_tenant:
        await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert missing.value.error_code == other_tenant.value.error_code
    assert missing.value.message == other_tenant.value.message


@pytest.mark.asyncio
async def test_repeating_a_completed_deletion_is_not_found_because_no_longer_a_member(repo):
    """Documented consequence of requiring current membership."""
    actor = _admin(uuid.uuid4())
    target = uuid.uuid4()
    await privacy_service.execute_user_data_deletion(None, actor, target)
    repo["is_member"] = False  # their roles are gone now
    repo["calls"].clear()
    with pytest.raises(NotFoundError):
        await privacy_service.execute_user_data_deletion(None, actor, target)
    assert repo["calls"] == []


@pytest.mark.asyncio
async def test_member_check_uses_the_actors_organization(repo, monkeypatch):
    seen: list[tuple] = []

    async def spy(session, user_id, organization_id):
        seen.append((user_id, organization_id))
        return True

    monkeypatch.setattr(privacy_service.users_service, "is_organization_member", spy)
    organization_id = uuid.uuid4()
    target = uuid.uuid4()
    await privacy_service.plan_user_data_deletion(None, _admin(organization_id), target)
    assert seen == [(target, organization_id)]


# --- V2: self-deletion and last admin --------------------------------------


@pytest.mark.asyncio
async def test_self_deletion_is_refused_even_when_another_admin_exists(repo):
    actor = _admin(uuid.uuid4())
    with pytest.raises(ValidationError) as excinfo:
        await privacy_service.execute_user_data_deletion(None, actor, actor.user_id)
    assert excinfo.value.error_code == "privacy.cannot_delete_self"
    assert repo["calls"] == []
    assert repo["admin_checks"] == []  # refused before any admin/lock logic
    assert repo["audit_events"] == []


@pytest.mark.asyncio
async def test_self_plan_is_refused_too(repo):
    actor = _admin(uuid.uuid4())
    with pytest.raises(ValidationError):
        await privacy_service.plan_user_data_deletion(None, actor, actor.user_id)


@pytest.mark.asyncio
async def test_last_admin_deletion_is_refused_and_mutates_nothing(repo):
    repo["last_admin"] = True
    actor = _admin(uuid.uuid4())
    with pytest.raises(ConflictError) as excinfo:
        await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert excinfo.value.error_code == "org.last_admin"
    assert repo["calls"] == []
    assert repo["revoked_sessions"] == 0
    assert repo["audit_events"] == []


@pytest.mark.asyncio
async def test_dry_run_reports_the_last_admin_refusal_without_locking(repo):
    repo["last_admin"] = True
    actor = _admin(uuid.uuid4())
    with pytest.raises(ConflictError):
        await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())
    assert [lock for (_o, _u, lock) in repo["admin_checks"]] == [False]
    assert "admin_lock" not in repo["calls"]


@pytest.mark.asyncio
async def test_execute_rechecks_last_admin_under_a_lock_before_any_mutation(repo):
    actor = _admin(uuid.uuid4())
    target = uuid.uuid4()
    await privacy_service.execute_user_data_deletion(None, actor, target)
    assert [lock for (_o, _u, lock) in repo["admin_checks"]] == [False, True]
    assert repo["admin_checks"][1][:2] == (actor.organization_id, target)
    assert repo["calls"][0] == "admin_lock"  # first thing that happens


# --- Rule D: multi-organization users keep their global account ------------


@pytest.mark.asyncio
async def test_multi_org_user_keeps_global_account(repo):
    repo["other_orgs"] = [uuid.uuid4()]
    actor = _admin(uuid.uuid4())
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())

    assert "anonymize_user_record" not in repo["calls"]
    # This organization's access and data are still removed.
    for call in ("delete_user_roles", "delete_refresh_tokens", "delete_project_memberships"):
        assert call in repo["calls"]
    step = next(s for s in result.steps if s.category == "user_record")
    assert step.action == "retain"
    assert step.succeeded
    assert step.rows_affected == 0
    assert result.status == "completed"
    assert repo["audit_events"][0]["metadata"]["account_retained_other_organizations"] is True


@pytest.mark.asyncio
async def test_multi_org_plan_marks_user_record_retain(repo):
    repo["other_orgs"] = [uuid.uuid4()]
    actor = _admin(uuid.uuid4())
    plan = await privacy_service.plan_user_data_deletion(None, actor, uuid.uuid4())
    step = next(s for s in plan.steps if s.category == "user_record")
    assert step.action == "retain"
    assert plan.total_rows_affected == 58  # 59 minus the retained users row


@pytest.mark.asyncio
async def test_single_org_user_is_still_fully_anonymized(repo):
    actor = _admin(uuid.uuid4())
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert "anonymize_user_record" in repo["calls"]
    step = next(s for s in result.steps if s.category == "user_record")
    assert step.action == "anonymize"
    assert repo["audit_events"][0]["metadata"]["account_retained_other_organizations"] is False


# --- Partial-deletion recovery: membership is removed LAST -----------------


@pytest.mark.asyncio
async def test_earlier_failure_keeps_membership_and_account(repo, monkeypatch):
    """If any cleanup step fails, neither the membership nor the global
    account may be touched -- the admin can retry the identical request."""
    actor = _admin(uuid.uuid4())

    async def boom(*args, **kwargs):
        raise RuntimeError("simulated cleanup failure")

    monkeypatch.setattr(privacy_service.repository, "delete_project_memberships", boom)
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())

    assert result.status == "partially_completed"
    assert "delete_user_roles" not in repo["calls"]
    assert "anonymize_user_record" not in repo["calls"]
    blocked = {s.category: s for s in result.steps if not s.succeeded}
    assert set(blocked) == {"project_memberships", "user_record", "organization_roles"}
    assert "retried" in blocked["organization_roles"].error


@pytest.mark.asyncio
async def test_retry_after_a_blocked_run_completes_while_still_a_member(repo, monkeypatch):
    """The scenario that motivated the reordering: because membership was kept,
    the second request passes the current-membership check and finishes."""
    actor = _admin(uuid.uuid4())
    target = uuid.uuid4()
    state = {"fail": True}
    real = privacy_service.repository.delete_project_memberships

    async def flaky(*args, **kwargs):
        if state["fail"]:
            raise RuntimeError("transient")
        return await real(*args, **kwargs)

    monkeypatch.setattr(privacy_service.repository, "delete_project_memberships", flaky)
    first = await privacy_service.execute_user_data_deletion(None, actor, target)
    assert first.status == "partially_completed"
    assert repo["is_member"] is True  # membership untouched, so a retry is allowed

    state["fail"] = False
    second = await privacy_service.execute_user_data_deletion(None, actor, target)
    assert second.status == "completed"
    assert repo["calls"][-1] == "delete_user_roles"
    assert "anonymize_user_record" in repo["calls"]


@pytest.mark.asyncio
async def test_failed_account_step_also_keeps_membership(repo, monkeypatch):
    actor = _admin(uuid.uuid4())

    async def boom(*args, **kwargs):
        raise RuntimeError("anonymize failed")

    monkeypatch.setattr(privacy_service.repository, "anonymize_user_record", boom)
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert result.status == "partially_completed"
    assert "delete_user_roles" not in repo["calls"]
    assert next(s for s in result.steps if s.category == "organization_roles").succeeded is False


@pytest.mark.asyncio
async def test_failure_of_the_role_deletion_itself_is_reported_and_retryable(repo, monkeypatch):
    actor = _admin(uuid.uuid4())

    async def boom(*args, **kwargs):
        raise RuntimeError("role delete failed")

    monkeypatch.setattr(privacy_service.repository, "delete_user_roles", boom)
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert result.status == "partially_completed"
    assert repo["is_member"] is True  # nothing changed membership, retry stays possible


# --- Locks, and decisions made under them ------------------------------------


@pytest.mark.asyncio
async def test_both_locks_are_taken_before_any_mutation_in_order(repo):
    actor = _admin(uuid.uuid4())
    await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert repo["calls"][:2] == ["admin_lock", "user_lock"]


@pytest.mark.asyncio
async def test_membership_is_rechecked_after_the_locks(repo, monkeypatch):
    """The plan's membership check runs before the locks; if the person was
    removed in between, execution must refuse and mutate nothing."""
    actor = _admin(uuid.uuid4())
    answers = iter([True, False])  # plan: member; after locks: no longer a member

    async def flip(session, user_id, organization_id):
        return next(answers)

    monkeypatch.setattr(privacy_service.users_service, "is_organization_member", flip)
    with pytest.raises(NotFoundError):
        await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert "delete_refresh_tokens" not in repo["calls"]
    assert "anonymize_user_record" not in repo["calls"]
    assert repo["audit_events"] == []


@pytest.mark.asyncio
async def test_anonymization_decision_uses_state_read_after_the_locks(repo, monkeypatch):
    """Plan saw no other organization; by execution time another membership
    exists (e.g. granted elsewhere). The account must be retained."""
    actor = _admin(uuid.uuid4())
    answers = iter([[], [uuid.uuid4()]])  # plan: none; execution: one other org

    async def changing(session, user_id):
        return next(answers)

    monkeypatch.setattr(privacy_service.users_service, "list_organizations_for_login", changing)
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert "anonymize_user_record" not in repo["calls"]
    assert next(s for s in result.steps if s.category == "user_record").action == "retain"
    assert "delete_user_roles" in repo["calls"]


# --- Final two steps are atomic (SAVEPOINT) ------------------------------------


def _db_error() -> Exception:
    from sqlalchemy.exc import DBAPIError

    return DBAPIError("DELETE FROM user_roles ...", {}, Exception("simulated database error"))


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("simulated non-database failure"), _db_error()],
    ids=["python-error", "database-error"],
)
@pytest.mark.asyncio
async def test_final_role_deletion_failure_rolls_back_the_account_anonymization(
    repo, monkeypatch, failure
):
    """Single-org target; everything earlier succeeds; the LAST step fails.
    The anonymization (done a moment earlier) must be undone with it."""
    actor = _admin(uuid.uuid4())

    async def boom(*args, **kwargs):
        raise failure

    monkeypatch.setattr(privacy_service.repository, "delete_user_roles", boom)
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())

    assert result.status == "partially_completed"
    # Atomic: the anonymization was rolled back together with the failed deletion.
    assert "anonymize_user_record" not in repo["calls"]
    assert "delete_user_roles" not in repo["calls"]
    assert repo["savepoint_rollbacks"] == 1
    # Earlier cleanup stays applied.
    for kept in (
        "delete_refresh_tokens",
        "delete_project_memberships",
        "anonymize_agent_executions",
    ):
        assert kept in repo["calls"]
    steps = {s.category: s for s in result.steps}
    assert steps["organization_roles"].succeeded is False
    assert steps["user_record"].succeeded is False
    assert "Rolled back together with organization_roles" in steps["user_record"].error
    assert "retried" in steps["user_record"].error
    # The failure is audited, and the person is still a member (retryable).
    event = repo["audit_events"][0]
    assert event["metadata"]["status"] == "partially_completed"
    assert set(event["metadata"]["failed_steps"]) == {"user_record", "organization_roles"}
    assert event["metadata"]["account_retained_other_organizations"] is False
    assert repo["is_member"] is True


@pytest.mark.asyncio
async def test_retry_after_a_rolled_back_final_step_completes(repo, monkeypatch):
    actor = _admin(uuid.uuid4())
    target = uuid.uuid4()
    real = privacy_service.repository.delete_user_roles
    state = {"fail": True}

    async def flaky(*args, **kwargs):
        if state["fail"]:
            raise RuntimeError("transient")
        return await real(*args, **kwargs)

    monkeypatch.setattr(privacy_service.repository, "delete_user_roles", flaky)
    first = await privacy_service.execute_user_data_deletion(None, actor, target)
    assert first.status == "partially_completed"
    assert "anonymize_user_record" not in repo["calls"]

    state["fail"] = False
    second = await privacy_service.execute_user_data_deletion(None, actor, target)
    assert second.status == "completed"
    assert repo["calls"][-2:] == ["anonymize_user_record", "delete_user_roles"]


@pytest.mark.asyncio
async def test_a_failing_anonymization_means_the_role_deletion_is_never_attempted(
    repo, monkeypatch
):
    actor = _admin(uuid.uuid4())

    async def boom(*args, **kwargs):
        raise RuntimeError("anonymize failed")

    monkeypatch.setattr(privacy_service.repository, "anonymize_user_record", boom)
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert result.status == "partially_completed"
    assert "delete_user_roles" not in repo["calls"]
    steps = {s.category: s for s in result.steps}
    assert "anonymize failed" in steps["user_record"].error
    assert "Rolled back together with user_record" in steps["organization_roles"].error


@pytest.mark.asyncio
async def test_multi_org_final_role_deletion_failure_is_also_rolled_back(repo, monkeypatch):
    repo["other_orgs"] = [uuid.uuid4()]
    actor = _admin(uuid.uuid4())

    async def boom(*args, **kwargs):
        raise RuntimeError("role delete failed")

    monkeypatch.setattr(privacy_service.repository, "delete_user_roles", boom)
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert result.status == "partially_completed"
    assert "anonymize_user_record" not in repo["calls"]
    # No "retained" claim is made for a decision that was rolled back.
    assert repo["audit_events"][0]["metadata"]["account_retained_other_organizations"] is False
    assert all(s.action != "retain" for s in result.steps if s.category == "user_record")


@pytest.mark.asyncio
async def test_successful_run_uses_exactly_one_savepoint_and_never_rolls_it_back(repo):
    actor = _admin(uuid.uuid4())
    result = await privacy_service.execute_user_data_deletion(None, actor, uuid.uuid4())
    assert result.status == "completed"
    assert repo["savepoints"] == 1
    assert repo["savepoint_rollbacks"] == 0


@pytest.mark.asyncio
async def test_no_savepoint_when_an_earlier_step_already_failed(repo, monkeypatch):
    async def boom(*args, **kwargs):
        raise RuntimeError("earlier failure")

    monkeypatch.setattr(privacy_service.repository, "delete_project_memberships", boom)
    await privacy_service.execute_user_data_deletion(None, _admin(uuid.uuid4()), uuid.uuid4())
    assert repo["savepoints"] == 0  # the final pair is held back entirely

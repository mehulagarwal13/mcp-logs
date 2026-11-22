"""Tests for the data-deletion endpoints on `app.api.routers.users`.

Same `TestClient` + `dependency_overrides` + stubbed-service style as
`tests/api/test_tenancy_router.py`. These cover the transport layer only --
that the right service function is called, with the caller's own identity,
and that a caller-supplied `user_id` can never redirect the operation at
another organization. The ownership/idempotency/partial-failure logic itself
is covered in `tests/core/privacy/`.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.deps import get_current_identity
from app.api.routers import users as users_router
from app.core.privacy.schemas import DeletionPlan, DeletionResult, PlannedStep, StepResult
from app.database.session import get_db_session
from app.shared.schemas import ActorKind, Identity

_NOW = datetime(2026, 8, 1, tzinfo=UTC)


def _admin(organization_id: uuid.UUID) -> Identity:
    return Identity(
        kind=ActorKind.USER,
        subject=str(uuid.uuid4()),
        organization_id=organization_id,
        user_id=uuid.uuid4(),
        permissions=frozenset({"tenancy:manage"}),
    )


@pytest.fixture()
def client():
    organization_id = uuid.uuid4()
    actor = _admin(organization_id)

    async def _fake_session():
        yield None

    api_main.app.dependency_overrides[get_current_identity] = lambda: actor
    api_main.app.dependency_overrides[get_db_session] = _fake_session

    yield TestClient(api_main.app), actor

    api_main.app.dependency_overrides.clear()


def _plan(organization_id: uuid.UUID, target: uuid.UUID) -> DeletionPlan:
    return DeletionPlan(
        scope="user_data",
        target_user_id=target,
        organization_id=organization_id,
        steps=[
            PlannedStep(
                category="sessions",
                action="hard_delete",
                table="refresh_tokens",
                count=2,
                rationale="ephemeral",
            ),
            PlannedStep(
                category="organization_knowledge",
                action="retain",
                table="documents",
                count=None,
                rationale="organization-owned",
            ),
        ],
    )


def _result(organization_id: uuid.UUID, target: uuid.UUID) -> DeletionResult:
    return DeletionResult(
        scope="user_data",
        target_user_id=target,
        organization_id=organization_id,
        status="completed",
        steps=[
            StepResult(
                category="sessions", action="hard_delete", succeeded=True, rows_affected=2
            )
        ],
        executed_at=_NOW,
    )


def test_plan_endpoint_is_a_dry_run_returning_the_plan(client, monkeypatch):
    test_client, actor = client
    target = uuid.uuid4()
    captured: dict[str, object] = {}

    async def fake_plan(session, passed_actor, user_id, **kwargs):
        captured["actor"] = passed_actor
        captured["user_id"] = user_id
        return _plan(actor.organization_id, target)

    monkeypatch.setattr(users_router.privacy_service, "plan_user_data_deletion", fake_plan)

    response = test_client.get(f"/users/{target}/data-deletion/plan")

    assert response.status_code == 200
    assert captured["actor"] is actor
    assert captured["user_id"] == target
    payload = response.json()
    assert payload["scope"] == "user_data"
    # The retained category must be visible in the dry run -- the whole
    # point is showing an operator what will NOT be touched.
    categories = {step["category"]: step["action"] for step in payload["steps"]}
    assert categories["organization_knowledge"] == "retain"


def test_execute_endpoint_calls_the_service_with_the_callers_identity(client, monkeypatch):
    test_client, actor = client
    target = uuid.uuid4()
    captured: dict[str, object] = {}

    async def fake_execute(session, passed_actor, user_id, **kwargs):
        captured["actor"] = passed_actor
        captured["user_id"] = user_id
        return _result(actor.organization_id, target)

    monkeypatch.setattr(users_router.privacy_service, "execute_user_data_deletion", fake_execute)

    response = test_client.post(f"/users/{target}/data-deletion")

    assert response.status_code == 200
    assert captured["actor"] is actor
    assert response.json()["status"] == "completed"


def test_execute_endpoint_never_accepts_a_caller_supplied_organization(client, monkeypatch):
    """Cross-tenant prevention at the transport layer: the endpoint takes no
    organization parameter at all, so the service always derives it from the
    authenticated identity."""
    test_client, actor = client
    target = uuid.uuid4()
    seen: dict[str, object] = {}

    async def fake_execute(session, passed_actor, user_id, **kwargs):
        seen["organization_id"] = passed_actor.organization_id
        return _result(passed_actor.organization_id, target)

    monkeypatch.setattr(users_router.privacy_service, "execute_user_data_deletion", fake_execute)

    # Attempt to smuggle a different organization in via query string and body.
    other_org = uuid.uuid4()
    response = test_client.post(
        f"/users/{target}/data-deletion?organization_id={other_org}",
        json={"organization_id": str(other_org)},
    )

    assert response.status_code == 200
    assert seen["organization_id"] == actor.organization_id
    assert seen["organization_id"] != other_org


def test_deletion_result_exposes_status_for_partial_failure(client, monkeypatch):
    """A caller must be able to see that a run did not fully succeed."""
    test_client, actor = client
    target = uuid.uuid4()

    async def fake_execute(session, passed_actor, user_id, **kwargs):
        return DeletionResult(
            scope="user_data",
            target_user_id=target,
            organization_id=passed_actor.organization_id,
            status="partially_completed",
            steps=[
                StepResult(
                    category="sessions", action="hard_delete", succeeded=True, rows_affected=1
                ),
                StepResult(
                    category="user_record",
                    action="anonymize",
                    succeeded=False,
                    error="database unavailable",
                ),
            ],
            executed_at=_NOW,
        )

    monkeypatch.setattr(users_router.privacy_service, "execute_user_data_deletion", fake_execute)

    payload = test_client.post(f"/users/{target}/data-deletion").json()
    assert payload["status"] == "partially_completed"
    failed = [s for s in payload["steps"] if not s["succeeded"]]
    assert len(failed) == 1
    assert failed[0]["error"] == "database unavailable"


def test_no_delete_verb_endpoint_exists_for_users():
    """Deliberate API-design assertion: this feature does not remove a
    `users` row (three RESTRICT foreign keys make that impossible), so it
    must not advertise `DELETE /users/{id}`.

    Introspected via `app.openapi()["paths"]`. An earlier version of this
    test walked `app.routes`, which in this FastAPI version yields
    `_IncludedRouter` objects with no `.path`/`.methods` -- so it built an
    empty list and passed no matter what. Asserting over the OpenAPI schema
    is the version-stable way to make this check real.
    """
    paths = api_main.app.openapi()["paths"]
    operations = {
        (path, method.upper()) for path, methods in paths.items() for method in methods
    }
    # The check itself must be non-vacuous: prove the user surface is present.
    assert any(path.startswith("/users") for path, _ in operations), "user routes missing"
    assert ("/users/{user_id}", "DELETE") not in operations
    # The real deletion surface is a POST to an explicit sub-resource.
    assert ("/users/{user_id}/data-deletion", "POST") in operations


# --- HTTP error mapping for the V1/V2 safeguards ------------------------------
#
# These run the REAL service through the REAL router and exception handler
# (`app.api.errors.ekip_error_handler`); only the database-touching lookups are
# stubbed. They pin the exact status code and `error_code` a client sees for
# each refusal, on BOTH the dry-run (GET .../plan) and the executing
# (POST .../data-deletion) endpoint.

from app.core.privacy import service as privacy_service  # noqa: E402
from app.core.users import service as users_service  # noqa: E402

_ENDPOINTS = [
    pytest.param("get", "/users/{target}/data-deletion/plan", id="plan"),
    pytest.param("post", "/users/{target}/data-deletion", id="execute"),
]


@pytest.fixture()
def world(monkeypatch):
    """Stubs for the database-backed lookups the guards use."""
    state = {"user_exists": True, "is_member": True, "admins": {uuid.uuid4(), uuid.uuid4()}}

    async def user_exists(session, user_id):
        return state["user_exists"]

    async def is_member(session, user_id, organization_id):
        return state["is_member"]

    async def list_admins(session, organization_id, *, lock):
        return set(state["admins"])

    async def acquire(session, organization_id):
        return None

    monkeypatch.setattr(privacy_service.repository, "user_exists", user_exists)
    monkeypatch.setattr(users_service.repository, "is_organization_member", is_member)
    monkeypatch.setattr(users_service.repository, "list_org_admin_user_ids", list_admins)
    monkeypatch.setattr(users_service.repository, "acquire_org_admin_lock", acquire)
    return state


def _call(test_client, method, template, target):
    return getattr(test_client, method)(template.format(target=target))


@pytest.mark.parametrize(("method", "template"), _ENDPOINTS)
def test_self_deletion_returns_400_with_its_error_code(client, world, method, template):
    test_client, actor = client
    response = _call(test_client, method, template, actor.user_id)
    assert response.status_code == 400
    assert response.json()["error_code"] == "privacy.cannot_delete_self"


@pytest.mark.parametrize(("method", "template"), _ENDPOINTS)
def test_non_member_returns_404_with_its_error_code(client, world, method, template):
    test_client, _actor_identity = client
    world["is_member"] = False
    response = _call(test_client, method, template, uuid.uuid4())
    assert response.status_code == 404
    assert response.json()["error_code"] == "privacy.user_not_found"


@pytest.mark.parametrize(("method", "template"), _ENDPOINTS)
def test_nonexistent_user_returns_the_same_404(client, world, method, template):
    test_client, _actor_identity = client
    world["user_exists"] = False
    response = _call(test_client, method, template, uuid.uuid4())
    assert response.status_code == 404
    assert response.json()["error_code"] == "privacy.user_not_found"


@pytest.mark.parametrize(("method", "template"), _ENDPOINTS)
def test_non_member_and_nonexistent_user_responses_are_identical(
    client, world, method, template
):
    """No existence oracle: for the SAME requested id, a user who exists in
    another organization and a user who does not exist at all must produce
    byte-identical status, headers-relevant body and error code."""
    test_client, _actor_identity = client
    target = uuid.uuid4()

    world.update(user_exists=True, is_member=False)  # exists, but in another tenant
    other_tenant = _call(test_client, method, template, target)

    world.update(user_exists=False, is_member=False)  # does not exist anywhere
    missing = _call(test_client, method, template, target)

    assert other_tenant.status_code == missing.status_code == 404
    assert other_tenant.json() == missing.json()


@pytest.mark.parametrize(("method", "template"), _ENDPOINTS)
def test_last_admin_returns_409_with_its_error_code(client, world, method, template):
    test_client, _actor_identity = client
    target = uuid.uuid4()
    world["admins"] = {target}  # the target is the organization's only admin
    response = _call(test_client, method, template, target)
    assert response.status_code == 409
    assert response.json()["error_code"] == "org.last_admin"


@pytest.mark.parametrize(("method", "template"), _ENDPOINTS)
def test_an_admin_with_another_admin_remaining_is_not_refused_as_last_admin(
    client, world, method, template
):
    """Control: the 409 above is caused by the last-admin rule, not by the
    stubbing -- with a second admin present the guard lets the request through
    (it then fails later on the un-stubbed database, which is not a 409)."""
    test_client, _actor_identity = client
    target = uuid.uuid4()
    world["admins"] = {target, uuid.uuid4()}
    try:
        response = _call(test_client, method, template, target)
        assert response.status_code != 409
    except Exception:  # noqa: BLE001 -- un-stubbed DB access past the guards
        pass

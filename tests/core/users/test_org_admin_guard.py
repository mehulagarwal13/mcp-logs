"""Unit tests for the last-admin guard and the membership check added for the
member-management remediation (V1/V2).

Mocked: these cover the decision logic and the SQL *shape* (organization
scoping, the `tenancy:manage` filter, `FOR UPDATE` only when asked for). The
real concurrency behavior is proven against PostgreSQL in
`tests/core/users/test_org_admin_guard_postgres.py`.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.dialects import postgresql

from app.core.exceptions import ConflictError
from app.core.users import repository
from app.core.users import service as users_service


def _compile(stmt) -> str:
    # PostgreSQL dialect: the generic dialect drops `FOR UPDATE OF ...`.
    return str(
        stmt.compile(dialect=postgresql.dialect(), compile_kwargs={"literal_binds": True})
    )


class _CapturingSession:
    def __init__(self) -> None:
        self.statements: list[object] = []

    async def execute(self, statement, *args, **kwargs):
        self.statements.append(statement)
        return _Result()


class _Result:
    def scalars(self):
        return self

    def all(self):
        return []

    def scalar_one(self):
        return 0


# --- decision logic ---------------------------------------------------------


@pytest.fixture()
def admins(monkeypatch):
    state: dict = {"admins": set(), "locked": 0, "lock_flags": []}

    async def fake_list(session, organization_id, *, lock):
        state["lock_flags"].append(lock)
        return set(state["admins"])

    async def fake_acquire(session, organization_id):
        state["locked"] += 1

    monkeypatch.setattr(users_service.repository, "list_org_admin_user_ids", fake_list)
    monkeypatch.setattr(users_service.repository, "acquire_org_admin_lock", fake_acquire)
    return state


@pytest.mark.asyncio
async def test_sole_admin_cannot_be_removed(admins):
    target = uuid.uuid4()
    admins["admins"] = {target}
    with pytest.raises(ConflictError) as excinfo:
        await users_service.assert_organization_keeps_an_admin(
            None, uuid.uuid4(), target, lock=True
        )
    assert excinfo.value.error_code == "org.last_admin"


@pytest.mark.asyncio
async def test_an_admin_can_be_removed_when_another_remains(admins):
    target = uuid.uuid4()
    admins["admins"] = {target, uuid.uuid4()}
    await users_service.assert_organization_keeps_an_admin(None, uuid.uuid4(), target, lock=True)


@pytest.mark.asyncio
async def test_a_non_admin_can_always_be_removed(admins):
    admins["admins"] = {uuid.uuid4()}
    await users_service.assert_organization_keeps_an_admin(
        None, uuid.uuid4(), uuid.uuid4(), lock=True
    )


@pytest.mark.asyncio
async def test_non_admin_removal_is_fine_even_with_no_admins_at_all(admins):
    await users_service.assert_organization_keeps_an_admin(
        None, uuid.uuid4(), uuid.uuid4(), lock=False
    )


@pytest.mark.asyncio
async def test_lock_true_takes_the_advisory_lock_and_row_locks(admins):
    await users_service.assert_organization_keeps_an_admin(
        None, uuid.uuid4(), uuid.uuid4(), lock=True
    )
    assert admins["locked"] == 1
    assert admins["lock_flags"] == [True]


@pytest.mark.asyncio
async def test_lock_false_takes_no_locks(admins):
    await users_service.assert_organization_keeps_an_admin(
        None, uuid.uuid4(), uuid.uuid4(), lock=False
    )
    assert admins["locked"] == 0
    assert admins["lock_flags"] == [False]


# --- SQL shape ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_query_is_scoped_to_org_permission_and_active_users():
    session = _CapturingSession()
    organization_id = uuid.uuid4()
    await repository.list_org_admin_user_ids(session, organization_id, lock=False)

    sql = _compile(session.statements[0])
    assert organization_id.hex in sql.replace("-", "")
    assert "tenancy:manage" in sql
    assert "is_active" in sql
    assert "email_verified_at IS NOT NULL" in sql
    assert "FOR UPDATE" not in sql
    assert "DISTINCT" not in sql  # Postgres rejects FOR UPDATE + DISTINCT


@pytest.mark.asyncio
async def test_admin_query_locks_only_user_roles_rows_when_asked():
    session = _CapturingSession()
    await repository.list_org_admin_user_ids(session, uuid.uuid4(), lock=True)
    sql = _compile(session.statements[0])
    assert "FOR UPDATE OF user_roles" in sql
    assert "ORDER BY user_roles.user_id" in sql  # consistent lock order


@pytest.mark.asyncio
async def test_membership_query_filters_by_user_and_organization():
    session = _CapturingSession()
    user_id, organization_id = uuid.uuid4(), uuid.uuid4()
    await repository.is_organization_member(session, user_id, organization_id)
    sql = _compile(session.statements[0]).replace("-", "")
    assert user_id.hex in sql
    assert organization_id.hex in sql
    assert "user_roles" in sql


@pytest.mark.asyncio
async def test_advisory_lock_is_transaction_scoped_and_keyed_by_organization():
    session = _CapturingSession()
    await repository.acquire_org_admin_lock(session, uuid.uuid4())
    sql = str(session.statements[0])
    assert "pg_advisory_xact_lock" in sql  # released at commit/rollback


# --- assign_role vs account anonymization ------------------------------------


@pytest.fixture()
def role_repo(monkeypatch):
    state: dict = {"existing": None, "active": True, "calls": []}

    async def get_user_role(session, user_id, organization_id, role_id):
        return state["existing"]

    async def lock_user_row(session, user_id, *, exclusive):
        state["calls"].append(("lock", exclusive))
        return state["active"]

    async def insert_user_role(session, **kwargs):
        state["calls"].append(("insert", kwargs["organization_id"]))

    monkeypatch.setattr(users_service.repository, "get_user_role", get_user_role)
    monkeypatch.setattr(users_service.repository, "lock_user_row", lock_user_row)
    monkeypatch.setattr(users_service.repository, "insert_user_role", insert_user_role)
    return state


@pytest.mark.asyncio
async def test_assign_role_takes_a_shared_user_lock_before_inserting(role_repo):
    org = uuid.uuid4()
    await users_service.assign_role(
        None, user_id=uuid.uuid4(), organization_id=org, role_id=uuid.uuid4()
    )
    assert role_repo["calls"] == [("lock", False), ("insert", org)]  # shared, then insert


@pytest.mark.asyncio
async def test_assign_role_refuses_an_inactive_or_anonymized_account(role_repo):
    from app.core.exceptions import PermissionDeniedError

    role_repo["active"] = False
    with pytest.raises(PermissionDeniedError) as excinfo:
        await users_service.assign_role(
            None, user_id=uuid.uuid4(), organization_id=uuid.uuid4(), role_id=uuid.uuid4()
        )
    assert excinfo.value.error_code == "user.inactive"
    assert all(call[0] != "insert" for call in role_repo["calls"])


@pytest.mark.asyncio
async def test_assign_role_refuses_a_missing_user(role_repo):
    from app.core.exceptions import PermissionDeniedError

    role_repo["active"] = None
    with pytest.raises(PermissionDeniedError):
        await users_service.assign_role(
            None, user_id=uuid.uuid4(), organization_id=uuid.uuid4(), role_id=uuid.uuid4()
        )


@pytest.mark.asyncio
async def test_assign_role_stays_idempotent_without_locking_when_already_assigned(role_repo):
    role_repo["existing"] = object()
    await users_service.assign_role(
        None, user_id=uuid.uuid4(), organization_id=uuid.uuid4(), role_id=uuid.uuid4()
    )
    assert role_repo["calls"] == []


@pytest.mark.asyncio
async def test_exclusive_vs_shared_user_row_lock_sql():
    session = _CapturingSession()

    class _R(_Result):
        def scalar_one_or_none(self):
            return True

    async def execute(statement, *a, **k):
        session.statements.append(statement)
        return _R()

    session.execute = execute  # type: ignore[method-assign]
    user_id = uuid.uuid4()
    await repository.lock_user_row(session, user_id, exclusive=True)
    await repository.lock_user_row(session, user_id, exclusive=False)
    exclusive_sql, shared_sql = (_compile(s) for s in session.statements)
    assert exclusive_sql.rstrip().endswith("FOR UPDATE")
    assert shared_sql.rstrip().endswith("FOR SHARE")
    assert user_id.hex in exclusive_sql.replace("-", "")

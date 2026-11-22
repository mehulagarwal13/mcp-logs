"""HTTP-level tests for the email-verification endpoints.

Service internals are stubbed (their behaviour is proven in
`tests/core/auth/test_email_verification*.py`, including against real Postgres);
what these pin is the API contract: status codes, the generic non-enumerating
bodies, that no token of any kind ever appears in a response, that mail is sent
only AFTER the response, and that the endpoints fail closed when email delivery
is not configured.
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.routers import auth as auth_router
from app.core.auth import email_verification
from app.core.exceptions import ConflictError, PermissionDeniedError, ValidationError
from app.database.session import get_db_session

_SIGNUP = {
    "email": "founder@example.com",
    "password": "a-strong-password",
    "display_name": "Founder",
    "organization_name": "Acme",
    "organization_slug": "acme",
}


@pytest.fixture()
def api(monkeypatch):
    async def _fake_session():
        yield None

    api_main.app.dependency_overrides[get_db_session] = _fake_session
    monkeypatch.setattr("app.api.deps.email_delivery_configured", lambda: True)
    delivered: list[email_verification.PendingDelivery] = []

    async def fake_deliver(delivery, **kwargs):
        delivered.append(delivery)

    monkeypatch.setattr(auth_router.email_verification, "deliver_verification_email", fake_deliver)
    yield TestClient(api_main.app), delivered
    api_main.app.dependency_overrides.clear()


def _delivery(token="RAW-VERIFICATION-TOKEN"):
    return email_verification.PendingDelivery(uuid.uuid4(), "registration", "founder@example.com", token)


# ------------------------------------------------------------------ signup


def test_signup_returns_202_generic_body_with_no_tokens_and_sends_mail_after(api, monkeypatch) -> None:
    client, delivered = api

    async def fake_signup(session, data):
        return _delivery()

    monkeypatch.setattr(auth_router.auth_service, "signup", fake_signup)

    response = client.post("/auth/signup", json=_SIGNUP)

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "verification_sent"
    assert not {"access_token", "refresh_token", "selection_token"} & set(body)
    assert "RAW-VERIFICATION-TOKEN" not in response.text
    assert [d.raw_token for d in delivered] == ["RAW-VERIFICATION-TOKEN"]


def test_signup_for_an_existing_address_is_indistinguishable_from_a_new_one(api, monkeypatch) -> None:
    """No enumeration: an address that already has an account (service returns None)
    yields byte-identical status and body to the success path, and sends nothing."""
    client, delivered = api
    results = iter([_delivery(), None])

    async def fake_signup(session, data):
        return next(results)

    monkeypatch.setattr(auth_router.auth_service, "signup", fake_signup)

    fresh = client.post("/auth/signup", json=_SIGNUP)
    existing = client.post("/auth/signup", json=_SIGNUP)

    assert (fresh.status_code, fresh.json()) == (existing.status_code, existing.json()) == (202, fresh.json())
    assert len(delivered) == 1  # only the fresh one triggered a mail


def test_signup_fails_closed_with_503_for_everyone_when_email_is_not_configured(monkeypatch) -> None:
    async def _fake_session():
        yield None

    api_main.app.dependency_overrides[get_db_session] = _fake_session
    monkeypatch.setattr("app.api.deps.email_delivery_configured", lambda: False)
    called: list[str] = []

    async def fake_signup(session, data):
        called.append("signup")

    monkeypatch.setattr(auth_router.auth_service, "signup", fake_signup)
    try:
        response = TestClient(api_main.app).post("/auth/signup", json=_SIGNUP)
    finally:
        api_main.app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json()["error_code"] == "email.not_configured"
    assert called == []  # nothing recorded for a registration nobody could verify


def test_signup_slug_conflict_is_still_reported(api, monkeypatch) -> None:
    client, _ = api

    async def fake_signup(session, data):
        raise ConflictError("taken", error_code="organization.slug_taken")

    monkeypatch.setattr(auth_router.auth_service, "signup", fake_signup)

    assert client.post("/auth/signup", json=_SIGNUP).status_code == 409


# ------------------------------------------------------------------ verify-email


def test_verify_email_returns_verified_and_no_session(api, monkeypatch) -> None:
    client, _ = api
    seen: list[str] = []

    async def fake_verify(session, raw_token):
        seen.append(raw_token)
        return email_verification.VerificationResult(kind="registration")

    monkeypatch.setattr(auth_router.email_verification, "verify_token", fake_verify)

    response = client.post("/auth/verify-email", json={"token": "tok"})

    assert response.status_code == 200
    assert response.json() == {"status": "verified"}  # no access/refresh token: login stays separate
    assert seen == ["tok"]


@pytest.mark.parametrize(
    ("exc", "status", "code"),
    [
        (ValidationError("x", error_code="email_verification.invalid"), 400, "email_verification.invalid"),
        (ValidationError("x", error_code="email_verification.expired"), 400, "email_verification.expired"),
        (ConflictError("x", error_code="email_verification.already_used"), 409, "email_verification.already_used"),
    ],
)
def test_verify_email_maps_each_failure_to_a_distinct_code(api, monkeypatch, exc, status, code) -> None:
    client, _ = api

    async def fake_verify(session, raw_token):
        raise exc

    monkeypatch.setattr(auth_router.email_verification, "verify_token", fake_verify)

    response = client.post("/auth/verify-email", json={"token": "tok"})

    assert response.status_code == status and code in response.text


def test_verify_email_is_post_only_so_link_scanners_cannot_consume_a_token(api) -> None:
    client, _ = api
    assert client.get("/auth/verify-email?token=abc").status_code in (404, 405)


@pytest.mark.parametrize("body", [{}, {"token": ""}, {"token": "x" * 600}])
def test_verify_email_rejects_malformed_bodies(api, body) -> None:
    client, _ = api
    assert client.post("/auth/verify-email", json=body).status_code == 422


# ------------------------------------------------------------------ resend


def test_resend_is_a_generic_202_whether_or_not_anything_was_sent(api, monkeypatch) -> None:
    client, delivered = api
    results = iter([_delivery(), None, None])

    async def fake_resend(session, *, email):
        return next(results)

    monkeypatch.setattr(auth_router.email_verification, "resend_verification", fake_resend)

    responses = [client.post("/auth/email-verification/resend", json={"email": e}) for e in
                 ("known@example.com", "unknown@example.com", "limited@example.com")]

    assert {r.status_code for r in responses} == {202}
    assert len({r.text for r in responses}) == 1  # identical bodies
    assert len(delivered) == 1


def test_resend_503_when_email_not_configured(monkeypatch) -> None:
    async def _fake_session():
        yield None

    api_main.app.dependency_overrides[get_db_session] = _fake_session
    monkeypatch.setattr("app.api.deps.email_delivery_configured", lambda: False)
    try:
        response = TestClient(api_main.app).post(
            "/auth/email-verification/resend", json={"email": "a@example.com"}
        )
    finally:
        api_main.app.dependency_overrides.clear()
    assert response.status_code == 503


# ------------------------------------------------------------------ login contract


def test_login_for_an_unverified_account_is_403_with_a_stable_code_and_no_tokens(api, monkeypatch) -> None:
    client, _ = api

    async def fake_login(session, data):
        raise PermissionDeniedError("not verified", error_code="auth.email_not_verified")

    monkeypatch.setattr(auth_router.auth_service, "login_with_password", fake_login)

    response = client.post("/auth/login", json={"email": "a@example.com", "password": "pw"})

    assert response.status_code == 403
    assert "auth.email_not_verified" in response.text
    assert "access_token" not in response.text

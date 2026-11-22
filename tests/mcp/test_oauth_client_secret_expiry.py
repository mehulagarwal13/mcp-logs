"""F6: dynamically issued client secrets expire, and the expiry is enforced."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from app.core.mcp_oauth import service as mcp_oauth_service
from app.mcp.servers import server as server_module
from app.shared.config.settings import get_settings
from tests.mcp.oauth_harness import full_flow, refresh_with, register, registered_client

_DAY = 86400


def _window() -> int:
    return get_settings().mcp_oauth_client_secret_expiry_days * _DAY


def test_the_window_is_wired_into_registration_options():
    options = server_module.mcp_server.settings.auth.client_registration_options
    assert options.client_secret_expiry_seconds == _window()
    assert _window() == 180 * _DAY  # default


def test_a_new_registration_gets_a_finite_expiry(client, store):
    before = int(time.time())
    info = registered_client(client)
    assert info["client_secret"]
    assert before + _window() <= info["client_secret_expires_at"] <= int(time.time()) + _window()
    stored = store.clients[info["client_id"]]
    assert stored["client_secret_expires_at"] == info["client_secret_expires_at"]


def test_an_unexpired_secret_works(client, store):
    info, tokens = full_flow(client, store)
    assert refresh_with(client, info, tokens["refresh_token"]).status_code == 200


def test_an_expired_secret_is_refused_at_the_token_endpoint(client, store):
    info, tokens = full_flow(client, store)
    store.clients[info["client_id"]]["client_secret_expires_at"] = int(time.time()) - 1
    refused = refresh_with(client, info, tokens["refresh_token"])
    assert refused.status_code == 401
    body = refused.json()
    assert body["error"] == "invalid_client"
    assert "expired" in body["error_description"].lower()


def test_an_expired_client_cannot_exchange_an_authorization_code_either(client, store):
    from urllib.parse import parse_qs, urlparse

    from tests.mcp.oauth_harness import (
        confirm,
        exchange_code,
        pkce_pair,
        rest_access_token,
        start_authorize,
    )

    info = registered_client(client)
    verifier, challenge = pkce_pair()
    resp = start_authorize(client, info, challenge=challenge)
    done = confirm(client, resp, rest_access_token(store))
    code = parse_qs(urlparse(done.headers["location"]).query)["code"][0]
    store.clients[info["client_id"]]["client_secret_expires_at"] = int(time.time()) - 1
    assert exchange_code(client, info, code, verifier).status_code == 401


def test_registration_still_succeeds_with_the_default_scope(client):
    assert register(client).status_code == 201


# --- legacy rows (stored with expiry 0 = "never") are held to the window ------


def _row(*, issued_at, expires_at, secret="enc"):
    return SimpleNamespace(
        client_id="legacy",
        client_secret_encrypted=secret,
        client_secret_expires_at=expires_at,
        client_id_issued_at=issued_at,
        client_metadata={"redirect_uris": ["http://localhost:1/cb"]},
    )


@pytest.fixture
def registry(monkeypatch):
    rows: dict[str, SimpleNamespace] = {}

    async def get_oauth_client(session, client_id):
        return rows.get(client_id)

    async def decrypt(kms, value):
        return "plaintext-secret"

    monkeypatch.setattr(mcp_oauth_service.repository, "get_oauth_client", get_oauth_client)
    monkeypatch.setattr(mcp_oauth_service, "decrypt_secret", decrypt)
    monkeypatch.setattr(mcp_oauth_service, "get_kms", lambda: object())
    return rows


async def test_legacy_zero_expiry_is_computed_from_the_issue_time(registry):
    issued = int(time.time()) - 10 * _DAY
    registry["legacy"] = _row(issued_at=issued, expires_at=0)
    data = await mcp_oauth_service.get_registered_client(None, "legacy")
    assert data["client_secret_expires_at"] == issued + _window()
    assert data["client_secret_expires_at"] > time.time()


async def test_a_legacy_client_older_than_the_window_is_expired(registry):
    issued = int(time.time()) - _window() - _DAY
    registry["legacy"] = _row(issued_at=issued, expires_at=0)
    data = await mcp_oauth_service.get_registered_client(None, "legacy")
    assert data["client_secret_expires_at"] < time.time()


async def test_a_legacy_client_with_no_issue_time_is_treated_as_expired(registry):
    registry["legacy"] = _row(issued_at=None, expires_at=None)
    data = await mcp_oauth_service.get_registered_client(None, "legacy")
    assert 0 < data["client_secret_expires_at"] < time.time()


async def test_an_explicit_expiry_is_left_alone(registry):
    registry["legacy"] = _row(issued_at=1, expires_at=2_000_000_000)
    data = await mcp_oauth_service.get_registered_client(None, "legacy")
    assert data["client_secret_expires_at"] == 2_000_000_000


async def test_a_public_client_without_a_secret_gets_no_expiry(registry):
    registry["legacy"] = _row(issued_at=1, expires_at=None, secret=None)
    data = await mcp_oauth_service.get_registered_client(None, "legacy")
    assert data["client_secret"] is None
    assert data["client_secret_expires_at"] is None

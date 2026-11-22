"""F3: only safe, explicitly permitted redirect URIs can be registered or used."""

from __future__ import annotations

import time

import pytest

from app.mcp.oauth.redirect_policy import redirect_uri_violation
from app.shared.config.settings import get_settings
from tests.mcp.oauth_harness import pkce_pair, register, registered_client, start_authorize

_HOSTS = ["claude.ai", "claude.com"]

ACCEPTED = [
    "https://claude.ai/api/mcp/auth_callback",
    "https://claude.com/api/mcp/auth_callback",
    "https://CLAUDE.AI/api/mcp/auth_callback",
    "http://localhost:3334/oauth/callback",
    "http://127.0.0.1:8123/cb",
    "http://[::1]:9000/cb",
    "http://localhost/cb",
]

REJECTED = [
    "http://evil.example/cb",  # plain http, non-loopback
    "https://attacker.example/cb",  # https, host not allow-listed
    "https://claude.ai.evil.example/cb",  # allow-listed name as a prefix only
    "https://evil.example/claude.ai",  # allow-listed name only in the path
    "https://claude.ai@evil.example/cb",  # userinfo trick
    "https://user:pw@claude.ai/cb",  # embedded credentials
    "https://claude.ai/cb#frag",  # fragment
    "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "myapp://callback",  # custom scheme
    "file:///etc/passwd",
    "ftp://claude.ai/cb",
    "http://localhost.evil.example/cb",  # loopback name as a prefix only
    "http://127.0.0.1.evil.example/cb",
]


@pytest.mark.parametrize("uri", ACCEPTED)
def test_policy_accepts(uri):
    assert redirect_uri_violation(uri, _HOSTS) is None


@pytest.mark.parametrize("uri", REJECTED)
def test_policy_rejects(uri):
    assert redirect_uri_violation(uri, _HOSTS) is not None


@pytest.mark.parametrize("uri", ACCEPTED)
def test_registration_accepts_permitted_uris(client, uri):
    assert register(client, redirect_uri=uri).status_code == 201


@pytest.mark.parametrize("uri", REJECTED)
def test_registration_refuses_unsafe_uris_and_stores_nothing(client, store, uri):
    response = register(client, redirect_uri=uri)
    assert response.status_code == 400, response.text
    assert response.json()["error"] == "invalid_redirect_uri"
    assert store.clients == {}


def test_one_bad_uri_among_good_ones_refuses_the_whole_registration(client, store):
    response = client.post(
        "/register",
        json={
            "redirect_uris": ["http://localhost:3334/cb", "https://attacker.example/cb"],
            "token_endpoint_auth_method": "client_secret_post",
        },
    )
    assert response.status_code == 400
    assert response.json()["error"] == "invalid_redirect_uri"
    assert store.clients == {}


def test_allowed_hosts_are_configurable_and_comma_separated(client, monkeypatch):
    uri = "https://mcp-connector.example.org/cb"
    assert register(client, redirect_uri=uri).status_code == 400
    hosts = ["mcp-connector.example.org"]
    monkeypatch.setattr(get_settings(), "mcp_oauth_allowed_redirect_hosts", hosts)
    assert register(client, redirect_uri=uri).status_code == 201
    # ...and the defaults no longer apply once it is overridden.
    assert register(client, redirect_uri="https://claude.ai/cb").status_code == 400


def test_settings_parse_a_comma_separated_env_value(monkeypatch):
    monkeypatch.setenv("MCP_OAUTH_ALLOWED_REDIRECT_HOSTS", "A.example, b.example")
    get_settings.cache_clear()
    try:
        assert get_settings().mcp_oauth_allowed_redirect_hosts == ["a.example", "b.example"]
    finally:
        get_settings.cache_clear()


def test_a_legacy_client_with_a_now_forbidden_uri_cannot_authorize_or_get_tokens(client, store):
    """A client registered before the policy existed (stored in the DB) is
    treated as unknown: `/authorize` answers directly (no redirect to the
    attacker's URI) and `/token` fails client authentication."""
    store.clients["legacy"] = {
        "client_id": "legacy",
        "client_secret": "s3cret",
        "client_id_issued_at": int(time.time()),
        "client_secret_expires_at": 0,
        "redirect_uris": ["https://attacker.example/steal"],
        "token_endpoint_auth_method": "client_secret_post",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
        "scope": "ekip",
    }
    _, challenge = pkce_pair()
    info = {"client_id": "legacy", "redirect_uris": ["https://attacker.example/steal"]}
    response = start_authorize(client, info, challenge=challenge)
    assert response.status_code == 400
    assert "location" not in response.headers

    token = client.post(
        "/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": "x",
            "client_id": "legacy",
            "client_secret": "s3cret",
        },
    )
    assert token.status_code == 401


def test_a_compliant_client_still_authorizes(client):
    info = registered_client(client)
    _, challenge = pkce_pair()
    assert start_authorize(client, info, challenge=challenge).status_code == 302

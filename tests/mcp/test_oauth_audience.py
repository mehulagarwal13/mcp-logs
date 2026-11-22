"""F7: tokens are bound to the MCP resource (`aud`) and issuer (`iss`); a token
intended for one resource is not accepted by the other."""

from __future__ import annotations

from urllib.parse import parse_qs, urlparse

import pytest
from jose import jwt as jose_jwt

from app.core.auth import service as auth_service
from app.core.exceptions import PermissionDeniedError
from app.mcp.auth import resolve_mcp_identity
from app.mcp.oauth.constants import accepted_resource_indicators, mcp_resource_identifier
from app.shared.config.settings import get_settings
from tests.mcp.oauth_harness import (
    BASE_URL,
    confirm,
    decode_unverified,
    full_flow,
    mcp_access_token,
    mcp_initialize,
    pkce_pair,
    refresh_with,
    registered_client,
    rest_access_token,
    start_authorize,
)


def test_resource_identifier_matches_the_advertised_metadata(client):
    metadata = client.get("/.well-known/oauth-authorization-server").json()
    resource = client.get("/.well-known/oauth-protected-resource").json()
    identifier = mcp_resource_identifier()
    assert metadata["issuer"].rstrip("/") == identifier
    assert resource["resource"].rstrip("/") == identifier
    assert identifier == BASE_URL


def test_issued_tokens_carry_audience_issuer_and_scope(client, store):
    info, tokens = full_flow(client, store)
    refreshed = refresh_with(client, info, tokens["refresh_token"]).json()
    for token in (tokens["access_token"], refreshed["access_token"]):
        claims = decode_unverified(token)
        assert claims["aud"] == mcp_resource_identifier()
        assert claims["iss"] == mcp_resource_identifier()
        assert claims["scope"] == "ekip"


# --- an MCP token is not a REST token ------------------------------------------


def test_rest_verification_rejects_an_mcp_audience_token(client, store):
    _, tokens = full_flow(client, store)
    with pytest.raises(PermissionDeniedError):
        auth_service.verify_access_token(tokens["access_token"])


async def test_rest_api_identity_dependency_rejects_an_mcp_audience_token(store):
    """The REST dependency itself (not just the verifier) refuses it, before any
    database lookup."""
    from app.api import deps

    with pytest.raises(PermissionDeniedError):
        await deps.get_current_identity(None, f"Bearer {mcp_access_token(store)}")


async def test_the_sign_in_page_does_not_accept_an_mcp_token_as_proof_of_identity(client, store):
    """An MCP-audience token cannot be pasted to mint a fresh REST-grade session."""
    info = registered_client(client)
    _, challenge = pkce_pair()
    response = start_authorize(client, info, challenge=challenge)
    page = confirm(client, response, mcp_access_token(store))
    assert page.status_code == 200  # re-rendered with an error, no redirect/code
    assert "could not be verified" in page.text
    assert "location" not in page.headers


def test_rest_tokens_are_unchanged(store):
    token = rest_access_token(store)
    claims = decode_unverified(token)
    assert not ({"aud", "iss", "scope"} & set(claims))
    parsed = auth_service.verify_access_token(token)
    assert parsed.user_id == store.user_id and parsed.scopes == ()


# --- a REST token is not an MCP token --------------------------------------------


def test_mcp_transport_rejects_a_rest_token_even_with_the_scope(client, store):
    token = rest_access_token(store, scope="ekip")  # scope, but no aud/iss
    assert mcp_initialize(client, token).status_code == 401


@pytest.mark.parametrize(
    "tamper",
    [
        {"audience": "https://other-resource.example"},
        {"audience": None},
        {"issuer": "https://other-issuer.example"},
        {"issuer": None},
    ],
)
def test_mcp_transport_rejects_wrong_or_missing_audience_or_issuer(client, store, tamper):
    assert mcp_initialize(client, mcp_access_token(store, **tamper)).status_code == 401


def test_mcp_transport_accepts_a_correctly_bound_token(client, store):
    response = mcp_initialize(client, mcp_access_token(store))
    assert response.status_code not in (401, 403), response.text


def test_a_token_signed_with_another_key_is_rejected(client, store):
    claims = decode_unverified(mcp_access_token(store))
    forged = jose_jwt.encode(claims, "not-the-real-secret", algorithm=get_settings().jwt_algorithm)
    assert mcp_initialize(client, forged).status_code == 401


async def test_identity_resolution_enforces_audience_and_issuer(store):
    with pytest.raises(PermissionDeniedError):
        await resolve_mcp_identity(None, rest_access_token(store, scope="ekip"))
    with pytest.raises(PermissionDeniedError):
        await resolve_mcp_identity(None, mcp_access_token(store, audience="https://x.example"))
    with pytest.raises(PermissionDeniedError):
        await resolve_mcp_identity(None, mcp_access_token(store, issuer="https://x.example"))
    assert (await resolve_mcp_identity(None, mcp_access_token(store))).user_id == store.user_id


# --- RFC 8707 resource indicator at /authorize -------------------------------------


@pytest.mark.parametrize("suffix", ["", "/", "/mcp", "/mcp/"])
def test_the_servers_own_resource_is_accepted(client, suffix):
    info = registered_client(client)
    _, challenge = pkce_pair()
    response = start_authorize(client, info, challenge=challenge, resource=BASE_URL + suffix)
    assert response.status_code == 302
    assert "flow_id=" in response.headers["location"]


@pytest.mark.parametrize(
    "resource",
    ["https://other-resource.example/mcp", BASE_URL + "/other", BASE_URL + ".evil.example", "x"],
)
def test_a_foreign_resource_is_refused_with_invalid_target(client, store, resource):
    info = registered_client(client)
    _, challenge = pkce_pair()
    response = start_authorize(client, info, challenge=challenge, resource=resource)
    assert response.status_code == 302
    query = parse_qs(urlparse(response.headers["location"]).query)
    assert query["error"] == ["invalid_target"]
    assert "flow_id" not in response.headers["location"]


def test_no_resource_indicator_still_works_and_binds_to_this_server(client, store):
    _, tokens = full_flow(client, store)  # full_flow sends no `resource`
    assert decode_unverified(tokens["access_token"])["aud"] == mcp_resource_identifier()


def test_accepted_indicator_set_is_derived_from_the_public_base_url():
    assert accepted_resource_indicators() == {
        BASE_URL,
        BASE_URL + "/",
        BASE_URL + "/mcp",
        BASE_URL + "/mcp/",
    }

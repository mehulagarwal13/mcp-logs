"""F1: the `ekip` scope is requested, granted, issued, and enforced end to end."""

from __future__ import annotations

import pytest

from app.core.exceptions import PermissionDeniedError
from app.mcp.auth import resolve_mcp_identity
from tests.mcp.oauth_harness import (
    decode_unverified,
    full_flow,
    mcp_access_token,
    mcp_initialize,
    refresh_with,
)


def test_metadata_advertises_the_scope(client):
    metadata = client.get("/.well-known/oauth-authorization-server").json()
    assert metadata["scopes_supported"] == ["ekip"]
    resource = client.get("/.well-known/oauth-protected-resource").json()
    assert resource["scopes_supported"] == ["ekip"]


def test_registration_defaults_to_the_scope(client):
    response = client.post(
        "/register",
        json={
            "redirect_uris": ["http://localhost:3334/oauth/callback"],
            "token_endpoint_auth_method": "client_secret_post",
        },
    )
    assert response.status_code == 201
    assert response.json()["scope"] == "ekip"


@pytest.mark.parametrize("requested", [None, "ekip"])
def test_token_response_and_jwt_carry_the_scope(client, store, requested):
    """A client that sends no `scope` (the reported failure) and one that sends
    `ekip` both end up with the scope on the token response AND in the JWT."""
    _, tokens = full_flow(client, store, scope=requested)
    assert tokens["scope"] == "ekip"
    assert decode_unverified(tokens["access_token"])["scope"] == "ekip"


def test_refresh_keeps_the_scope(client, store):
    info, tokens = full_flow(client, store)
    refreshed = refresh_with(client, info, tokens["refresh_token"])
    assert refreshed.status_code == 200, refreshed.text
    assert refreshed.json()["scope"] == "ekip"
    assert decode_unverified(refreshed.json()["access_token"])["scope"] == "ekip"


def test_an_unknown_scope_is_refused_at_authorize(client, store):
    from tests.mcp.oauth_harness import pkce_pair, registered_client, start_authorize

    info = registered_client(client)
    _, challenge = pkce_pair()
    response = start_authorize(client, info, challenge=challenge, scope="ekip admin")
    assert response.status_code == 302
    assert "error=invalid_scope" in response.headers["location"]


def test_mcp_endpoint_requires_the_scope_at_the_transport(client, store):
    """A valid, correctly bound EKIP token WITHOUT the scope is refused by the
    bearer middleware with 403 insufficient_scope; one with the scope passes."""
    assert mcp_initialize(client, None).status_code == 401
    assert "resource_metadata" in mcp_initialize(client, None).headers["www-authenticate"]

    scopeless = mcp_initialize(client, mcp_access_token(store, scope=None))
    assert scopeless.status_code == 403
    assert scopeless.json()["error"] == "insufficient_scope"

    _, tokens = full_flow(client, store)
    allowed = mcp_initialize(client, tokens["access_token"])
    assert allowed.status_code not in (401, 403), allowed.text


async def test_identity_resolution_rechecks_the_scope(store):
    """Defence in depth: even if something bypassed the transport middleware,
    a tool call with a scope-less token is refused before any identity lookup."""
    with pytest.raises(PermissionDeniedError) as excinfo:
        await resolve_mcp_identity(None, mcp_access_token(store, scope=None))
    assert excinfo.value.error_code == "auth.insufficient_scope"

    identity = await resolve_mcp_identity(None, mcp_access_token(store))
    assert identity.user_id == store.user_id

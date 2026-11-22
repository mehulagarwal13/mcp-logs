"""F4: the consent page identifies the client and destination, is bound to the
browser that loaded it (CSRF), and escapes everything it renders."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import pytest
from starlette.testclient import TestClient

from app.core.auth import service as auth_service
from app.core.exceptions import PermissionDeniedError
from app.shared.config.settings import get_settings
from tests.mcp.oauth_harness import (
    BASE_URL,
    confirm,
    pkce_pair,
    register,
    registered_client,
    rest_access_token,
    start_authorize,
)

_XSS = '<script>alert("x")</script>'


def _begin(client, info, **kw):
    _, challenge = pkce_pair()
    response = start_authorize(client, info, challenge=challenge, **kw)
    assert response.status_code == 302, response.text
    location = response.headers["location"]
    return location, parse_qs(urlparse(location).query)["flow_id"][0]


def _csrf_field(page_text: str) -> str:
    return re.search(r'name="csrf_token" value="([^"]+)"', page_text).group(1)


def test_page_names_the_client_and_the_redirect_destination(client):
    info = registered_client(client, client_name="Acme Connector")
    location, _ = _begin(client, info)
    page = client.get(location)
    assert page.status_code == 200
    assert "Acme Connector" in page.text
    assert info["client_id"] in page.text
    assert "localhost:3334" in page.text  # the host the browser is sent to
    assert "http://localhost:3334/oauth/callback" in page.text
    assert "ekip" in page.text  # requested access


def test_page_is_not_cacheable_or_frameable_and_sets_an_httponly_cookie(client):
    info = registered_client(client)
    location, flow_id = _begin(client, info)
    page = client.get(location)
    assert page.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in page.headers["content-security-policy"]
    assert page.headers["x-frame-options"] == "DENY"
    cookie = page.headers["set-cookie"]
    assert f"ekip_oauth_csrf_{flow_id}=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert "Path=/ekip/oauth/authorize" in cookie


@pytest.mark.parametrize("https", [True, False])
def test_cookie_is_secure_when_served_over_https(client, monkeypatch, https):
    base = "https://mcp.example.com" if https else "http://localhost:8001"
    monkeypatch.setattr(get_settings(), "mcp_public_base_url", base)
    location, _ = _begin(client, registered_client(client))
    cookie = client.get(location).headers["set-cookie"]
    assert ("Secure" in cookie) is https


def test_post_without_the_cookie_is_refused_and_does_not_burn_the_flow(client, store):
    info = registered_client(client)
    location, flow_id = _begin(client, info)
    page = client.get(location)
    csrf = _csrf_field(page.text)
    token = rest_access_token(store)

    cookieless = TestClient(client.app, base_url=BASE_URL, follow_redirects=False)
    refused = cookieless.post(
        "/ekip/oauth/authorize",
        data={"flow_id": flow_id, "ekip_access_token": token, "csrf_token": csrf},
    )
    assert refused.status_code == 403
    assert "location" not in refused.headers

    # The legitimate browser (which holds the cookie) can still finish.
    ok = client.post(
        "/ekip/oauth/authorize",
        data={"flow_id": flow_id, "ekip_access_token": token, "csrf_token": csrf},
    )
    assert ok.status_code == 302
    assert "code=" in ok.headers["location"]


@pytest.mark.parametrize("form_token", ["", "wrong", None])
def test_post_with_a_missing_or_wrong_form_token_is_refused(client, store, form_token):
    info = registered_client(client)
    location, flow_id = _begin(client, info)
    client.get(location)  # browser now holds the cookie
    data = {"flow_id": flow_id, "ekip_access_token": rest_access_token(store)}
    if form_token is not None:
        data["csrf_token"] = form_token
    refused = client.post("/ekip/oauth/authorize", data=data)
    assert refused.status_code == 403
    assert "code=" not in refused.headers.get("location", "")


def test_a_cookie_from_another_flow_does_not_authorize_this_one(client, store):
    info = registered_client(client)
    loc_a, flow_a = _begin(client, info)
    loc_b, flow_b = _begin(client, info)
    page_a = client.get(loc_a)
    page_b = client.get(loc_b)
    # Present flow A's cookie/token pair against flow B.
    attacker = TestClient(client.app, base_url=BASE_URL, follow_redirects=False)
    attacker.cookies.set(f"ekip_oauth_csrf_{flow_b}", _csrf_field(page_a.text), path="/")
    refused = attacker.post(
        "/ekip/oauth/authorize",
        data={
            "flow_id": flow_b,
            "ekip_access_token": rest_access_token(store),
            "csrf_token": _csrf_field(page_a.text),
        },
    )
    assert refused.status_code == 403
    assert _csrf_field(page_a.text) != _csrf_field(page_b.text)


def test_a_flow_can_only_be_completed_once(client, store):
    info = registered_client(client)
    verifier, challenge = pkce_pair()
    resp = start_authorize(client, info, challenge=challenge)
    token = rest_access_token(store)
    first = confirm(client, resp, token)
    assert first.status_code == 302
    flow_id = parse_qs(urlparse(resp.headers["location"]).query)["flow_id"][0]
    again = client.post(
        "/ekip/oauth/authorize", data={"flow_id": flow_id, "ekip_access_token": token}
    )
    assert again.status_code == 400


def test_client_supplied_text_is_html_escaped(client):
    info = registered_client(client, client_name=_XSS)
    location, _ = _begin(client, info)
    page = client.get(location).text
    assert _XSS not in page
    assert "&lt;script&gt;" in page


def test_the_error_message_is_html_escaped(client, store, monkeypatch):
    def reject(_token):
        raise PermissionDeniedError(
            '<img src=x onerror="alert(1)">', error_code="auth.invalid_token"
        )

    monkeypatch.setattr(auth_service, "verify_access_token", reject)
    info = registered_client(client)
    location, flow_id = _begin(client, info)
    csrf = _csrf_field(client.get(location).text)
    page = client.post(
        "/ekip/oauth/authorize",
        data={"flow_id": flow_id, "ekip_access_token": "x", "csrf_token": csrf},
    )
    assert page.status_code == 200
    assert "<img" not in page.text
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in page.text
    # a typo is recoverable: the same flow is still pending and the form is intact
    assert 'name="csrf_token"' in page.text


def test_a_redirect_uri_with_markup_is_escaped_on_the_page(client):
    uri = 'http://localhost:3334/cb?x="><script>1</script>'
    info = registered_client(client, redirect_uri=uri)
    location, _ = _begin(client, info)
    page = client.get(location).text
    assert "<script>1</script>" not in page


def test_unknown_flow_gets_the_hardened_expired_page(client):
    page = client.get("/ekip/oauth/authorize?flow_id=nope")
    assert page.status_code == 400
    assert page.headers["cache-control"] == "no-store"
    assert register(client).status_code == 201  # sanity: server still healthy

"""F2: unauthenticated dynamic client registration is rate limited and capped."""

from __future__ import annotations

import pytest
from mcp.server.transport_security import TransportSecuritySettings
from starlette.testclient import TestClient

from app.mcp.servers import server as server_module
from app.shared.config.settings import get_settings
from scripts import run_mcp_server
from tests.mcp.oauth_harness import BASE_URL, register


@pytest.fixture
def limited_client(store, monkeypatch):
    """The real app exactly as `scripts/run_mcp_server.py` builds it."""
    monkeypatch.setattr(get_settings(), "mcp_oauth_registration_per_ip_per_hour", 3)
    app = run_mcp_server.build_app("0.0.0.0", TransportSecuritySettings())
    with TestClient(app, base_url=BASE_URL, follow_redirects=False) as c:
        yield c


def test_registration_is_throttled_per_ip_then_returns_429(limited_client):
    statuses = [register(limited_client).status_code for _ in range(5)]
    assert statuses == [201, 201, 201, 429, 429]
    blocked = register(limited_client)
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) >= 1
    assert blocked.json()["error"] == "temporarily_unavailable"


def test_a_throttled_caller_does_not_reach_the_database(limited_client, store):
    for _ in range(3):
        register(limited_client)
    before = len(store.clients)
    register(limited_client)
    assert len(store.clients) == before


def test_only_post_register_is_throttled(limited_client):
    for _ in range(3):
        register(limited_client)
    assert register(limited_client).status_code == 429
    assert limited_client.get("/.well-known/oauth-authorization-server").status_code == 200
    assert limited_client.get("/authorize").status_code != 429
    assert limited_client.post("/token", data={}).status_code != 429


def test_global_daily_cap_refuses_registration_when_reached(client, store, monkeypatch):
    monkeypatch.setattr(get_settings(), "mcp_oauth_registrations_per_day", 2)
    assert register(client).status_code == 201
    assert register(client).status_code == 201
    refused = register(client)
    assert refused.status_code == 400
    assert refused.json()["error"] == "invalid_client_metadata"
    assert len(store.clients) == 2, "a refused registration must not persist a client"


def test_cap_clears_when_old_registrations_age_out(client, store, monkeypatch):
    monkeypatch.setattr(get_settings(), "mcp_oauth_registrations_per_day", 1)
    assert register(client).status_code == 201
    assert register(client).status_code == 400
    for client_id in store.client_created_at:
        store.client_created_at[client_id] -= 25 * 3600  # now outside the 24h window
    assert register(client).status_code == 201


def test_server_module_is_the_singleton_the_entrypoint_builds():
    assert run_mcp_server.server_module is server_module

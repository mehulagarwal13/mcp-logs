"""`top_k` bounds on `POST /search/similar-incidents` and
`POST /search/recent-changes` (review finding F5).

Same pattern as `tests/api/test_ask_router.py`: the real app via `TestClient`
with identity/session overridden and the `agents.service` call stubbed, so what
is asserted is exactly what reaches the service (and that a rejected request
never does).
"""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.deps import get_current_identity
from app.api.routers import ask as ask_router
from app.database.session import get_db_session
from app.retrieval.schemas import MAX_TOP_K
from app.shared.schemas import Identity

_ENDPOINTS = {
    "similar": (
        "/search/similar-incidents",
        {"description": "db pool exhausted"},
        "search_similar_incidents",
    ),
    "recent": ("/search/recent-changes", {"query": "pool size"}, "search_recent_changes"),
}


@pytest.fixture()
def client(monkeypatch):
    actor = Identity.for_agent("test_actor", uuid.uuid4())
    seen: list[int] = []

    async def _fake_session():
        yield None

    async def fake_service(session, text, passed_actor, *, top_k, **_kwargs):
        seen.append(top_k)
        return []

    for attr in ("search_similar_incidents", "search_recent_changes"):
        monkeypatch.setattr(ask_router.agents_service, attr, fake_service)
    api_main.app.dependency_overrides[get_current_identity] = lambda: actor
    api_main.app.dependency_overrides[get_db_session] = _fake_session
    yield TestClient(api_main.app), seen
    api_main.app.dependency_overrides.clear()


def _post(test_client, endpoint, **extra):
    path, body, _ = _ENDPOINTS[endpoint]
    return test_client.post(path, json={**body, **extra})


@pytest.mark.parametrize("endpoint", list(_ENDPOINTS))
def test_the_default_top_k_is_10_when_omitted(client, endpoint):
    test_client, seen = client
    assert _post(test_client, endpoint).status_code == 200
    assert seen == [10]


@pytest.mark.parametrize("endpoint", list(_ENDPOINTS))
@pytest.mark.parametrize("value", [1, 2, 10, 50, MAX_TOP_K])
def test_values_inside_the_range_are_forwarded_unchanged(client, endpoint, value):
    test_client, seen = client
    assert _post(test_client, endpoint, top_k=value).status_code == 200
    assert seen == [value]


@pytest.mark.parametrize("endpoint", list(_ENDPOINTS))
@pytest.mark.parametrize(
    "value",
    [0, -1, -(10**9), MAX_TOP_K + 1, 1000, 10**9, 2**63, "abc", 5.5, None, [1]],
    ids=lambda v: repr(v)[:20],
)
def test_out_of_range_or_malformed_values_are_rejected_with_422_and_never_reach_the_service(
    client, endpoint, value
):
    test_client, seen = client
    response = _post(test_client, endpoint, top_k=value)
    assert response.status_code == 422, (value, response.text[:200])
    assert seen == [], "a rejected request must not call the service"


@pytest.mark.parametrize("endpoint", list(_ENDPOINTS))
def test_the_validation_error_names_the_field(client, endpoint):
    test_client, _ = client
    detail = _post(test_client, endpoint, top_k=MAX_TOP_K + 1).json()
    assert "top_k" in str(detail)


def test_the_openapi_schema_advertises_the_bounds():
    schema = api_main.app.openapi()["components"]["schemas"]
    for model in ("SimilarIncidentsRequest", "RecentChangesRequest"):
        top_k = schema[model]["properties"]["top_k"]
        assert (top_k["minimum"], top_k["maximum"], top_k["default"]) == (1, MAX_TOP_K, 10)


@pytest.mark.parametrize("endpoint", list(_ENDPOINTS))
def test_json_true_is_coerced_to_1_by_pydantic_lax_mode_and_stays_in_range(client, endpoint):
    """Documented behaviour, not a bypass: `true` becomes `1`, which is a valid
    `top_k`. Every OUT-OF-RANGE integer is rejected (see above), so nothing that
    could widen the query gets through."""
    test_client, seen = client
    assert _post(test_client, endpoint, top_k=True).status_code == 200
    assert seen == [1]

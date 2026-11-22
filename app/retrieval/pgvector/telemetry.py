"""Observability for dense vector search: how often the exact-search fallback
fires, and why.

Owned by: retrieval/pgvector/.

`PgVectorStore.search`/`search_all` run an approximate (HNSW) query first and
re-run it exactly when it returns fewer than `top_k` rows. That fallback is
always *correct*, but it is silent by nature -- it can mean two very different
things, and the difference is the whole point of this module:

* `outcome="complete"`: the exact retry returned the SAME number of rows. Then
  the tenant simply has fewer than `top_k` permitted chunks (exact search
  returns `min(top_k, permitted)` rows, so equal counts below `top_k` mean the
  approximate pass already returned all of them). Expected for small tenants
  and narrow filters; high volume, logged at DEBUG.
* `outcome="recovered"`: the exact retry returned MORE rows. The approximate
  scan under-retrieved (iterative scan unavailable or capped, an old pgvector,
  a bad plan choice). That is a real signal and is logged at WARNING.

Two OpenTelemetry counters make the rate computable:
`ekip.retrieval.dense_searches` (every dense search) and
`ekip.retrieval.exact_fallbacks` (those that fell back, labelled by outcome).
Like the tracing in `app.shared.config.tracing`, they are no-ops until a
`MeterProvider` is configured -- no metrics backend exists in this project yet.

What is deliberately NOT recorded, in logs or metric attributes: the query
text, the query embedding, chunk content, titles or URLs. Metric attributes are
low-cardinality only (path, collection, outcome); the log additionally carries
the organization id, which identifies a tenant but no content, and whether a
project/repository filter was in play.
"""

from __future__ import annotations

from typing import Literal

from opentelemetry import metrics

from app.shared.config.logging import get_logger

logger = get_logger(__name__)

SearchPath = Literal["search", "search_all"]
FallbackOutcome = Literal["recovered", "complete"]

_meter = metrics.get_meter("ekip.retrieval")
dense_searches = _meter.create_counter(
    "ekip.retrieval.dense_searches",
    unit="1",
    description="Dense (vector) searches executed, before any exact-search fallback.",
)
exact_fallbacks = _meter.create_counter(
    "ekip.retrieval.exact_fallbacks",
    unit="1",
    description=(
        "Dense searches re-run as an exact search because the approximate pass "
        "returned fewer than top_k rows. outcome=recovered means the approximate "
        "pass under-retrieved; outcome=complete means the tenant has fewer than "
        "top_k permitted chunks."
    ),
)


def record_dense_search(*, path: SearchPath, collection: str) -> None:
    dense_searches.add(1, {"path": path, "collection": collection})


def record_exact_fallback(
    *,
    path: SearchPath,
    collection: str,
    top_k: int,
    approximate_rows: int,
    exact_rows: int,
    organization_id: str,
    has_project_filter: bool,
    has_repository_filter: bool,
) -> FallbackOutcome:
    """Count and log one exact-search fallback; returns its outcome."""
    outcome: FallbackOutcome = "recovered" if exact_rows > approximate_rows else "complete"
    exact_fallbacks.add(1, {"path": path, "collection": collection, "outcome": outcome})
    log = logger.warning if outcome == "recovered" else logger.debug
    log(
        "dense_search_exact_fallback",
        path=path,
        collection=collection,
        top_k=top_k,
        approximate_rows=approximate_rows,
        exact_rows=exact_rows,
        outcome=outcome,
        organization_id=organization_id,
        has_project_filter=has_project_filter,
        has_repository_filter=has_repository_filter,
    )
    return outcome

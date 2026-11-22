"""pgvector-backed `VectorStore` implementation (PROJECT_PLAN.md section 5,
Milestone 5's "pgvector backend" bullet -- the only backend this milestone
builds; see `app.retrieval.interfaces.base`'s module docstring for why
Qdrant stays unbuilt for now).

Owned by: retrieval/pgvector/. Backs all four collections
(`documentation_chunks`, `code_chunks`, `conversations_chunks`,
`incidents_chunks`) today -- `_COLLECTION_MODELS` (searched by the
all-collections default) and `_SINGLE_COLLECTION_MODELS` (searched by an
explicit-`collection` call; a superset of `_COLLECTION_MODELS` -- see that
dict's own comment for why `"incidents"` is single-collection-only) are the
only places that would need to change if a future collection moved to the
Qdrant backend instead (PROJECT_PLAN.md section 8.4: a per-collection
configuration choice, not a code fork).

Joins back to `documents` (ingestion-owned) for `title`/`source_url` at
query time -- the same direct-database-read exception already established
for ingestion reading `connector_configs` (`app.ingestion.repository`'s
module docstring) and anticipated by section 9.9's own dependency list:
"database (metadata joins), shared".
"""

from __future__ import annotations

import uuid

from sqlalchemy import delete as sql_delete
from sqlalchemy import func, literal, literal_column, or_, select, text, union_all
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models.ingestion_models import Document, DocumentMetadata
from app.database.models.retrieval_models import (
    CodeChunk,
    ConversationChunk,
    DocumentationChunk,
    IncidentChunk,
)
from app.retrieval.pgvector import telemetry
from app.retrieval.schemas import (
    CollectionName,
    ScoredChunk,
    SearchFilters,
    UpsertChunk,
    validate_top_k,
)

# Backs `search_all`/`lexical_search_all` (the "search every collection"
# fast path `retrieval.service.search(collection=None)` uses) -- these three
# only, deliberately. "incidents" is NOT here: every existing caller of the
# all-collections default (the Answer Agent's normal `answer_question` path,
# the Investigation Agent's evidence gathering) must keep searching exactly
# what it already searches, per this fix's own scope -- see
# `retrieval.schemas.CollectionName`'s comment.
_COLLECTION_MODELS: dict[CollectionName, type] = {
    "documentation": DocumentationChunk,
    "code": CodeChunk,
    "conversations": ConversationChunk,
}

# Backs the *single*-collection methods below (`search`, `lexical_search`,
# `upsert`, `delete`) instead of `_COLLECTION_MODELS` directly -- every
# collection `_COLLECTION_MODELS` has, plus `"incidents"`, reachable only
# when a caller passes an explicit `collection` (never via the all-
# collections default above). `IncidentChunk` needs no special-cased query
# logic here: it is a plain `_ChunkColumns` table with a real `document_id`
# FK into `documents`, populated by the normal ingestion pipeline exactly
# like every other collection (`app.ingestion.connectors.incidents.
# IncidentsConnector`), so the existing `.join(Document, ...)` logic below
# already works for it unchanged.
_SINGLE_COLLECTION_MODELS: dict[CollectionName, type] = {
    **_COLLECTION_MODELS,
    "incidents": IncidentChunk,
}


_REPO_SCOPED_COLLECTIONS: frozenset[CollectionName] = frozenset({"code", "documentation"})

# --- HNSW search settings ----------------------------------------------------
#
# The chunk tables carry HNSW indexes (migration `b7c1d9e4f2a3`). Every EKIP
# vector query is tenant-filtered, and an HNSW scan applies such filters AFTER
# it has fetched its candidate list, so without these settings a filtered query
# can return fewer rows than exist (verified by experiment). All three are
# `set_config(..., is_local => true)`: transaction-scoped, so they die with the
# caller's transaction exactly as `set_tenant_context`'s GUC does, and cannot
# leak across pooled connections. They are issued on the CALLER's session, i.e.
# in the same transaction as the tenant context and the vector query.
#
# - `plan_cache_mode = force_custom_plan`: asyncpg prepares every statement,
#   and after a few executions PostgreSQL may switch to a *generic* plan that
#   cannot see which tenant is being served; it then picks the HNSW scan even
#   for tiny tenants, which is both slow and incomplete (measured: 700-1000 ms
#   and short results). A custom plan per execution lets the planner use the
#   tenant-specific statistics and choose the btree path for small tenants.
# - `hnsw.iterative_scan = strict_order`: keep scanning until `LIMIT` rows
#   that pass the filters are found, and return them in exact distance order
#   (`relaxed_order` can return rows out of order, which would break the
#   "dense[0] is the best hit" assumption in `retrieval.service`).
#   pgvector >= 0.8.0 only: on older versions this setting either does nothing
#   or raises "reserved prefix" depending on whether the connection has already
#   loaded the library, so it is applied conditionally *inside the SQL* (CASE
#   short-circuits), never by try/except on a failed statement.
# - `hnsw.ef_search = 100`: the candidate-list size; 100 measurably improved
#   recall over the default 40 at similar latency.
_VECTOR_SEARCH_EF_SEARCH = 100
# True when the installed pgvector is >= 0.8.0 (the release that introduced
# iterative index scans). Compared as integer arrays, so 0.10.x > 0.8.x.
_PGVECTOR_AT_LEAST_0_8 = (
    "string_to_array(regexp_replace(extversion, '[^0-9.].*$', ''), '.')::int[] >= ARRAY[0, 8, 0]"
)
_VECTOR_SEARCH_SETTINGS_SQL = text(
    "SELECT set_config('plan_cache_mode', 'force_custom_plan', true), "
    "set_config('hnsw.ef_search', :ef_search, true), "
    f"CASE WHEN (SELECT {_PGVECTOR_AT_LEAST_0_8} FROM pg_extension WHERE extname = 'vector') "
    "THEN set_config('hnsw.iterative_scan', 'strict_order', true) END"
)


async def _apply_vector_search_settings(session: AsyncSession) -> None:
    await session.execute(
        _VECTOR_SEARCH_SETTINGS_SQL, {"ef_search": str(_VECTOR_SEARCH_EF_SEARCH)}
    )


def _exact_order(distance):
    """`distance + 0` is a different expression from the one the HNSW index
    orders by, so the planner cannot use the index for it: used only for the
    exact fallback below, which must be an exhaustive search."""
    return distance + literal_column("0")


def _repository_filter_clause(collection: CollectionName, model: type, repository: str):
    """`SearchFilters.repository`'s WHERE clause -- only `CodeChunk`/
    `DocumentationChunk` carry a `repo_full_name` column (real GitHub
    connector output lands in both -- source files in `"code"`, but
    issues/PRs/commit messages/READMEs, which is most of what a typical
    repo actually produces, in `"documentation"`). `ConversationChunk` has
    no such column, so filtering by repository there is a caller error, not
    a silent no-op.
    """
    if collection not in _REPO_SCOPED_COLLECTIONS:
        raise ValueError(
            f"SearchFilters.repository is only supported for the 'code' and "
            f"'documentation' collections, got collection={collection!r}."
        )
    return model.repo_full_name == repository


class PgVectorStore:
    """The pgvector `VectorStore` implementation. Stateless (holds no
    per-instance connection or config) -- every method takes the session
    and collection it needs, so one shared instance serves every collection
    and every caller's session.
    """

    async def search(
        self,
        session: AsyncSession,
        collection: CollectionName,
        query_embedding: list[float],
        filters: SearchFilters,
        top_k: int,
        *,
        include_metadata: bool = False,
    ) -> list[ScoredChunk]:
        """See `VectorStore.search`'s docstring for the contract.

        Uses pgvector's max-inner-product operator (`<#>`), not cosine
        distance (`<=>`): `retrieval.embedding` L2-normalizes every vector
        at generation time, so inner product and cosine similarity rank
        identically for normalized vectors, and inner product is the
        cheaper of the two to compute (see `embedding.py`'s docstring).
        pgvector's `max_inner_product` returns the *negated* inner product
        (so that, like its other distance operators, smaller means "more
        similar" -- consistent `ORDER BY ... ASC` semantics across every
        pgvector operator) -- negated back to a normal similarity score
        before returning.

        Tenant/project/ACL filters are applied as `WHERE` clauses on this
        query, never as a post-filter on already-fetched rows (PROJECT_PLAN.md
        sections 5.4-5.5).

        `include_metadata` is opt-in (default `False`) -- see
        `ScoredChunk.metadata`'s docstring for why this isn't joined
        unconditionally.
        """
        validate_top_k(top_k)
        model = _SINGLE_COLLECTION_MODELS[collection]
        distance = model.embedding.max_inner_product(query_embedding)
        await _apply_vector_search_settings(session)

        stmt = (
            select(
                model.id,
                model.document_id,
                model.content,
                model.source_offset_start,
                model.source_offset_end,
                distance.label("distance"),
                Document.title,
                Document.source_url,
            )
            .join(Document, Document.id == model.document_id)
            .where(model.organization_id == filters.organization_id)
            # A soft-deleted document's chunks must never be retrievable.
            # `documents.deleted_at` is set by `core.knowledge.service.
            # reject_document`; before this predicate existed, the rejected
            # document's chunk rows stayed in `<collection>_chunks` (nothing
            # purged them) and this join happily returned them with their
            # title/source_url attached -- so the Answer Agent could quote
            # and cite content a human had explicitly rejected. Enforced
            # here, in the store, rather than only in the caller: this is
            # the single chokepoint every retrieval path goes through, and
            # a filter the caller could forget is not a guarantee.
            .where(Document.deleted_at.is_(None))
            # Only the current version of a document is searchable: a
            # superseded version's chunks are removed at supersession, and
            # this predicate keeps any not-yet-purged ones (older rows,
            # an in-progress backfill) out of results. Same reasoning, and
            # same need to repeat it in every query below, as `deleted_at`.
            .where(Document.superseded_at.is_(None))
            .where(
                or_(
                    model.acl_permission_code.is_(None),
                    model.acl_permission_code.in_(filters.permission_codes),
                )
            )
        )
        if filters.project_ids is not None:
            stmt = stmt.where(model.project_id.in_(filters.project_ids))
        if filters.repository is not None:
            stmt = stmt.where(_repository_filter_clause(collection, model, filters.repository))

        rows = (await session.execute(stmt.order_by(distance.asc()).limit(top_k))).all()
        telemetry.record_dense_search(path="search", collection=collection)
        if len(rows) < top_k:
            # An approximate (HNSW) scan can come up short of what the filters
            # actually permit; fewer than `top_k` rows is also the normal result
            # for a small tenant. Either way an exact search is correct, and
            # cheap exactly when this triggers. Completeness is not negotiable.
            # Counted and logged (`telemetry`) so that "approximate pass
            # under-retrieved" is distinguishable from "tenant is just small".
            approximate_rows = len(rows)
            rows = (
                await session.execute(stmt.order_by(_exact_order(distance).asc()).limit(top_k))
            ).all()
            telemetry.record_exact_fallback(
                path="search",
                collection=collection,
                top_k=top_k,
                approximate_rows=approximate_rows,
                exact_rows=len(rows),
                organization_id=str(filters.organization_id),
                has_project_filter=filters.project_ids is not None,
                has_repository_filter=filters.repository is not None,
            )
        metadata_by_document = await self._load_metadata_by_document(
            session, [row.document_id for row in rows] if include_metadata else []
        )
        return [
            ScoredChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                collection=collection,
                content=row.content,
                score=-row.distance,
                source_offset_start=row.source_offset_start,
                source_offset_end=row.source_offset_end,
                title=row.title,
                source_url=row.source_url,
                metadata=metadata_by_document.get(row.document_id, {}),
            )
            for row in rows
        ]

    @staticmethod
    def _dense_all_statement(
        query_embedding: list[float], filters: SearchFilters, top_k: int, *, exact: bool
    ):
        branches = []
        for collection, model in _COLLECTION_MODELS.items():
            distance = model.embedding.max_inner_product(query_embedding)
            stmt = (
                select(
                    model.id.label("id"),
                    model.document_id.label("document_id"),
                    literal(collection).label("collection"),
                    model.content.label("content"),
                    model.source_offset_start.label("source_offset_start"),
                    model.source_offset_end.label("source_offset_end"),
                    distance.label("distance"),
                    Document.title.label("title"),
                    Document.source_url.label("source_url"),
                )
                .join(Document, Document.id == model.document_id)
                .where(model.organization_id == filters.organization_id)
                .where(Document.deleted_at.is_(None))
                .where(Document.superseded_at.is_(None))
                .where(
                    or_(
                        model.acl_permission_code.is_(None),
                        model.acl_permission_code.in_(filters.permission_codes),
                    )
                )
            )
            if filters.project_ids is not None:
                stmt = stmt.where(model.project_id.in_(filters.project_ids))
            if filters.repository is not None:
                stmt = stmt.where(_repository_filter_clause(collection, model, filters.repository))
            order = _exact_order(distance) if exact else distance
            branches.append(stmt.order_by(order.asc()).limit(top_k))
        combined = union_all(*branches).subquery()
        return select(combined).order_by(combined.c.distance.asc()).limit(top_k)

    async def search_all(
        self,
        session: AsyncSession,
        query_embedding: list[float],
        filters: SearchFilters,
        top_k: int,
    ) -> list[ScoredChunk]:
        """Dense-search all collections in one database round-trip.

        Each `UNION ALL` branch carries its OWN `ORDER BY distance LIMIT
        top_k`, followed by the global `ORDER BY distance LIMIT top_k`. For an
        exact scan this returns exactly what an unlimited union would (every
        member of the global top-k is, by definition, in its own branch's
        top-k), but it lets each branch use its collection's HNSW index: with
        the limit only on the outer query PostgreSQL cannot push the ordering
        into the branches and falls back to scanning every row of the tenant
        (verified by `EXPLAIN`). Every filter is unchanged and repeated in each
        branch, never applied to already-fetched rows.

        If the (approximate) result is short of `top_k` rows, the identical
        query is re-run in a form the planner cannot serve from an HNSW index,
        i.e. exact -- see `_exact_order`.
        """
        validate_top_k(top_k)
        # Built (and validated -- an unsupported repository filter raises here)
        # before any statement is sent.
        approximate = self._dense_all_statement(query_embedding, filters, top_k, exact=False)
        await _apply_vector_search_settings(session)
        rows = (await session.execute(approximate)).all()
        telemetry.record_dense_search(path="search_all", collection="all")
        if len(rows) < top_k:
            approximate_rows = len(rows)
            exact = self._dense_all_statement(query_embedding, filters, top_k, exact=True)
            rows = (await session.execute(exact)).all()
            telemetry.record_exact_fallback(
                path="search_all",
                collection="all",
                top_k=top_k,
                approximate_rows=approximate_rows,
                exact_rows=len(rows),
                organization_id=str(filters.organization_id),
                has_project_filter=filters.project_ids is not None,
                has_repository_filter=filters.repository is not None,
            )
        return [
            ScoredChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                collection=row.collection,
                content=row.content,
                score=-row.distance,
                source_offset_start=row.source_offset_start,
                source_offset_end=row.source_offset_end,
                title=row.title,
                source_url=row.source_url,
            )
            for row in rows
        ]

    async def lexical_search_all(
        self,
        session: AsyncSession,
        query_text: str,
        filters: SearchFilters,
        top_k: int,
    ) -> list[ScoredChunk]:
        """Lexically search all collections in one database round-trip."""
        validate_top_k(top_k)
        branches = []
        for collection, model in _COLLECTION_MODELS.items():
            tsquery = func.websearch_to_tsquery("english", query_text)
            rank = func.ts_rank_cd(model.content_tsv, tsquery)
            stmt = (
                select(
                    model.id.label("id"),
                    model.document_id.label("document_id"),
                    literal(collection).label("collection"),
                    model.content.label("content"),
                    model.source_offset_start.label("source_offset_start"),
                    model.source_offset_end.label("source_offset_end"),
                    rank.label("rank"),
                    Document.title.label("title"),
                    Document.source_url.label("source_url"),
                )
                .join(Document, Document.id == model.document_id)
                .where(model.content_tsv.op("@@")(tsquery))
                .where(model.organization_id == filters.organization_id)
                .where(Document.deleted_at.is_(None))
                .where(Document.superseded_at.is_(None))
                .where(
                    or_(
                        model.acl_permission_code.is_(None),
                        model.acl_permission_code.in_(filters.permission_codes),
                    )
                )
            )
            if filters.project_ids is not None:
                stmt = stmt.where(model.project_id.in_(filters.project_ids))
            if filters.repository is not None:
                stmt = stmt.where(_repository_filter_clause(collection, model, filters.repository))
            branches.append(stmt)
        combined = union_all(*branches).subquery()
        rows = (
            await session.execute(select(combined).order_by(combined.c.rank.desc()).limit(top_k))
        ).all()
        return [
            ScoredChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                collection=row.collection,
                content=row.content,
                score=row.rank,
                source_offset_start=row.source_offset_start,
                source_offset_end=row.source_offset_end,
                title=row.title,
                source_url=row.source_url,
            )
            for row in rows
        ]

    async def lexical_search(
        self,
        session: AsyncSession,
        collection: CollectionName,
        query_text: str,
        filters: SearchFilters,
        top_k: int,
        *,
        include_metadata: bool = False,
    ) -> list[ScoredChunk]:
        """See `VectorStore.lexical_search`'s docstring for the contract.

        Postgres full-text search against the generated `content_tsv`
        column (`app.database.models.retrieval_models`'s module docstring
        explains why this is `ts_rank_cd`-based ranking, not literally
        BM25). `plainto_tsquery` (not `websearch_to_tsquery` or a raw
        `to_tsquery`) treats `query_text` as a plain phrase -- no operator
        syntax (`&`, `|`, `:*`) a caller would need to know about or escape.

        `include_metadata` -- see `search()`'s docstring, same contract.
        """
        validate_top_k(top_k)
        model = _SINGLE_COLLECTION_MODELS[collection]
        tsquery = func.plainto_tsquery("english", query_text)
        rank = func.ts_rank_cd(model.content_tsv, tsquery).label("rank")

        stmt = (
            select(
                model.id,
                model.document_id,
                model.content,
                model.source_offset_start,
                model.source_offset_end,
                rank,
                Document.title,
                Document.source_url,
            )
            .join(Document, Document.id == model.document_id)
            .where(model.content_tsv.op("@@")(tsquery))
            .where(model.organization_id == filters.organization_id)
            # Same soft-delete exclusion as `search()` above -- see that
            # method's comment. Both halves of hybrid search need it
            # independently: fusing a filtered dense list with an
            # unfiltered lexical one would reintroduce the exact leak.
            .where(Document.deleted_at.is_(None))
            .where(Document.superseded_at.is_(None))
            .where(
                or_(
                    model.acl_permission_code.is_(None),
                    model.acl_permission_code.in_(filters.permission_codes),
                )
            )
        )
        if filters.project_ids is not None:
            stmt = stmt.where(model.project_id.in_(filters.project_ids))
        if filters.repository is not None:
            stmt = stmt.where(_repository_filter_clause(collection, model, filters.repository))

        stmt = stmt.order_by(rank.desc()).limit(top_k)

        result = await session.execute(stmt)
        rows = result.all()
        metadata_by_document = await self._load_metadata_by_document(
            session, [row.document_id for row in rows] if include_metadata else []
        )
        return [
            ScoredChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                collection=collection,
                content=row.content,
                score=row.rank,
                source_offset_start=row.source_offset_start,
                source_offset_end=row.source_offset_end,
                title=row.title,
                source_url=row.source_url,
                metadata=metadata_by_document.get(row.document_id, {}),
            )
            for row in rows
        ]

    async def _load_metadata_by_document(
        self, session: AsyncSession, document_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, dict[str, str]]:
        """Fetch every `document_metadata` row for `document_ids` in one
        query, folded into `{document_id: {key: value, ...}, ...}`.

        One extra query per `search()`/`lexical_search()` call (not one per
        chunk) -- called only when `include_metadata=True`; `document_ids`
        is passed in already-empty by both callers otherwise, so this
        short-circuits without a round-trip.
        """
        if not document_ids:
            return {}

        stmt = select(
            DocumentMetadata.document_id, DocumentMetadata.key, DocumentMetadata.value
        ).where(DocumentMetadata.document_id.in_(set(document_ids)))
        result = await session.execute(stmt)

        metadata_by_document: dict[uuid.UUID, dict[str, str]] = {}
        for document_id, key, value in result.all():
            metadata_by_document.setdefault(document_id, {})[key] = value
        return metadata_by_document

    async def upsert(
        self,
        session: AsyncSession,
        collection: CollectionName,
        chunks: list[UpsertChunk],
        embeddings: list[list[float]],
    ) -> None:
        """See `VectorStore.upsert`'s docstring for the contract.

        One `INSERT ... ON CONFLICT (document_id, chunk_index) DO UPDATE`
        per chunk, targeting each table's `uq_<table>_document_chunk_index`
        unique constraint (`app.database.models.retrieval_models`) -- a
        real upsert, not a delete-then-insert, so re-embedding an unchanged
        chunk doesn't churn its primary key or briefly leave it absent from
        the index mid-write.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks and embeddings must be the same length "
                f"(got {len(chunks)} chunks, {len(embeddings)} embeddings)."
            )

        model = _SINGLE_COLLECTION_MODELS[collection]
        values_batch: list[dict] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            values: dict = dict(
                organization_id=chunk.organization_id,
                project_id=chunk.project_id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                embedding=embedding,
                source_offset_start=chunk.source_offset_start,
                source_offset_end=chunk.source_offset_end,
                acl_permission_code=chunk.acl_permission_code,
            )
            if collection in _REPO_SCOPED_COLLECTIONS:
                values["repo_full_name"] = chunk.repo_full_name
            values_batch.append(values)

        update_columns = [
            "organization_id",
            "project_id",
            "content",
            "embedding",
            "source_offset_start",
            "source_offset_end",
            "acl_permission_code",
        ]
        if collection in _REPO_SCOPED_COLLECTIONS:
            update_columns.append("repo_full_name")

        # One multi-values statement per document/collection rather than one
        # database round-trip per chunk. Sentence-transformers already embeds
        # the same list as a batch; persistence now preserves that batching.
        stmt = pg_insert(model).values(values_batch)
        stmt = stmt.on_conflict_do_update(
            index_elements=["document_id", "chunk_index"],
            set_={column: getattr(stmt.excluded, column) for column in update_columns},
        )
        await session.execute(stmt)
        await session.flush()

    async def delete(
        self, session: AsyncSession, collection: CollectionName, document_id: uuid.UUID
    ) -> None:
        """See `VectorStore.delete`'s docstring for the contract."""
        model = _SINGLE_COLLECTION_MODELS[collection]
        await session.execute(sql_delete(model).where(model.document_id == document_id))
        await session.flush()

"""HNSW approximate-nearest-neighbour indexes on <collection>_chunks.embedding

Revision ID: b7c1d9e4f2a3
Revises: d5f2a8c1e9b4
Create Date: 2026-11-14

Dense retrieval (`PgVectorStore.search` / `search_all`) orders by
`embedding <#> :query` -- negative inner product; every embedding is
L2-normalized, so this ranks like cosine similarity. The matching operator
class is `vector_ip_ops`: an index built with `vector_cosine_ops` would NEVER
be used by that operator.

Parameters `m = 16, ef_construction = 128` were chosen by experiment on
clustered 384-dimension data (see the HNSW design notes): at the same index
size, `ef_construction = 128` gave higher recall than 64 for a similar build
time.

Requires pgvector >= 0.8.0. Filtered HNSW queries (every EKIP query filters by
tenant) are only complete with 0.8's iterative index scans; on an older
pgvector the index would silently return fewer rows than exist. The migration
therefore refuses to run rather than ship that.

Built with CREATE INDEX CONCURRENTLY so ingestion writes are not blocked.
CONCURRENTLY cannot run inside a transaction, hence `autocommit_block()` -- note
that this COMMITS anything earlier in the same `alembic upgrade` run first
(`base.py` runs all pending revisions in one transaction).

A concurrent build that fails or is cancelled leaves an INVALID index, which
`CREATE INDEX IF NOT EXISTS` would silently keep. So each step first drops an
existing index of the same name if it is invalid or has a different
definition, and verifies validity afterwards. Re-running the migration after a
failure is therefore safe.

Long builds: the migration engine (`base.py`) is built with no asyncpg
`command_timeout`; with the 30 s request-serving timeout a real-sized HNSW
build would be cancelled mid-way (leaving an INVALID index, which a re-run
repairs). Tests cover this with a bounded-timeout engine.

Row-level security: indexes do not interact with the tenant_isolation policies
on these tables, which are untouched. `agent_memories` is deliberately NOT
indexed (per-organization volume is small; an exact scan is faster and exact).
"""

import sqlalchemy as sa
from alembic import op

revision = 'b7c1d9e4f2a3'
down_revision = 'd5f2a8c1e9b4'
branch_labels = None
depends_on = None

_TABLES = ('documentation_chunks', 'code_chunks', 'conversations_chunks', 'incidents_chunks')
_M = 16
_EF_CONSTRUCTION = 128
_MIN_PGVECTOR = (0, 8, 0)
# Same SQL text `pg_get_indexdef` renders for the index created below.
_DEFINITION = (
    f"USING hnsw (embedding vector_ip_ops) WITH (m='{_M}', ef_construction='{_EF_CONSTRUCTION}')"
)


def _index_name(table: str) -> str:
    return f'ix_{table}_embedding_hnsw'


def _index_state(bind, name: str):
    """`(is_valid, definition)` of index `name` in the current schema, or None."""
    row = bind.execute(
        sa.text(
            'SELECT i.indisvalid, pg_get_indexdef(i.indexrelid) '
            'FROM pg_index i '
            'JOIN pg_class c ON c.oid = i.indexrelid '
            'JOIN pg_namespace n ON n.oid = c.relnamespace '
            'WHERE c.relname = :name AND n.nspname = current_schema()'
        ),
        {'name': name},
    ).first()
    return None if row is None else (row[0], row[1])


def _check_pgvector(bind) -> None:
    version = bind.execute(
        sa.text("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    ).scalar()
    if version is None:
        raise RuntimeError('The pgvector extension is not installed.')
    numeric = []
    for part in version.split('.')[:3]:
        digits = ''.join(ch for ch in part if ch.isdigit())
        numeric.append(int(digits) if digits else 0)
    if tuple(numeric) < _MIN_PGVECTOR:
        raise RuntimeError(
            f'pgvector {version} is older than 0.8.0: HNSW without iterative index '
            'scans returns incomplete results for tenant-filtered queries. '
            'Upgrade pgvector (ALTER EXTENSION vector UPDATE) and re-run.'
        )


def upgrade() -> None:
    with op.get_context().autocommit_block():
        bind = op.get_bind()
        _check_pgvector(bind)
        # Both are session settings on this (autocommit) connection. They are
        # reset only after success: if a build fails or is cancelled the
        # connection is unusable (a RESET would raise PendingRollbackError and
        # mask the real error) and is discarded with the migration run anyway.
        op.execute('SET statement_timeout = 0')
        op.execute("SET maintenance_work_mem = '512MB'")
        for table in _TABLES:
            name = _index_name(table)
            state = _index_state(bind, name)
            if state is not None and (not state[0] or _DEFINITION not in state[1]):
                # A failed/cancelled earlier build, or a different definition
                # under our name: remove it so the build below is real.
                op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS {name}')
            op.execute(
                f'CREATE INDEX CONCURRENTLY IF NOT EXISTS {name} ON {table} '
                f'USING hnsw (embedding vector_ip_ops) '
                f'WITH (m = {_M}, ef_construction = {_EF_CONSTRUCTION})'
            )
            state = _index_state(bind, name)
            if state is None or not state[0]:
                raise RuntimeError(
                    f'{name} is missing or INVALID after the build; drop it and re-run.'
                )
        op.execute('RESET maintenance_work_mem')
        op.execute('RESET statement_timeout')


def downgrade() -> None:
    with op.get_context().autocommit_block():
        for table in _TABLES:
            op.execute(f'DROP INDEX CONCURRENTLY IF EXISTS {_index_name(table)}')

"""HNSW migration `b7c1d9e4f2a3` -- the parts that need no database.

Real-PostgreSQL behaviour (upgrade/downgrade with data, invalid-index recovery,
the asyncpg command timeout, `alembic check`) is in
`test_hnsw_migration_postgres.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from app.database.models import retrieval_models
from app.database.session import Base

_FILE = (
    Path(__file__).resolve().parents[2]
    / "app/database/migrations/versions/b7c1d9e4f2a3_hnsw_chunk_embedding_indexes.py"
)
_TABLES = ("documentation_chunks", "code_chunks", "conversations_chunks", "incidents_chunks")


@pytest.fixture(scope="module")
def migration():
    spec = importlib.util.spec_from_file_location("hnsw_migration", _FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Bind:
    def __init__(self, version):
        self._version = version

    def execute(self, *_args, **_kwargs):
        return self

    def scalar(self):
        return self._version


@pytest.mark.parametrize("version", ["0.8.0", "0.8.7", "0.8.10", "0.10.1", "1.0.0"])
def test_pgvector_0_8_and_later_are_accepted(migration, version):
    migration._check_pgvector(_Bind(version))


@pytest.mark.parametrize("version", ["0.5.1", "0.6.2", "0.7.0", "0.7.4"])
def test_older_pgvector_is_refused_with_an_actionable_message(migration, version):
    with pytest.raises(RuntimeError, match=r"older than 0\.8\.0.*iterative"):
        migration._check_pgvector(_Bind(version))


def test_missing_extension_is_refused(migration):
    with pytest.raises(RuntimeError, match="not installed"):
        migration._check_pgvector(_Bind(None))


def test_migration_chain_position_and_scope(migration):
    assert migration.revision == "b7c1d9e4f2a3"
    assert migration.down_revision == "d5f2a8c1e9b4"
    # exactly the four chunk tables -- never agent_memories
    assert migration._TABLES == _TABLES
    assert (migration._M, migration._EF_CONSTRUCTION) == (16, 128)


def test_models_declare_the_same_indexes_the_migration_creates(migration):
    """`alembic check` compares models to the database: the declaration must
    match the migration exactly or CI reports drift."""
    for table in _TABLES:
        indexes = {i.name: i for i in Base.metadata.tables[table].indexes}
        index = indexes[f"ix_{table}_embedding_hnsw"]
        assert index.name == migration._index_name(table)
        assert index.dialect_options["postgresql"]["using"] == "hnsw"
        assert index.dialect_options["postgresql"]["with"] == {
            "m": migration._M,
            "ef_construction": migration._EF_CONSTRUCTION,
        }
        assert index.dialect_options["postgresql"]["ops"] == {"embedding": "vector_ip_ops"}
        assert [c.name for c in index.columns] == ["embedding"]


def test_agent_memories_is_not_indexed_for_ann():
    from app.database.models import memory_models  # noqa: F401  (registers the table)

    for index in Base.metadata.tables["agent_memories"].indexes:
        assert index.dialect_options["postgresql"]["using"] not in {"hnsw", "ivfflat"}


def test_models_module_is_the_declaration_site():
    assert "ix_" in retrieval_models._base_chunk_table_args("documentation_chunks")[-1].name


# --- the asyncpg command timeout the HNSW build must not inherit ---------------


def _captured_connect_args(monkeypatch, **build_kwargs):
    from app.database import session as session_module

    captured: dict[str, object] = {}

    def fake_create_async_engine(url, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(session_module, "create_async_engine", fake_create_async_engine)
    session_module._build_engine("postgresql+asyncpg://u:p@localhost:1/x", **build_kwargs)
    return captured["connect_args"]


def test_request_engines_keep_the_30_second_command_timeout(monkeypatch):
    assert _captured_connect_args(monkeypatch)["command_timeout"] == 30.0


def test_a_migration_engine_can_disable_the_client_side_command_timeout(monkeypatch):
    assert _captured_connect_args(monkeypatch, command_timeout=None)["command_timeout"] is None


def test_migration_environment_builds_a_timeout_free_engine_in_both_branches():
    """`base.py` must never reuse the shared 30 s engine: with or without
    `migration_database_url`, the engine it migrates with has no timeout."""
    source = (Path(__file__).resolve().parents[2] / "app/database/migrations/base.py").read_text(
        encoding="utf-8"
    )
    assert source.count("command_timeout=None") == 2
    assert "connectable = engine" not in source

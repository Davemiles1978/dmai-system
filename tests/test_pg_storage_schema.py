"""PR BBB-1: Postgres insights schema drift regression tests.

The Postgres `insights` schema in `pg_storage.py` must include every
column that any caller INSERTs into `insights` via `_get_db_conn()`
(save_knowledge, insight_promoter, seeder, etc). When these columns
are missing prod hits:

    save_knowledge FAILED for '...': SQLite Error: column "source_topic"
    of relation "insights" does not exist

Both the initial CREATE TABLE DDL AND the idempotent
ADD COLUMN IF NOT EXISTS migration list must cover every column. The
CREATE-DDL path handles fresh databases; the migration path handles
older deployments whose CREATE ran before the column was added.
"""
from __future__ import annotations

import importlib
import re

_REQUIRED_COLUMNS = (
    "id",
    "insight_text",
    "entity_type",
    "entities",
    "relationship",
    "confidence",
    "source_topic",
    "target_topic",
    "source_url",
    "source_title",
    "source_type",
    "created_at",
    "occurrence_count",
    "last_used",
    "neuron_level",
    "parent_macro_id",
    "domain",
    "provenance",
)


def _schema_ddl() -> str:
    """Return the CREATE TABLE insights (...) block from _SCHEMA_SQL."""
    pg_storage = importlib.import_module("components.pg_storage")
    sql = pg_storage._SCHEMA_SQL
    # Grab the CREATE TABLE ... insights ... block only, to avoid
    # matching column names in later table definitions.
    m = re.search(
        r"CREATE TABLE IF NOT EXISTS insights\s*\((.*?)\);",
        sql,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert m, "insights CREATE TABLE block not found in _SCHEMA_SQL"
    return m.group(1)


def test_schema_ddl_declares_all_required_columns():
    """Every canonical column must appear in the CREATE TABLE block."""
    ddl = _schema_ddl()
    missing = [c for c in _REQUIRED_COLUMNS if not re.search(rf"\b{c}\b", ddl)]
    assert not missing, (
        f"PG insights CREATE TABLE missing columns: {missing}\n"
        f"CREATE block was:\n{ddl}"
    )


def test_migrations_cover_all_required_columns_idempotently():
    """The idempotent migration list must ADD COLUMN IF NOT EXISTS for
    every column beyond the primordial (id, insight_text) set. This is
    what heals older prod DBs whose CREATE ran with the pre-BBB-1
    schema."""
    import inspect
    pg_storage = importlib.import_module("components.pg_storage")
    src = inspect.getsource(pg_storage.PGStorage._init_schema)

    # Columns that MUST appear as ADD COLUMN IF NOT EXISTS statements.
    must_migrate = (
        "source_topic",
        "target_topic",
        "occurrence_count",
        "last_used",
        "neuron_level",
        "parent_macro_id",
        "domain",
        "provenance",
    )
    for col in must_migrate:
        pat = (
            rf"ALTER\s+TABLE\s+insights\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS"
            rf"\s+{col}\b"
        )
        assert re.search(pat, src, flags=re.IGNORECASE), (
            f"pg_storage._init_schema missing idempotent migration for "
            f"insights.{col}. Older prod DBs won't auto-heal."
        )


def test_migrations_declare_source_topic_index():
    """Every insight lookup by topic needs an index. The migration
    must ensure it exists on older DBs too."""
    import inspect
    pg_storage = importlib.import_module("components.pg_storage")
    src = inspect.getsource(pg_storage.PGStorage._init_schema)
    assert re.search(
        r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+idx_insights_source_topic",
        src,
        flags=re.IGNORECASE,
    ), "pg_storage._init_schema missing idx_insights_source_topic migration"


def test_module_imports_clean():
    """Sanity: no import-time errors after the BBB-1 edit."""
    mod = importlib.import_module("components.pg_storage")
    assert hasattr(mod, "PGStorage")
    assert hasattr(mod, "_SCHEMA_SQL")


# ---------------------------------------------------------------------------
# PR CCC-1a: api_keys + external_api_calls schema regression tests
# ---------------------------------------------------------------------------
_API_KEYS_REQUIRED = (
    "key",
    "service",
    "source",
    "validated",
    "created_at",
    "last_used",
    "key_hash",
    "scope",
    "rate_limit_per_min",
    "revoked",
    "label",
)

_EXTERNAL_CALLS_REQUIRED = (
    "id",
    "key_hash",
    "service",
    "endpoint",
    "status_code",
    "ts",
    "duration_ms",
)


def _table_ddl(table: str) -> str:
    pg_storage = importlib.import_module("components.pg_storage")
    sql = pg_storage._SCHEMA_SQL
    m = re.search(
        rf"CREATE TABLE IF NOT EXISTS {table}\s*\((.*?)\);",
        sql,
        flags=re.DOTALL | re.IGNORECASE,
    )
    assert m, f"{table} CREATE TABLE block not found in _SCHEMA_SQL"
    return m.group(1)


def test_api_keys_ddl_declares_all_ccc1a_columns():
    ddl = _table_ddl("api_keys")
    missing = [c for c in _API_KEYS_REQUIRED if not re.search(rf"\b{c}\b", ddl)]
    assert not missing, (
        f"api_keys CREATE TABLE missing columns: {missing}\n{ddl}"
    )


def test_external_api_calls_ddl_declares_all_columns():
    ddl = _table_ddl("external_api_calls")
    missing = [c for c in _EXTERNAL_CALLS_REQUIRED if not re.search(rf"\b{c}\b", ddl)]
    assert not missing, (
        f"external_api_calls CREATE TABLE missing columns: {missing}\n{ddl}"
    )


def test_migrations_cover_all_api_keys_ccc1a_columns():
    import inspect
    pg_storage = importlib.import_module("components.pg_storage")
    src = inspect.getsource(pg_storage.PGStorage._init_schema)
    must_migrate = ("key_hash", "scope", "rate_limit_per_min", "revoked", "label")
    for col in must_migrate:
        pat = (
            rf"ALTER\s+TABLE\s+api_keys\s+ADD\s+COLUMN\s+IF\s+NOT\s+EXISTS"
            rf"\s+{col}\b"
        )
        assert re.search(pat, src, flags=re.IGNORECASE), (
            f"pg_storage._init_schema missing api_keys.{col} migration"
        )


def test_migrations_ensure_external_api_calls_and_hash_index():
    import inspect
    pg_storage = importlib.import_module("components.pg_storage")
    src = inspect.getsource(pg_storage.PGStorage._init_schema)
    assert re.search(
        r"CREATE TABLE IF NOT EXISTS external_api_calls",
        src, flags=re.IGNORECASE,
    ), "missing external_api_calls CREATE in migrations"
    assert re.search(
        r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+idx_api_keys_hash",
        src, flags=re.IGNORECASE,
    ), "missing idx_api_keys_hash migration"
    assert re.search(
        r"CREATE\s+INDEX\s+IF\s+NOT\s+EXISTS\s+idx_ext_calls_key_ts",
        src, flags=re.IGNORECASE,
    ), "missing idx_ext_calls_key_ts migration"

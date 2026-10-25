"""Postgres-backed ByteStore for LangChain CacheBackedEmbeddings.

Uses Neon (DATABASE_URL / POSTGRES_URL / DATABASE_URL_UNPOOLED) when available,
falls back to raising a clear error otherwise.
"""
import os

from dotenv import load_dotenv
from langchain_core.stores import ByteStore
from sqlalchemy import Column, LargeBinary, String, Table, MetaData, create_engine, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

load_dotenv(override=True)  # .env is source of truth — overrides any stray env vars
def _resolve_postgres_url() -> str:
    for key in ("DATABASE_URL", "POSTGRES_URL", "DATABASE_URL_UNPOOLED"):
        url = os.environ.get(key, "").strip()
        if url:
            return url
    raise RuntimeError(
        "No Postgres connection string found. Set DATABASE_URL (pooled) "
        "or DATABASE_URL_UNPOOLED (direct) in .env via `neon env pull`."
    )


class PostgresByteStore(ByteStore):
    """Minimal ByteStore backed by a single Postgres table (key TEXT PK, value BYTEA)."""

    def __init__(self, connection_string: str, table_name: str = "embedding_cache"):
        # PgBouncer (pooled -pooler URL) does not support some session-level
        # statements; use plain autocommit statements only â†’ safe on both.
        if connection_string.startswith("postgres://"):
            connection_string = "postgresql://" + connection_string[len("postgres://"):]
        # Route SQLAlchemy through psycopg v3 (installed) instead of psycopg2.
        if connection_string.startswith("postgresql://"):
            connection_string = "postgresql+psycopg://" + connection_string[len("postgresql://"):]
        elif connection_string.startswith("postgresql+psycopg://"):
            pass
        self.engine = create_engine(connection_string, pool_pre_ping=True)
        self.metadata = MetaData()
        self.table = Table(
            table_name,
            self.metadata,
            Column("key", String, primary_key=True),
            Column("value", LargeBinary, nullable=False),
        )
        self.metadata.create_all(self.engine, checkfirst=True)

    def mget(self, keys):
        with self.engine.begin() as conn:
            rows = conn.execute(
                select(self.table.c.key, self.table.c.value).where(
                    self.table.c.key.in_(list(keys))
                )
            ).all()
        found = {k: bytes(v) for k, v in rows}
        return [found.get(k) for k in keys]

    def mset(self, key_value_pairs) -> None:
        pairs = list(key_value_pairs)
        if not pairs:
            return
        with self.engine.begin() as conn:
            for k, v in pairs:
                stmt = pg_insert(self.table).values(key=k, value=bytes(v))
                stmt = stmt.on_conflict_do_update(
                    index_elements=["key"], set_={"value": stmt.excluded.value}
                )
                conn.execute(stmt)

    def mdelete(self, keys) -> None:
        keys = list(keys)
        if not keys:
            return
        with self.engine.begin() as conn:
            conn.execute(self.table.delete().where(self.table.c.key.in_(keys)))

    def yield_keys(self, prefix=None):
        with self.engine.connect() as conn:
            q = select(self.table.c.key)
            if prefix:
                q = q.where(self.table.c.key.like(f"{prefix}%"))
            for (k,) in conn.execute(q):
                yield k


def get_embedding_byte_store(table_name: str = "embedding_cache"):
    """Return PostgresByteStore if a Postgres URL is configured, else None."""
    try:
        url = _resolve_postgres_url()
    except RuntimeError:
        return None
    return PostgresByteStore(url, table_name=table_name)

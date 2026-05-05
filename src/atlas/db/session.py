from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from atlas.core.config import get_settings
from atlas.db.models import Base

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _apply_lightweight_migrations()


def _apply_lightweight_migrations() -> None:
    statements = [
        """
        create table if not exists parent_blocks (
            parent_id varchar(64) primary key,
            document_id varchar(64) not null references documents(document_id),
            parent_type varchar(32) not null,
            page_start integer not null,
            page_end integer not null,
            text text not null,
            child_ids_json jsonb not null default '[]'::jsonb,
            metadata_json jsonb not null default '{}'::jsonb
        )
        """,
        "create index if not exists ix_parent_blocks_document_id on parent_blocks (document_id)",
        """
        create index if not exists ix_parent_blocks_page_range
        on parent_blocks (document_id, page_start, page_end)
        """,
        "alter table chunks add column if not exists parent_id varchar(64)",
        "create index if not exists ix_chunks_parent_id on chunks (parent_id)",
        "alter table chunks add column if not exists page_start integer",
        "alter table chunks add column if not exists page_end integer",
        "alter table ingestion_runs add column if not exists summary_json jsonb not null default '{}'::jsonb",
        "alter table query_runs add column if not exists details_json jsonb not null default '{}'::jsonb",
        """
        create table if not exists query_cache (
            "key" varchar(64) primary key,
            answer text not null,
            confidence varchar(32),
            citations_json jsonb not null default '[]'::jsonb,
            metadata_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now(),
            updated_at timestamp with time zone not null default now(),
            expires_at timestamp with time zone,
            hit_count integer not null default 0
        )
        """,
        "create index if not exists ix_query_cache_expires_at on query_cache (expires_at)",
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def check_db() -> bool:
    with engine.connect() as conn:
        conn.execute(text("select 1"))
    return True


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

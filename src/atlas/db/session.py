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


def _sql_text_array(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _primary_key_constraint_sql(
    table: str,
    constraint: str,
    columns: tuple[str, ...],
    *,
    require_name: bool = False,
) -> str:
    column_list = ", ".join(columns)
    null_predicate = " or ".join(f"{column} is null" for column in columns)
    expected_columns = _sql_text_array(columns)
    name_predicate = f" or existing_pk_name is distinct from '{constraint}'" if require_name else ""
    return f"""
        do $$
        declare
            existing_pk_name text;
            existing_pk_columns text[];
        begin
            select con.conname, array_agg(att.attname::text order by keys.ordinality)
            into existing_pk_name, existing_pk_columns
            from pg_constraint con
            join unnest(con.conkey) with ordinality as keys(attnum, ordinality) on true
            join pg_attribute att
              on att.attrelid = con.conrelid
             and att.attnum = keys.attnum
            where con.conrelid = '{table}'::regclass
              and con.contype = 'p'
            group by con.conname;

            if existing_pk_columns is distinct from array[{expected_columns}]::text[]{name_predicate} then
                if exists (select 1 from {table} where {null_predicate}) then
                    raise exception
                        'Cannot rebuild primary key {constraint} on {table}: target columns contain null values';
                end if;

                if exists (
                    select 1
                    from (
                        select {column_list}
                        from {table}
                        group by {column_list}
                        having count(*) > 1
                    ) duplicate_keys
                ) then
                    raise exception
                        'Cannot rebuild primary key {constraint} on {table}: target columns contain duplicate values';
                end if;

                if existing_pk_name is not null then
                    execute format('alter table {table} drop constraint %I', existing_pk_name);
                end if;

                alter table {table}
                add constraint {constraint} primary key ({column_list});
            end if;
        end $$;
        """


def _unique_constraint_sql(table: str, constraint: str, columns: tuple[str, ...]) -> str:
    column_list = ", ".join(columns)
    expected_columns = _sql_text_array(columns)
    return f"""
        do $$
        declare
            existing_columns text[];
        begin
            select array_agg(att.attname::text order by keys.ordinality)
            into existing_columns
            from pg_constraint con
            join unnest(con.conkey) with ordinality as keys(attnum, ordinality) on true
            join pg_attribute att
              on att.attrelid = con.conrelid
             and att.attnum = keys.attnum
            where con.conrelid = '{table}'::regclass
              and con.contype = 'u'
              and con.conname = '{constraint}'
            group by con.conname;

            if existing_columns is distinct from array[{expected_columns}]::text[] then
                if existing_columns is not null then
                    alter table {table} drop constraint {constraint};
                end if;

                drop index if exists {constraint};

                alter table {table}
                add constraint {constraint} unique ({column_list});
            end if;
        end $$;
        """


def _foreign_key_constraint_sql(
    table: str,
    constraint: str,
    columns: tuple[str, ...],
    target_table: str,
    target_columns: tuple[str, ...],
) -> str:
    column_list = ", ".join(columns)
    target_column_list = ", ".join(target_columns)
    expected_columns = _sql_text_array(columns)
    expected_target_columns = _sql_text_array(target_columns)
    return f"""
        do $$
        declare
            existing_columns text[];
            existing_target_table regclass;
            existing_target_columns text[];
            existing_delete_action "char";
        begin
            select
                array_agg(att.attname::text order by keys.ordinality),
                con.confrelid::regclass,
                array_agg(target_att.attname::text order by target_keys.ordinality),
                con.confdeltype
            into
                existing_columns,
                existing_target_table,
                existing_target_columns,
                existing_delete_action
            from pg_constraint con
            join unnest(con.conkey) with ordinality as keys(attnum, ordinality) on true
            join pg_attribute att
              on att.attrelid = con.conrelid
             and att.attnum = keys.attnum
            join unnest(con.confkey) with ordinality as target_keys(attnum, ordinality)
              on target_keys.ordinality = keys.ordinality
            join pg_attribute target_att
              on target_att.attrelid = con.confrelid
             and target_att.attnum = target_keys.attnum
            where con.conrelid = '{table}'::regclass
              and con.contype = 'f'
              and con.conname = '{constraint}'
            group by con.conname, con.confrelid, con.confdeltype;

            if existing_columns is distinct from array[{expected_columns}]::text[]
                or existing_target_table is distinct from '{target_table}'::regclass
                or existing_target_columns is distinct from array[{expected_target_columns}]::text[]
                or existing_delete_action is distinct from 'c' then
                if existing_columns is not null then
                    alter table {table} drop constraint {constraint};
                end if;

                alter table {table}
                add constraint {constraint}
                foreign key ({column_list})
                references {target_table}({target_column_list})
                on delete cascade;
            end if;
        end $$;
        """


def _not_null_columns_sql(table: str, columns: tuple[str, ...], message: str) -> str:
    null_predicate = " or ".join(f"{column} is null" for column in columns)
    alter_statements = "\n".join(
        f"                alter table {table} alter column {column} set not null;"
        for column in columns
    )
    return f"""
        do $$
        begin
            if exists (select 1 from {table} where {null_predicate}) then
                raise exception '{message}';
            end if;

{alter_statements}
        end $$;
        """


def _drop_fk_constraints_sql(table: str, constraints: tuple[str, ...]) -> str:
    constraint_names = _sql_text_array(constraints)
    return f"""
        do $$
        declare
            existing_constraint_name text;
        begin
            if to_regclass('{table}') is not null then
                for existing_constraint_name in
                    select con.conname
                    from pg_constraint con
                    where con.conrelid = '{table}'::regclass
                      and con.contype = 'f'
                      and con.conname = any(array[{constraint_names}]::text[])
                loop
                    execute format(
                        'alter table {table} drop constraint %I',
                        existing_constraint_name
                    );
                end loop;
            end if;
        end $$;
        """


def _drop_legacy_unique_constraint_sql(
    table: str,
    constraint: str,
    legacy_columns: tuple[str, ...],
) -> str:
    expected_columns = _sql_text_array(legacy_columns)
    return f"""
        do $$
        declare
            existing_constraint_columns text[];
            existing_index_columns text[];
        begin
            if to_regclass('{table}') is not null then
                select array_agg(att.attname::text order by keys.ordinality)
                into existing_constraint_columns
                from pg_constraint con
                join unnest(con.conkey) with ordinality as keys(attnum, ordinality) on true
                join pg_attribute att
                  on att.attrelid = con.conrelid
                 and att.attnum = keys.attnum
                where con.conrelid = '{table}'::regclass
                  and con.contype = 'u'
                  and con.conname = '{constraint}'
                group by con.conname;

                if existing_constraint_columns = array[{expected_columns}]::text[] then
                    alter table {table} drop constraint {constraint};
                end if;

                select array_agg(att.attname::text order by keys.ordinality)
                into existing_index_columns
                from pg_class idx
                join pg_index index_info on index_info.indexrelid = idx.oid
                join unnest(index_info.indkey) with ordinality as keys(attnum, ordinality) on true
                join pg_attribute att
                  on att.attrelid = index_info.indrelid
                 and att.attnum = keys.attnum
                where index_info.indrelid = '{table}'::regclass
                  and index_info.indisunique
                  and idx.relname = '{constraint}'
                group by idx.oid;

                if existing_index_columns = array[{expected_columns}]::text[] then
                    drop index if exists {constraint};
                end if;
            end if;
        end $$;
        """


def _drop_legacy_foreign_key_constraints_sql(
    table: str,
    columns: tuple[str, ...],
    target_table: str,
    target_columns: tuple[str, ...],
) -> str:
    expected_columns = _sql_text_array(columns)
    expected_target_columns = _sql_text_array(target_columns)
    return f"""
        do $$
        declare
            child_table regclass := to_regclass('{table}');
            parent_table regclass := to_regclass('{target_table}');
            legacy_constraint_name text;
        begin
            if child_table is null or parent_table is null then
                return;
            end if;

            for legacy_constraint_name in
                select con.conname
                from pg_constraint con
                where con.conrelid = child_table
                  and con.contype = 'f'
                  and con.confrelid = parent_table
                  and (
                      select array_agg(att.attname::text order by keys.ordinality)
                      from unnest(con.conkey) with ordinality as keys(attnum, ordinality)
                      join pg_attribute att
                        on att.attrelid = con.conrelid
                       and att.attnum = keys.attnum
                  ) = array[{expected_columns}]::text[]
                  and (
                      select array_agg(att.attname::text order by keys.ordinality)
                      from unnest(con.confkey) with ordinality as keys(attnum, ordinality)
                      join pg_attribute att
                        on att.attrelid = con.confrelid
                       and att.attnum = keys.attnum
                  ) = array[{expected_target_columns}]::text[]
            loop
                execute format('alter table %s drop constraint %I', child_table, legacy_constraint_name);
            end loop;
        end $$;
        """


def _legacy_graph_version_backfill_sql() -> str:
    return """
        do $$
        begin
            if to_regclass('graph_relationships') is not null then
                alter table graph_relationships
                add column if not exists graph_version varchar(128);
            end if;

            if to_regclass('graph_entity_anchors') is not null then
                alter table graph_entity_anchors
                add column if not exists graph_version varchar(128);
            end if;

            if to_regclass('graph_relationship_anchors') is not null then
                alter table graph_relationship_anchors
                add column if not exists graph_version varchar(128);
            end if;

            if to_regclass('graph_relationships') is not null
                and to_regclass('graph_entities') is not null
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_relationships'::regclass
                      and attname = 'relationship_id'
                      and not attisdropped
                )
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_relationships'::regclass
                      and attname = 'source_entity_id'
                      and not attisdropped
                )
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_relationships'::regclass
                      and attname = 'target_entity_id'
                      and not attisdropped
                )
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_entities'::regclass
                      and attname = 'graph_version'
                      and not attisdropped
                )
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_entities'::regclass
                      and attname = 'entity_id'
                      and not attisdropped
                )
            then
                execute $sql$
                    with entity_versions as (
                        select entity_id, min(graph_version) as graph_version
                        from graph_entities
                        where graph_version is not null
                        group by entity_id
                        having count(distinct graph_version) = 1
                    ),
                    relationship_versions as (
                        select rel.relationship_id, source_versions.graph_version
                        from graph_relationships rel
                        join entity_versions source_versions
                          on source_versions.entity_id = rel.source_entity_id
                        join entity_versions target_versions
                          on target_versions.entity_id = rel.target_entity_id
                         and target_versions.graph_version = source_versions.graph_version
                    )
                    update graph_relationships rel
                    set graph_version = relationship_versions.graph_version
                    from relationship_versions
                    where rel.graph_version is null
                      and rel.relationship_id = relationship_versions.relationship_id
                $sql$;
            end if;

            if to_regclass('graph_entity_anchors') is not null
                and to_regclass('graph_entities') is not null
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_entity_anchors'::regclass
                      and attname = 'entity_id'
                      and not attisdropped
                )
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_entities'::regclass
                      and attname = 'graph_version'
                      and not attisdropped
                )
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_entities'::regclass
                      and attname = 'entity_id'
                      and not attisdropped
                )
            then
                execute $sql$
                    with entity_versions as (
                        select entity_id, min(graph_version) as graph_version
                        from graph_entities
                        where graph_version is not null
                        group by entity_id
                        having count(distinct graph_version) = 1
                    )
                    update graph_entity_anchors anchor
                    set graph_version = entity_versions.graph_version
                    from entity_versions
                    where anchor.graph_version is null
                      and anchor.entity_id = entity_versions.entity_id
                $sql$;
            end if;

            if to_regclass('graph_relationship_anchors') is not null
                and to_regclass('graph_relationships') is not null
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_relationship_anchors'::regclass
                      and attname = 'relationship_id'
                      and not attisdropped
                )
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_relationships'::regclass
                      and attname = 'graph_version'
                      and not attisdropped
                )
                and exists (
                    select 1
                    from pg_attribute
                    where attrelid = 'graph_relationships'::regclass
                      and attname = 'relationship_id'
                      and not attisdropped
                )
            then
                execute $sql$
                    with relationship_versions as (
                        select relationship_id, min(graph_version) as graph_version
                        from graph_relationships
                        where graph_version is not null
                        group by relationship_id
                        having count(distinct graph_version) = 1
                    )
                    update graph_relationship_anchors anchor
                    set graph_version = relationship_versions.graph_version
                    from relationship_versions
                    where anchor.graph_version is null
                      and anchor.relationship_id = relationship_versions.relationship_id
                $sql$;
            end if;
        end $$;
        """


def _legacy_graph_upgrade_preflight_sql() -> str:
    return """
        do $$
        declare
            has_entity_key_columns boolean;
            has_relationship_key_columns boolean;
            has_relationship_entity_columns boolean;
            has_entity_anchor_key_columns boolean;
            has_entity_anchor_entity_columns boolean;
            has_entity_anchor_entity_id_column boolean;
            has_entity_anchor_chunk_id_column boolean;
            has_relationship_anchor_key_columns boolean;
            has_relationship_anchor_relationship_columns boolean;
            has_relationship_anchor_relationship_id_column boolean;
            has_relationship_anchor_chunk_id_column boolean;
        begin
            select to_regclass('graph_entities') is not null
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_entities')
                      and attname = 'graph_version'
                      and not attisdropped
                )
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_entities')
                      and attname = 'entity_id'
                      and not attisdropped
                )
            into has_entity_key_columns;

            select to_regclass('graph_relationships') is not null
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_relationships')
                      and attname = 'graph_version'
                      and not attisdropped
                )
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_relationships')
                      and attname = 'relationship_id'
                      and not attisdropped
                )
            into has_relationship_key_columns;

            select has_relationship_key_columns
                and has_entity_key_columns
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_relationships')
                      and attname = 'source_entity_id'
                      and not attisdropped
                )
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_relationships')
                      and attname = 'target_entity_id'
                      and not attisdropped
                )
            into has_relationship_entity_columns;

            select to_regclass('graph_entity_anchors') is not null
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_entity_anchors')
                      and attname = 'graph_version'
                      and not attisdropped
                )
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_entity_anchors')
                      and attname = 'anchor_id'
                      and not attisdropped
                )
            into has_entity_anchor_key_columns;

            select has_entity_anchor_key_columns
                and has_entity_key_columns
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_entity_anchors')
                      and attname = 'entity_id'
                      and not attisdropped
                )
            into has_entity_anchor_entity_columns;

            select to_regclass('graph_relationship_anchors') is not null
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_relationship_anchors')
                      and attname = 'graph_version'
                      and not attisdropped
                )
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_relationship_anchors')
                      and attname = 'anchor_id'
                      and not attisdropped
                )
            into has_relationship_anchor_key_columns;

            select has_relationship_anchor_key_columns
                and has_relationship_key_columns
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_relationship_anchors')
                      and attname = 'relationship_id'
                      and not attisdropped
                )
            into has_relationship_anchor_relationship_columns;

            select to_regclass('graph_entity_anchors') is not null
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_entity_anchors')
                      and attname = 'entity_id'
                      and not attisdropped
                )
            into has_entity_anchor_entity_id_column;

            select to_regclass('graph_entity_anchors') is not null
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_entity_anchors')
                      and attname = 'chunk_id'
                      and not attisdropped
                )
            into has_entity_anchor_chunk_id_column;

            select to_regclass('graph_relationship_anchors') is not null
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_relationship_anchors')
                      and attname = 'relationship_id'
                      and not attisdropped
                )
            into has_relationship_anchor_relationship_id_column;

            select to_regclass('graph_relationship_anchors') is not null
                and exists (
                    select 1 from pg_attribute
                    where attrelid = to_regclass('graph_relationship_anchors')
                      and attname = 'chunk_id'
                      and not attisdropped
                )
            into has_relationship_anchor_chunk_id_column;

            if to_regclass('graph_entities') is not null
                and exists (
                    select 1
                    from graph_entities
                    where graph_version is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_entities.graph_version contains null values';
            end if;

            if has_entity_key_columns
                and exists (
                    select 1
                    from (
                        select graph_version, entity_id
                        from graph_entities
                        group by graph_version, entity_id
                        having count(*) > 1
                    ) duplicate_keys
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_entities has duplicate graph_version/entity_id values';
            end if;

            if has_entity_key_columns
                and to_regclass('graph_indexes') is not null
                and exists (
                    select 1
                    from graph_entities entity
                    left join graph_indexes graph_index
                      on graph_index.graph_version = entity.graph_version
                    where graph_index.graph_version is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_entities.graph_version values are missing from graph_indexes';
            end if;

            if has_relationship_key_columns
                and exists (
                    select 1
                    from graph_relationships
                    where graph_version is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_relationships.graph_version contains null values';
            end if;

            if has_relationship_key_columns
                and exists (
                    select 1
                    from (
                        select graph_version, relationship_id
                        from graph_relationships
                        group by graph_version, relationship_id
                        having count(*) > 1
                    ) duplicate_keys
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_relationships has duplicate graph_version/relationship_id values';
            end if;

            if has_entity_anchor_key_columns
                and exists (
                    select 1
                    from graph_entity_anchors
                    where graph_version is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_entity_anchors.graph_version contains null values';
            end if;

            if has_entity_anchor_entity_id_column
                and exists (
                    select 1
                    from graph_entity_anchors
                    where entity_id is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_entity_anchors.entity_id contains null values';
            end if;

            if has_entity_anchor_chunk_id_column
                and exists (
                    select 1
                    from graph_entity_anchors
                    where chunk_id is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_entity_anchors.chunk_id contains null values';
            end if;

            if has_entity_anchor_key_columns
                and exists (
                    select 1
                    from (
                        select graph_version, anchor_id
                        from graph_entity_anchors
                        group by graph_version, anchor_id
                        having count(*) > 1
                    ) duplicate_keys
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_entity_anchors has duplicate graph_version/anchor_id values';
            end if;

            if has_relationship_anchor_key_columns
                and exists (
                    select 1
                    from graph_relationship_anchors
                    where graph_version is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_relationship_anchors.graph_version contains null values';
            end if;

            if has_relationship_anchor_relationship_id_column
                and exists (
                    select 1
                    from graph_relationship_anchors
                    where relationship_id is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_relationship_anchors.relationship_id contains null values';
            end if;

            if has_relationship_anchor_chunk_id_column
                and exists (
                    select 1
                    from graph_relationship_anchors
                    where chunk_id is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_relationship_anchors.chunk_id contains null values';
            end if;

            if has_relationship_anchor_key_columns
                and exists (
                    select 1
                    from (
                        select graph_version, anchor_id
                        from graph_relationship_anchors
                        group by graph_version, anchor_id
                        having count(*) > 1
                    ) duplicate_keys
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_relationship_anchors has duplicate graph_version/anchor_id values';
            end if;

            if has_relationship_entity_columns
                and exists (
                    select 1
                    from graph_relationships rel
                    left join graph_entities entity
                      on entity.graph_version = rel.graph_version
                     and entity.entity_id = rel.source_entity_id
                    where entity.entity_id is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_relationships source entities do not match graph_version';
            end if;

            if has_relationship_entity_columns
                and exists (
                    select 1
                    from graph_relationships rel
                    left join graph_entities entity
                      on entity.graph_version = rel.graph_version
                     and entity.entity_id = rel.target_entity_id
                    where entity.entity_id is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_relationships target entities do not match graph_version';
            end if;

            if has_entity_anchor_entity_columns
                and exists (
                    select 1
                    from graph_entity_anchors anchor
                    left join graph_entities entity
                      on entity.graph_version = anchor.graph_version
                     and entity.entity_id = anchor.entity_id
                    where entity.entity_id is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_entity_anchors entities do not match graph_version';
            end if;

            if has_relationship_anchor_relationship_columns
                and exists (
                    select 1
                    from graph_relationship_anchors anchor
                    left join graph_relationships rel
                      on rel.graph_version = anchor.graph_version
                     and rel.relationship_id = anchor.relationship_id
                    where rel.relationship_id is null
                )
            then
                raise exception
                    'Cannot upgrade legacy graph schema: graph_relationship_anchors relationships do not match graph_version';
            end if;
        end $$;
        """


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
        create table if not exists graph_indexes (
            graph_version varchar(128) primary key,
            corpus_version varchar(128),
            fixture_schema_version varchar(64) not null,
            fixture_hash varchar(128) not null,
            loader_version varchar(64) not null,
            row_counts_json jsonb not null default '{}'::jsonb,
            status varchar(32) not null default 'pending',
            loaded_at timestamp with time zone,
            metadata_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "alter table graph_indexes add column if not exists graph_version varchar(128)",
        "alter table graph_indexes add column if not exists corpus_version varchar(128)",
        """
        alter table graph_indexes
        add column if not exists fixture_schema_version varchar(64)
        """,
        "alter table graph_indexes add column if not exists fixture_hash varchar(128)",
        "alter table graph_indexes add column if not exists loader_version varchar(64)",
        """
        alter table graph_indexes
        add column if not exists row_counts_json jsonb not null default '{}'::jsonb
        """,
        """
        alter table graph_indexes
        add column if not exists status varchar(32) not null default 'pending'
        """,
        """
        alter table graph_indexes
        add column if not exists loaded_at timestamp with time zone
        """,
        """
        alter table graph_indexes
        add column if not exists metadata_json jsonb not null default '{}'::jsonb
        """,
        """
        alter table graph_indexes
        add column if not exists created_at timestamp with time zone not null default now()
        """,
        "update graph_indexes set row_counts_json = '{}'::jsonb where row_counts_json is null",
        "update graph_indexes set status = 'pending' where status is null",
        "update graph_indexes set metadata_json = '{}'::jsonb where metadata_json is null",
        "update graph_indexes set created_at = now() where created_at is null",
        """
        update graph_indexes
        set fixture_schema_version = 'legacy_unknown'
        where fixture_schema_version is null
        """,
        """
        update graph_indexes
        set fixture_hash = 'legacy_unknown'
        where fixture_hash is null
        """,
        """
        update graph_indexes
        set loader_version = 'legacy_unknown'
        where loader_version is null
        """,
        "alter table graph_indexes alter column fixture_schema_version set not null",
        "alter table graph_indexes alter column fixture_hash set not null",
        "alter table graph_indexes alter column loader_version set not null",
        "alter table graph_indexes alter column row_counts_json set default '{}'::jsonb",
        "alter table graph_indexes alter column row_counts_json set not null",
        "alter table graph_indexes alter column status set default 'pending'",
        "alter table graph_indexes alter column status set not null",
        "alter table graph_indexes alter column metadata_json set default '{}'::jsonb",
        "alter table graph_indexes alter column metadata_json set not null",
        "alter table graph_indexes alter column created_at set default now()",
        "alter table graph_indexes alter column created_at set not null",
        _primary_key_constraint_sql(
            "graph_indexes",
            "graph_indexes_pkey",
            ("graph_version",),
        ),
        """
        create table if not exists graph_entities (
            graph_version varchar(128) not null
                constraint fk_graph_entities_graph_version
                references graph_indexes(graph_version) on delete cascade,
            entity_id varchar(128) not null,
            canonical_name varchar(512) not null,
            canonical_name_norm varchar(512) not null,
            entity_type varchar(64) not null,
            aliases_json jsonb not null default '[]'::jsonb,
            metadata_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now(),
            constraint pk_graph_entities primary key (graph_version, entity_id),
            constraint uq_graph_entities_version_type_name
                unique (graph_version, entity_type, canonical_name_norm)
        )
        """,
        "alter table graph_entities add column if not exists graph_version varchar(128)",
        "alter table graph_entities add column if not exists entity_id varchar(128)",
        "alter table graph_entities add column if not exists canonical_name varchar(512)",
        """
        alter table graph_entities
        add column if not exists canonical_name_norm varchar(512)
        """,
        "alter table graph_entities add column if not exists entity_type varchar(64)",
        """
        alter table graph_entities
        add column if not exists aliases_json jsonb not null default '[]'::jsonb
        """,
        """
        alter table graph_entities
        add column if not exists metadata_json jsonb not null default '{}'::jsonb
        """,
        """
        alter table graph_entities
        add column if not exists created_at timestamp with time zone not null default now()
        """,
        "update graph_entities set aliases_json = '[]'::jsonb where aliases_json is null",
        "update graph_entities set metadata_json = '{}'::jsonb where metadata_json is null",
        "update graph_entities set created_at = now() where created_at is null",
        "alter table graph_entities alter column aliases_json set default '[]'::jsonb",
        "alter table graph_entities alter column aliases_json set not null",
        "alter table graph_entities alter column metadata_json set default '{}'::jsonb",
        "alter table graph_entities alter column metadata_json set not null",
        "alter table graph_entities alter column created_at set default now()",
        "alter table graph_entities alter column created_at set not null",
        _drop_legacy_unique_constraint_sql(
            "graph_entities",
            "uq_graph_entities_version_entity",
            ("graph_version", "entity_id"),
        ),
        _legacy_graph_version_backfill_sql(),
        _legacy_graph_upgrade_preflight_sql(),
        _drop_legacy_foreign_key_constraints_sql(
            "graph_relationships",
            ("source_entity_id",),
            "graph_entities",
            ("entity_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "graph_relationships",
            ("target_entity_id",),
            "graph_entities",
            ("entity_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "graph_entity_anchors",
            ("entity_id",),
            "graph_entities",
            ("entity_id",),
        ),
        _primary_key_constraint_sql(
            "graph_entities",
            "pk_graph_entities",
            ("graph_version", "entity_id"),
        ),
        _foreign_key_constraint_sql(
            "graph_entities",
            "fk_graph_entities_graph_version",
            ("graph_version",),
            "graph_indexes",
            ("graph_version",),
        ),
        _unique_constraint_sql(
            "graph_entities",
            "uq_graph_entities_version_type_name",
            ("graph_version", "entity_type", "canonical_name_norm"),
        ),
        "create index if not exists ix_graph_entities_graph_version on graph_entities (graph_version)",
        "create index if not exists ix_graph_entities_entity_type on graph_entities (entity_type)",
        """
        create index if not exists ix_graph_entities_canonical_name_norm
        on graph_entities (canonical_name_norm)
        """,
        """
        create table if not exists graph_relationships (
            graph_version varchar(128) not null
                constraint fk_graph_relationships_graph_version
                references graph_indexes(graph_version) on delete cascade,
            relationship_id varchar(128) not null,
            source_entity_id varchar(128) not null,
            target_entity_id varchar(128) not null,
            relation_type varchar(128) not null,
            confidence double precision not null default 1.0,
            metadata_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now(),
            constraint pk_graph_relationships primary key (graph_version, relationship_id),
            constraint fk_graph_relationships_source_entity
                foreign key (graph_version, source_entity_id)
                references graph_entities(graph_version, entity_id)
                on delete cascade,
            constraint fk_graph_relationships_target_entity
                foreign key (graph_version, target_entity_id)
                references graph_entities(graph_version, entity_id)
                on delete cascade
        )
        """,
        "alter table graph_relationships add column if not exists graph_version varchar(128)",
        """
        alter table graph_relationships
        add column if not exists relationship_id varchar(128)
        """,
        """
        alter table graph_relationships
        add column if not exists source_entity_id varchar(128)
        """,
        """
        alter table graph_relationships
        add column if not exists target_entity_id varchar(128)
        """,
        """
        alter table graph_relationships
        add column if not exists relation_type varchar(128)
        """,
        """
        alter table graph_relationships
        add column if not exists confidence double precision not null default 1.0
        """,
        """
        alter table graph_relationships
        add column if not exists metadata_json jsonb not null default '{}'::jsonb
        """,
        """
        alter table graph_relationships
        add column if not exists created_at timestamp with time zone not null default now()
        """,
        "update graph_relationships set confidence = 1.0 where confidence is null",
        "update graph_relationships set metadata_json = '{}'::jsonb where metadata_json is null",
        "update graph_relationships set created_at = now() where created_at is null",
        "alter table graph_relationships alter column confidence set default 1.0",
        "alter table graph_relationships alter column confidence set not null",
        "alter table graph_relationships alter column metadata_json set default '{}'::jsonb",
        "alter table graph_relationships alter column metadata_json set not null",
        "alter table graph_relationships alter column created_at set default now()",
        "alter table graph_relationships alter column created_at set not null",
        """
        do $$
        begin
            if not exists (
                select 1 from graph_relationships where graph_version is null
            ) then
                alter table graph_relationships alter column graph_version set not null;
            end if;
        end $$;
        """,
        _drop_legacy_foreign_key_constraints_sql(
            "graph_relationship_anchors",
            ("relationship_id",),
            "graph_relationships",
            ("relationship_id",),
        ),
        _primary_key_constraint_sql(
            "graph_relationships",
            "pk_graph_relationships",
            ("graph_version", "relationship_id"),
        ),
        _foreign_key_constraint_sql(
            "graph_relationships",
            "fk_graph_relationships_graph_version",
            ("graph_version",),
            "graph_indexes",
            ("graph_version",),
        ),
        _foreign_key_constraint_sql(
            "graph_relationships",
            "fk_graph_relationships_source_entity",
            ("graph_version", "source_entity_id"),
            "graph_entities",
            ("graph_version", "entity_id"),
        ),
        _foreign_key_constraint_sql(
            "graph_relationships",
            "fk_graph_relationships_target_entity",
            ("graph_version", "target_entity_id"),
            "graph_entities",
            ("graph_version", "entity_id"),
        ),
        """
        create index if not exists ix_graph_relationships_graph_version
        on graph_relationships (graph_version)
        """,
        """
        create index if not exists ix_graph_relationships_source
        on graph_relationships (source_entity_id)
        """,
        """
        create index if not exists ix_graph_relationships_target
        on graph_relationships (target_entity_id)
        """,
        """
        create index if not exists ix_graph_relationships_relation_type
        on graph_relationships (relation_type)
        """,
        """
        create index if not exists ix_graph_relationships_graph_source
        on graph_relationships (graph_version, source_entity_id)
        """,
        """
        create index if not exists ix_graph_relationships_graph_target
        on graph_relationships (graph_version, target_entity_id)
        """,
        """
        create index if not exists ix_graph_relationships_graph_relation
        on graph_relationships (graph_version, relation_type)
        """,
        """
        create table if not exists graph_entity_anchors (
            graph_version varchar(128) not null,
            anchor_id varchar(128) not null,
            entity_id varchar(128) not null,
            chunk_id varchar(64) not null
                constraint fk_graph_entity_anchors_chunk
                references chunks(chunk_id) on delete cascade,
            text_span text,
            text_span_hash varchar(128) not null default 'whole_chunk',
            metadata_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now(),
            constraint fk_graph_entity_anchors_entity
                foreign key (graph_version, entity_id)
                references graph_entities(graph_version, entity_id)
                on delete cascade,
            constraint pk_graph_entity_anchors primary key (graph_version, anchor_id),
            constraint uq_graph_entity_anchors_entity_chunk_span
                unique (graph_version, entity_id, chunk_id, text_span_hash)
        )
        """,
        "alter table graph_entity_anchors add column if not exists anchor_id varchar(128)",
        "alter table graph_entity_anchors add column if not exists graph_version varchar(128)",
        "alter table graph_entity_anchors add column if not exists entity_id varchar(128)",
        "alter table graph_entity_anchors add column if not exists chunk_id varchar(64)",
        "alter table graph_entity_anchors add column if not exists text_span text",
        """
        alter table graph_entity_anchors
        add column if not exists text_span_hash varchar(128) not null default 'whole_chunk'
        """,
        """
        alter table graph_entity_anchors
        add column if not exists metadata_json jsonb not null default '{}'::jsonb
        """,
        """
        alter table graph_entity_anchors
        add column if not exists created_at timestamp with time zone not null default now()
        """,
        """
        with entity_versions as (
            select entity_id, min(graph_version) as graph_version
            from graph_entities
            group by entity_id
            having count(*) = 1
        )
        update graph_entity_anchors anchor
        set graph_version = entity_versions.graph_version
        from entity_versions
        where anchor.graph_version is null
          and anchor.entity_id = entity_versions.entity_id
        """,
        """
        update graph_entity_anchors
        set text_span_hash = 'whole_chunk'
        where text_span_hash is null
        """,
        """
        do $$
        begin
            if not exists (
                select 1 from graph_entity_anchors where graph_version is null
            ) then
                alter table graph_entity_anchors alter column graph_version set not null;
            end if;
        end $$;
        """,
        _not_null_columns_sql(
            "graph_entity_anchors",
            ("entity_id", "chunk_id"),
            "Cannot upgrade legacy graph schema: graph_entity_anchors "
            "entity_id/chunk_id contain null values",
        ),
        "alter table graph_entity_anchors alter column text_span_hash set default 'whole_chunk'",
        "alter table graph_entity_anchors alter column text_span_hash set not null",
        "update graph_entity_anchors set metadata_json = '{}'::jsonb where metadata_json is null",
        "update graph_entity_anchors set created_at = now() where created_at is null",
        "alter table graph_entity_anchors alter column metadata_json set default '{}'::jsonb",
        "alter table graph_entity_anchors alter column metadata_json set not null",
        "alter table graph_entity_anchors alter column created_at set default now()",
        "alter table graph_entity_anchors alter column created_at set not null",
        _drop_legacy_unique_constraint_sql(
            "graph_entity_anchors",
            "uq_graph_entity_anchors_entity_chunk_span",
            ("entity_id", "chunk_id", "text_span_hash"),
        ),
        _drop_fk_constraints_sql(
            "graph_entity_anchors",
            (
                "graph_entity_anchors_chunk_id_fkey",
                "fk_graph_entity_anchors_chunk",
            ),
        ),
        _primary_key_constraint_sql(
            "graph_entity_anchors",
            "pk_graph_entity_anchors",
            ("graph_version", "anchor_id"),
            require_name=True,
        ),
        _foreign_key_constraint_sql(
            "graph_entity_anchors",
            "fk_graph_entity_anchors_entity",
            ("graph_version", "entity_id"),
            "graph_entities",
            ("graph_version", "entity_id"),
        ),
        _foreign_key_constraint_sql(
            "graph_entity_anchors",
            "fk_graph_entity_anchors_chunk",
            ("chunk_id",),
            "chunks",
            ("chunk_id",),
        ),
        _unique_constraint_sql(
            "graph_entity_anchors",
            "uq_graph_entity_anchors_entity_chunk_span",
            ("graph_version", "entity_id", "chunk_id", "text_span_hash"),
        ),
        """
        create index if not exists ix_graph_entity_anchors_graph_version
        on graph_entity_anchors (graph_version)
        """,
        """
        create index if not exists ix_graph_entity_anchors_entity_id
        on graph_entity_anchors (entity_id)
        """,
        """
        create index if not exists ix_graph_entity_anchors_graph_entity
        on graph_entity_anchors (graph_version, entity_id)
        """,
        """
        create index if not exists ix_graph_entity_anchors_chunk_id
        on graph_entity_anchors (chunk_id)
        """,
        """
        create table if not exists graph_relationship_anchors (
            graph_version varchar(128) not null,
            anchor_id varchar(128) not null,
            relationship_id varchar(128) not null,
            chunk_id varchar(64) not null
                constraint fk_graph_relationship_anchors_chunk
                references chunks(chunk_id) on delete cascade,
            text_span text,
            text_span_hash varchar(128) not null default 'whole_chunk',
            metadata_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now(),
            constraint fk_graph_relationship_anchors_relationship
                foreign key (graph_version, relationship_id)
                references graph_relationships(graph_version, relationship_id)
                on delete cascade,
            constraint pk_graph_relationship_anchors primary key (graph_version, anchor_id),
            constraint uq_graph_relationship_anchors_relationship_chunk_span
                unique (graph_version, relationship_id, chunk_id, text_span_hash)
        )
        """,
        """
        alter table graph_relationship_anchors
        add column if not exists anchor_id varchar(128)
        """,
        "alter table graph_relationship_anchors add column if not exists graph_version varchar(128)",
        """
        alter table graph_relationship_anchors
        add column if not exists relationship_id varchar(128)
        """,
        "alter table graph_relationship_anchors add column if not exists chunk_id varchar(64)",
        "alter table graph_relationship_anchors add column if not exists text_span text",
        """
        alter table graph_relationship_anchors
        add column if not exists text_span_hash varchar(128) not null default 'whole_chunk'
        """,
        """
        alter table graph_relationship_anchors
        add column if not exists metadata_json jsonb not null default '{}'::jsonb
        """,
        """
        alter table graph_relationship_anchors
        add column if not exists created_at timestamp with time zone not null default now()
        """,
        """
        with relationship_versions as (
            select relationship_id, min(graph_version) as graph_version
            from graph_relationships
            group by relationship_id
            having count(*) = 1
        )
        update graph_relationship_anchors anchor
        set graph_version = relationship_versions.graph_version
        from relationship_versions
        where anchor.graph_version is null
          and anchor.relationship_id = relationship_versions.relationship_id
        """,
        """
        update graph_relationship_anchors
        set text_span_hash = 'whole_chunk'
        where text_span_hash is null
        """,
        """
        do $$
        begin
            if not exists (
                select 1 from graph_relationship_anchors where graph_version is null
            ) then
                alter table graph_relationship_anchors
                alter column graph_version set not null;
            end if;
        end $$;
        """,
        _not_null_columns_sql(
            "graph_relationship_anchors",
            ("relationship_id", "chunk_id"),
            "Cannot upgrade legacy graph schema: graph_relationship_anchors "
            "relationship_id/chunk_id contain null values",
        ),
        """
        alter table graph_relationship_anchors
        alter column text_span_hash set default 'whole_chunk'
        """,
        "alter table graph_relationship_anchors alter column text_span_hash set not null",
        """
        update graph_relationship_anchors
        set metadata_json = '{}'::jsonb
        where metadata_json is null
        """,
        "update graph_relationship_anchors set created_at = now() where created_at is null",
        "alter table graph_relationship_anchors alter column metadata_json set default '{}'::jsonb",
        "alter table graph_relationship_anchors alter column metadata_json set not null",
        "alter table graph_relationship_anchors alter column created_at set default now()",
        "alter table graph_relationship_anchors alter column created_at set not null",
        _drop_legacy_unique_constraint_sql(
            "graph_relationship_anchors",
            "uq_graph_relationship_anchors_relationship_chunk_span",
            ("relationship_id", "chunk_id", "text_span_hash"),
        ),
        _drop_fk_constraints_sql(
            "graph_relationship_anchors",
            (
                "graph_relationship_anchors_chunk_id_fkey",
                "fk_graph_relationship_anchors_chunk",
            ),
        ),
        _primary_key_constraint_sql(
            "graph_relationship_anchors",
            "pk_graph_relationship_anchors",
            ("graph_version", "anchor_id"),
            require_name=True,
        ),
        _foreign_key_constraint_sql(
            "graph_relationship_anchors",
            "fk_graph_relationship_anchors_relationship",
            ("graph_version", "relationship_id"),
            "graph_relationships",
            ("graph_version", "relationship_id"),
        ),
        _foreign_key_constraint_sql(
            "graph_relationship_anchors",
            "fk_graph_relationship_anchors_chunk",
            ("chunk_id",),
            "chunks",
            ("chunk_id",),
        ),
        _unique_constraint_sql(
            "graph_relationship_anchors",
            "uq_graph_relationship_anchors_relationship_chunk_span",
            ("graph_version", "relationship_id", "chunk_id", "text_span_hash"),
        ),
        """
        create index if not exists ix_graph_relationship_anchors_graph_version
        on graph_relationship_anchors (graph_version)
        """,
        """
        create index if not exists ix_graph_relationship_anchors_relationship_id
        on graph_relationship_anchors (relationship_id)
        """,
        """
        create index if not exists ix_graph_relationship_anchors_graph_relationship
        on graph_relationship_anchors (graph_version, relationship_id)
        """,
        """
        create index if not exists ix_graph_relationship_anchors_chunk_id
        on graph_relationship_anchors (chunk_id)
        """,
        """
        create table if not exists graph_communities (
            graph_version varchar(128) not null
                constraint fk_graph_communities_graph_version
                references graph_indexes(graph_version) on delete cascade,
            community_id varchar(128) not null,
            level integer not null,
            summary text,
            metadata_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now(),
            constraint pk_graph_communities primary key (graph_version, community_id)
        )
        """,
        "alter table graph_communities add column if not exists graph_version varchar(128)",
        "alter table graph_communities add column if not exists community_id varchar(128)",
        "alter table graph_communities add column if not exists level integer",
        "alter table graph_communities add column if not exists summary text",
        """
        alter table graph_communities
        add column if not exists metadata_json jsonb not null default '{}'::jsonb
        """,
        """
        alter table graph_communities
        add column if not exists created_at timestamp with time zone not null default now()
        """,
        "update graph_communities set metadata_json = '{}'::jsonb where metadata_json is null",
        "update graph_communities set created_at = now() where created_at is null",
        "alter table graph_communities alter column metadata_json set default '{}'::jsonb",
        "alter table graph_communities alter column metadata_json set not null",
        "alter table graph_communities alter column created_at set default now()",
        "alter table graph_communities alter column created_at set not null",
        _primary_key_constraint_sql(
            "graph_communities",
            "pk_graph_communities",
            ("graph_version", "community_id"),
        ),
        _foreign_key_constraint_sql(
            "graph_communities",
            "fk_graph_communities_graph_version",
            ("graph_version",),
            "graph_indexes",
            ("graph_version",),
        ),
        """
        create index if not exists ix_graph_communities_graph_version
        on graph_communities (graph_version)
        """,
        """
        create index if not exists ix_graph_communities_graph_level
        on graph_communities (graph_version, level)
        """,
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
        """
        create table if not exists query_plans (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            plan_id varchar(64),
            planner varchar(128),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "create index if not exists ix_query_plans_query_id on query_plans (query_id)",
        """
        create table if not exists retrieval_tasks (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            task_id varchar(64),
            unit_id varchar(64),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "create index if not exists ix_retrieval_tasks_query_id on retrieval_tasks (query_id)",
        """
        create table if not exists retrieval_results (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            status varchar(64),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "create index if not exists ix_retrieval_results_query_id on retrieval_results (query_id)",
        """
        create table if not exists candidates (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            chunk_id varchar(64),
            rank integer,
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "create index if not exists ix_candidates_query_id on candidates (query_id)",
        """
        create table if not exists evidence_blocks (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            evidence_id varchar(64),
            chunk_id varchar(64),
            rank integer,
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "create index if not exists ix_evidence_blocks_query_id on evidence_blocks (query_id)",
        """
        create table if not exists evidence_packs (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            pack_id varchar(64),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "create index if not exists ix_evidence_packs_query_id on evidence_packs (query_id)",
        """
        create table if not exists evidence_evaluations (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            status varchar(64),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "create index if not exists ix_evidence_evaluations_query_id on evidence_evaluations (query_id)",
        """
        create table if not exists answers (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            confidence varchar(32),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "create index if not exists ix_answers_query_id on answers (query_id)",
        """
        create table if not exists citations (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            citation_id varchar(64),
            evidence_id varchar(64),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "create index if not exists ix_citations_query_id on citations (query_id)",
        """
        create table if not exists citation_verifications (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            status varchar(64),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now()
        )
        """,
        "create index if not exists ix_citation_verifications_query_id on citation_verifications (query_id)",
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

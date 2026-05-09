from collections.abc import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from atlas.core.config import get_settings
from atlas.db.models import (
    CITATION_AUDIT_SNAPSHOT_GOVERNANCE_CHECK,
    LLM_CALL_EVIDENCE_SNAPSHOT_GOVERNANCE_CHECK,
    LLM_CALL_RAW_GOVERNANCE_CHECK,
    QUALITY_REVIEW_PAYLOAD_GOVERNANCE_CHECK,
    Base,
)

settings = get_settings()

engine = create_engine(settings.database_url, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)
async_engine = create_async_engine(settings.database_url, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    autoflush=False,
    expire_on_commit=False,
)


def init_db() -> None:
    _apply_pre_create_observability_migrations()
    Base.metadata.create_all(bind=engine)
    _apply_lightweight_migrations()


def _sql_text_array(values: tuple[str, ...]) -> str:
    return ", ".join(f"'{value}'" for value in values)


def _add_column_if_table_exists_sql(table: str, column_definition: str) -> str:
    return f"""
        do $$
        begin
            if to_regclass('{table}') is not null then
                alter table {table} add column if not exists {column_definition};
            end if;
        end $$;
        """


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
            target_table regclass := to_regclass('{table}');
            expected_columns text[] := array[{expected_columns}]::text[];
            duplicate_constraint record;
            duplicate_index record;
            dependent_fk_names text;
            existing_columns text[];
            existing_constraint_oid oid;
            existing_constraint_index_oid oid;
        begin
            if target_table is null then
                return;
            end if;

            for duplicate_constraint in
                select
                    con.conname,
                    (
                        select string_agg(
                            format('%s.%I', fk.conrelid::regclass::text, fk.conname),
                            ', '
                            order by fk.conrelid::regclass::text, fk.conname
                        )
                        from pg_constraint fk
                        where fk.contype = 'f'
                          and fk.confrelid = target_table
                          and (
                              exists (
                                  select 1
                                  from pg_depend dep
                                  where dep.classid = 'pg_constraint'::regclass
                                    and dep.objid = fk.oid
                                    and (
                                        (
                                            dep.refclassid = 'pg_constraint'::regclass
                                            and dep.refobjid = con.oid
                                        )
                                        or (
                                            dep.refclassid = 'pg_class'::regclass
                                            and dep.refobjid = con.conindid
                                        )
                                    )
                              )
                              or (
                                  select array_agg(
                                      target_att.attname::text
                                      order by target_keys.ordinality
                                  )
                                  from unnest(fk.confkey) with ordinality
                                      as target_keys(attnum, ordinality)
                                  join pg_attribute target_att
                                    on target_att.attrelid = fk.confrelid
                                   and target_att.attnum = target_keys.attnum
                              ) = expected_columns
                          )
                    ) as dependent_fk_names
                from pg_constraint con
                where con.conrelid = target_table
                  and con.contype = 'u'
                  and con.conname <> '{constraint}'
                  and (
                      select array_agg(att.attname::text order by keys.ordinality)
                      from unnest(con.conkey) with ordinality as keys(attnum, ordinality)
                      join pg_attribute att
                        on att.attrelid = con.conrelid
                       and att.attnum = keys.attnum
                  ) = expected_columns
            loop
                if duplicate_constraint.dependent_fk_names is null then
                    execute format(
                        'alter table %s drop constraint %I',
                        target_table,
                        duplicate_constraint.conname
                    );
                end if;
            end loop;

            for duplicate_index in
                select
                    idx.oid::regclass::text as index_name,
                    (
                        select string_agg(
                            format('%s.%I', fk.conrelid::regclass::text, fk.conname),
                            ', '
                            order by fk.conrelid::regclass::text, fk.conname
                        )
                        from pg_constraint fk
                        where fk.contype = 'f'
                          and fk.confrelid = target_table
                          and (
                              exists (
                                  select 1
                                  from pg_depend dep
                                  where dep.classid = 'pg_constraint'::regclass
                                    and dep.objid = fk.oid
                                    and dep.refclassid = 'pg_class'::regclass
                                    and dep.refobjid = idx.oid
                              )
                              or (
                                  select array_agg(
                                      target_att.attname::text
                                      order by target_keys.ordinality
                                  )
                                  from unnest(fk.confkey) with ordinality
                                      as target_keys(attnum, ordinality)
                                  join pg_attribute target_att
                                    on target_att.attrelid = fk.confrelid
                                   and target_att.attnum = target_keys.attnum
                              ) = expected_columns
                          )
                    ) as dependent_fk_names
                from pg_class idx
                join pg_index index_info on index_info.indexrelid = idx.oid
                where index_info.indrelid = target_table
                  and index_info.indisunique
                  and not index_info.indisprimary
                  and index_info.indpred is null
                  and index_info.indexprs is null
                  and idx.relname <> '{constraint}'
                  and not exists (
                      select 1
                      from pg_constraint con
                      where con.conindid = idx.oid
                  )
                  and (
                      select array_agg(att.attname::text order by keys.ordinality)
                      from unnest(index_info.indkey) with ordinality as keys(attnum, ordinality)
                      join pg_attribute att
                        on att.attrelid = index_info.indrelid
                       and att.attnum = keys.attnum
                      where keys.ordinality <= index_info.indnkeyatts
                  ) = expected_columns
            loop
                if duplicate_index.dependent_fk_names is null then
                    execute format('drop index %s', duplicate_index.index_name);
                end if;
            end loop;

            select
                array_agg(att.attname::text order by keys.ordinality),
                con.oid,
                con.conindid
            into
                existing_columns,
                existing_constraint_oid,
                existing_constraint_index_oid
            from pg_constraint con
            join unnest(con.conkey) with ordinality as keys(attnum, ordinality) on true
            join pg_attribute att
              on att.attrelid = con.conrelid
             and att.attnum = keys.attnum
            where con.conrelid = target_table
              and con.contype = 'u'
              and con.conname = '{constraint}'
            group by con.conname, con.oid, con.conindid;

            if existing_columns is distinct from expected_columns then
                if existing_columns is not null then
                    select string_agg(
                        format('%s.%I', fk.conrelid::regclass::text, fk.conname),
                        ', '
                        order by fk.conrelid::regclass::text, fk.conname
                    )
                    into dependent_fk_names
                    from pg_constraint fk
                    where fk.contype = 'f'
                      and fk.confrelid = target_table
                      and (
                          exists (
                              select 1
                              from pg_depend dep
                              where dep.classid = 'pg_constraint'::regclass
                                and dep.objid = fk.oid
                                and (
                                    (
                                        dep.refclassid = 'pg_constraint'::regclass
                                        and dep.refobjid = existing_constraint_oid
                                    )
                                    or (
                                        dep.refclassid = 'pg_class'::regclass
                                        and dep.refobjid = existing_constraint_index_oid
                                    )
                                )
                          )
                          or (
                              select array_agg(
                                  target_att.attname::text
                                  order by target_keys.ordinality
                              )
                              from unnest(fk.confkey) with ordinality
                                  as target_keys(attnum, ordinality)
                              join pg_attribute target_att
                                on target_att.attrelid = fk.confrelid
                               and target_att.attnum = target_keys.attnum
                          ) = existing_columns
                      );

                    if dependent_fk_names is not null then
                        raise exception
                            'Cannot rebuild unique constraint {constraint} on {table}: existing columns % are referenced by foreign keys: %',
                            existing_columns,
                            dependent_fk_names;
                    end if;

                    execute format('alter table %s drop constraint %I', target_table, '{constraint}');
                end if;

                for duplicate_index in
                    select
                        idx.oid::regclass::text as index_name,
                        (
                            select array_agg(att.attname::text order by keys.ordinality)
                            from unnest(index_info.indkey) with ordinality
                                as keys(attnum, ordinality)
                            join pg_attribute att
                              on att.attrelid = index_info.indrelid
                             and att.attnum = keys.attnum
                            where keys.ordinality <= index_info.indnkeyatts
                        ) as index_columns,
                        index_info.indpred is null
                            and index_info.indexprs is null as is_plain_index,
                        (
                            select string_agg(
                                format('%s.%I', fk.conrelid::regclass::text, fk.conname),
                                ', '
                                order by fk.conrelid::regclass::text, fk.conname
                            )
                            from pg_constraint fk
                            where fk.contype = 'f'
                              and fk.confrelid = target_table
                              and (
                                  exists (
                                      select 1
                                      from pg_depend dep
                                      where dep.classid = 'pg_constraint'::regclass
                                        and dep.objid = fk.oid
                                        and dep.refclassid = 'pg_class'::regclass
                                        and dep.refobjid = idx.oid
                                  )
                                  or (
                                      select array_agg(
                                          target_att.attname::text
                                          order by target_keys.ordinality
                                      )
                                      from unnest(fk.confkey) with ordinality
                                          as target_keys(attnum, ordinality)
                                      join pg_attribute target_att
                                        on target_att.attrelid = fk.confrelid
                                       and target_att.attnum = target_keys.attnum
                                  ) = (
                                      select array_agg(att.attname::text order by keys.ordinality)
                                      from unnest(index_info.indkey) with ordinality
                                          as keys(attnum, ordinality)
                                      join pg_attribute att
                                        on att.attrelid = index_info.indrelid
                                       and att.attnum = keys.attnum
                                      where keys.ordinality <= index_info.indnkeyatts
                                  )
                              )
                        ) as dependent_fk_names
                    from pg_class idx
                    join pg_index index_info on index_info.indexrelid = idx.oid
                    where index_info.indrelid = target_table
                      and index_info.indisunique
                      and not index_info.indisprimary
                      and idx.relname = '{constraint}'
                      and not exists (
                          select 1
                          from pg_constraint con
                          where con.conindid = idx.oid
                      )
                loop
                    if not duplicate_index.is_plain_index then
                        raise exception
                            'Cannot rebuild unique constraint {constraint} on {table}: existing index % is partial or expression and cannot be reused automatically',
                            duplicate_index.index_name;
                    end if;

                    if duplicate_index.index_columns = expected_columns
                    then
                        execute format(
                            'alter table %s add constraint %I unique using index %s',
                            target_table,
                            '{constraint}',
                            duplicate_index.index_name
                        );
                        return;
                    end if;

                    if duplicate_index.dependent_fk_names is not null then
                        raise exception
                            'Cannot rebuild unique constraint {constraint} on {table}: existing index columns % are referenced by foreign keys: %',
                            duplicate_index.index_columns,
                            duplicate_index.dependent_fk_names;
                    end if;

                    execute format('drop index %s', duplicate_index.index_name);
                end loop;

                execute format(
                    'alter table %s add constraint %I unique ({column_list})',
                    target_table,
                    '{constraint}'
                );
            end if;
        end $$;
        """


def _foreign_key_constraint_sql(
    table: str,
    constraint: str,
    columns: tuple[str, ...],
    target_table: str,
    target_columns: tuple[str, ...],
    *,
    on_delete: str = "cascade",
) -> str:
    normalized_on_delete = on_delete.lower().replace("_", " ")
    if normalized_on_delete == "cascade":
        delete_action = "c"
        delete_clause = "on delete cascade"
    elif normalized_on_delete == "set null":
        delete_action = "n"
        delete_clause = "on delete set null"
    elif normalized_on_delete == "no action":
        delete_action = "a"
        delete_clause = ""
    else:
        raise ValueError(f"Unsupported FK on_delete action: {on_delete}")

    column_list = ", ".join(columns)
    target_column_list = ", ".join(target_columns)
    expected_columns = _sql_text_array(columns)
    expected_target_columns = _sql_text_array(target_columns)
    return f"""
        do $$
        declare
            child_table regclass := to_regclass('{table}');
            parent_table regclass := to_regclass('{target_table}');
            duplicate_constraint_name text;
            existing_columns text[];
            existing_target_table regclass;
            existing_target_columns text[];
            existing_delete_action "char";
        begin
            if child_table is null or parent_table is null then
                return;
            end if;

            for duplicate_constraint_name in
                select con.conname
                from pg_constraint con
                where con.conrelid = child_table
                  and con.contype = 'f'
                  and con.confrelid = parent_table
                  and con.conname <> '{constraint}'
                  and (
                      select array_agg(att.attname::text order by keys.ordinality)
                      from unnest(con.conkey) with ordinality as keys(attnum, ordinality)
                      join pg_attribute att
                        on att.attrelid = con.conrelid
                       and att.attnum = keys.attnum
                  ) = array[{expected_columns}]::text[]
                  and (
                      select array_agg(target_att.attname::text order by target_keys.ordinality)
                      from unnest(con.confkey) with ordinality as target_keys(attnum, ordinality)
                      join pg_attribute target_att
                        on target_att.attrelid = con.confrelid
                       and target_att.attnum = target_keys.attnum
                  ) = array[{expected_target_columns}]::text[]
            loop
                execute format(
                    'alter table %s drop constraint %I',
                    child_table,
                    duplicate_constraint_name
                );
            end loop;

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
            where con.conrelid = child_table
              and con.contype = 'f'
              and con.conname = '{constraint}'
            group by con.conname, con.confrelid, con.confdeltype;

            if existing_columns is distinct from array[{expected_columns}]::text[]
                or existing_target_table is distinct from parent_table
                or existing_target_columns is distinct from array[{expected_target_columns}]::text[]
                or existing_delete_action is distinct from '{delete_action}' then
                if existing_columns is not null then
                    execute format('alter table %s drop constraint %I', child_table, '{constraint}');
                end if;

                alter table {table}
                add constraint {constraint}
                foreign key ({column_list})
                references {target_table}({target_column_list})
                {delete_clause};
            end if;
        end $$;
        """


def _nullable_composite_fk_preflight_sql(
    table: str,
    constraint: str,
    columns: tuple[str, ...],
    target_table: str,
    target_columns: tuple[str, ...],
) -> str:
    child_columns_present = " and ".join(f"child.{column} is not null" for column in columns)
    join_predicate = " and ".join(
        f"target.{target_column} = child.{column}"
        for column, target_column in zip(columns, target_columns, strict=True)
    )
    target_null_check = f"target.{target_columns[0]} is null"
    target_signature = ", ".join(target_columns)
    return f"""
        do $$
        begin
            if to_regclass('{table}') is null
                or to_regclass('{target_table}') is null then
                return;
            end if;

            if exists (
                select 1
                from {table} child
                left join {target_table} target
                  on {join_predicate}
                where {child_columns_present}
                  and {target_null_check}
            ) then
                raise exception
                    'Cannot upgrade {table} schema: existing rows violate {constraint}; nullable provenance columns must either be null or match {target_table}({target_signature})';
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


def _check_constraint_sql(table: str, constraint: str, expression: str) -> str:
    return f"""
        do $$
        begin
            if to_regclass('{table}') is null then
                return;
            end if;

            if exists (select 1 from {table} where not ({expression})) then
                raise exception
                    'Cannot upgrade {table} schema: existing rows violate {constraint}';
            end if;

            if exists (
                select 1
                from pg_constraint
                where conrelid = '{table}'::regclass
                  and conname = '{constraint}'
                  and contype = 'c'
            ) then
                alter table {table} drop constraint {constraint};
            end if;

            alter table {table}
            add constraint {constraint}
            check ({expression});
        end $$;
        """


def _llm_call_evidence_rank_backfill_sql() -> str:
    return """
        do $$
        begin
            if to_regclass('llm_call_evidence') is null then
                return;
            end if;

            if not exists (
                select 1
                from pg_attribute
                where attrelid = 'llm_call_evidence'::regclass
                  and attname = 'rank'
                  and not attisdropped
            ) then
                return;
            end if;

            if exists (
                select 1
                from (
                    select call_id, rank
                    from llm_call_evidence
                    where rank is not null
                    group by call_id, rank
                    having count(*) > 1
                ) duplicate_ranks
            ) then
                raise exception
                    'Cannot upgrade llm_call_evidence schema: duplicate non-null call_id/rank values prevent uq_llm_call_evidence_call_rank';
            end if;

            with rank_max as (
                select call_id, max(rank) as max_rank
                from llm_call_evidence
                where rank is not null
                group by call_id
            ),
            ranked_evidence as (
                select
                    evidence.record_id,
                    coalesce(rank_max.max_rank, 0)
                        + row_number() over (
                            partition by evidence.call_id
                            order by evidence.created_at, evidence.record_id
                        ) as backfilled_rank
                from llm_call_evidence evidence
                left join rank_max
                  on rank_max.call_id = evidence.call_id
                where evidence.rank is null
            )
            update llm_call_evidence evidence
            set rank = ranked_evidence.backfilled_rank
            from ranked_evidence
            where evidence.record_id = ranked_evidence.record_id;
        end $$;
        """


def _llm_call_evidence_query_id_backfill_sql() -> str:
    return """
        do $$
        begin
            if to_regclass('llm_call_evidence') is null
                or to_regclass('llm_calls') is null then
                return;
            end if;

            if not exists (
                select 1
                from pg_attribute
                where attrelid = 'llm_call_evidence'::regclass
                  and attname = 'call_id'
                  and not attisdropped
            ) or not exists (
                select 1
                from pg_attribute
                where attrelid = 'llm_call_evidence'::regclass
                  and attname = 'query_id'
                  and not attisdropped
            ) or not exists (
                select 1
                from pg_attribute
                where attrelid = 'llm_calls'::regclass
                  and attname = 'call_id'
                  and not attisdropped
            ) or not exists (
                select 1
                from pg_attribute
                where attrelid = 'llm_calls'::regclass
                  and attname = 'query_id'
                  and not attisdropped
            ) then
                return;
            end if;

            update llm_call_evidence evidence
            set query_id = calls.query_id
            from llm_calls calls
            where evidence.query_id is null
              and evidence.call_id = calls.call_id
              and calls.query_id is not null;

            if exists (
                select 1
                from llm_call_evidence
                where query_id is null
            ) then
                raise exception
                    'Cannot upgrade llm_call_evidence schema: query_id could not be backfilled from llm_calls.query_id';
            end if;

            if exists (
                select 1
                from llm_call_evidence evidence
                left join llm_calls calls
                  on calls.call_id = evidence.call_id
                 and calls.query_id = evidence.query_id
                where evidence.call_id is not null
                  and evidence.query_id is not null
                  and calls.call_id is null
            ) then
                raise exception
                    'Cannot upgrade llm_call_evidence schema: call_id/query_id values do not match llm_calls';
            end if;
        end $$;
        """


def _llm_call_evidence_block_backfill_sql() -> str:
    return """
        do $$
        begin
            if to_regclass('llm_call_evidence') is null
                or to_regclass('evidence_blocks') is null then
                return;
            end if;

            with unique_blocks as (
                select query_id, evidence_id, min(record_id) as record_id
                from evidence_blocks
                where evidence_id is not null
                group by query_id, evidence_id
                having count(*) = 1
            )
            update llm_call_evidence evidence
            set evidence_block_record_id = block.record_id
            from unique_blocks block
            where evidence.evidence_block_record_id is null
              and evidence.query_id = block.query_id
              and evidence.evidence_id = block.evidence_id;
        end $$;
        """


def _evidence_block_evidence_id_disambiguation_sql() -> str:
    return """
        do $$
        begin
            if to_regclass('evidence_blocks') is null then
                return;
            end if;

            with duplicate_blocks as (
                select
                    record_id,
                    left(evidence_id, 48) || '_' || substr(md5(record_id), 1, 15)
                        as disambiguated_evidence_id,
                    count(*) over (partition by query_id, evidence_id) as duplicate_count
                from evidence_blocks
                where record_id is not null
                  and query_id is not null
                  and evidence_id is not null
            )
            update evidence_blocks block
            set evidence_id = duplicate_blocks.disambiguated_evidence_id
            from duplicate_blocks
            where block.record_id = duplicate_blocks.record_id
              and duplicate_blocks.duplicate_count > 1;
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


def _apply_pre_create_observability_migrations() -> None:
    statements = [
        _add_column_if_table_exists_sql("answers", "query_id varchar(64)"),
        _add_column_if_table_exists_sql("answers", "answer_call_id varchar(64)"),
        _drop_legacy_foreign_key_constraints_sql(
            "answers",
            ("answer_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _unique_constraint_sql("answers", "uq_answers_record_query", ("record_id", "query_id")),
        _add_column_if_table_exists_sql("citations", "query_id varchar(64)"),
        _unique_constraint_sql("citations", "uq_citations_record_query", ("record_id", "query_id")),
        _add_column_if_table_exists_sql("llm_calls", "call_id varchar(64)"),
        _add_column_if_table_exists_sql("llm_calls", "query_id varchar(64)"),
        _add_column_if_table_exists_sql("query_plans", "query_id varchar(64)"),
        _add_column_if_table_exists_sql("query_plans", "planner_call_id varchar(64)"),
        _drop_legacy_foreign_key_constraints_sql(
            "query_plans",
            ("planner_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _unique_constraint_sql(
            "query_plans",
            "uq_query_plans_record_query",
            ("record_id", "query_id"),
        ),
        _add_column_if_table_exists_sql("evidence_blocks", "query_id varchar(64)"),
        _add_column_if_table_exists_sql("evidence_blocks", "evidence_id varchar(64)"),
        _unique_constraint_sql(
            "evidence_blocks",
            "uq_evidence_blocks_record_query",
            ("record_id", "query_id"),
        ),
        _evidence_block_evidence_id_disambiguation_sql(),
        _unique_constraint_sql(
            "evidence_blocks",
            "uq_evidence_blocks_evidence_query",
            ("evidence_id", "query_id"),
        ),
        _add_column_if_table_exists_sql("citation_verifications", "query_id varchar(64)"),
        _unique_constraint_sql(
            "citation_verifications",
            "uq_citation_verifications_record_query",
            ("record_id", "query_id"),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "llm_call_evidence",
            ("call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("planner_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("answer_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("answer_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("review_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _unique_constraint_sql("llm_calls", "uq_llm_calls_call_query", ("call_id", "query_id")),
        _add_column_if_table_exists_sql("llm_call_evidence", "record_id varchar(64)"),
        _add_column_if_table_exists_sql("llm_call_evidence", "query_id varchar(64)"),
        _add_column_if_table_exists_sql(
            "llm_call_evidence",
            "evidence_block_record_id varchar(64)",
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "llm_call_evidence",
            ("evidence_block_record_id",),
            "evidence_blocks",
            ("record_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("citation_record_id",),
            "citations",
            ("record_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("llm_call_evidence_record_id",),
            "llm_call_evidence",
            ("record_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("answer_record_id",),
            "answers",
            ("record_id",),
        ),
        _add_column_if_table_exists_sql(
            "citation_audits",
            "citation_verification_record_id varchar(64)",
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("citation_verification_record_id",),
            "citation_verifications",
            ("record_id",),
        ),
        _unique_constraint_sql(
            "llm_call_evidence",
            "uq_llm_call_evidence_record_query",
            ("record_id", "query_id"),
        ),
    ]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


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
        f"""
        create table if not exists llm_calls (
            call_id varchar(64) not null,
            query_id varchar(64) not null,
            stage varchar(32) not null,
            attempt_index integer,
            sequence_index integer,
            model_name varchar(256) not null,
            prompt_version varchar(128),
            planner_version varchar(128),
            status varchar(32) not null,
            error_message text,
            latency_ms integer,
            input_tokens integer,
            output_tokens integer,
            request_json jsonb not null default '{{}}'::jsonb,
            response_json jsonb not null default '{{}}'::jsonb,
            usage_json jsonb not null default '{{}}'::jsonb,
            metadata_json jsonb not null default '{{}}'::jsonb,
            instructions_text text,
            input_text text,
            raw_output_text text,
            parsed_answer_text text,
            parsed_confidence varchar(32),
            parsed_plan_id varchar(64),
            validation_status varchar(64),
            max_output_tokens integer,
            reasoning_effort varchar(64),
            store boolean,
            raw_payload_hash varchar(128),
            raw_redaction_status varchar(64) not null default 'unredacted',
            raw_encryption_status varchar(64) not null default 'plaintext',
            raw_retention_expires_at timestamp with time zone,
            created_at timestamp with time zone not null default now(),
            constraint pk_llm_calls primary key (call_id),
            constraint fk_llm_calls_query
                foreign key (query_id) references query_runs(query_id) on delete cascade,
            constraint uq_llm_calls_call_query unique (call_id, query_id),
            constraint ck_llm_calls_raw_governance
                check {LLM_CALL_RAW_GOVERNANCE_CHECK}
        )
        """,
        "alter table llm_calls add column if not exists call_id varchar(64)",
        "alter table llm_calls add column if not exists query_id varchar(64)",
        "alter table llm_calls add column if not exists stage varchar(32)",
        "alter table llm_calls add column if not exists attempt_index integer",
        "alter table llm_calls add column if not exists sequence_index integer",
        "alter table llm_calls add column if not exists model_name varchar(256)",
        "alter table llm_calls add column if not exists prompt_version varchar(128)",
        "alter table llm_calls add column if not exists planner_version varchar(128)",
        "alter table llm_calls add column if not exists status varchar(32)",
        "alter table llm_calls add column if not exists error_message text",
        "alter table llm_calls add column if not exists latency_ms integer",
        "alter table llm_calls add column if not exists input_tokens integer",
        "alter table llm_calls add column if not exists output_tokens integer",
        "alter table llm_calls add column if not exists request_json jsonb default '{}'::jsonb",
        "alter table llm_calls add column if not exists response_json jsonb default '{}'::jsonb",
        "alter table llm_calls add column if not exists usage_json jsonb default '{}'::jsonb",
        "alter table llm_calls add column if not exists metadata_json jsonb default '{}'::jsonb",
        "alter table llm_calls add column if not exists instructions_text text",
        "alter table llm_calls add column if not exists input_text text",
        "alter table llm_calls add column if not exists raw_output_text text",
        "alter table llm_calls add column if not exists parsed_answer_text text",
        "alter table llm_calls add column if not exists parsed_confidence varchar(32)",
        "alter table llm_calls add column if not exists parsed_plan_id varchar(64)",
        "alter table llm_calls add column if not exists validation_status varchar(64)",
        "alter table llm_calls add column if not exists max_output_tokens integer",
        "alter table llm_calls add column if not exists reasoning_effort varchar(64)",
        "alter table llm_calls add column if not exists store boolean",
        "alter table llm_calls add column if not exists raw_payload_hash varchar(128)",
        """
        alter table llm_calls
        add column if not exists raw_redaction_status varchar(64) default 'unredacted'
        """,
        """
        alter table llm_calls
        add column if not exists raw_encryption_status varchar(64) default 'plaintext'
        """,
        """
        alter table llm_calls
        add column if not exists raw_retention_expires_at timestamp with time zone
        """,
        """
        alter table llm_calls
        add column if not exists created_at timestamp with time zone default now()
        """,
        """
        update llm_calls
        set request_json = '{}'::jsonb
        where request_json is null
        """,
        """
        update llm_calls
        set response_json = '{}'::jsonb
        where response_json is null
        """,
        """
        update llm_calls
        set usage_json = '{}'::jsonb
        where usage_json is null
        """,
        """
        update llm_calls
        set metadata_json = '{}'::jsonb
        where metadata_json is null
        """,
        """
        update llm_calls
        set raw_redaction_status = 'unredacted'
        where nullif(btrim(raw_redaction_status), '') is null
           or lower(raw_redaction_status) = 'unknown'
        """,
        """
        update llm_calls
        set raw_encryption_status = 'plaintext'
        where nullif(btrim(raw_encryption_status), '') is null
           or lower(raw_encryption_status) = 'unknown'
        """,
        "alter table llm_calls alter column request_json set default '{}'::jsonb",
        "alter table llm_calls alter column response_json set default '{}'::jsonb",
        "alter table llm_calls alter column usage_json set default '{}'::jsonb",
        "alter table llm_calls alter column metadata_json set default '{}'::jsonb",
        "alter table llm_calls alter column raw_redaction_status set default 'unredacted'",
        "alter table llm_calls alter column raw_encryption_status set default 'plaintext'",
        "update llm_calls set created_at = now() where created_at is null",
        "alter table llm_calls alter column created_at set default now()",
        "alter table llm_calls alter column created_at set not null",
        _drop_legacy_foreign_key_constraints_sql(
            "llm_call_evidence",
            ("call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("planner_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("answer_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("answer_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("review_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _primary_key_constraint_sql(
            "llm_calls",
            "pk_llm_calls",
            ("call_id",),
            require_name=True,
        ),
        _foreign_key_constraint_sql(
            "llm_calls",
            "fk_llm_calls_query",
            ("query_id",),
            "query_runs",
            ("query_id",),
        ),
        _unique_constraint_sql("llm_calls", "uq_llm_calls_call_query", ("call_id", "query_id")),
        _not_null_columns_sql(
            "llm_calls",
            (
                "call_id",
                "query_id",
                "stage",
                "model_name",
                "status",
                "request_json",
                "response_json",
                "usage_json",
                "metadata_json",
                "raw_redaction_status",
                "raw_encryption_status",
                "created_at",
            ),
            "Cannot upgrade llm_calls schema: required columns contain null values",
        ),
        _check_constraint_sql(
            "llm_calls",
            "ck_llm_calls_raw_governance",
            LLM_CALL_RAW_GOVERNANCE_CHECK,
        ),
        "create index if not exists ix_llm_calls_query_id on llm_calls (query_id)",
        "create index if not exists ix_llm_calls_stage on llm_calls (stage)",
        """
        create index if not exists ix_llm_calls_query_stage
        on llm_calls (query_id, stage)
        """,
        """
        create index if not exists ix_llm_calls_query_stage_attempt
        on llm_calls (query_id, stage, attempt_index)
        """,
        """
        create index if not exists ix_llm_calls_query_sequence
        on llm_calls (query_id, sequence_index)
        """,
        f"""
        create table if not exists llm_call_evidence (
            record_id varchar(64) not null,
            call_id varchar(64) not null,
            query_id varchar(64) not null,
            evidence_id varchar(64) not null,
            evidence_block_record_id varchar(64),
            rank integer not null,
            provider varchar(64),
            chunk_id varchar(64),
            document_id varchar(64),
            page_start integer,
            page_end integer,
            retrieval_score double precision,
            token_count integer,
            text_snapshot text,
            text_hash varchar(128),
            snapshot_redaction_status varchar(64) not null default 'unredacted',
            snapshot_encryption_status varchar(64) not null default 'plaintext',
            snapshot_retention_expires_at timestamp with time zone,
            created_at timestamp with time zone not null default now(),
            constraint pk_llm_call_evidence primary key (record_id),
            constraint fk_llm_call_evidence_call_query
                foreign key (call_id, query_id)
                references llm_calls(call_id, query_id)
                on delete cascade,
            constraint fk_llm_call_evidence_query
                foreign key (query_id) references query_runs(query_id) on delete cascade,
            constraint fk_llm_call_evidence_evidence_block_query
                foreign key (evidence_block_record_id, query_id)
                references evidence_blocks(record_id, query_id),
            constraint uq_llm_call_evidence_record_query unique (record_id, query_id),
            constraint uq_llm_call_evidence_call_rank unique (call_id, rank),
            constraint ck_llm_call_evidence_snapshot_governance
                check {LLM_CALL_EVIDENCE_SNAPSHOT_GOVERNANCE_CHECK}
        )
        """,
        "alter table llm_call_evidence add column if not exists record_id varchar(64)",
        "alter table llm_call_evidence add column if not exists call_id varchar(64)",
        "alter table llm_call_evidence add column if not exists query_id varchar(64)",
        "alter table llm_call_evidence add column if not exists evidence_id varchar(64)",
        """
        alter table llm_call_evidence
        add column if not exists evidence_block_record_id varchar(64)
        """,
        "alter table llm_call_evidence add column if not exists rank integer",
        "alter table llm_call_evidence add column if not exists provider varchar(64)",
        "alter table llm_call_evidence add column if not exists chunk_id varchar(64)",
        "alter table llm_call_evidence add column if not exists document_id varchar(64)",
        "alter table llm_call_evidence add column if not exists page_start integer",
        "alter table llm_call_evidence add column if not exists page_end integer",
        """
        alter table llm_call_evidence
        add column if not exists retrieval_score double precision
        """,
        "alter table llm_call_evidence add column if not exists token_count integer",
        "alter table llm_call_evidence add column if not exists text_snapshot text",
        "alter table llm_call_evidence add column if not exists text_hash varchar(128)",
        """
        alter table llm_call_evidence
        add column if not exists snapshot_redaction_status varchar(64) default 'unredacted'
        """,
        """
        alter table llm_call_evidence
        add column if not exists snapshot_encryption_status varchar(64) default 'plaintext'
        """,
        """
        alter table llm_call_evidence
        add column if not exists snapshot_retention_expires_at timestamp with time zone
        """,
        """
        alter table llm_call_evidence
        add column if not exists created_at timestamp with time zone default now()
        """,
        """
        update llm_call_evidence
        set snapshot_redaction_status = 'unredacted'
        where nullif(btrim(snapshot_redaction_status), '') is null
           or lower(snapshot_redaction_status) = 'unknown'
        """,
        """
        update llm_call_evidence
        set snapshot_encryption_status = 'plaintext'
        where nullif(btrim(snapshot_encryption_status), '') is null
           or lower(snapshot_encryption_status) = 'unknown'
        """,
        """
        alter table llm_call_evidence
        alter column snapshot_redaction_status set default 'unredacted'
        """,
        """
        alter table llm_call_evidence
        alter column snapshot_encryption_status set default 'plaintext'
        """,
        "update llm_call_evidence set created_at = now() where created_at is null",
        "alter table llm_call_evidence alter column created_at set default now()",
        "alter table llm_call_evidence alter column created_at set not null",
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("llm_call_evidence_record_id",),
            "llm_call_evidence",
            ("record_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "llm_call_evidence",
            ("evidence_block_record_id",),
            "evidence_blocks",
            ("record_id",),
        ),
        _primary_key_constraint_sql(
            "llm_call_evidence",
            "pk_llm_call_evidence",
            ("record_id",),
            require_name=True,
        ),
        _llm_call_evidence_query_id_backfill_sql(),
        _llm_call_evidence_block_backfill_sql(),
        _foreign_key_constraint_sql(
            "llm_call_evidence",
            "fk_llm_call_evidence_call_query",
            ("call_id", "query_id"),
            "llm_calls",
            ("call_id", "query_id"),
        ),
        _foreign_key_constraint_sql(
            "llm_call_evidence",
            "fk_llm_call_evidence_query",
            ("query_id",),
            "query_runs",
            ("query_id",),
        ),
        _unique_constraint_sql(
            "evidence_blocks",
            "uq_evidence_blocks_record_query",
            ("record_id", "query_id"),
        ),
        _nullable_composite_fk_preflight_sql(
            "llm_call_evidence",
            "fk_llm_call_evidence_evidence_block_query",
            ("evidence_block_record_id", "query_id"),
            "evidence_blocks",
            ("record_id", "query_id"),
        ),
        _foreign_key_constraint_sql(
            "llm_call_evidence",
            "fk_llm_call_evidence_evidence_block_query",
            ("evidence_block_record_id", "query_id"),
            "evidence_blocks",
            ("record_id", "query_id"),
            on_delete="no action",
        ),
        _unique_constraint_sql(
            "llm_call_evidence",
            "uq_llm_call_evidence_record_query",
            ("record_id", "query_id"),
        ),
        _llm_call_evidence_rank_backfill_sql(),
        _not_null_columns_sql(
            "llm_call_evidence",
            (
                "record_id",
                "call_id",
                "query_id",
                "evidence_id",
                "rank",
                "snapshot_redaction_status",
                "snapshot_encryption_status",
                "created_at",
            ),
            "Cannot upgrade llm_call_evidence schema: required columns contain null values",
        ),
        _unique_constraint_sql(
            "llm_call_evidence",
            "uq_llm_call_evidence_call_rank",
            ("call_id", "rank"),
        ),
        _check_constraint_sql(
            "llm_call_evidence",
            "ck_llm_call_evidence_snapshot_governance",
            LLM_CALL_EVIDENCE_SNAPSHOT_GOVERNANCE_CHECK,
        ),
        """
        create index if not exists ix_llm_call_evidence_query_id
        on llm_call_evidence (query_id)
        """,
        """
        create index if not exists ix_llm_call_evidence_call_id
        on llm_call_evidence (call_id)
        """,
        """
        create index if not exists ix_llm_call_evidence_call_rank
        on llm_call_evidence (call_id, rank)
        """,
        """
        create index if not exists ix_llm_call_evidence_chunk_id
        on llm_call_evidence (chunk_id)
        """,
        """
        create index if not exists ix_llm_call_evidence_evidence_id
        on llm_call_evidence (evidence_id)
        """,
        """
        create index if not exists ix_llm_call_evidence_evidence_block_record_id
        on llm_call_evidence (evidence_block_record_id)
        """,
        """
        create index if not exists ix_llm_call_evidence_query_evidence_block_record
        on llm_call_evidence (query_id, evidence_block_record_id)
        """,
        _unique_constraint_sql("answers", "uq_answers_record_query", ("record_id", "query_id")),
        _unique_constraint_sql("citations", "uq_citations_record_query", ("record_id", "query_id")),
        f"""
        create table if not exists citation_audits (
            record_id varchar(64) not null,
            query_id varchar(64) not null,
            citation_id varchar(64),
            citation_record_id varchar(64),
            citation_verification_record_id varchar(64),
            answer_record_id varchar(64),
            answer_call_id varchar(64),
            evidence_id varchar(64),
            llm_call_evidence_record_id varchar(64),
            chunk_id varchar(64),
            verifier_status varchar(64),
            unsupported_numbers_text text,
            issue_text text,
            supporting_text_snapshot text,
            supporting_text_hash varchar(128),
            snapshot_redaction_status varchar(64) not null default 'unredacted',
            snapshot_encryption_status varchar(64) not null default 'plaintext',
            snapshot_retention_expires_at timestamp with time zone,
            created_at timestamp with time zone not null default now(),
            constraint pk_citation_audits primary key (record_id),
            constraint fk_citation_audits_query
                foreign key (query_id) references query_runs(query_id) on delete cascade,
            constraint fk_citation_audits_citation_record_query
                foreign key (citation_record_id, query_id)
                references citations(record_id, query_id),
            constraint fk_citation_audits_llm_call_evidence_query
                foreign key (llm_call_evidence_record_id, query_id)
                references llm_call_evidence(record_id, query_id),
            constraint fk_citation_audits_answer_record_query
                foreign key (answer_record_id, query_id)
                references answers(record_id, query_id),
            constraint fk_citation_audits_answer_call_query
                foreign key (answer_call_id, query_id)
                references llm_calls(call_id, query_id),
            constraint fk_citation_audits_citation_verification_query
                foreign key (citation_verification_record_id, query_id)
                references citation_verifications(record_id, query_id),
            constraint ck_citation_audits_snapshot_governance
                check {CITATION_AUDIT_SNAPSHOT_GOVERNANCE_CHECK}
        )
        """,
        "alter table citation_audits add column if not exists record_id varchar(64)",
        "alter table citation_audits add column if not exists query_id varchar(64)",
        "alter table citation_audits add column if not exists citation_id varchar(64)",
        "alter table citation_audits add column if not exists citation_record_id varchar(64)",
        """
        alter table citation_audits
        add column if not exists citation_verification_record_id varchar(64)
        """,
        "alter table citation_audits add column if not exists answer_record_id varchar(64)",
        "alter table citation_audits add column if not exists answer_call_id varchar(64)",
        "alter table citation_audits add column if not exists evidence_id varchar(64)",
        """
        alter table citation_audits
        add column if not exists llm_call_evidence_record_id varchar(64)
        """,
        "alter table citation_audits add column if not exists chunk_id varchar(64)",
        "alter table citation_audits add column if not exists verifier_status varchar(64)",
        "alter table citation_audits add column if not exists unsupported_numbers_text text",
        "alter table citation_audits add column if not exists issue_text text",
        "alter table citation_audits add column if not exists supporting_text_snapshot text",
        "alter table citation_audits add column if not exists supporting_text_hash varchar(128)",
        """
        alter table citation_audits
        add column if not exists snapshot_redaction_status varchar(64) default 'unredacted'
        """,
        """
        alter table citation_audits
        add column if not exists snapshot_encryption_status varchar(64) default 'plaintext'
        """,
        """
        alter table citation_audits
        add column if not exists snapshot_retention_expires_at timestamp with time zone
        """,
        """
        alter table citation_audits
        add column if not exists created_at timestamp with time zone default now()
        """,
        """
        update citation_audits
        set snapshot_redaction_status = 'unredacted'
        where nullif(btrim(snapshot_redaction_status), '') is null
           or lower(snapshot_redaction_status) = 'unknown'
        """,
        """
        update citation_audits
        set snapshot_encryption_status = 'plaintext'
        where nullif(btrim(snapshot_encryption_status), '') is null
           or lower(snapshot_encryption_status) = 'unknown'
        """,
        """
        alter table citation_audits
        alter column snapshot_redaction_status set default 'unredacted'
        """,
        """
        alter table citation_audits
        alter column snapshot_encryption_status set default 'plaintext'
        """,
        "update citation_audits set created_at = now() where created_at is null",
        "alter table citation_audits alter column created_at set default now()",
        "alter table citation_audits alter column created_at set not null",
        _primary_key_constraint_sql(
            "citation_audits",
            "pk_citation_audits",
            ("record_id",),
            require_name=True,
        ),
        _foreign_key_constraint_sql(
            "citation_audits",
            "fk_citation_audits_query",
            ("query_id",),
            "query_runs",
            ("query_id",),
        ),
        _drop_fk_constraints_sql(
            "citation_audits",
            (
                "fk_citation_audits_citation_record",
                "fk_citation_audits_llm_call_evidence",
                "fk_citation_audits_answer_record",
                "fk_citation_audits_answer_call",
                "fk_citation_audits_citation_verification",
            ),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("citation_record_id",),
            "citations",
            ("record_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("llm_call_evidence_record_id",),
            "llm_call_evidence",
            ("record_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("answer_record_id",),
            "answers",
            ("record_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("answer_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "citation_audits",
            ("citation_verification_record_id",),
            "citation_verifications",
            ("record_id",),
        ),
        _foreign_key_constraint_sql(
            "citation_audits",
            "fk_citation_audits_citation_record_query",
            ("citation_record_id", "query_id"),
            "citations",
            ("record_id", "query_id"),
            on_delete="no action",
        ),
        _foreign_key_constraint_sql(
            "citation_audits",
            "fk_citation_audits_llm_call_evidence_query",
            ("llm_call_evidence_record_id", "query_id"),
            "llm_call_evidence",
            ("record_id", "query_id"),
            on_delete="no action",
        ),
        _foreign_key_constraint_sql(
            "citation_audits",
            "fk_citation_audits_answer_record_query",
            ("answer_record_id", "query_id"),
            "answers",
            ("record_id", "query_id"),
            on_delete="no action",
        ),
        _foreign_key_constraint_sql(
            "citation_audits",
            "fk_citation_audits_answer_call_query",
            ("answer_call_id", "query_id"),
            "llm_calls",
            ("call_id", "query_id"),
            on_delete="no action",
        ),
        _unique_constraint_sql(
            "citation_verifications",
            "uq_citation_verifications_record_query",
            ("record_id", "query_id"),
        ),
        _nullable_composite_fk_preflight_sql(
            "citation_audits",
            "fk_citation_audits_citation_verification_query",
            ("citation_verification_record_id", "query_id"),
            "citation_verifications",
            ("record_id", "query_id"),
        ),
        _foreign_key_constraint_sql(
            "citation_audits",
            "fk_citation_audits_citation_verification_query",
            ("citation_verification_record_id", "query_id"),
            "citation_verifications",
            ("record_id", "query_id"),
            on_delete="no action",
        ),
        _not_null_columns_sql(
            "citation_audits",
            (
                "record_id",
                "query_id",
                "snapshot_redaction_status",
                "snapshot_encryption_status",
                "created_at",
            ),
            "Cannot upgrade citation_audits schema: required columns contain null values",
        ),
        _check_constraint_sql(
            "citation_audits",
            "ck_citation_audits_snapshot_governance",
            CITATION_AUDIT_SNAPSHOT_GOVERNANCE_CHECK,
        ),
        """
        create index if not exists ix_citation_audits_query_id
        on citation_audits (query_id)
        """,
        """
        create index if not exists ix_citation_audits_citation_id
        on citation_audits (citation_id)
        """,
        """
        create index if not exists ix_citation_audits_citation_record_id
        on citation_audits (citation_record_id)
        """,
        """
        create index if not exists ix_citation_audits_evidence_id
        on citation_audits (evidence_id)
        """,
        """
        create index if not exists ix_citation_audits_llm_call_evidence_record_id
        on citation_audits (llm_call_evidence_record_id)
        """,
        """
        create index if not exists ix_citation_audits_answer_record_id
        on citation_audits (answer_record_id)
        """,
        """
        create index if not exists ix_citation_audits_answer_call_id
        on citation_audits (answer_call_id)
        """,
        """
        create index if not exists ix_citation_audits_citation_verification_record_id
        on citation_audits (citation_verification_record_id)
        """,
        """
        create index if not exists ix_citation_audits_query_citation
        on citation_audits (query_id, citation_id)
        """,
        """
        create index if not exists ix_citation_audits_query_citation_record
        on citation_audits (query_id, citation_record_id)
        """,
        """
        create index if not exists ix_citation_audits_query_evidence
        on citation_audits (query_id, evidence_id)
        """,
        """
        create index if not exists ix_citation_audits_query_llm_call_evidence_record
        on citation_audits (query_id, llm_call_evidence_record_id)
        """,
        """
        create index if not exists ix_citation_audits_query_answer_record
        on citation_audits (query_id, answer_record_id)
        """,
        """
        create index if not exists ix_citation_audits_query_answer_call
        on citation_audits (query_id, answer_call_id)
        """,
        """
        create index if not exists ix_citation_audits_query_citation_verification
        on citation_audits (query_id, citation_verification_record_id)
        """,
        f"""
        create table if not exists quality_reviews (
            record_id varchar(64) not null,
            query_id varchar(64) not null,
            answer_record_id varchar(64),
            planner_call_id varchar(64),
            answer_call_id varchar(64),
            review_call_id varchar(64),
            reviewer varchar(128) not null,
            status varchar(64) not null,
            planner_verdict text,
            evidence_relevance_verdict text,
            answer_faithfulness_verdict text,
            citation_verdict text,
            issues_text text,
            recommendations_text text,
            payload_json jsonb not null default '{{}}'::jsonb,
            payload_hash varchar(128),
            payload_redaction_status varchar(64) not null default 'unredacted',
            payload_encryption_status varchar(64) not null default 'plaintext',
            payload_retention_expires_at timestamp with time zone,
            created_at timestamp with time zone not null default now(),
            constraint pk_quality_reviews primary key (record_id),
            constraint fk_quality_reviews_query
                foreign key (query_id) references query_runs(query_id) on delete cascade,
            constraint fk_quality_reviews_answer_record_query
                foreign key (answer_record_id, query_id)
                references answers(record_id, query_id),
            constraint fk_quality_reviews_planner_call_query
                foreign key (planner_call_id, query_id)
                references llm_calls(call_id, query_id),
            constraint fk_quality_reviews_answer_call_query
                foreign key (answer_call_id, query_id)
                references llm_calls(call_id, query_id),
            constraint fk_quality_reviews_review_call_query
                foreign key (review_call_id, query_id)
                references llm_calls(call_id, query_id),
            constraint ck_quality_reviews_payload_governance
                check {QUALITY_REVIEW_PAYLOAD_GOVERNANCE_CHECK}
        )
        """,
        "alter table quality_reviews add column if not exists record_id varchar(64)",
        "alter table quality_reviews add column if not exists query_id varchar(64)",
        "alter table quality_reviews add column if not exists answer_record_id varchar(64)",
        "alter table quality_reviews add column if not exists planner_call_id varchar(64)",
        "alter table quality_reviews add column if not exists answer_call_id varchar(64)",
        "alter table quality_reviews add column if not exists review_call_id varchar(64)",
        "alter table quality_reviews add column if not exists reviewer varchar(128)",
        "alter table quality_reviews add column if not exists status varchar(64)",
        "alter table quality_reviews add column if not exists planner_verdict text",
        "alter table quality_reviews add column if not exists evidence_relevance_verdict text",
        "alter table quality_reviews add column if not exists answer_faithfulness_verdict text",
        "alter table quality_reviews add column if not exists citation_verdict text",
        "alter table quality_reviews add column if not exists issues_text text",
        "alter table quality_reviews add column if not exists recommendations_text text",
        "alter table quality_reviews add column if not exists payload_json jsonb default '{}'::jsonb",
        "alter table quality_reviews add column if not exists payload_hash varchar(128)",
        """
        alter table quality_reviews
        add column if not exists payload_redaction_status varchar(64) default 'unredacted'
        """,
        """
        alter table quality_reviews
        add column if not exists payload_encryption_status varchar(64) default 'plaintext'
        """,
        """
        alter table quality_reviews
        add column if not exists payload_retention_expires_at timestamp with time zone
        """,
        """
        alter table quality_reviews
        add column if not exists created_at timestamp with time zone default now()
        """,
        """
        update quality_reviews
        set payload_json = '{}'::jsonb
        where payload_json is null
        """,
        """
        update quality_reviews
        set payload_redaction_status = 'unredacted'
        where nullif(btrim(payload_redaction_status), '') is null
           or lower(payload_redaction_status) = 'unknown'
        """,
        """
        update quality_reviews
        set payload_encryption_status = 'plaintext'
        where nullif(btrim(payload_encryption_status), '') is null
           or lower(payload_encryption_status) = 'unknown'
        """,
        """
        update quality_reviews
        set reviewer = 'legacy_unknown'
        where reviewer is null
        """,
        """
        update quality_reviews
        set status = 'legacy_unreviewed'
        where status is null
        """,
        "alter table quality_reviews alter column payload_json set default '{}'::jsonb",
        "alter table quality_reviews alter column payload_redaction_status set default 'unredacted'",
        "alter table quality_reviews alter column payload_encryption_status set default 'plaintext'",
        "update quality_reviews set created_at = now() where created_at is null",
        "alter table quality_reviews alter column created_at set default now()",
        "alter table quality_reviews alter column created_at set not null",
        _primary_key_constraint_sql(
            "quality_reviews",
            "pk_quality_reviews",
            ("record_id",),
            require_name=True,
        ),
        _foreign_key_constraint_sql(
            "quality_reviews",
            "fk_quality_reviews_query",
            ("query_id",),
            "query_runs",
            ("query_id",),
        ),
        _drop_fk_constraints_sql(
            "quality_reviews",
            (
                "fk_quality_reviews_answer_record",
                "fk_quality_reviews_planner_call",
                "fk_quality_reviews_answer_call",
                "fk_quality_reviews_review_call",
            ),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("answer_record_id",),
            "answers",
            ("record_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("planner_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("answer_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _drop_legacy_foreign_key_constraints_sql(
            "quality_reviews",
            ("review_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _foreign_key_constraint_sql(
            "quality_reviews",
            "fk_quality_reviews_answer_record_query",
            ("answer_record_id", "query_id"),
            "answers",
            ("record_id", "query_id"),
            on_delete="no action",
        ),
        _foreign_key_constraint_sql(
            "quality_reviews",
            "fk_quality_reviews_planner_call_query",
            ("planner_call_id", "query_id"),
            "llm_calls",
            ("call_id", "query_id"),
            on_delete="no action",
        ),
        _foreign_key_constraint_sql(
            "quality_reviews",
            "fk_quality_reviews_answer_call_query",
            ("answer_call_id", "query_id"),
            "llm_calls",
            ("call_id", "query_id"),
            on_delete="no action",
        ),
        _foreign_key_constraint_sql(
            "quality_reviews",
            "fk_quality_reviews_review_call_query",
            ("review_call_id", "query_id"),
            "llm_calls",
            ("call_id", "query_id"),
            on_delete="no action",
        ),
        _not_null_columns_sql(
            "quality_reviews",
            (
                "record_id",
                "query_id",
                "reviewer",
                "status",
                "payload_json",
                "payload_redaction_status",
                "payload_encryption_status",
                "created_at",
            ),
            "Cannot upgrade quality_reviews schema: required columns contain null values",
        ),
        _check_constraint_sql(
            "quality_reviews",
            "ck_quality_reviews_payload_governance",
            QUALITY_REVIEW_PAYLOAD_GOVERNANCE_CHECK,
        ),
        """
        create index if not exists ix_quality_reviews_query_id
        on quality_reviews (query_id)
        """,
        """
        create index if not exists ix_quality_reviews_answer_record_id
        on quality_reviews (answer_record_id)
        """,
        """
        create index if not exists ix_quality_reviews_planner_call_id
        on quality_reviews (planner_call_id)
        """,
        """
        create index if not exists ix_quality_reviews_answer_call_id
        on quality_reviews (answer_call_id)
        """,
        """
        create index if not exists ix_quality_reviews_review_call_id
        on quality_reviews (review_call_id)
        """,
        """
        create index if not exists ix_quality_reviews_query_answer_record
        on quality_reviews (query_id, answer_record_id)
        """,
        """
        create index if not exists ix_quality_reviews_query_planner_call
        on quality_reviews (query_id, planner_call_id)
        """,
        """
        create index if not exists ix_quality_reviews_query_answer_call
        on quality_reviews (query_id, answer_call_id)
        """,
        """
        create index if not exists ix_quality_reviews_query_review_call
        on quality_reviews (query_id, review_call_id)
        """,
        "create index if not exists ix_quality_reviews_status on quality_reviews (status)",
        """
        create table if not exists query_plans (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            planner_call_id varchar(64),
            plan_id varchar(64),
            planner varchar(128),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now(),
            constraint fk_query_plans_planner_call_query
                foreign key (planner_call_id, query_id)
                references llm_calls(call_id, query_id),
            constraint uq_query_plans_record_query unique (record_id, query_id)
        )
        """,
        "alter table query_plans add column if not exists planner_call_id varchar(64)",
        _drop_legacy_foreign_key_constraints_sql(
            "query_plans",
            ("planner_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _unique_constraint_sql(
            "query_plans",
            "uq_query_plans_record_query",
            ("record_id", "query_id"),
        ),
        _nullable_composite_fk_preflight_sql(
            "query_plans",
            "fk_query_plans_planner_call_query",
            ("planner_call_id", "query_id"),
            "llm_calls",
            ("call_id", "query_id"),
        ),
        _foreign_key_constraint_sql(
            "query_plans",
            "fk_query_plans_planner_call_query",
            ("planner_call_id", "query_id"),
            "llm_calls",
            ("call_id", "query_id"),
            on_delete="no action",
        ),
        "create index if not exists ix_query_plans_query_id on query_plans (query_id)",
        """
        create index if not exists ix_query_plans_planner_call_id
        on query_plans (planner_call_id)
        """,
        """
        create index if not exists ix_query_plans_query_planner_call
        on query_plans (query_id, planner_call_id)
        """,
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
            created_at timestamp with time zone not null default now(),
            constraint uq_evidence_blocks_record_query unique (record_id, query_id),
            constraint uq_evidence_blocks_evidence_query unique (evidence_id, query_id)
        )
        """,
        _unique_constraint_sql(
            "evidence_blocks",
            "uq_evidence_blocks_record_query",
            ("record_id", "query_id"),
        ),
        _evidence_block_evidence_id_disambiguation_sql(),
        _unique_constraint_sql(
            "evidence_blocks",
            "uq_evidence_blocks_evidence_query",
            ("evidence_id", "query_id"),
        ),
        "create index if not exists ix_evidence_blocks_query_id on evidence_blocks (query_id)",
        """
        create index if not exists ix_evidence_blocks_evidence_id
        on evidence_blocks (evidence_id)
        """,
        """
        create index if not exists ix_evidence_blocks_query_evidence
        on evidence_blocks (query_id, evidence_id)
        """,
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
            answer_call_id varchar(64),
            confidence varchar(32),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now(),
            constraint fk_answers_answer_call_query
                foreign key (answer_call_id, query_id)
                references llm_calls(call_id, query_id),
            constraint uq_answers_record_query unique (record_id, query_id)
        )
        """,
        "alter table answers add column if not exists answer_call_id varchar(64)",
        _drop_legacy_foreign_key_constraints_sql(
            "answers",
            ("answer_call_id",),
            "llm_calls",
            ("call_id",),
        ),
        _unique_constraint_sql("answers", "uq_answers_record_query", ("record_id", "query_id")),
        _nullable_composite_fk_preflight_sql(
            "answers",
            "fk_answers_answer_call_query",
            ("answer_call_id", "query_id"),
            "llm_calls",
            ("call_id", "query_id"),
        ),
        _foreign_key_constraint_sql(
            "answers",
            "fk_answers_answer_call_query",
            ("answer_call_id", "query_id"),
            "llm_calls",
            ("call_id", "query_id"),
            on_delete="no action",
        ),
        "create index if not exists ix_answers_query_id on answers (query_id)",
        "create index if not exists ix_answers_answer_call_id on answers (answer_call_id)",
        """
        create index if not exists ix_answers_query_answer_call
        on answers (query_id, answer_call_id)
        """,
        """
        create table if not exists citations (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            citation_id varchar(64),
            evidence_id varchar(64),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now(),
            constraint uq_citations_record_query unique (record_id, query_id)
        )
        """,
        _unique_constraint_sql("citations", "uq_citations_record_query", ("record_id", "query_id")),
        "create index if not exists ix_citations_query_id on citations (query_id)",
        """
        create table if not exists citation_verifications (
            record_id varchar(64) primary key,
            query_id varchar(64) not null references query_runs(query_id),
            status varchar(64),
            payload_json jsonb not null default '{}'::jsonb,
            created_at timestamp with time zone not null default now(),
            constraint uq_citation_verifications_record_query unique (record_id, query_id)
        )
        """,
        _unique_constraint_sql(
            "citation_verifications",
            "uq_citation_verifications_record_query",
            ("record_id", "query_id"),
        ),
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


async def get_async_db():
    async with AsyncSessionLocal() as db:
        yield db

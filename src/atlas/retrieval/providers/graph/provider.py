from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import time
from typing import Any

from sqlalchemy.orm import Session

from atlas.db import repositories
from atlas.query_orchestrator.schema import QueryPlan
from atlas.retrieval.contracts import ProviderResult, SourceAnchor
from atlas.retrieval.models.candidate import Candidate
from atlas.retrieval.models.retrieval_task import RetrievalTask
from atlas.retrieval.providers.graph.models import (
    DEFAULT_DEGREE_CAP,
    DEFAULT_MAX_HOPS,
    DEFAULT_MAX_PATHS,
    DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
    GRAPH_PROVIDER,
    GRAPH_PROVIDER_VERSION,
    SUPPORTED_GRAPH_MODES,
    UNSUPPORTED_GRAPH_MODES,
    GraphEntityMatch,
    GraphFilters,
    GraphNeighborhood,
    GraphPath,
)
from atlas.retrieval.providers.graph.evidence import (
    build_graph_evidence_from_candidates,
    graph_source_anchor_payload,
    sanitize_graph_metadata,
)
from atlas.retrieval.providers.graph.postgres_store import PostgresGraphStore
from atlas.retrieval.providers.graph.store import GraphStore


SUPPORTED_PROVIDER_STATUSES = frozenset({"ready", "supported", "ok", "enabled"})


class GraphProvider:
    """V3 graph retrieval provider boundary.

    Phase 3 emits normal source-grounded Candidate objects when graph anchors can be
    hydrated to chunk text. Graph-only summaries stay out of evidence.
    """

    provider_name = GRAPH_PROVIDER

    def __init__(
        self,
        *,
        store: GraphStore | None = None,
        default_graph_version: str = "default",
        degree_cap: int = DEFAULT_DEGREE_CAP,
        max_hops: int = DEFAULT_MAX_HOPS,
        max_paths: int = DEFAULT_MAX_PATHS,
        max_source_chunks_per_result: int = DEFAULT_MAX_SOURCE_CHUNKS_PER_RESULT,
        max_context_tokens: int | None = 6000,
        max_blocks: int | None = None,
    ) -> None:
        self.store = store or PostgresGraphStore()
        self.default_graph_version = default_graph_version
        self.degree_cap = degree_cap
        self.max_hops = max_hops
        self.max_paths = max_paths
        self.max_source_chunks_per_result = max_source_chunks_per_result
        self.max_context_tokens = max_context_tokens
        self.max_blocks = max_blocks

    def retrieve_provider_result(
        self,
        db: Session,
        *,
        query: str,
        top_k: int,
        filters: dict | None,
        options: dict,
        query_plan: QueryPlan,
        retrieval_tasks: list[RetrievalTask],
    ) -> ProviderResult:
        started = time.perf_counter()
        candidates: list[Candidate] = []
        task_traces: list[dict[str, Any]] = []
        graph_items: list[dict[str, Any]] = []
        reasons: list[str] = []

        for task in retrieval_tasks:
            if str(task.provider).strip().lower() != GRAPH_PROVIDER:
                trace = _base_task_trace(task, query_plan=query_plan)
                trace.update(
                    {
                        "status": "skipped",
                        "reason": "non_graph_task",
                        "skip_reason": "non_graph_task",
                    }
                )
                task_traces.append(trace)
                continue

            if not _task_is_ready(task):
                reason = task.unsupported_reason or f"provider_status:{task.provider_status}"
                trace = _base_task_trace(task, query_plan=query_plan)
                trace.update(
                    {
                        "status": "skipped",
                        "reason": reason,
                        "skip_reason": reason,
                    }
                )
                task_traces.append(trace)
                continue

            task_trace, task_candidates, task_graph_items, reason = self._run_task(
                db,
                task=task,
                query_plan=query_plan,
                filters=filters,
                options=options or {},
                top_k=max(0, top_k - len(candidates)),
            )
            task_traces.append(task_trace)
            graph_items.extend(task_graph_items)
            candidates.extend(task_candidates)
            if reason:
                reasons.append(reason)
            if len(candidates) >= top_k:
                break

        candidates = candidates[:top_k]
        evidence_result = (
            build_graph_evidence_from_candidates(
                candidates,
                max_context_tokens=self.max_context_tokens,
                max_blocks=_evidence_max_blocks(self.max_blocks, top_k),
                plan_id=query_plan.plan_id,
            )
            if candidates
            else None
        )
        evidence = evidence_result.evidence if evidence_result is not None else ()
        evidence_pack = evidence_result.evidence_pack if evidence_result is not None else None

        latency_ms = int((time.perf_counter() - started) * 1000)
        status = "executed" if candidates else "empty"
        reason = None if candidates else _first_reason(reasons, task_traces)
        trace = _provider_trace(
            query_plan=query_plan,
            task_traces=task_traces,
            graph_items=graph_items,
            candidates=candidates,
            evidence_count=len(evidence),
            evidence_pack_id=evidence_pack.pack_id if evidence_pack else None,
            dropped_evidence_count=len(evidence_pack.dropped_blocks) if evidence_pack else 0,
            latency_ms=latency_ms,
            reason=reason,
            default_cap_config={
                "degree_cap": self.degree_cap,
                "max_hops": self.max_hops,
                "max_paths": self.max_paths,
                "max_source_chunks_per_result": self.max_source_chunks_per_result,
                "max_context_tokens": self.max_context_tokens,
                "max_blocks": self.max_blocks,
            },
        )
        return ProviderResult(
            provider=GRAPH_PROVIDER,
            task_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].task_id,
            unit_id=None if len(retrieval_tasks) != 1 else retrieval_tasks[0].unit_id,
            status=status,
            candidates=tuple(candidates),
            evidence=evidence,
            evidence_pack=evidence_pack,
            latency_ms=latency_ms,
            reason=reason,
            trace=trace,
        )

    def _run_task(
        self,
        db: Session,
        *,
        task: RetrievalTask,
        query_plan: QueryPlan,
        filters: dict | None,
        options: dict[str, Any],
        top_k: int,
    ) -> tuple[dict[str, Any], list[Candidate], list[dict[str, Any]], str | None]:
        task_started = time.perf_counter()
        graph_filters = _graph_filters(
            filters=filters,
            options=options,
            query_plan=query_plan,
            task=task,
            default_graph_version=self.default_graph_version,
        )
        cap_config = _cap_config(
            task,
            options=options,
            degree_cap=self.degree_cap,
            max_hops=self.max_hops,
            max_paths=self.max_paths,
            max_source_chunks_per_result=self.max_source_chunks_per_result,
        )
        trace = _base_task_trace(
            task,
            query_plan=query_plan,
            graph_filters=graph_filters,
            cap_config=cap_config,
        )

        explicit_mode = _explicit_graph_mode(task)
        if explicit_mode in UNSUPPORTED_GRAPH_MODES or (
            explicit_mode and explicit_mode not in SUPPORTED_GRAPH_MODES
        ):
            reason = f"unsupported_graph_mode:{explicit_mode}"
            trace.update(
                {
                    "status": "empty",
                    "mode": explicit_mode,
                    "reason": reason,
                    "latency_ms": int((time.perf_counter() - task_started) * 1000),
                }
            )
            return trace, [], [], reason

        resolved, resolution_trace = self._resolve_entities(
            db,
            task=task,
            query_plan=query_plan,
            graph_filters=graph_filters,
        )
        trace["entity_resolution"] = resolution_trace
        trace["resolved_entities"] = [
            _resolved_entity_payload(match) for match in resolved
        ]

        mode = explicit_mode or _mode_from_resolved_entities(resolved)
        trace["mode"] = mode

        if not resolved:
            reason = "no_resolved_entity"
            trace.update(
                {
                    "status": "empty",
                    "reason": reason,
                    "latency_ms": int((time.perf_counter() - task_started) * 1000),
                }
            )
            return trace, [], [], reason

        if mode == "local":
            task_candidates, graph_items = self._run_local(
                db,
                task=task,
                query_plan=query_plan,
                graph_filters=graph_filters,
                cap_config=cap_config,
                resolved=resolved,
                top_k=top_k,
                trace=trace,
            )
        elif mode == "path":
            if len(resolved) < 2:
                reason = "path_requires_two_resolved_entities"
                trace.update(
                    {
                        "status": "empty",
                        "reason": reason,
                        "latency_ms": int((time.perf_counter() - task_started) * 1000),
                    }
                )
                return trace, [], [], reason
            task_candidates, graph_items = self._run_path(
                db,
                task=task,
                query_plan=query_plan,
                graph_filters=graph_filters,
                cap_config=cap_config,
                resolved=resolved,
                top_k=top_k,
                trace=trace,
            )
        else:
            reason = f"unsupported_graph_mode:{mode}"
            trace.update(
                {
                    "status": "empty",
                    "reason": reason,
                    "latency_ms": int((time.perf_counter() - task_started) * 1000),
                }
            )
            return trace, [], [], reason

        reason = None if task_candidates else "no_grounded_graph_candidate"
        trace.update(
            {
                "status": "executed" if task_candidates else "empty",
                "reason": reason,
                "latency_ms": int((time.perf_counter() - task_started) * 1000),
            }
        )
        return trace, task_candidates, graph_items, reason

    def _resolve_entities(
        self,
        db: Session,
        *,
        task: RetrievalTask,
        query_plan: QueryPlan,
        graph_filters: GraphFilters,
    ) -> tuple[tuple[GraphEntityMatch, ...], dict[str, Any]]:
        attempts: list[dict[str, Any]] = []

        plan_queries = _plan_entity_queries(query_plan)
        if plan_queries:
            matches = self._resolve_entity_queries(
                db,
                queries=plan_queries,
                graph_filters=graph_filters,
                source="query_plan.entities",
                attempts=attempts,
            )
            if matches:
                return matches, {"source": "query_plan.entities", "attempts": attempts}

        metadata_queries = _task_metadata_entity_queries(task.metadata)
        if metadata_queries:
            matches = self._resolve_entity_queries(
                db,
                queries=metadata_queries,
                graph_filters=graph_filters,
                source="task.metadata",
                attempts=attempts,
            )
            if matches:
                return matches, {"source": "task.metadata", "attempts": attempts}

        matches = self.store.find_entities(
            db,
            query_text=task.query_text,
            filters=graph_filters,
            aliases=(),
            limit=max(2, DEFAULT_MAX_PATHS),
        )
        attempts.append(
            _resolution_attempt_payload(
                source="task.query_text",
                query_text=task.query_text,
                aliases=(),
                matches=matches,
            )
        )
        return _dedupe_matches(matches), {"source": "task.query_text", "attempts": attempts}

    def _resolve_entity_queries(
        self,
        db: Session,
        *,
        queries: Sequence[dict[str, Any]],
        graph_filters: GraphFilters,
        source: str,
        attempts: list[dict[str, Any]],
    ) -> tuple[GraphEntityMatch, ...]:
        resolved: list[GraphEntityMatch] = []
        for query in queries:
            entity_id = _optional_str(query.get("entity_id"))
            if entity_id:
                entity = self.store.get_entity(
                    db,
                    entity_id,
                    graph_version=graph_filters.graph_version,
                )
                matches = (
                    (
                        GraphEntityMatch(
                            entity=entity,
                            score=1.0,
                            matched_name=entity.canonical_name,
                            match_type="entity_id",
                            rank=1,
                        ),
                    )
                    if entity is not None
                    else ()
                )
                attempts.append(
                    _resolution_attempt_payload(
                        source=source,
                        query_text=entity_id,
                        aliases=(),
                        matches=matches,
                    )
                )
                resolved.extend(matches)
                continue

            query_text = _optional_str(query.get("query_text"))
            aliases = _string_tuple(query.get("aliases"))
            if not query_text and aliases:
                query_text, aliases = aliases[0], aliases[1:]
            if not query_text:
                continue
            matches = self.store.find_entities(
                db,
                query_text=query_text,
                filters=graph_filters,
                aliases=aliases,
                limit=1,
            )
            attempts.append(
                _resolution_attempt_payload(
                    source=source,
                    query_text=query_text,
                    aliases=aliases,
                    matches=matches,
                )
            )
            resolved.extend(matches[:1])
        return _dedupe_matches(resolved)

    def _run_local(
        self,
        db: Session,
        *,
        task: RetrievalTask,
        query_plan: QueryPlan,
        graph_filters: GraphFilters,
        cap_config: dict[str, int],
        resolved: tuple[GraphEntityMatch, ...],
        top_k: int,
        trace: dict[str, Any],
    ) -> tuple[list[Candidate], list[dict[str, Any]]]:
        entity = resolved[0].entity
        neighborhood = self.store.get_neighbors(
            db,
            entity_id=entity.entity_id,
            degree_cap=cap_config["degree_cap"],
            relation_types=graph_filters.relation_types,
            filters=graph_filters,
        )
        graph_item = _neighborhood_graph_item(
            task=task,
            graph_filters=graph_filters,
            neighborhood=neighborhood,
        )
        trace.update(_local_trace(neighborhood, graph_filters=graph_filters, cap_config=cap_config))
        candidates = _candidates_from_graph_items(
            db,
            graph_items=(graph_item,),
            query_plan=query_plan,
            task=task,
            limit=top_k,
            max_source_chunks_per_result=cap_config["max_source_chunks_per_result"],
        )
        trace["graph_candidate_count"] = 1
        trace["grounded_candidate_count"] = len(candidates)
        return candidates, [graph_item]

    def _run_path(
        self,
        db: Session,
        *,
        task: RetrievalTask,
        query_plan: QueryPlan,
        graph_filters: GraphFilters,
        cap_config: dict[str, int],
        resolved: tuple[GraphEntityMatch, ...],
        top_k: int,
        trace: dict[str, Any],
    ) -> tuple[list[Candidate], list[dict[str, Any]]]:
        source = resolved[0].entity
        target = resolved[1].entity
        paths = self.store.find_paths(
            db,
            source_entity_id=source.entity_id,
            target_entity_id=target.entity_id,
            max_hops=cap_config["max_hops"],
            degree_cap=cap_config["degree_cap"],
            relation_types=graph_filters.relation_types,
            filters=graph_filters,
            max_paths=cap_config["max_paths"],
        )
        graph_items = tuple(
            _path_graph_item(task=task, graph_filters=graph_filters, path=path)
            for path in paths
        )
        trace.update(_path_trace(paths, graph_filters=graph_filters, cap_config=cap_config))
        candidates = _candidates_from_graph_items(
            db,
            graph_items=graph_items,
            query_plan=query_plan,
            task=task,
            limit=top_k,
            max_source_chunks_per_result=cap_config["max_source_chunks_per_result"],
        )
        trace["graph_candidate_count"] = len(graph_items)
        trace["grounded_candidate_count"] = len(candidates)
        return candidates, list(graph_items)


def _task_is_ready(task: RetrievalTask) -> bool:
    return str(task.provider_status).strip().lower() in SUPPORTED_PROVIDER_STATUSES


def _evidence_max_blocks(configured: int | None, top_k: int) -> int | None:
    if configured is None:
        return max(0, top_k)
    return max(0, min(configured, top_k))


def _explicit_graph_mode(task: RetrievalTask) -> str | None:
    metadata = dict(task.metadata or {})
    mode = _first_text(
        metadata.get("graph_mode"),
        metadata.get("graph_search_mode"),
        metadata.get("graph_retrieval_mode"),
    )
    text = _optional_str(mode)
    return text.strip().lower() if text else None


def _mode_from_resolved_entities(
    resolved: tuple[GraphEntityMatch, ...],
) -> str | None:
    if len(resolved) >= 2:
        return "path"
    if len(resolved) == 1:
        return "local"
    return None


def _graph_filters(
    *,
    filters: dict | None,
    options: Mapping[str, Any],
    query_plan: QueryPlan,
    task: RetrievalTask,
    default_graph_version: str,
) -> GraphFilters:
    task_metadata = dict(task.metadata or {})
    task_filter = dict(task.metadata_filter or {})
    merged_filter: dict[str, Any] = {}
    if isinstance(filters, dict):
        merged_filter.update(filters)
    merged_filter.update(task_filter)
    metadata_filters = task_metadata.get("filters")
    if isinstance(metadata_filters, dict):
        merged_filter.update(metadata_filters)

    graph_version = _first_text(
        task_metadata.get("graph_version"),
        merged_filter.get("graph_version"),
        options.get("graph_version"),
        query_plan.metadata.get("graph_version"),
        default_graph_version,
    )
    corpus_version = _first_text(
        task_metadata.get("corpus_version"),
        merged_filter.get("corpus_version"),
        options.get("corpus_version"),
        query_plan.metadata.get("corpus_version"),
    )
    entity_types = _first_sequence(
        task_metadata.get("entity_types"),
        merged_filter.get("entity_types"),
        options.get("graph_entity_types"),
    )
    relation_types = _first_sequence(
        task_metadata.get("relation_types"),
        merged_filter.get("relation_types"),
        options.get("graph_relation_types"),
    )
    document_ids = _first_sequence(
        merged_filter.get("document_ids"),
        task_metadata.get("document_ids"),
        options.get("document_ids"),
    )
    chunk_ids = _first_sequence(
        merged_filter.get("chunk_ids"),
        task_metadata.get("chunk_ids"),
        options.get("chunk_ids"),
    )

    passthrough_metadata = {
        key: value
        for key, value in merged_filter.items()
        if key
        not in {
            "graph_version",
            "corpus_version",
            "entity_types",
            "relation_types",
            "document_ids",
            "chunk_ids",
        }
    }
    return GraphFilters(
        graph_version=graph_version or default_graph_version,
        corpus_version=corpus_version,
        entity_types=entity_types,
        relation_types=relation_types,
        document_ids=document_ids,
        chunk_ids=chunk_ids,
        metadata=passthrough_metadata,
    )


def _cap_config(
    task: RetrievalTask,
    *,
    options: Mapping[str, Any],
    degree_cap: int,
    max_hops: int,
    max_paths: int,
    max_source_chunks_per_result: int,
) -> dict[str, int]:
    metadata = dict(task.metadata or {})
    return {
        "degree_cap": _positive_int(
            metadata.get("degree_cap"),
            options.get("graph_degree_cap"),
            options.get("degree_cap"),
            default=degree_cap,
        ),
        "max_hops": _positive_int(
            metadata.get("max_hops"),
            options.get("graph_max_hops"),
            options.get("max_hops"),
            default=max_hops,
        ),
        "max_paths": _positive_int(
            metadata.get("max_paths"),
            options.get("graph_max_paths"),
            options.get("max_paths"),
            default=max_paths,
        ),
        "max_source_chunks_per_result": _positive_int(
            metadata.get("max_source_chunks_per_result"),
            options.get("graph_max_source_chunks_per_result"),
            options.get("max_source_chunks_per_result"),
            default=max_source_chunks_per_result,
        ),
    }


def _plan_entity_queries(query_plan: QueryPlan) -> tuple[dict[str, Any], ...]:
    queries: list[dict[str, Any]] = []
    for entity in query_plan.entities:
        query_text = _optional_str(getattr(entity, "value", None))
        aliases = _string_tuple(getattr(entity, "aliases", ()))
        if query_text or aliases:
            queries.append({"query_text": query_text, "aliases": aliases})
    return tuple(queries)


def _task_metadata_entity_queries(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    queries: list[dict[str, Any]] = []
    aliases = _string_tuple(
        metadata.get("aliases")
        or metadata.get("entity_aliases")
        or metadata.get("graph_entity_aliases")
    )
    raw_entities = metadata.get("entities") or metadata.get("graph_entities") or ()
    for raw in _as_sequence(raw_entities):
        if isinstance(raw, Mapping):
            entity_id = _optional_str(raw.get("entity_id") or raw.get("id"))
            query_text = _optional_str(
                raw.get("value")
                or raw.get("name")
                or raw.get("canonical_name")
                or raw.get("query_text")
            )
            item_aliases = _string_tuple(raw.get("aliases")) or aliases
            queries.append(
                {
                    "entity_id": entity_id,
                    "query_text": query_text,
                    "aliases": item_aliases,
                }
            )
            continue
        query_text = _optional_str(raw)
        if query_text:
            queries.append({"query_text": query_text, "aliases": aliases})
    if not queries and aliases:
        queries.append({"query_text": aliases[0], "aliases": aliases[1:]})
    return tuple(queries)


def _neighborhood_graph_item(
    *,
    task: RetrievalTask,
    graph_filters: GraphFilters,
    neighborhood: GraphNeighborhood,
) -> dict[str, Any]:
    relationship_ids = tuple(
        relationship.relationship_id for relationship in neighborhood.relationships
    )
    entity_ids = (
        neighborhood.center_entity.entity_id,
        *(neighbor.entity_id for neighbor in neighborhood.neighbors),
    )
    score = _average(
        relationship.confidence for relationship in neighborhood.relationships
    )
    return {
        "candidate_id": f"graph_local:{task.task_id}:{neighborhood.center_entity.entity_id}",
        "provider": GRAPH_PROVIDER,
        "mode": "local",
        "source_type": "graph_neighborhood",
        "graph_version": graph_filters.graph_version,
        "entity_ids": _dedupe_strings(entity_ids),
        "relationship_ids": relationship_ids,
        "source_anchors": [
            graph_source_anchor_payload(anchor) for anchor in neighborhood.source_anchors
        ],
        "graph_score": score,
        "grounding_strength": 1.0 if neighborhood.source_anchors else 0.0,
        "metadata": {
            "center_entity_id": neighborhood.center_entity.entity_id,
            "degree_cap": neighborhood.degree_cap,
            "truncated": neighborhood.truncated,
            "degree_seen": neighborhood.metadata.get("degree_seen"),
        },
        "_source_anchors": neighborhood.source_anchors,
    }


def _path_graph_item(
    *,
    task: RetrievalTask,
    graph_filters: GraphFilters,
    path: GraphPath,
) -> dict[str, Any]:
    return {
        "candidate_id": f"graph_path:{task.task_id}:{path.path_id or ':'.join(path.relationship_ids)}",
        "provider": GRAPH_PROVIDER,
        "mode": "path",
        "source_type": "graph_path",
        "graph_version": graph_filters.graph_version,
        "entity_ids": path.entity_ids,
        "relationship_ids": path.relationship_ids,
        "source_anchors": [
            graph_source_anchor_payload(anchor) for anchor in path.source_anchors
        ],
        "graph_score": path.score,
        "grounding_strength": 1.0 if path.source_anchors else 0.0,
        "metadata": {
            "path_id": path.path_id,
            "hops": path.hops,
            "degree_cap": path.metadata.get("degree_cap"),
        },
        "_source_anchors": path.source_anchors,
    }


def _candidates_from_graph_items(
    db: Session,
    *,
    graph_items: Sequence[dict[str, Any]],
    query_plan: QueryPlan,
    task: RetrievalTask,
    limit: int,
    max_source_chunks_per_result: int,
) -> list[Candidate]:
    if limit <= 0:
        return []

    chunk_ids = _dedupe_strings(
        anchor.chunk_id
        for item in graph_items
        for anchor in _limited_anchors(item, max_source_chunks_per_result)
        if anchor.chunk_id
    )
    chunk_map = _get_chunks_by_ids(db, chunk_ids)
    candidates: list[Candidate] = []
    seen_chunks: set[str] = set()
    for item in graph_items:
        for anchor in _limited_anchors(item, max_source_chunks_per_result):
            if not anchor.chunk_id or anchor.chunk_id in seen_chunks:
                continue
            chunk = chunk_map.get(anchor.chunk_id)
            if chunk is None:
                continue
            seen_chunks.add(anchor.chunk_id)
            rank = len(candidates) + 1
            candidates.append(
                _candidate_from_chunk(
                    chunk,
                    anchor=anchor,
                    graph_item=item,
                    query_plan=query_plan,
                    task=task,
                    rank=rank,
                )
            )
            if len(candidates) >= limit:
                return candidates
    return candidates


def _candidate_from_chunk(
    chunk: Any,
    *,
    anchor: SourceAnchor,
    graph_item: Mapping[str, Any],
    query_plan: QueryPlan,
    task: RetrievalTask,
    rank: int,
) -> Candidate:
    document = getattr(chunk, "document", None)
    metadata = sanitize_graph_metadata(_chunk_metadata(chunk, document))
    graph_score = _float_or_zero(graph_item.get("graph_score"))
    grounding_strength = _float_or_zero(graph_item.get("grounding_strength"))
    source_anchor = graph_source_anchor_payload(anchor)
    candidate_chunk_id = str(getattr(chunk, "chunk_id", None) or anchor.chunk_id or "")
    graph_candidate_id = _optional_str(graph_item.get("candidate_id"))
    entity_ids = _dedupe_strings(graph_item.get("entity_ids") or ())
    relationship_ids = _dedupe_strings(graph_item.get("relationship_ids") or ())
    grounded_source_chunk_ids = _dedupe_strings((candidate_chunk_id,))
    graph_path = _graph_path_metadata(graph_item)
    graph_metadata = sanitize_graph_metadata(
        {
            "provider": GRAPH_PROVIDER,
            "graph_candidate_id": graph_candidate_id,
            "mode": graph_item.get("mode"),
            "source_type": graph_item.get("source_type"),
            "graph_version": graph_item.get("graph_version"),
            "entity_ids": entity_ids,
            "relationship_ids": relationship_ids,
            "grounded_source_chunk_ids": grounded_source_chunk_ids,
            "graph_score": graph_score,
            "grounding_strength": grounding_strength,
            "retrieval_task_id": task.task_id,
            "retrieval_unit_id": task.unit_id,
            "metadata": dict(graph_item.get("metadata") or {}),
        }
    )
    if graph_path is not None:
        graph_metadata["graph_path"] = sanitize_graph_metadata(graph_path)
    metadata.update(
        {
            "provider": GRAPH_PROVIDER,
            "query_plan_id": query_plan.plan_id,
            "graph_candidate_id": graph_candidate_id,
            "entity_ids": entity_ids,
            "relationship_ids": relationship_ids,
            "graph_score": graph_score,
            "grounding_strength": grounding_strength,
            "grounded_source_chunk_ids": grounded_source_chunk_ids,
            "source_anchor": source_anchor,
            "retrieval_task_id": task.task_id,
            "retrieval_unit_id": task.unit_id,
            "graph": graph_metadata,
            "graph_provider": {
                "provider": GRAPH_PROVIDER,
                "provider_version": GRAPH_PROVIDER_VERSION,
            },
        }
    )
    if graph_path is not None:
        metadata["graph_path"] = sanitize_graph_metadata(graph_path)
    metadata = sanitize_graph_metadata(metadata)
    document_id = _first_text(
        getattr(document, "document_id", None),
        getattr(chunk, "document_id", None),
        anchor.document_id,
    ) or ""
    source_title = _first_text(
        getattr(document, "title", None),
        getattr(document, "source_title", None),
        document_id,
    ) or document_id
    return Candidate(
        candidate_id=f"{graph_item.get('candidate_id')}:{anchor.chunk_id}:{rank}",
        chunk_id=candidate_chunk_id,
        document_id=document_id,
        doc_name=source_title,
        source_title=source_title,
        company=_metadata_value(metadata, "company"),
        text=str(getattr(chunk, "text", "")),
        page_start=_int_or_none(getattr(chunk, "page_start", None), anchor.page_start),
        page_end=_int_or_none(getattr(chunk, "page_end", None), anchor.page_end),
        chunk_index=_int_or_zero(
            getattr(chunk, "chunk_index", None),
            anchor.metadata.get("chunk_index") if isinstance(anchor.metadata, dict) else None,
        ),
        token_count=_int_or_zero(
            getattr(chunk, "token_count", None),
            len(str(getattr(chunk, "text", "")).split()),
        ),
        retrieved_by=(GRAPH_PROVIDER,),
        dense_rank=None,
        dense_score=None,
        fusion_rank=rank,
        fusion_score=graph_score,
        final_rank=rank,
        metadata=metadata,
        source_uri=getattr(document, "source_uri", None),
        section_title=getattr(chunk, "section_title", None),
        parent_id=getattr(chunk, "parent_id", None) or anchor.parent_id,
        provider=GRAPH_PROVIDER,
        source_type="text_chunk",
        lane=str(graph_item.get("mode") or GRAPH_PROVIDER),
        retrieval_task_id=task.task_id,
        retrieval_unit_id=task.unit_id,
        unit_weight=task.weight,
        lane_weight=1.0,
        lane_rank=rank,
        lane_score=graph_score,
    )


def _graph_path_metadata(graph_item: Mapping[str, Any]) -> dict[str, Any] | None:
    if graph_item.get("mode") != "path":
        return None
    item_metadata = dict(graph_item.get("metadata") or {})
    entity_ids = list(graph_item.get("entity_ids") or ())
    return {
        "path_id": item_metadata.get("path_id"),
        "graph_version": graph_item.get("graph_version"),
        "source_entity_id": entity_ids[0] if entity_ids else None,
        "target_entity_id": entity_ids[-1] if entity_ids else None,
        "entity_ids": entity_ids,
        "relationship_ids": list(graph_item.get("relationship_ids") or ()),
        "hops": item_metadata.get("hops"),
    }


def _get_chunks_by_ids(db: Session, chunk_ids: Iterable[str]) -> dict[str, Any]:
    ids = tuple(_dedupe_strings(chunk_ids))
    if not ids:
        return {}
    fake_chunks = getattr(db, "chunks_by_id", None)
    if isinstance(fake_chunks, Mapping):
        return {
            chunk_id: fake_chunks[chunk_id]
            for chunk_id in ids
            if chunk_id in fake_chunks
        }
    fake_chunks = getattr(db, "chunks", None)
    if isinstance(fake_chunks, Mapping):
        return {
            chunk_id: fake_chunks[chunk_id]
            for chunk_id in ids
            if chunk_id in fake_chunks
        }
    return repositories.get_chunks_by_ids(db, ids)


def _limited_anchors(
    graph_item: Mapping[str, Any],
    max_source_chunks_per_result: int,
) -> tuple[SourceAnchor, ...]:
    anchors = graph_item.get("_source_anchors") or ()
    deduped: list[SourceAnchor] = []
    seen_chunks: set[str] = set()
    for anchor in anchors:
        if not isinstance(anchor, SourceAnchor) or not anchor.chunk_id:
            continue
        if anchor.chunk_id in seen_chunks:
            continue
        seen_chunks.add(anchor.chunk_id)
        deduped.append(anchor)
        if len(deduped) >= max_source_chunks_per_result:
            break
    return tuple(deduped)


def _local_trace(
    neighborhood: GraphNeighborhood,
    *,
    graph_filters: GraphFilters,
    cap_config: dict[str, int],
) -> dict[str, Any]:
    degree_seen = _int_or_zero(
        neighborhood.metadata.get("degree_seen"),
        len(neighborhood.relationships),
    )
    neighbors_examined = len(neighborhood.relationships)
    neighbors_returned = _int_or_zero(
        neighborhood.metadata.get("neighbors_returned"),
        len(neighborhood.neighbors),
    )
    truncated = bool(neighborhood.truncated)
    hub_cap_applied = truncated or degree_seen > cap_config["degree_cap"]
    return {
        "truncated": truncated,
        "hub_cap_applied": hub_cap_applied,
        "degree_seen": degree_seen,
        "neighbors_examined": neighbors_examined,
        "neighbors_returned": neighbors_returned,
        "paths_seen": 0,
        "paths_returned": 0,
        "truncated_reason": "degree_cap" if truncated else None,
        "cap_config": dict(cap_config),
        "relation_types": list(graph_filters.relation_types),
    }


def _path_trace(
    paths: Sequence[GraphPath],
    *,
    graph_filters: GraphFilters,
    cap_config: dict[str, int],
) -> dict[str, Any]:
    paths_returned = len(paths)
    truncated = cap_config["max_paths"] > 0 and paths_returned >= cap_config["max_paths"]
    return {
        "truncated": truncated,
        "hub_cap_applied": False,
        "degree_seen": 0,
        "neighbors_examined": 0,
        "neighbors_returned": 0,
        "paths_seen": paths_returned,
        "paths_returned": paths_returned,
        "truncated_reason": "max_paths" if truncated else None,
        "cap_config": dict(cap_config),
        "relation_types": list(graph_filters.relation_types),
    }


def _base_task_trace(
    task: RetrievalTask,
    *,
    query_plan: QueryPlan,
    graph_filters: GraphFilters | None = None,
    cap_config: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "provider": GRAPH_PROVIDER,
        "provider_version": GRAPH_PROVIDER_VERSION,
        "query_plan_id": query_plan.plan_id,
        "task_id": task.task_id,
        "unit_id": task.unit_id,
        "query_text": task.query_text,
        "provider_status": task.provider_status,
        "unsupported_reason": task.unsupported_reason,
        "mode": None,
        "status": "pending",
        "reason": None,
        "skip_reason": None,
        "resolved_entities": [],
        "entity_resolution": {"source": None, "attempts": []},
        "graph_filters": _graph_filters_payload(graph_filters) if graph_filters else {},
        "truncated": False,
        "hub_cap_applied": False,
        "degree_seen": 0,
        "neighbors_examined": 0,
        "neighbors_returned": 0,
        "paths_seen": 0,
        "paths_returned": 0,
        "truncated_reason": None,
        "cap_config": dict(cap_config or {}),
        "relation_types": list(graph_filters.relation_types) if graph_filters else [],
        "graph_candidate_count": 0,
        "grounded_candidate_count": 0,
        "latency_ms": 0,
    }


def _provider_trace(
    *,
    query_plan: QueryPlan,
    task_traces: list[dict[str, Any]],
    graph_items: list[dict[str, Any]],
    candidates: list[Candidate],
    evidence_count: int,
    evidence_pack_id: str | None,
    dropped_evidence_count: int,
    latency_ms: int,
    reason: str | None,
    default_cap_config: dict[str, Any],
) -> dict[str, Any]:
    truncated_reasons = _dedupe_strings(
        item.get("truncated_reason") for item in task_traces if item.get("truncated_reason")
    )
    return {
        "provider": GRAPH_PROVIDER,
        "provider_version": GRAPH_PROVIDER_VERSION,
        "query_plan_id": query_plan.plan_id,
        "planner": query_plan.planner,
        "status": "executed" if candidates else "empty",
        "reason": reason,
        "tasks": task_traces,
        "graph_candidates": [_trace_graph_item(item) for item in graph_items],
        "candidate_count": len(candidates),
        "evidence_count": evidence_count,
        "evidence_pack_id": evidence_pack_id,
        "dropped_evidence_count": dropped_evidence_count,
        "retrieval_latency_ms": latency_ms,
        "truncated": any(bool(item.get("truncated")) for item in task_traces),
        "hub_cap_applied": any(bool(item.get("hub_cap_applied")) for item in task_traces),
        "degree_seen": max(
            (_int_or_zero(item.get("degree_seen")) for item in task_traces),
            default=0,
        ),
        "neighbors_examined": sum(
            _int_or_zero(item.get("neighbors_examined")) for item in task_traces
        ),
        "neighbors_returned": sum(
            _int_or_zero(item.get("neighbors_returned")) for item in task_traces
        ),
        "paths_seen": sum(_int_or_zero(item.get("paths_seen")) for item in task_traces),
        "paths_returned": sum(
            _int_or_zero(item.get("paths_returned")) for item in task_traces
        ),
        "truncated_reason": truncated_reasons[0] if truncated_reasons else None,
        "truncated_reasons": truncated_reasons,
        "cap_config": dict(default_cap_config),
        "relation_types": _dedupe_strings(
            relation_type
            for item in task_traces
            for relation_type in item.get("relation_types", ())
        ),
    }


def _trace_graph_item(item: Mapping[str, Any]) -> dict[str, Any]:
    return sanitize_graph_metadata(
        {
            key: value
            for key, value in item.items()
            if key != "_source_anchors"
        }
    )


def _graph_filters_payload(graph_filters: GraphFilters | None) -> dict[str, Any]:
    if graph_filters is None:
        return {}
    return {
        "graph_version": graph_filters.graph_version,
        "corpus_version": graph_filters.corpus_version,
        "entity_types": list(graph_filters.entity_types),
        "relation_types": list(graph_filters.relation_types),
        "document_ids": list(graph_filters.document_ids),
        "chunk_ids": list(graph_filters.chunk_ids),
        "metadata": dict(graph_filters.metadata),
    }


def _resolution_attempt_payload(
    *,
    source: str,
    query_text: str,
    aliases: tuple[str, ...],
    matches: Sequence[GraphEntityMatch],
) -> dict[str, Any]:
    return {
        "source": source,
        "query_text": query_text,
        "aliases": list(aliases),
        "match_count": len(matches),
        "matches": [_resolved_entity_payload(match) for match in matches],
    }


def _resolved_entity_payload(match: GraphEntityMatch) -> dict[str, Any]:
    entity = match.entity
    return {
        "entity_id": entity.entity_id,
        "canonical_name": entity.canonical_name,
        "entity_type": entity.entity_type,
        "matched_name": match.matched_name,
        "match_type": match.match_type,
        "score": match.score,
        "rank": match.rank,
    }


def _dedupe_matches(
    matches: Iterable[GraphEntityMatch],
) -> tuple[GraphEntityMatch, ...]:
    seen: set[str] = set()
    deduped: list[GraphEntityMatch] = []
    for match in matches:
        entity_id = match.entity.entity_id
        if entity_id in seen:
            continue
        seen.add(entity_id)
        deduped.append(match)
    return tuple(deduped)


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _chunk_metadata(chunk: Any, document: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    document_metadata = getattr(document, "metadata_json", None)
    if isinstance(document_metadata, dict):
        metadata.update(document_metadata)
    chunk_metadata = getattr(chunk, "metadata_json", None)
    if isinstance(chunk_metadata, dict):
        metadata.update(chunk_metadata)
    parent_id = getattr(chunk, "parent_id", None)
    if parent_id:
        metadata["parent_id"] = parent_id
    return metadata


def _first_reason(
    reasons: list[str],
    task_traces: list[dict[str, Any]],
) -> str | None:
    for reason in reasons:
        if reason:
            return reason
    for trace in task_traces:
        reason = _optional_str(trace.get("reason") or trace.get("skip_reason"))
        if reason:
            return reason
    return "no_ready_graph_task" if task_traces else "no_graph_task"


def _first_text(*values: Any) -> str | None:
    for value in values:
        text = _optional_str(value)
        if text:
            return text
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _string_tuple(value: Any) -> tuple[str, ...]:
    return tuple(_dedupe_strings(_as_sequence(value)))


def _first_sequence(*values: Any) -> tuple[str, ...]:
    for value in values:
        sequence = _string_tuple(value)
        if sequence:
            return sequence
    return ()


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    if isinstance(value, set):
        return tuple(value)
    return (value,)


def _positive_int(*values: Any, default: int) -> int:
    for value in values:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return default


def _int_or_none(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _int_or_zero(*values: Any) -> int:
    value = _int_or_none(*values)
    return value if value is not None else 0


def _float_or_zero(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _average(values: Iterable[float]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return sum(items) / len(items)


def _metadata_value(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    return str(value) if value is not None else None


__all__ = ["GraphProvider"]

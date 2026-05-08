from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


GRAPH_INDEX_STATUS_PENDING = "pending"
WHOLE_CHUNK_TEXT_SPAN_HASH = "whole_chunk"
LLM_CALL_RAW_GOVERNANCE_CHECK = """
(
    (
        nullif(instructions_text, '') is null
        and nullif(input_text, '') is null
        and nullif(raw_output_text, '') is null
        and nullif(parsed_answer_text, '') is null
        and coalesce(request_json, '{}'::jsonb) = '{}'::jsonb
        and coalesce(response_json, '{}'::jsonb) = '{}'::jsonb
    )
    or (
        nullif(raw_payload_hash, '') is not null
        and raw_retention_expires_at is not null
        and raw_retention_expires_at > created_at
        and nullif(btrim(raw_redaction_status), '') is not null
        and lower(raw_redaction_status) <> 'unknown'
        and nullif(btrim(raw_encryption_status), '') is not null
        and lower(raw_encryption_status) <> 'unknown'
    )
)
"""
LLM_CALL_EVIDENCE_SNAPSHOT_GOVERNANCE_CHECK = """
(
    nullif(text_snapshot, '') is null
    or (
        nullif(text_hash, '') is not null
        and snapshot_retention_expires_at is not null
        and snapshot_retention_expires_at > created_at
        and nullif(btrim(snapshot_redaction_status), '') is not null
        and lower(snapshot_redaction_status) <> 'unknown'
        and nullif(btrim(snapshot_encryption_status), '') is not null
        and lower(snapshot_encryption_status) <> 'unknown'
    )
)
"""
CITATION_AUDIT_SNAPSHOT_GOVERNANCE_CHECK = """
(
    nullif(supporting_text_snapshot, '') is null
    or (
        nullif(supporting_text_hash, '') is not null
        and snapshot_retention_expires_at is not null
        and snapshot_retention_expires_at > created_at
        and nullif(btrim(snapshot_redaction_status), '') is not null
        and lower(snapshot_redaction_status) <> 'unknown'
        and nullif(btrim(snapshot_encryption_status), '') is not null
        and lower(snapshot_encryption_status) <> 'unknown'
    )
)
"""
QUALITY_REVIEW_PAYLOAD_GOVERNANCE_CHECK = """
(
    (
        coalesce(payload_json, '{}'::jsonb) = '{}'::jsonb
        and nullif(planner_verdict, '') is null
        and nullif(evidence_relevance_verdict, '') is null
        and nullif(answer_faithfulness_verdict, '') is null
        and nullif(citation_verdict, '') is null
        and nullif(issues_text, '') is null
        and nullif(recommendations_text, '') is null
    )
    or (
        nullif(payload_hash, '') is not null
        and payload_retention_expires_at is not null
        and payload_retention_expires_at > created_at
        and nullif(btrim(payload_redaction_status), '') is not null
        and lower(payload_redaction_status) <> 'unknown'
        and nullif(btrim(payload_encryption_status), '') is not null
        and lower(payload_encryption_status) <> 'unknown'
    )
)
"""


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    source_uri: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_type: Mapped[str] = mapped_column(String(32), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, unique=True, index=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    parent_blocks: Mapped[list["ParentBlock"]] = relationship(back_populates="document")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="document")


class ParentBlock(Base):
    __tablename__ = "parent_blocks"
    __table_args__ = (
        Index("ix_parent_blocks_document_id", "document_id"),
        Index("ix_parent_blocks_page_range", "document_id", "page_start", "page_end"),
    )

    parent_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id"), nullable=False)
    parent_type: Mapped[str] = mapped_column(String(32), nullable=False)
    page_start: Mapped[int] = mapped_column(Integer, nullable=False)
    page_end: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    child_ids_json: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)

    document: Mapped[Document] = relationship(back_populates="parent_blocks")
    chunks: Mapped[list["Chunk"]] = relationship(back_populates="parent_block")


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunks_document_index"),
        Index("ix_chunks_document_id", "document_id"),
        Index("ix_chunks_parent_id", "parent_id"),
        Index("ix_chunks_text_hash", "text_hash"),
    )

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id"), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(
        ForeignKey("parent_blocks.parent_id"),
        nullable=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    text_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    section_title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(256), nullable=False)
    embedding_dim: Mapped[int] = mapped_column(Integer, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    document: Mapped[Document] = relationship(back_populates="chunks")
    parent_block: Mapped[ParentBlock | None] = relationship(back_populates="chunks")


class GraphIndex(Base):
    __tablename__ = "graph_indexes"

    graph_version: Mapped[str] = mapped_column(String(128), primary_key=True)
    corpus_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    fixture_schema_version: Mapped[str] = mapped_column(String(64), nullable=False)
    fixture_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    loader_version: Mapped[str] = mapped_column(String(64), nullable=False)
    row_counts_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(
        String(32),
        default=GRAPH_INDEX_STATUS_PENDING,
        server_default=text("'pending'"),
        nullable=False,
    )
    loaded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=text("now()"),
        nullable=False,
    )


class GraphEntityRecord(Base):
    __tablename__ = "graph_entities"
    __table_args__ = (
        PrimaryKeyConstraint("graph_version", "entity_id", name="pk_graph_entities"),
        UniqueConstraint(
            "graph_version",
            "entity_type",
            "canonical_name_norm",
            name="uq_graph_entities_version_type_name",
        ),
        Index("ix_graph_entities_graph_version", "graph_version"),
        Index("ix_graph_entities_entity_type", "entity_type"),
        Index("ix_graph_entities_canonical_name_norm", "canonical_name_norm"),
    )

    graph_version: Mapped[str] = mapped_column(
        ForeignKey(
            "graph_indexes.graph_version",
            name="fk_graph_entities_graph_version",
            ondelete="CASCADE",
        ),
        primary_key=True,
        nullable=False,
    )
    entity_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(512), nullable=False)
    canonical_name_norm: Mapped[str] = mapped_column(String(512), nullable=False)
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False)
    aliases_json: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        server_default=text("'[]'::jsonb"),
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=text("now()"),
        nullable=False,
    )


class GraphRelationshipRecord(Base):
    __tablename__ = "graph_relationships"
    __table_args__ = (
        PrimaryKeyConstraint("graph_version", "relationship_id", name="pk_graph_relationships"),
        ForeignKeyConstraint(
            ["graph_version", "source_entity_id"],
            ["graph_entities.graph_version", "graph_entities.entity_id"],
            name="fk_graph_relationships_source_entity",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["graph_version", "target_entity_id"],
            ["graph_entities.graph_version", "graph_entities.entity_id"],
            name="fk_graph_relationships_target_entity",
            ondelete="CASCADE",
        ),
        Index("ix_graph_relationships_graph_version", "graph_version"),
        Index("ix_graph_relationships_source", "source_entity_id"),
        Index("ix_graph_relationships_target", "target_entity_id"),
        Index("ix_graph_relationships_relation_type", "relation_type"),
        Index("ix_graph_relationships_graph_source", "graph_version", "source_entity_id"),
        Index("ix_graph_relationships_graph_target", "graph_version", "target_entity_id"),
        Index("ix_graph_relationships_graph_relation", "graph_version", "relation_type"),
    )

    graph_version: Mapped[str] = mapped_column(
        ForeignKey(
            "graph_indexes.graph_version",
            name="fk_graph_relationships_graph_version",
            ondelete="CASCADE",
        ),
        primary_key=True,
        nullable=False,
    )
    relationship_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source_entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    target_entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    relation_type: Mapped[str] = mapped_column(String(128), nullable=False)
    confidence: Mapped[float] = mapped_column(
        Float,
        default=1.0,
        server_default=text("1.0"),
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=text("now()"),
        nullable=False,
    )


class GraphEntityAnchor(Base):
    __tablename__ = "graph_entity_anchors"
    __table_args__ = (
        PrimaryKeyConstraint("graph_version", "anchor_id", name="pk_graph_entity_anchors"),
        ForeignKeyConstraint(
            ["graph_version", "entity_id"],
            ["graph_entities.graph_version", "graph_entities.entity_id"],
            name="fk_graph_entity_anchors_entity",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "graph_version",
            "entity_id",
            "chunk_id",
            "text_span_hash",
            name="uq_graph_entity_anchors_entity_chunk_span",
        ),
        Index("ix_graph_entity_anchors_graph_version", "graph_version"),
        Index("ix_graph_entity_anchors_entity_id", "entity_id"),
        Index("ix_graph_entity_anchors_graph_entity", "graph_version", "entity_id"),
        Index("ix_graph_entity_anchors_chunk_id", "chunk_id"),
    )

    anchor_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    graph_version: Mapped[str] = mapped_column(String(128), primary_key=True, nullable=False)
    entity_id: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey(
            "chunks.chunk_id",
            name="fk_graph_entity_anchors_chunk",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    text_span: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_span_hash: Mapped[str] = mapped_column(
        String(128),
        default=WHOLE_CHUNK_TEXT_SPAN_HASH,
        server_default=text("'whole_chunk'"),
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=text("now()"),
        nullable=False,
    )


class GraphRelationshipAnchor(Base):
    __tablename__ = "graph_relationship_anchors"
    __table_args__ = (
        PrimaryKeyConstraint(
            "graph_version",
            "anchor_id",
            name="pk_graph_relationship_anchors",
        ),
        ForeignKeyConstraint(
            ["graph_version", "relationship_id"],
            ["graph_relationships.graph_version", "graph_relationships.relationship_id"],
            name="fk_graph_relationship_anchors_relationship",
            ondelete="CASCADE",
        ),
        UniqueConstraint(
            "graph_version",
            "relationship_id",
            "chunk_id",
            "text_span_hash",
            name="uq_graph_relationship_anchors_relationship_chunk_span",
        ),
        Index("ix_graph_relationship_anchors_graph_version", "graph_version"),
        Index("ix_graph_relationship_anchors_relationship_id", "relationship_id"),
        Index(
            "ix_graph_relationship_anchors_graph_relationship",
            "graph_version",
            "relationship_id",
        ),
        Index("ix_graph_relationship_anchors_chunk_id", "chunk_id"),
    )

    anchor_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    graph_version: Mapped[str] = mapped_column(String(128), primary_key=True, nullable=False)
    relationship_id: Mapped[str] = mapped_column(String(128), nullable=False)
    chunk_id: Mapped[str] = mapped_column(
        ForeignKey(
            "chunks.chunk_id",
            name="fk_graph_relationship_anchors_chunk",
            ondelete="CASCADE",
        ),
        nullable=False,
    )
    text_span: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_span_hash: Mapped[str] = mapped_column(
        String(128),
        default=WHOLE_CHUNK_TEXT_SPAN_HASH,
        server_default=text("'whole_chunk'"),
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=text("now()"),
        nullable=False,
    )


class GraphCommunity(Base):
    __tablename__ = "graph_communities"
    __table_args__ = (
        PrimaryKeyConstraint("graph_version", "community_id", name="pk_graph_communities"),
        Index("ix_graph_communities_graph_version", "graph_version"),
        Index("ix_graph_communities_graph_level", "graph_version", "level"),
    )

    graph_version: Mapped[str] = mapped_column(
        ForeignKey(
            "graph_indexes.graph_version",
            name="fk_graph_communities_graph_version",
            ondelete="CASCADE",
        ),
        primary_key=True,
        nullable=False,
    )
    community_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=text("now()"),
        nullable=False,
    )


class IngestionRun(Base):
    __tablename__ = "ingestion_runs"

    ingestion_run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    input_paths_json: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    document_ids_json: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class QueryRun(Base):
    __tablename__ = "query_runs"
    __table_args__ = (Index("ix_query_runs_trace_id", "trace_id"),)

    query_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    user_query: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    model_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    retrieval_events: Mapped[list["RetrievalEvent"]] = relationship(back_populates="query_run")
    generation_events: Mapped[list["GenerationEvent"]] = relationship(back_populates="query_run")
    query_plans: Mapped[list["QueryPlanRecord"]] = relationship(back_populates="query_run")
    retrieval_tasks: Mapped[list["RetrievalTaskRecord"]] = relationship(back_populates="query_run")
    retrieval_results: Mapped[list["RetrievalResultRecord"]] = relationship(back_populates="query_run")
    candidates: Mapped[list["CandidateRecord"]] = relationship(back_populates="query_run")
    evidence_blocks: Mapped[list["EvidenceBlockRecord"]] = relationship(back_populates="query_run")
    evidence_packs: Mapped[list["EvidencePackRecord"]] = relationship(back_populates="query_run")
    evidence_evaluations: Mapped[list["EvidenceEvaluationRecord"]] = relationship(back_populates="query_run")
    answers: Mapped[list["AnswerRecord"]] = relationship(back_populates="query_run")
    citations: Mapped[list["CitationRecord"]] = relationship(back_populates="query_run")
    citation_verifications: Mapped[list["CitationVerificationRecord"]] = relationship(back_populates="query_run")
    llm_calls: Mapped[list["LLMCallRecord"]] = relationship(back_populates="query_run")
    llm_call_evidence: Mapped[list["LLMCallEvidenceRecord"]] = relationship(
        back_populates="query_run",
        foreign_keys="LLMCallEvidenceRecord.query_id",
        overlaps="evidence_records,llm_call",
    )
    citation_audits: Mapped[list["CitationAuditRecord"]] = relationship(
        back_populates="query_run"
    )
    quality_reviews: Mapped[list["QualityReviewRecord"]] = relationship(
        back_populates="query_run"
    )


class RetrievalEvent(Base):
    __tablename__ = "retrieval_events"
    __table_args__ = (Index("ix_retrieval_events_query_id", "query_id"),)

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_score: Mapped[float] = mapped_column(Float, nullable=False)
    retriever_type: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="retrieval_events")


class GenerationEvent(Base):
    __tablename__ = "generation_events"
    __table_args__ = (Index("ix_generation_events_query_id", "query_id"),)

    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="generation_events")


class LLMCallRecord(Base):
    __tablename__ = "llm_calls"
    __table_args__ = (
        PrimaryKeyConstraint("call_id", name="pk_llm_calls"),
        ForeignKeyConstraint(
            ["query_id"],
            ["query_runs.query_id"],
            name="fk_llm_calls_query",
            ondelete="CASCADE",
        ),
        UniqueConstraint("call_id", "query_id", name="uq_llm_calls_call_query"),
        CheckConstraint(
            LLM_CALL_RAW_GOVERNANCE_CHECK,
            name="ck_llm_calls_raw_governance",
        ),
        Index("ix_llm_calls_query_id", "query_id"),
        Index("ix_llm_calls_stage", "stage"),
        Index("ix_llm_calls_query_stage", "query_id", "stage"),
        Index("ix_llm_calls_query_stage_attempt", "query_id", "stage", "attempt_index"),
        Index("ix_llm_calls_query_sequence", "query_id", "sequence_index"),
    )

    call_id: Mapped[str] = mapped_column(String(64), nullable=False)
    query_id: Mapped[str] = mapped_column(String(64), nullable=False)
    stage: Mapped[str] = mapped_column(String(32), nullable=False)
    attempt_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    sequence_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    planner_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    request_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    response_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    usage_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    instructions_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    raw_output_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_answer_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    parsed_plan_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    validation_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    max_output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reasoning_effort: Mapped[str | None] = mapped_column(String(64), nullable=True)
    store: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    raw_payload_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    raw_redaction_status: Mapped[str] = mapped_column(
        String(64),
        default="unredacted",
        server_default=text("'unredacted'"),
        nullable=False,
    )
    raw_encryption_status: Mapped[str] = mapped_column(
        String(64),
        default="plaintext",
        server_default=text("'plaintext'"),
        nullable=False,
    )
    raw_retention_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=text("now()"),
        nullable=False,
    )

    query_run: Mapped[QueryRun] = relationship(back_populates="llm_calls")
    evidence_records: Mapped[list["LLMCallEvidenceRecord"]] = relationship(
        back_populates="llm_call",
        foreign_keys="[LLMCallEvidenceRecord.call_id, LLMCallEvidenceRecord.query_id]",
        overlaps="llm_call_evidence,query_run",
    )


class LLMCallEvidenceRecord(Base):
    __tablename__ = "llm_call_evidence"
    __table_args__ = (
        PrimaryKeyConstraint("record_id", name="pk_llm_call_evidence"),
        UniqueConstraint(
            "record_id",
            "query_id",
            name="uq_llm_call_evidence_record_query",
        ),
        UniqueConstraint("call_id", "rank", name="uq_llm_call_evidence_call_rank"),
        CheckConstraint(
            LLM_CALL_EVIDENCE_SNAPSHOT_GOVERNANCE_CHECK,
            name="ck_llm_call_evidence_snapshot_governance",
        ),
        ForeignKeyConstraint(
            ["call_id", "query_id"],
            ["llm_calls.call_id", "llm_calls.query_id"],
            name="fk_llm_call_evidence_call_query",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["query_id"],
            ["query_runs.query_id"],
            name="fk_llm_call_evidence_query",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["evidence_block_record_id", "query_id"],
            ["evidence_blocks.record_id", "evidence_blocks.query_id"],
            name="fk_llm_call_evidence_evidence_block_query",
            ondelete="NO ACTION",
        ),
        Index("ix_llm_call_evidence_query_id", "query_id"),
        Index("ix_llm_call_evidence_call_id", "call_id"),
        Index("ix_llm_call_evidence_call_rank", "call_id", "rank"),
        Index("ix_llm_call_evidence_chunk_id", "chunk_id"),
        Index("ix_llm_call_evidence_evidence_id", "evidence_id"),
        Index("ix_llm_call_evidence_evidence_block_record_id", "evidence_block_record_id"),
        Index(
            "ix_llm_call_evidence_query_evidence_block_record",
            "query_id",
            "evidence_block_record_id",
        ),
    )

    record_id: Mapped[str] = mapped_column(String(64), nullable=False)
    call_id: Mapped[str] = mapped_column(String(64), nullable=False)
    query_id: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_id: Mapped[str] = mapped_column(String(64), nullable=False)
    evidence_block_record_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    document_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    text_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    snapshot_redaction_status: Mapped[str] = mapped_column(
        String(64),
        default="unredacted",
        server_default=text("'unredacted'"),
        nullable=False,
    )
    snapshot_encryption_status: Mapped[str] = mapped_column(
        String(64),
        default="plaintext",
        server_default=text("'plaintext'"),
        nullable=False,
    )
    snapshot_retention_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=text("now()"),
        nullable=False,
    )

    query_run: Mapped[QueryRun] = relationship(
        back_populates="llm_call_evidence",
        foreign_keys=[query_id],
        overlaps="evidence_records,llm_call",
    )
    llm_call: Mapped[LLMCallRecord] = relationship(
        back_populates="evidence_records",
        foreign_keys=[call_id, query_id],
        overlaps="llm_call_evidence,query_run",
    )


class QueryPlanRecord(Base):
    __tablename__ = "query_plans"
    __table_args__ = (
        ForeignKeyConstraint(
            ["planner_call_id", "query_id"],
            ["llm_calls.call_id", "llm_calls.query_id"],
            name="fk_query_plans_planner_call_query",
            ondelete="NO ACTION",
        ),
        UniqueConstraint("record_id", "query_id", name="uq_query_plans_record_query"),
        Index("ix_query_plans_query_id", "query_id"),
        Index("ix_query_plans_planner_call_id", "planner_call_id"),
        Index("ix_query_plans_query_planner_call", "query_id", "planner_call_id"),
    )

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    planner_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    plan_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    planner: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="query_plans")


class RetrievalTaskRecord(Base):
    __tablename__ = "retrieval_tasks"
    __table_args__ = (Index("ix_retrieval_tasks_query_id", "query_id"),)

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    unit_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="retrieval_tasks")


class RetrievalResultRecord(Base):
    __tablename__ = "retrieval_results"
    __table_args__ = (Index("ix_retrieval_results_query_id", "query_id"),)

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="retrieval_results")


class CandidateRecord(Base):
    __tablename__ = "candidates"
    __table_args__ = (Index("ix_candidates_query_id", "query_id"),)

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="candidates")


class EvidenceBlockRecord(Base):
    __tablename__ = "evidence_blocks"
    __table_args__ = (
        UniqueConstraint("record_id", "query_id", name="uq_evidence_blocks_record_query"),
        UniqueConstraint("evidence_id", "query_id", name="uq_evidence_blocks_evidence_query"),
        Index("ix_evidence_blocks_query_id", "query_id"),
        Index("ix_evidence_blocks_evidence_id", "evidence_id"),
        Index("ix_evidence_blocks_query_evidence", "query_id", "evidence_id"),
    )

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    evidence_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="evidence_blocks")


class EvidencePackRecord(Base):
    __tablename__ = "evidence_packs"
    __table_args__ = (Index("ix_evidence_packs_query_id", "query_id"),)

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    pack_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="evidence_packs")


class EvidenceEvaluationRecord(Base):
    __tablename__ = "evidence_evaluations"
    __table_args__ = (Index("ix_evidence_evaluations_query_id", "query_id"),)

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="evidence_evaluations")


class AnswerRecord(Base):
    __tablename__ = "answers"
    __table_args__ = (
        ForeignKeyConstraint(
            ["answer_call_id", "query_id"],
            ["llm_calls.call_id", "llm_calls.query_id"],
            name="fk_answers_answer_call_query",
            ondelete="NO ACTION",
        ),
        UniqueConstraint("record_id", "query_id", name="uq_answers_record_query"),
        Index("ix_answers_query_id", "query_id"),
        Index("ix_answers_answer_call_id", "answer_call_id"),
        Index("ix_answers_query_answer_call", "query_id", "answer_call_id"),
    )

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    answer_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="answers")


class CitationRecord(Base):
    __tablename__ = "citations"
    __table_args__ = (
        UniqueConstraint("record_id", "query_id", name="uq_citations_record_query"),
        Index("ix_citations_query_id", "query_id"),
    )

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    citation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="citations")


class CitationVerificationRecord(Base):
    __tablename__ = "citation_verifications"
    __table_args__ = (
        UniqueConstraint(
            "record_id",
            "query_id",
            name="uq_citation_verifications_record_query",
        ),
        Index("ix_citation_verifications_query_id", "query_id"),
    )

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="citation_verifications")


class CitationAuditRecord(Base):
    __tablename__ = "citation_audits"
    __table_args__ = (
        PrimaryKeyConstraint("record_id", name="pk_citation_audits"),
        ForeignKeyConstraint(
            ["query_id"],
            ["query_runs.query_id"],
            name="fk_citation_audits_query",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["citation_record_id", "query_id"],
            ["citations.record_id", "citations.query_id"],
            name="fk_citation_audits_citation_record_query",
        ),
        ForeignKeyConstraint(
            ["llm_call_evidence_record_id", "query_id"],
            ["llm_call_evidence.record_id", "llm_call_evidence.query_id"],
            name="fk_citation_audits_llm_call_evidence_query",
        ),
        ForeignKeyConstraint(
            ["answer_record_id", "query_id"],
            ["answers.record_id", "answers.query_id"],
            name="fk_citation_audits_answer_record_query",
            ondelete="NO ACTION",
        ),
        ForeignKeyConstraint(
            ["answer_call_id", "query_id"],
            ["llm_calls.call_id", "llm_calls.query_id"],
            name="fk_citation_audits_answer_call_query",
            ondelete="NO ACTION",
        ),
        ForeignKeyConstraint(
            ["citation_verification_record_id", "query_id"],
            ["citation_verifications.record_id", "citation_verifications.query_id"],
            name="fk_citation_audits_citation_verification_query",
            ondelete="NO ACTION",
        ),
        CheckConstraint(
            CITATION_AUDIT_SNAPSHOT_GOVERNANCE_CHECK,
            name="ck_citation_audits_snapshot_governance",
        ),
        Index("ix_citation_audits_query_id", "query_id"),
        Index("ix_citation_audits_citation_id", "citation_id"),
        Index("ix_citation_audits_citation_record_id", "citation_record_id"),
        Index("ix_citation_audits_evidence_id", "evidence_id"),
        Index(
            "ix_citation_audits_llm_call_evidence_record_id",
            "llm_call_evidence_record_id",
        ),
        Index("ix_citation_audits_answer_record_id", "answer_record_id"),
        Index("ix_citation_audits_answer_call_id", "answer_call_id"),
        Index(
            "ix_citation_audits_citation_verification_record_id",
            "citation_verification_record_id",
        ),
        Index("ix_citation_audits_query_citation", "query_id", "citation_id"),
        Index(
            "ix_citation_audits_query_citation_record",
            "query_id",
            "citation_record_id",
        ),
        Index("ix_citation_audits_query_evidence", "query_id", "evidence_id"),
        Index(
            "ix_citation_audits_query_llm_call_evidence_record",
            "query_id",
            "llm_call_evidence_record_id",
        ),
        Index("ix_citation_audits_query_answer_record", "query_id", "answer_record_id"),
        Index("ix_citation_audits_query_answer_call", "query_id", "answer_call_id"),
        Index(
            "ix_citation_audits_query_citation_verification",
            "query_id",
            "citation_verification_record_id",
        ),
    )

    record_id: Mapped[str] = mapped_column(String(64), nullable=False)
    query_id: Mapped[str] = mapped_column(String(64), nullable=False)
    citation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    citation_record_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    citation_verification_record_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    answer_record_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    answer_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    llm_call_evidence_record_id: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    chunk_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    verifier_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    unsupported_numbers_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    issue_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    supporting_text_snapshot: Mapped[str | None] = mapped_column(Text, nullable=True)
    supporting_text_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    snapshot_redaction_status: Mapped[str] = mapped_column(
        String(64),
        default="unredacted",
        server_default=text("'unredacted'"),
        nullable=False,
    )
    snapshot_encryption_status: Mapped[str] = mapped_column(
        String(64),
        default="plaintext",
        server_default=text("'plaintext'"),
        nullable=False,
    )
    snapshot_retention_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=text("now()"),
        nullable=False,
    )

    query_run: Mapped[QueryRun] = relationship(back_populates="citation_audits")


class QualityReviewRecord(Base):
    __tablename__ = "quality_reviews"
    __table_args__ = (
        PrimaryKeyConstraint("record_id", name="pk_quality_reviews"),
        ForeignKeyConstraint(
            ["query_id"],
            ["query_runs.query_id"],
            name="fk_quality_reviews_query",
            ondelete="CASCADE",
        ),
        ForeignKeyConstraint(
            ["answer_record_id", "query_id"],
            ["answers.record_id", "answers.query_id"],
            name="fk_quality_reviews_answer_record_query",
        ),
        ForeignKeyConstraint(
            ["planner_call_id", "query_id"],
            ["llm_calls.call_id", "llm_calls.query_id"],
            name="fk_quality_reviews_planner_call_query",
        ),
        ForeignKeyConstraint(
            ["answer_call_id", "query_id"],
            ["llm_calls.call_id", "llm_calls.query_id"],
            name="fk_quality_reviews_answer_call_query",
        ),
        ForeignKeyConstraint(
            ["review_call_id", "query_id"],
            ["llm_calls.call_id", "llm_calls.query_id"],
            name="fk_quality_reviews_review_call_query",
            ondelete="NO ACTION",
        ),
        CheckConstraint(
            QUALITY_REVIEW_PAYLOAD_GOVERNANCE_CHECK,
            name="ck_quality_reviews_payload_governance",
        ),
        Index("ix_quality_reviews_query_id", "query_id"),
        Index("ix_quality_reviews_answer_record_id", "answer_record_id"),
        Index("ix_quality_reviews_planner_call_id", "planner_call_id"),
        Index("ix_quality_reviews_answer_call_id", "answer_call_id"),
        Index("ix_quality_reviews_review_call_id", "review_call_id"),
        Index("ix_quality_reviews_query_answer_record", "query_id", "answer_record_id"),
        Index("ix_quality_reviews_query_planner_call", "query_id", "planner_call_id"),
        Index("ix_quality_reviews_query_answer_call", "query_id", "answer_call_id"),
        Index("ix_quality_reviews_query_review_call", "query_id", "review_call_id"),
        Index("ix_quality_reviews_status", "status"),
    )

    record_id: Mapped[str] = mapped_column(String(64), nullable=False)
    query_id: Mapped[str] = mapped_column(String(64), nullable=False)
    answer_record_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    planner_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    answer_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    review_call_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    reviewer: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(64), nullable=False)
    planner_verdict: Mapped[str | None] = mapped_column(Text, nullable=True)
    evidence_relevance_verdict: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer_faithfulness_verdict: Mapped[str | None] = mapped_column(Text, nullable=True)
    citation_verdict: Mapped[str | None] = mapped_column(Text, nullable=True)
    issues_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommendations_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        server_default=text("'{}'::jsonb"),
        nullable=False,
    )
    payload_hash: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload_redaction_status: Mapped[str] = mapped_column(
        String(64),
        default="unredacted",
        server_default=text("'unredacted'"),
        nullable=False,
    )
    payload_encryption_status: Mapped[str] = mapped_column(
        String(64),
        default="plaintext",
        server_default=text("'plaintext'"),
        nullable=False,
    )
    payload_retention_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        server_default=text("now()"),
        nullable=False,
    )

    query_run: Mapped[QueryRun] = relationship(back_populates="quality_reviews")


class QueryCache(Base):
    __tablename__ = "query_cache"
    __table_args__ = (Index("ix_query_cache_expires_at", "expires_at"),)

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    citations_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    hit_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)


class EvalRun(Base):
    __tablename__ = "eval_runs"

    eval_run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    cases_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_cases: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source_hits: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    confidence_hits: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    average_keyword_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    average_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    results: Mapped[list["EvalResult"]] = relationship(back_populates="eval_run")


class EvalResult(Base):
    __tablename__ = "eval_results"
    __table_args__ = (Index("ix_eval_results_eval_run_id", "eval_run_id"),)

    eval_result_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    eval_run_id: Mapped[str] = mapped_column(ForeignKey("eval_runs.eval_run_id"), nullable=False)
    case_id: Mapped[str] = mapped_column(String(128), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    query_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    trace_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    expected_confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    actual_confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    source_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    confidence_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    keyword_score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    details_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    eval_run: Mapped[EvalRun] = relationship(back_populates="results")

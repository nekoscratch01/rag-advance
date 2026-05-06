from datetime import UTC, datetime
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


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


class QueryPlanRecord(Base):
    __tablename__ = "query_plans"
    __table_args__ = (Index("ix_query_plans_query_id", "query_id"),)

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
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
    __table_args__ = (Index("ix_evidence_blocks_query_id", "query_id"),)

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
    __table_args__ = (Index("ix_answers_query_id", "query_id"),)

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="answers")


class CitationRecord(Base):
    __tablename__ = "citations"
    __table_args__ = (Index("ix_citations_query_id", "query_id"),)

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    citation_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    evidence_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="citations")


class CitationVerificationRecord(Base):
    __tablename__ = "citation_verifications"
    __table_args__ = (Index("ix_citation_verifications_query_id", "query_id"),)

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    query_id: Mapped[str] = mapped_column(ForeignKey("query_runs.query_id"), nullable=False)
    status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    query_run: Mapped[QueryRun] = relationship(back_populates="citation_verifications")


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

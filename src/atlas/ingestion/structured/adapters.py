from __future__ import annotations

from dataclasses import asdict, is_dataclass

from atlas.ingestion.contracts import LoadedDocument, StructuredArtifact
from atlas.ingestion.structured.tables import csv_to_table_ir_and_cards


def csv_structured_artifacts(loaded: LoadedDocument) -> list[StructuredArtifact]:
    result = csv_to_table_ir_and_cards(
        loaded.path,
        source_uri=f"local:{loaded.path}",
        table_name=loaded.title,
    )
    artifacts = [
        _structured_artifact(
            artifact_type="table",
            payload=_contract_payload(result.table_ir),
            metadata={"source_file_type": loaded.file_type},
        )
    ]
    artifacts.extend(
        _structured_artifact(
            artifact_type="schema_routing_card",
            payload=_contract_payload(card),
            metadata={"source_file_type": loaded.file_type},
        )
        for card in result.cards
    )
    return artifacts


def _structured_artifact(
    *,
    artifact_type: str,
    payload: dict,
    metadata: dict,
) -> StructuredArtifact:
    artifact_id = (
        payload.get("artifact_id")
        or payload.get("card_id")
        or payload.get("table_id")
        or payload.get("id")
    )
    try:
        return StructuredArtifact(
            artifact_type=artifact_type,
            payload=payload,
            artifact_id=str(artifact_id) if artifact_id else None,
            content_hash=payload.get("content_hash"),
            source_locator=payload.get("source_locator"),
            schema_routing_card=payload if artifact_type == "schema_routing_card" else None,
            envelope_version="v4.phase1",
            metadata=metadata,
        )
    except TypeError:
        return StructuredArtifact(artifact_type=artifact_type, payload=payload)


def _contract_payload(value: object) -> dict:
    if hasattr(value, "to_payload"):
        payload = value.to_payload()
    elif is_dataclass(value):
        payload = asdict(value)
    elif isinstance(value, dict):
        payload = dict(value)
    else:
        payload = dict(getattr(value, "__dict__", {}) or {})
    return payload if isinstance(payload, dict) else {"value": payload}

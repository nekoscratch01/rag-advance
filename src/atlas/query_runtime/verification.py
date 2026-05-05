from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


VerificationStage = Literal["pre_generation", "post_generation", "citation"]
VerificationStatus = Literal["supported", "insufficient", "contradicted", "partially_supported", "unsupported", "warning", "ok"]


@dataclass(frozen=True)
class VerificationResult:
    verification_id: str
    stage: VerificationStage
    status: VerificationStatus
    confidence_override: str | None = None
    warnings: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    supported_evidence_ids: tuple[str, ...] = ()
    unsupported_evidence_ids: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        return self.status in {"insufficient", "contradicted", "unsupported"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "verification_id": self.verification_id,
            "stage": self.stage,
            "status": self.status,
            "confidence_override": self.confidence_override,
            "warnings": list(self.warnings),
            "reasons": list(self.reasons),
            "supported_evidence_ids": list(self.supported_evidence_ids),
            "unsupported_evidence_ids": list(self.unsupported_evidence_ids),
            "details": dict(self.details),
        }

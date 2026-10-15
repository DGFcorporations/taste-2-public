"""taste-2 wire shapes — mirrors contract v1 (taste-bench/contract/models.py)
with A2 rich doc extension: Standard + examples[] + counterexamples[] in variant_meta.

Self-contained by design: variants are independent implementations, the
conformance suite is the arbiter of contract-equivalence.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(UTC)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


def _require_rationale(v: str) -> str:
    """NEVER rule 1: a rationale must carry content, not whitespace."""
    if not v or not v.strip():
        raise ValueError("rationale is required (NEVER rule 1)")
    return v


class Polarity(str, Enum):
    DO = "do"
    NEVER = "never"
    WHY = "why"


class Category(str, Enum):
    STRUCTURE = "structure"
    AESTHETIC = "aesthetic"
    PROCEDURAL = "procedural"
    DOCTRINE = "doctrine"
    ANTI = "anti"


class Provenance(str, Enum):
    PERSONA = "persona"
    DECISION = "decision"
    CORRECTION = "correction"
    INFERENCE = "inference"
    AMENDMENT = "amendment"


class LayerState(str, Enum):
    PROVISIONAL = "provisional"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETIRED = "retired"


class Standard(BaseModel):
    id: str = Field(default_factory=lambda: _id("std"))
    profile_id: str = "owner"
    category: Category
    polarity: Polarity
    directive: str
    rationale: str = ""
    confidence: float = 0.5
    provenance: Provenance = Provenance.PERSONA
    state: LayerState = LayerState.PROVISIONAL
    supersedes: list[str] = Field(default_factory=list)
    created: datetime = Field(default_factory=_now)
    updated: datetime = Field(default_factory=_now)
    variant_meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def examples(self) -> list[str]:
        return self.variant_meta.get("examples", [])

    @property
    def counterexamples(self) -> list[str]:
        return self.variant_meta.get("counterexamples", [])

    def with_examples(self, examples: list[str],
                      counterexamples: list[str] | None = None) -> Standard:
        meta = dict(self.variant_meta)
        meta["examples"] = examples
        if counterexamples is not None:
            meta["counterexamples"] = counterexamples
        self.variant_meta = meta
        return self


class Directive(BaseModel):
    polarity: Polarity
    text: str
    standard_id: str | None = None


class Exemplar(BaseModel):
    text: str
    why: str = ""


class Constraint(BaseModel):
    hard: bool = True
    text: str


class Packet(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    directives: list[Directive] = Field(default_factory=list)
    exemplars: list[Exemplar] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    free_degrees: list[str] = Field(default_factory=list)
    variant_meta: dict[str, Any] = Field(default_factory=dict)


class Candidate(BaseModel):
    label: str
    content_ref: str = ""
    content: str | None = None
    score: float = 0.0


class DecisionRecord(BaseModel):
    id: str = Field(default_factory=lambda: _id("dec"))
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    candidates: list[Candidate] = Field(default_factory=list)
    pick: str
    rationale: str
    agent_role: str = ""
    provenance: str = "owner_pick"
    ts: datetime = Field(default_factory=_now)
    variant_meta: dict[str, Any] = Field(default_factory=dict)

    _rationale_required = field_validator("rationale")(_require_rationale)


class InterviewRequest(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)


class InterviewQuestion(BaseModel):
    id: str
    text: str
    options: list[str] = Field(default_factory=list)
    resolves: str = ""


class InterviewResponse(BaseModel):
    questions: list[InterviewQuestion] = Field(default_factory=list)
    coverage: dict[str, list[str]] = Field(
        default_factory=lambda: {"resolved": [], "gaps": []}
    )


class MirrorRequest(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    interview_answers: dict[str, str] = Field(default_factory=dict)


class MirrorResponse(BaseModel):
    mirror: str
    assumptions: list[str] = Field(default_factory=list)
    vetoes_hit: list[str] = Field(default_factory=list)
    inferred: bool = False
    gate: str = "required"


class PacketRequest(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    answers: dict[str, str] = Field(default_factory=dict)
    guidelines: list[str] = Field(default_factory=list)


class ScoreRequest(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    candidates: list[Candidate]


class CandidateScore(BaseModel):
    ref: str
    score: float
    breakdown: dict[str, float] = Field(default_factory=dict)
    violations: list[str] = Field(default_factory=list)


class ScoreResult(BaseModel):
    scores: list[CandidateScore] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    profile_id: str = "owner"
    task_ctx: dict[str, Any] = Field(default_factory=dict)
    pick: str
    rationale: str = Field(min_length=1)  # empty -> 422, NEVER rule 1
    rejected: list[str] = Field(default_factory=list)

    _rationale_required = field_validator("rationale")(_require_rationale)


class FeedbackResponse(BaseModel):
    accepted: bool
    record_id: str | None = None
    extracted: dict[str, int] = Field(default_factory=dict)


class HoneRequest(BaseModel):
    profile_id: str = "owner"
    segment_ctx: dict[str, Any] = Field(default_factory=dict)
    draft: str | None = None


class HoneResponse(BaseModel):
    candidates: list[Candidate] = Field(default_factory=list)
    pick: str = ""
    score: float = 0.0
    iterate: bool = False


class IngestRequest(BaseModel):
    profile_id: str = "owner"
    docs: list[dict[str, str]] = Field(default_factory=list)


class Restatement(BaseModel):
    layer: Standard
    implications: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    candidates: list[Standard] = Field(default_factory=list)
    restatements: list[Restatement] = Field(default_factory=list)


class ImportRequest(BaseModel):
    profile_id: str = "owner"
    standards: list[Standard] = Field(default_factory=list)


class MetaResponse(BaseModel):
    name: str
    version: str
    contract: str = "v1"
    permutation: dict[str, str] = Field(default_factory=dict)

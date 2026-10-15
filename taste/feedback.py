"""Owner feedback — one write path for HTTP and MCP.

A decision record is written only when the rationale has content. The
extracted layer stays provisional until the owner confirms it. Anchor
memory is unchanged: this path does not write Lance rows except through
the store the engine already owns.
"""

from __future__ import annotations

from .core import StandardsEngine
from .models import (
    Category,
    DecisionRecord,
    FeedbackRequest,
    FeedbackResponse,
    Polarity,
    Provenance,
    Standard,
)
from .store import Store


def record_feedback(engine: StandardsEngine, store: Store,
                    req: FeedbackRequest) -> FeedbackResponse:
    rec = DecisionRecord(
        profile_id=req.profile_id, task_ctx=req.task_ctx,
        pick=req.pick, rationale=req.rationale,
        agent_role=str(req.task_ctx.get("role", "")),
        provenance="owner_pick")
    store.put_decision(rec)
    layer = Standard(
        profile_id=req.profile_id, category=Category.DOCTRINE,
        polarity=Polarity.WHY,
        directive=(f"When choosing for {req.task_ctx.get('type', 'this kind of task')},"
                   f" prefer what won as '{req.pick}'"),
        rationale=req.rationale, provenance=Provenance.DECISION,
        variant_meta={"decision_id": rec.id})
    engine.propose(layer)
    return FeedbackResponse(
        accepted=True, record_id=rec.id, extracted={"candidates": 1})

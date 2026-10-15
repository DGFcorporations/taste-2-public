"""mirror — intent echo before execution.

Attended: gate=required — nothing runs until the owner confirms.
Unattended: the engine infers from accumulated layers, marks inferred=true,
queues the inference for later review (every correction is a delta).
Includes elimination filter (flags active never-rules violated by task context).
"""

from __future__ import annotations

import re
from uuid import uuid4

from .core import StandardsEngine
from .models import MirrorRequest, MirrorResponse, Polarity
from .store import Store

_STOP = {"never", "avoid", "with", "from", "that", "this", "they", "have",
         "been", "when", "where", "what", "your", "their", "would", "should",
         "could", "don't", "like", "into", "them", "than"}


def _significant_terms(directive: str) -> set[str]:
    """Never-rule -> content terms for task-veto matching."""
    text = re.sub(r"^(never|avoid|do not|don't|no)\s+", "", directive.lower())
    terms = set()
    for w in re.findall(r"[a-z]{4,}", text):
        w = w.rstrip("s")
        if w not in _STOP:
            terms.add(w)
    return terms


def build_mirror(engine: StandardsEngine, store: Store,
                 req: MirrorRequest) -> MirrorResponse:
    ctx = req.task_ctx
    attended = bool(ctx.get("attended", True))
    task_type = ctx.get("type", "task")
    role = ctx.get("role", "agent")
    brief = ctx.get("brief", ctx.get("goal", ""))

    hits = engine.retrieve(
        f"{task_type} {role} {brief}".strip(), req.profile_id, top_n=6
    )
    do_rules = [s.directive for s, score in hits
                if s.polarity == Polarity.DO and score >= 0.012]
    never_rules = [s.directive for s, score in hits
                   if s.polarity == Polarity.NEVER and score >= 0.012]

    lines = [
        f"Goal: {brief or ('complete ' + task_type)}",
        f"Output shape: {task_type} via {role}.",
    ]
    if do_rules:
        lines.append("Will follow: " + "; ".join(do_rules[:4]))
    if never_rules:
        lines.append("Will avoid: " + "; ".join(never_rules[:4]))
    if req.interview_answers:
        lines.append(
            "Your answers: "
            + "; ".join(f"{k}={v}" for k, v in req.interview_answers.items())
        )
    mirror = "\n".join(lines)

    assumptions = []
    if not brief:
        assumptions.append("No explicit brief — goal inferred from task type")
    if not hits:
        assumptions.append("No taste coverage yet — first-principles run")

    # Elimination filter — detect if task itself violates an active never-rule
    task_terms = {
        t.rstrip("s")
        for t in re.findall(
            r"[a-z]{4,}",
            " ".join(str(v) for v in ctx.values() if isinstance(v, str)).lower(),
        )
    }
    task_terms -= _STOP
    vetoes: list[str] = []
    for s in engine.active(req.profile_id):
        if s.polarity != Polarity.NEVER:
            continue
        terms = _significant_terms(s.directive)
        if not terms or not task_terms:
            continue
        hits = terms & task_terms
        rule_cov = len(hits) / len(terms)
        task_cov = len(hits) / len(task_terms)
        # Two ways to fire, covering the two failure shapes:
        #   rule_cov >= 0.5 — half+ of the RULE's terms appear in the task:
        #     catches long rules whose terms are diluted by a big brief.
        #   >= 3 hits AND task_cov >= 0.5 — rule terms dominate a SHORT
        #     task brief even when the rule itself is longer: catches
        #     briefs that are mostly the forbidden thing.
        if hits and (rule_cov >= 0.5 or (len(hits) >= 3 and task_cov >= 0.5)):
            vetoes.append(
                f"task hits never-rule {s.id}: {sorted(hits)} of '{s.directive[:60]}'"
            )

    inferred = not attended
    resp = MirrorResponse(
        mirror=mirror,
        assumptions=assumptions,
        vetoes_hit=vetoes,
        inferred=inferred,
        gate="queued" if inferred else "required",
    )

    if inferred:
        store.enqueue_inferred(
            f"mir_{uuid4().hex[:8]}",
            req.profile_id,
            "mirror",
            {"task_ctx": ctx, "mirror": mirror, "assumptions": assumptions},
        )

    return resp

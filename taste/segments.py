"""segments — J2 permutation: dynamic task decomposition.

Proposes 3-5 segments from task_ctx; persisted per task for audit.
"""

from __future__ import annotations

from collections.abc import Callable

from .store import Store

Driver = Callable[[dict], list[str]]


def _default_dynamic_decomposer(task_ctx: dict) -> list[str]:
    """Dynamically propose 3-5 segments tailored to the specific task context."""
    # If explicitly supplied in context, validate and clamp to 3-5
    if "segments" in task_ctx and isinstance(task_ctx["segments"], list):
        custom = [str(s).strip() for s in task_ctx["segments"] if str(s).strip()]
        if 3 <= len(custom) <= 5:
            return custom
        if len(custom) > 5:
            return custom[:5]
        if 1 <= len(custom) < 3:
            # pad to 3
            while len(custom) < 3:
                custom.append(f"segment_{len(custom) + 1}")
            return custom

    task_type = str(task_ctx.get("type", "")).lower()
    role = str(task_ctx.get("role", "")).lower()
    goal = str(task_ctx.get("goal", task_ctx.get("brief", ""))).lower()

    if any(k in task_type for k in ("thread", "social", "post", "hook")):
        return ["hook", "premise", "evidence", "payoff"]
    if any(k in task_type for k in ("doc", "brief", "memo", "essay")):
        return ["executive_verdict", "structural_core", "elaboration", "next_actions"]
    if any(k in task_type for k in ("review", "audit", "security", "inspect")):
        return ["triage_scope", "critical_findings", "remediation_verdict"]
    if any(k in task_type for k in ("digest", "telemetry", "metric", "report")):
        return ["core_metrics", "anomaly_detection", "recommended_shift"]
    if any(k in task_type for k in ("code", "engineer", "build", "refactor")):
        return ["contract_spec", "core_implementation", "verification_gate"]

    # Role/goal based decomposition
    if "publisher" in role or "write" in goal:
        return ["lead_angle", "evidence_body", "concluding_synthesis"]

    return ["lead_hook", "core_argument", "operational_close"]


def propose_segments(task_ctx: dict, driver: Driver | None = None,
                     store: Store | None = None,
                     profile_id: str = "owner") -> list[str]:
    """Propose 3-5 segments dynamically and persist for audit."""
    if driver is not None:
        proposed = driver(task_ctx)
    else:
        proposed = _default_dynamic_decomposer(task_ctx)

    # Ensure bounds: 3 to 5 segments
    if len(proposed) < 3:
        while len(proposed) < 3:
            proposed.append(f"stage_{len(proposed) + 1}")
    elif len(proposed) > 5:
        proposed = proposed[:5]

    # Persist for audit if store is provided
    if store is not None:
        import hashlib
        import json
        ctx_hash = hashlib.sha256(json.dumps(task_ctx, sort_keys=True, default=str).encode()).hexdigest()[:12]
        task_id = str(task_ctx.get("id") or task_ctx.get("task_id") or task_ctx.get("ref") or f"task_{ctx_hash}")
        store.put_task_segments(task_id, profile_id, proposed)

    return proposed


def segments_for(task_type: str) -> list[str]:
    """Compatibility entry point."""
    return propose_segments({"type": task_type})

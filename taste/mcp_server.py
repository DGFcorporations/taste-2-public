"""taste-2 MCP server — the fleet interface.

The four contract tools call the same functions as their HTTP routes and
return the same JSON. Anchor retrieval stays inside compile_packet and
score_candidates. Importing this module does not open LanceDB.

Run: python -m taste.mcp_server   (env TASTE2_DB selects the store)
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

from mcp.server.mcpserver import MCPServer

from .core import StandardsEngine
from .feedback import record_feedback
from .inject import compile_packet
from .models import Candidate, FeedbackRequest, LayerState, PacketRequest
from .score import score_candidates
from .store import Store

TOOL_DESCRIPTIONS = {
    "taste_packet":
        "Compile the taste packet for a task — the bundle that travels "
        "with the work order.",
    "taste_score": "Score candidate output(s) against the owner's taste.",
    "taste_feedback":
        "Log an owner decision — rationale REQUIRED. This is how the "
        "corpus grows.",
    "taste_recall":
        "Recall taste standards for a profile. Same body as GET /standards.",
}



def _error(message: str) -> str:
    return json.dumps({"error": {"code": "e422", "message": message}})


def create_tools(store: Store, engine: StandardsEngine) -> dict[str, Callable]:
    def taste_check(profile_id: str = "owner") -> str:
        active = engine.active(profile_id)
        all_stds = store.list_standards(profile_id)
        cats: dict[str, int] = {}
        pols: dict[str, int] = {}
        for s in active:
            c = s.category.value if hasattr(s.category, "value") else str(s.category)
            p = s.polarity.value if hasattr(s.polarity, "value") else str(s.polarity)
            cats[c] = cats.get(c, 0) + 1
            pols[p] = pols.get(p, 0) + 1
        return json.dumps({
            "status": "healthy",
            "name": "taste-2",
            "active_standards": len(active),
            "total_standards": len(all_stds),
            "categories": cats,
            "polarities": pols,
        })

    def taste_packet(task_ctx: dict, profile_id: str = "owner",
                     answers: dict | None = None,
                     guidelines: list[str] | None = None) -> str:
        pkt = compile_packet(engine, PacketRequest(
            profile_id=profile_id, task_ctx=task_ctx,
            answers=answers or {}, guidelines=guidelines or []))
        return pkt.model_dump_json()

    def taste_score(candidates: list[dict], task_ctx: dict | None = None,
                    profile_id: str = "owner") -> str:
        cands = [Candidate(**c) for c in candidates]
        scored = score_candidates(engine, profile_id, task_ctx or {}, cands)
        return json.dumps({"scores": [s.model_dump(mode="json") for s in scored]})

    def taste_feedback(pick: str, rationale: str, task_ctx: dict | None = None,
                       profile_id: str = "owner",
                       rejected: list[str] | None = None) -> str:
        if not rationale or not str(rationale).strip():
            return _error("rationale is required (NEVER rule 1)")
        resp = record_feedback(engine, store, FeedbackRequest(
            profile_id=profile_id, task_ctx=task_ctx or {}, pick=pick,
            rationale=rationale, rejected=rejected or []))
        return resp.model_dump_json()

    def taste_recall(profile_id: str = "owner", state: str | None = None,
                     category: str | None = None) -> str:
        if category == "check":
            return taste_check(profile_id)
        from .models import Category
        try:
            st = LayerState(state) if state else None
        except ValueError:
            return _error(
                f"invalid state: {state!r} — expected one of "
                f"{[item.value for item in LayerState]}")
        try:
            cat = Category(category) if category else None
        except ValueError:
            return _error(f"invalid category: {category}")
        rows = store.list_standards(profile_id, state=st, category=cat)
        dumped = [row.model_dump(mode="json") for row in rows]
        return json.dumps({"standards": dumped, "count": len(dumped)})

    return {
        "taste_packet": taste_packet,
        "taste_score": taste_score,
        "taste_feedback": taste_feedback,
        "taste_recall": taste_recall,
    }




def create_server(store: Store, engine: StandardsEngine) -> MCPServer:
    server = MCPServer("taste-2")
    for name, fn in create_tools(store, engine).items():
        server.tool(description=TOOL_DESCRIPTIONS[name])(fn)
    return server


def main():
    store = Store()
    asyncio.run(create_server(store, StandardsEngine(store)).run_stdio_async())


if __name__ == "__main__":
    main()

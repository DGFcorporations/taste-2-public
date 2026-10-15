"""hone — F2 permutation: critique-revise (generate 1 -> self-critique vs directives -> revise -> score).

Fewer tokens, single lineage. The offline default critiques and revises
deterministically so the loop is fully testable offline without spend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable

from . import llm
from .core import StandardsEngine
from .inject import compile_packet
from .models import (
    Candidate,
    HoneRequest,
    HoneResponse,
    PacketRequest,
)
from .score import score_candidates

SCORE_THRESHOLD = 0.55
HONE_LIMIT = 15
log = logging.getLogger("taste.hone")


def task_key(ctx: dict) -> str:
    """Segments of one task share a hone budget. Attendance does not reset it."""
    for field in ("task_ref", "ref"):
        value = ctx.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    relevant = {key: value for key, value in ctx.items() if key not in {"segment", "attended"}}
    canonical = json.dumps(relevant, sort_keys=True, separators=(",", ":"), default=str)
    return "task_" + hashlib.sha256(canonical.encode()).hexdigest()

CritiqueReviseDriver = Callable[[str | None, dict, list[str]], tuple[str, str, str]]

HEDGE_WORDS = ["just", "really", "very", "basically", "actually",
               "in today's world", "it's important to note"]


def _offline_critique_revise(draft: str | None, directives: list[str],
                             segment: str) -> tuple[str, str, str]:
    """Deterministic generate -> critique -> revise implementation.

    Returns (initial_draft, critique, revised_draft).
    """
    base = (draft or "").strip()
    if not base:
        base = f"Draft for segment: {segment}."

    critique_items = []
    for w in HEDGE_WORDS:
        if re.search(rf"\b{re.escape(w)}\b", base, flags=re.I):
            critique_items.append(f"Remove hedge word '{w}'")

    if directives and not any(d.lower()[:15] in base.lower() for d in directives):
        critique_items.append(f"Incorporate directive: {directives[0]}")

    critique = "; ".join(critique_items) if critique_items else "Tighten phrasing for authority."

    # Revise according to critique
    revised = base
    for w in HEDGE_WORDS:
        revised = re.sub(rf"\b{re.escape(w)}\b[:,]?\s*", "", revised, flags=re.I)
    revised = re.sub(r"\s+", " ", revised).strip()

    if directives and not any(d.lower()[:15] in revised.lower() for d in directives):
        revised = f"{directives[0].rstrip('.')} — {revised}"

    return base, critique, revised


_CR_SYSTEM = ("You are a precise editor. Output JSON {draft: str, critique: str, revised: str}. "
              "draft = a first attempt (or the given draft unchanged), critique = how it "
              "violates the directives or misses their style, revised = the corrected text.")


def _llm_critique_revise(model: str, draft: str | None, directives: list[str],
                         segment: str) -> tuple[str, str, str]:
    """F2 with a real model: generate -> self-critique vs directives -> revise."""
    prompt = (f"SEGMENT: {segment}\n"
              f"DIRECTIVES:\n" + "\n".join(f"- {d}" for d in directives) + "\n"
              f"GIVEN DRAFT (may be empty): {draft or ''}")
    res = llm.chat_json(model, _CR_SYSTEM, prompt)
    first = str(res.get("draft") or draft or "").strip()
    revised = str(res.get("revised") or "").strip()
    if not revised:
        raise ValueError("model returned no revision")
    return first or revised, str(res.get("critique", "")), revised


def _default_driver(segment: str) -> CritiqueReviseDriver:
    def drive(draft, _ctx, rules):
        model = llm.judge_model()
        if model:
            try:
                return _llm_critique_revise(model, draft, rules, segment)
            except Exception as exc:  # degrade, never silently
                log.warning("LLM critique-revise failed (%s: %s); using offline driver",
                            type(exc).__name__, exc)
        return _offline_critique_revise(draft, rules, segment)
    return drive


def hone_segment(engine: StandardsEngine, req: HoneRequest,
                 driver: CritiqueReviseDriver | None = None) -> HoneResponse:
    reservation = engine.store.reserve_hone(
        req.profile_id, task_key(req.segment_ctx), HONE_LIMIT)
    ctx = {**req.segment_ctx, "attended": False}
    packet = compile_packet(engine, PacketRequest(
        profile_id=req.profile_id, task_ctx=ctx
    ))
    do_rules = [d.text for d in packet.directives if d.polarity.value == "do"]
    segment_name = str(req.segment_ctx.get("segment", "main"))

    gen = driver or _default_driver(segment_name)

    initial_draft, critique, revised_draft = gen(req.draft, req.segment_ctx, do_rules)

    candidates = [
        Candidate(label=chr(65 + i), content=t)
        for i, t in enumerate(
            t for t in (initial_draft, revised_draft) if t and t.strip())
    ]

    if not candidates:
        # driver returned nothing — an empty result is not an error
        return HoneResponse(candidates=[], pick="", score=0.0, iterate=False)

    scored = score_candidates(engine, req.profile_id, req.segment_ctx, candidates)
    if not scored:
        return HoneResponse(candidates=candidates, pick="", score=0.0,
                            iterate=False)
    score_map = {s.ref: s.score for s in scored}
    for c in candidates:
        c.score = score_map.get(c.label, 0.0)
    best = max(scored, key=lambda s: s.score)

    return HoneResponse(
        candidates=candidates,
        pick=best.ref,
        score=best.score,
        iterate=best.score < SCORE_THRESHOLD and reservation < HONE_LIMIT,
    )

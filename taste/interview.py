"""interview — E4 permutation: lazy interview.

Interview ONLY fires when coverage gaps exist AND (confidence < 0.7 on retrieved
layers OR retrieved layers conflict). Otherwise empty questions[] — zero friction.
"""

from __future__ import annotations

import re
from uuid import uuid4

from .core import StandardsEngine
from .models import InterviewQuestion, InterviewResponse, Polarity, Standard

GAP_QUESTIONS = {
    "structure": (
        "How should this be organized — what leads, what follows?",
        ["verdict first", "problem→solution", "chronological", "hierarchical"]),
    "aesthetic": (
        "What should this feel like — tone, density, register?",
        ["direct/operational", "warm/conversational", "formal/precise", "bold/punchy"]),
    "procedural": (
        "Any process rules for this — must-do steps or must-avoid moves?",
        []),
    "doctrine": (
        "Is there a principle at stake here — a line this output shouldn't cross?",
        []),
    "anti": (
        "What would make this output feel like slop to you — the thing to never do?",
        []),
}

MAX_QUESTIONS = 3
CONFIDENCE_THRESHOLD = 0.7


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]{4,}", text.lower())
    stop = {"never", "avoid", "with", "from", "that", "this", "they", "have",
            "been", "when", "where", "what", "your", "their", "would", "should",
            "could", "directive", "standard", "rule", "task", "tasks"}
    return {w.rstrip("s") for w in words if w not in stop}


def _layers_conflict(layers: list[Standard]) -> bool:
    """Detect if any retrieved layers have opposing polarity or contradictory directives."""
    nevers = [s for s in layers if s.polarity == Polarity.NEVER]
    dos = [s for s in layers if s.polarity == Polarity.DO]
    for n in nevers:
        n_words = _extract_keywords(n.directive)
        for d in dos:
            d_words = _extract_keywords(d.directive)
            overlap = n_words & d_words
            if not overlap:
                continue
            # Significant overlap or single shared substantive term for short rules indicates conflict
            if len(overlap) >= 2 or (len(overlap) >= 1 and (len(n_words) <= 2 or len(d_words) <= 2)):
                return True
    return False


def run_interview(engine: StandardsEngine, profile_id: str,
                  task_ctx: dict) -> InterviewResponse:
    query = " ".join(
        str(v) for v in task_ctx.values() if isinstance(v, str)
    ).strip() or "general task"
    cov = engine.coverage(profile_id, query)

    # If no coverage gaps exist, zero questions fired
    has_gaps = len(cov["gaps"]) > 0
    if not has_gaps:
        return InterviewResponse(questions=[], coverage=cov)

    hits = engine.retrieve(query, profile_id)
    retrieved_layers = [s for s, _ in hits]

    # E4 Lazy logic:
    # If thin corpus (no hits retrieved), confidence is 0 (< 0.7) -> questions fire.
    # If hits exist: ONLY fire if at least one layer has confidence < 0.7 OR layers conflict.
    if not retrieved_layers:
        needs_questions = True
    else:
        low_confidence = any(s.confidence < CONFIDENCE_THRESHOLD for s in retrieved_layers)
        conflict = _layers_conflict(retrieved_layers)
        needs_questions = low_confidence or conflict

    if not needs_questions:
        # Confident, conflict-free corpus -> zero friction
        return InterviewResponse(questions=[], coverage=cov)

    questions: list[InterviewQuestion] = []
    for gap in cov["gaps"][:MAX_QUESTIONS]:
        text, options = GAP_QUESTIONS.get(gap, (f"Preference for {gap}?", []))
        questions.append(InterviewQuestion(
            id=f"q_{uuid4().hex[:8]}", text=text, options=options, resolves=gap
        ))

    return InterviewResponse(questions=questions, coverage=cov)

"""inject — I2 permutation: few-shot exemplars as primary signal, directives secondary.

The packet carries `exemplars` (winning output texts + why) as the primary signal;
directives are secondary. Deterministic: same corpus + same input -> same packet.
"""

from __future__ import annotations

from .core import StandardsEngine
from .models import (
    Constraint,
    Directive,
    Exemplar,
    Packet,
    PacketRequest,
    Polarity,
)

import re

SIM_THRESHOLD = 0.012         # RRF score threshold for relevant layers
MAX_EXEMPLARS = 5             # I2: exemplars are primary signal
MAX_DIRECTIVES = 8            # directives secondary
MAX_NEVERS = 4                # guardrails always apply (bounded)

# Domain synonym keywords for boosting relevance
DOMAIN_KEYWORDS = {
    "media": ["video", "audio", "mp4", "render", "youtube", "tiktok", "reels", "voice", "sync", "lip", "thumbnail", "kallaway", "hormozi", "elevenlabs", "qwen", "deepgram", "camera", "frame", "cut"],
    "content": ["copy", "text", "headline", "hook", "story", "framing", "script", "editorial", "email", "angle", "jargon", "fluff"],
    "engineering": ["code", "bug", "fix", "feature", "git", "commit", "test", "tdd", "branch", "refactor", "security", "fail-closed", "clean", "architecture"],
    "systems": ["container", "docker", "service", "systemd", "server", "port", "daemon", "proxy", "cloudflare", "postgres", "redis", "bridge", "rclone"],
    "governance": ["gate", "audit", "compliance", "phi", "legal", "secret", "env", "credential", "permission", "sterile-field"],
}


def score_standard_relevance(s, query: str, tokens: set[str], hit_score: float = 0.0) -> float:
    text = (s.directive + " " + (s.rationale or "")).lower()
    score = hit_score * 10.0
    for t in tokens:
        if len(t) > 3 and t in text:
            score += 3.0
        elif t in text:
            score += 1.0
    q_low = query.lower()
    for domain, kws in DOMAIN_KEYWORDS.items():
        if any(k in q_low for k in kws):
            if any(k in text for k in kws):
                score += 5.0
    conf = float(s.confidence if s.confidence is not None else 0.5)
    return score * (0.5 + conf)



def compile_packet(engine: StandardsEngine, req: PacketRequest) -> Packet:
    query_parts = [
        str(v) for v in req.task_ctx.values() if isinstance(v, str)
    ]
    for ans in req.answers.values():
        query_parts.append(ans)
    query = " ".join(query_parts).strip() or "general task"

    hits = engine.retrieve(query, req.profile_id)

    exemplars: list[Exemplar] = []
    directives: list[Directive] = []
    seen_exemplar_texts: set[str] = set()

    # 1. Primary Signal: Exemplars from Decisions & Rich Docs
    # A) Check decisions for winning outputs matching task type
    task_type = str(req.task_ctx.get("type", "")).lower()
    decisions = engine.store.list_decisions(req.profile_id)
    # Check most recent relevant decisions
    for d in reversed(decisions):
        d_type = str(d.task_ctx.get("type", "")).lower()
        if d_type and d_type != task_type:
            continue
        # Find winning candidate text
        win_cand = next((c for c in d.candidates if c.label == d.pick), None)
        text = (win_cand.content or win_cand.content_ref) if win_cand else (d.pick if len(d.pick) > 4 else "")
        if text and text not in seen_exemplar_texts:
            exemplars.append(Exemplar(text=text, why=d.rationale))
            seen_exemplar_texts.add(text)
            if len(exemplars) >= MAX_EXEMPLARS:
                break

    # B) Exemplars from retrieved rich doc standards (examples in variant_meta)
    if len(exemplars) < MAX_EXEMPLARS:
        for s, _score in hits:
            doc_examples = s.variant_meta.get("examples", [])
            for ex in doc_examples:
                if ex not in seen_exemplar_texts:
                    exemplars.append(Exemplar(text=ex, why=s.rationale or s.directive))
                    seen_exemplar_texts.add(ex)
                    if len(exemplars) >= MAX_EXEMPLARS:
                        break
            if len(exemplars) >= MAX_EXEMPLARS:
                break

    # C) WHY layers as exemplars if still under quota
    if len(exemplars) < MAX_EXEMPLARS:
        for s, _score in hits:
            if s.polarity == Polarity.WHY:
                if s.directive not in seen_exemplar_texts:
                    exemplars.append(Exemplar(text=s.directive, why=s.rationale))
                    seen_exemplar_texts.add(s.directive)
                    if len(exemplars) >= MAX_EXEMPLARS:
                        break

    # 2. Secondary Signal: Directives
    hit_dict = {s.id: score for s, score in hits}
    tokens = set(re.findall(r"\w+", query.lower()))

    # Rank NEVER rules by task relevance + confidence, break ties by s.id for determinism
    active_nevers = [
        s for s in engine.active(req.profile_id)
        if s.polarity == Polarity.NEVER
    ]
    ranked_nevers = sorted(
        active_nevers,
        key=lambda s: (score_standard_relevance(s, query, tokens, hit_dict.get(s.id, 0.0)), s.id),
        reverse=True,
    )
    selected_nevers = sorted(ranked_nevers[:MAX_NEVERS], key=lambda s: s.id) if ranked_nevers else []
    directives.extend(
        Directive(polarity=s.polarity, text=s.directive, standard_id=s.id)
        for s in selected_nevers)

    ranked_hits = sorted(
        hits,
        key=lambda item: (score_standard_relevance(item[0], query, tokens, item[1]), item[1]),
        reverse=True,
    )
    for s, score in ranked_hits:
        if s.polarity == Polarity.NEVER:
            continue  # already sourced above — guardrails are not hits

        if score < SIM_THRESHOLD:
            continue

        if s.polarity == Polarity.WHY:
            continue

        if len(directives) >= MAX_DIRECTIVES:
            break

        directives.append(Directive(
            polarity=s.polarity, text=s.directive, standard_id=s.id
        ))

    # Dynamic J2 segments decomposition & audit persistence
    from .segments import propose_segments
    segments = propose_segments(req.task_ctx, store=engine.store, profile_id=req.profile_id)

    # Owner-supplied guidelines land as hard constraints (fresh signal law)
    constraints = [Constraint(hard=True, text=g) for g in req.guidelines]

    # NEVER rule 6: one bounded free degree per segment minimum, always
    free_degrees = [
        f"expression for '{seg}': match cadence and density within exemplars, keep authentic voice"
        for seg in segments
    ]

    return Packet(
        profile_id=req.profile_id,
        task_ctx=req.task_ctx,
        directives=directives,
        exemplars=exemplars,
        constraints=constraints,
        free_degrees=free_degrees,
        variant_meta={
            "engine": "taste-2",
            "hits": len(hits),
            "exemplar_count": len(exemplars),
            "directives_count": len(directives),
            "segments": segments,
        },
    )

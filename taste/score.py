"""score — D2 permutation: LLM-judge scorer with rules fallback.

Prompt = candidate + packet directives + exemplars -> judge returns
{score 0-1, violations[], one_line_why}.
Breakdown includes {"positive": score, "negative": len(violations), "judge_conf": conf}.

If TASTE_JUDGE_MODEL is unset or fails, degrades to deterministic rules scorer.
"""

from __future__ import annotations

import logging
import os
import re

from . import llm
from .core import StandardsEngine
from .embed import cosine, embed
from .inject import compile_packet
from .models import Candidate, CandidateScore, PacketRequest, Polarity

log = logging.getLogger("taste.score")

QUOTED = re.compile(
    r'"([^"]{4,})"|“([^”]{4,})”'      # double / smart quotes
    r"|(?<![\w'])'([^']{4,})'(?!\w)"  # single quotes, apostrophe-safe
)
LEADIN = re.compile(r"^(never|avoid|do not|don't|no)\s+", re.I)


def _quoted_strings(text: str) -> list[str]:
    return [next(g for g in m if g) for m in QUOTED.findall(text)]


def _forbidden_strings(directive: str) -> list[str]:
    """Extract concrete forbidden material from a never-directive."""
    quoted = _quoted_strings(directive)
    stripped = LEADIN.sub("", directive).strip().rstrip(".")
    # include the residual whenever it carries ANY content — a single-term
    # rule like "No emoji" is still a guardrail that must fire
    phrases = quoted + ([stripped] if stripped else [])
    return [p.lower() for p in phrases]


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", t.lower())


def _rules_scorer(engine: StandardsEngine, profile_id: str,
                  task_ctx: dict, candidates: list[Candidate]
                  ) -> list[CandidateScore]:
    """Deterministic rules scorer fallback (D1 mechanics)."""
    query = " ".join(str(v) for v in task_ctx.values() if isinstance(v, str))
    hits = engine.retrieve(query or "general", profile_id)

    nevers = [s for s in engine.active(profile_id)
              if s.polarity == Polarity.NEVER]
    positives = [s for s, sim in hits
                 if s.polarity in (Polarity.DO, Polarity.WHY) and sim >= 0.012]

    centroid = None
    if positives:
        vecs = embed([s.directive + " " + s.rationale for s in positives])
        dim = len(vecs[0])
        centroid = [sum(v[i] for v in vecs) / len(vecs) for i in range(dim)]

    out: list[CandidateScore] = []
    for cand in candidates:
        text = cand.content or cand.content_ref
        norm = _norm(text)

        violations: list[str] = []
        for s in nevers:
            for phrase in _forbidden_strings(s.directive):
                if phrase and phrase in norm:
                    violations.append(f"violates {s.id}: '{phrase}'")

        positive = 0.0
        if centroid is not None and text.strip():
            positive = max(0.0, cosine(embed([text])[0], centroid))

        score = max(-1.0, min(1.0, positive - 0.5 * len(violations)))
        out.append(CandidateScore(
            ref=cand.label,
            score=round(score, 4),
            breakdown={
                "positive": round(positive, 4),
                "negative": float(len(violations)),
                "judge_conf": 1.0,
                "bloat_veto": float(len(violations)),
                "leverage_fit": round(positive, 4),
                "cost_tokens": float(len(text) // 4),
            },
            violations=violations,
        ))
    return out


_JUDGE_SYSTEM = ("You are an objective judge evaluating candidate text against taste "
                 "directives and exemplars. Output JSON with {score: float (0.0 to 1.0), "
                 "violations: list[str], why: str, judge_conf: float}.")


def _parse_llm_json(raw_msg: str) -> dict:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    return llm.parse_json(raw_msg)


def _call_llm_provider(model: str, prompt: str) -> dict:
    """Call the judge provider (OpenAI-compatible: OpenAI / Qwen / Groq)."""
    return llm.chat_json(model, _JUDGE_SYSTEM, prompt)


def _llm_judge(engine: StandardsEngine, profile_id: str,
               task_ctx: dict, candidates: list[Candidate],
               model: str) -> list[CandidateScore]:
    """D2 LLM judge scorer."""
    pkt = compile_packet(engine, PacketRequest(profile_id=profile_id, task_ctx=task_ctx))
    directives_text = "\n".join(f"- [{d.polarity.value}] {d.text}" for d in pkt.directives)
    exemplars_text = "\n".join(f"- EXAG: {e.text} (WHY: {e.why})" for e in pkt.exemplars)

    scores: list[CandidateScore] = []
    for cand in candidates:
        text = cand.content or cand.content_ref
        if not text or not text.strip():
            scores.append(CandidateScore(
                ref=cand.label,
                score=0.0,
                breakdown={
                    "positive": 0.0,
                    "negative": 0.0,
                    "judge_conf": 1.0,
                },
                violations=[],
            ))
            continue

        prompt = (
            f"EVALUATE THIS CANDIDATE TEXT:\n'''{text}'''\n\n"
            f"DIRECTIVES TO ENFORCE:\n{directives_text or 'None'}\n\n"
            f"EXEMPLARS TO MATCH:\n{exemplars_text or 'None'}\n\n"
            f"Return JSON:\n"
            f'{{"score": float between 0.0 and 1.0, "violations": ["violation 1", ...], "why": "one line reason", "judge_conf": 0.0 to 1.0}}'
        )

        res = _call_llm_provider(model, prompt)
        score_val = max(0.0, min(1.0, float(res.get("score", 0.5))))
        violations = [str(v) for v in res.get("violations", [])]
        judge_conf = max(0.0, min(1.0, float(res.get("judge_conf", 0.9))))

        scores.append(CandidateScore(
            ref=cand.label,
            score=round(score_val, 4),
            breakdown={
                "positive": round(score_val, 4),
                "negative": float(len(violations)),
                "judge_conf": round(judge_conf, 4),
            },
            violations=violations,
        ))
    return scores


def score_candidates(engine: StandardsEngine, profile_id: str,
                      task_ctx: dict, candidates: list[Candidate]
                      ) -> list[CandidateScore]:
    judge_model = os.environ.get("TASTE_JUDGE_MODEL")
    if judge_model:
        try:
            return _llm_judge(engine, profile_id, task_ctx, candidates, judge_model)
        except Exception as exc:  # degrade, but never silently
            log.warning("LLM judge failed (%s: %s); using rules scorer",
                        type(exc).__name__, exc)
            return _rules_scorer(engine, profile_id, task_ctx, candidates)

    return _rules_scorer(engine, profile_id, task_ctx, candidates)

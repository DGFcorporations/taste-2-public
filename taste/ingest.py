"""ingest — cold-start pipeline: docs -> candidate layers + restatements.
Extended with A2 rich docs: extracts examples and counterexamples into variant_meta.

Pairs of "good/bad" phrasing in personas land as exemplars.
Provisional standards marked provenance=persona; nothing activates here (contract NEVER rules).
"""

from __future__ import annotations

import json
import re

from .comprehend import restate
from .models import (
    Category,
    IngestResponse,
    Polarity,
    Provenance,
    Restatement,
    Standard,
)

NEVER_RE = re.compile(
    r"\b(never|avoid|do not|don't|no\b.*\bever|ban|cut|ditch|scrap)\b", re.I)
WHY_RE = re.compile(r"\b(because|why|reason|since)\b", re.I)
RULE_RE = re.compile(
    r"\b(never|always|avoid|prefer|lead|keep|use|must|should|"
    r"cut|ban|require|start|end|write|organize|verify)\b", re.I)
BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")

CAT_HINTS = [
    (Category.ANTI, re.compile(r"slop|generic|bloat|never|forbidden", re.I)),
    (Category.STRUCTURE, re.compile(
        r"structure|organiz|lead|order|first|section|hook|close", re.I)),
    (Category.AESTHETIC, re.compile(
        r"tone|voice|style|rhythm|sentence|word|punchy|warm", re.I)),
    (Category.PROCEDURAL, re.compile(
        r"step|process|protocol|verify|check|test|commit|handoff", re.I)),
    (Category.DOCTRINE, re.compile(
        r"principle|doctrine|belief|value|rule\b", re.I)),
]

EXAMPLE_RE = re.compile(r"(?:example|e\.g\.|good|like this|prefer):\s*[\"“']?([^\"”\n]+)[\"”']?", re.I)
COUNTER_RE = re.compile(r"(?:bad|counterexample|not like this|avoid):\s*[\"“']?([^\"”\n]+)[\"”']?", re.I)


def _classify(line: str, section: str) -> tuple[Category, Polarity]:
    text = line.lower()
    for cat, rx in CAT_HINTS:
        if rx.search(section) or rx.search(text):
            category = cat
            break
    else:
        category = Category.DOCTRINE
    if NEVER_RE.search(text):
        return (Category.ANTI if category == Category.DOCTRINE else category,
                Polarity.NEVER)
    if WHY_RE.search(text):
        return category, Polarity.WHY
    return category, Polarity.DO


def _extract_rich_examples(line: str) -> tuple[str, list[str], list[str]]:
    """Extract inline examples and counterexamples from rule line.
    Returns (cleaned_line, examples, counterexamples).
    """
    examples: list[str] = []
    counterexamples: list[str] = []

    m_ex = EXAMPLE_RE.search(line)
    if m_ex:
        examples.append(m_ex.group(1).strip(" ."))

    m_cnt = COUNTER_RE.search(line)
    if m_cnt:
        counterexamples.append(m_cnt.group(1).strip(" ."))

    # Check for quote pairs like "Good phrasing" vs "Bad phrasing"
    vs_match = re.search(r'["“]([^"”]+)["”]\s+(?:vs\.?|instead of|not)\s+["“]([^"”]+)["”]', line, re.I)
    if vs_match:
        examples.append(vs_match.group(1).strip())
        counterexamples.append(vs_match.group(2).strip())

    cleaned = line
    if m_ex:
        cleaned = cleaned[:m_ex.start()].strip(" ;,-")
    if m_cnt:
        cleaned = cleaned[:m_cnt.start()].strip(" ;,-")

    return cleaned, examples, counterexamples


def _extract_lines(text: str) -> list[tuple[str, str, list[str], list[str]]]:
    """(rule_text, nearest_section_header, examples, counterexamples) tuples."""
    results, section = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            section = line.lstrip("#").strip()
            continue
        if BULLET_RE.match(line):
            line = BULLET_RE.sub("", line)

        cleaned, examples, counterexamples = _extract_rich_examples(line)
        if RULE_RE.search(cleaned) and 4 <= len(cleaned.split()) <= 80:
            results.append((cleaned.rstrip("."), section, examples, counterexamples))
    return results


def _mk(profile_id: str, category: Category, polarity: Polarity,
        directive: str, rationale: str, source: str,
        examples: list[str] | None = None,
        counterexamples: list[str] | None = None,
        **meta) -> Standard:
    variant_meta = {
        "source_doc": source,
        "examples": examples or [],
        "counterexamples": counterexamples or [],
        **meta,
    }
    return Standard(
        profile_id=profile_id,
        category=category,
        polarity=polarity,
        directive=directive.rstrip("."),
        rationale=rationale,
        provenance=Provenance.PERSONA,
        variant_meta=variant_meta,
    )


def _ingest_profile(profile_id: str, doc_id: str, data: dict
                    ) -> list[Standard]:
    """taste_profile.json -> candidate standards."""
    out: list[Standard] = []
    why = "persona-declared preference"

    for trig in data.get("veto_triggers", []):
        out.append(_mk(profile_id, Category.ANTI, Polarity.NEVER,
                       trig, why, doc_id, trigger="veto"))

    for k, v in (data.get("axioms") or {}).items():
        if v is True:
            d = k.replace("_", " ")
        elif v is False or v is None:
            continue
        elif isinstance(v, (int, float)):
            d = f"{k.replace('_', ' ')}: {v}"
        else:
            d = str(v)
        out.append(_mk(profile_id, Category.DOCTRINE, Polarity.DO,
                       d, why, doc_id, axiom=k))

    for k, v in (data.get("author_profile") or {}).items():
        out.append(_mk(profile_id, Category.DOCTRINE, Polarity.DO,
                       f"{k.replace('_', ' ')}: {v}", why, doc_id,
                       profile_key=k))

    for k, v in (data.get("aesthetic_and_tone") or {}).items():
        out.append(_mk(profile_id, Category.AESTHETIC, Polarity.DO,
                       f"{k.replace('_', ' ')}: {v}", why, doc_id,
                       aesthetic_key=k))

    for lane, caps in (data.get("agent_lanes") or {}).items():
        out.append(_mk(profile_id, Category.PROCEDURAL, Polarity.DO,
                       f"{lane} lane operates: {', '.join(caps)}",
                       why, doc_id, lane=lane, capabilities=caps))

    return out


def ingest_docs(profile_id: str, docs: list[dict[str, str]]
                ) -> IngestResponse:
    candidates: list[Standard] = []
    restatements: list[Restatement] = []

    for doc in docs:
        text = doc.get("text", "")
        try:
            parsed = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and (
                "axioms" in parsed or "veto_triggers" in parsed):
            new = _ingest_profile(profile_id, doc.get("id", ""), parsed)
            candidates.extend(new)
            restatements.extend(restate(s) for s in new)
            continue

        for line, section, examples, counterexamples in _extract_lines(text):
            category, polarity = _classify(line, section)
            rationale = ""
            m = re.search(r"because\s+(.+)$", line, re.I)
            if m:
                line, rationale = line[: m.start()].strip(" .—-"), m.group(1)
            if not rationale:
                rationale = "persona-declared preference"

            std = Standard(
                profile_id=profile_id,
                category=category,
                polarity=polarity,
                directive=line,
                rationale=rationale,
                provenance=Provenance.PERSONA,
                variant_meta={
                    "source_doc": doc.get("id", ""),
                    "source_section": section,
                    "examples": examples,
                    "counterexamples": counterexamples,
                },
            )
            candidates.append(std)
            restatements.append(restate(std))

    return IngestResponse(candidates=candidates, restatements=restatements)

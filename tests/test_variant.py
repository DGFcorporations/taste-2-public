"""Variant-specific checks from docs/BLUEPRINT.md for taste-2:

- RRF: craft a doc where lexical-only and vector-only rank differently; fusion must pick the union correctly
- Judge-fallback: unset TASTE_JUDGE_MODEL -> scorer degrades to rules, not error
- Lazy interview: dense corpus -> 0 questions; thin corpus -> questions fire
- Dynamic segments: proposes 3-5 segments from task_ctx; persisted for audit
- Confidence math: owner confirm -> +0.10; correction contradicting -> -0.25; 30d unused decay
- LanceDB versioning: versioned snapshots per cycle
"""

import pytest

from taste.models import Category, LayerState, Polarity, Standard
from taste.segments import propose_segments


def test_rrf_hybrid_fusion(app):
    """RRF: craft docs where lexical-only and vector-only rank differently;
    fusion must pick the union correctly."""
    engine = app.state.engine

    # Standard 1: strong lexical match for 'xylophone', moderate vector
    s1 = Standard(
        id="std_lexical",
        profile_id="owner",
        category=Category.DOCTRINE,
        polarity=Polarity.DO,
        directive="Play the xylophone loudly every morning",
        rationale="rhythmic resonance",
        confidence=0.8,
        state=LayerState.ACTIVE,
    )
    # Standard 2: semantic/vector match for music/melody, but does not contain the word 'xylophone'
    s2 = Standard(
        id="std_semantic",
        profile_id="owner",
        category=Category.DOCTRINE,
        polarity=Polarity.DO,
        directive="Harmonize auditory frequencies and acoustic pitch",
        rationale="vibrational balance",
        confidence=0.8,
        state=LayerState.ACTIVE,
    )

    engine.propose(s1)
    engine.confirm(s1.id, rationale="rhythmic resonance")
    engine.propose(s2)
    engine.confirm(s2.id, rationale="vibrational balance")

    # Query with lexical keyword 'xylophone'
    hits = engine.retrieve("xylophone acoustic music", "owner")
    hit_ids = [s.id for s, score in hits]
    assert "std_lexical" in hit_ids, "lexical lane must pick up xylophone"
    # Both items should be present in the hybrid union
    assert len(hit_ids) >= 2, "hybrid RRF must return the union of both lanes"


def test_judge_fallback_when_model_unset(client, monkeypatch):
    """Judge-fallback: unset TASTE_JUDGE_MODEL -> scorer degrades to rules, not error."""
    monkeypatch.delenv("TASTE_JUDGE_MODEL", raising=False)
    r = client.post("/score", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc"},
        "candidates": [{"label": "A", "content": "Just shipping clean code."}],
    })
    assert r.status_code == 200
    res = r.json()
    assert len(res["scores"]) == 1
    s = res["scores"][0]
    assert "positive" in s["breakdown"] and "negative" in s["breakdown"]
    assert s["score"] is not None


def test_lazy_interview(client, app):
    """Lazy interview: dense corpus -> 0 questions; thin corpus -> questions fire."""
    # 1. Thin corpus: no standards confirmed yet -> questions fire
    r = client.post("/interview", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "role": "publisher"},
    })
    assert r.status_code == 200
    assert len(r.json()["questions"]) > 0, "thin corpus must fire questions"

    # 2. Make corpus dense and high confidence (all 5 categories covered with conf >= 0.7)
    engine = app.state.engine
    categories = [Category.STRUCTURE, Category.AESTHETIC, Category.PROCEDURAL,
                  Category.DOCTRINE, Category.ANTI]
    for cat in categories:
        s = Standard(
            profile_id="owner",
            category=cat,
            polarity=Polarity.DO if cat != Category.ANTI else Polarity.NEVER,
            directive=f"Directive for {cat.value} standards in write_doc tasks",
            rationale=f"Rationale for {cat.value}",
            confidence=0.8,  # > 0.7 threshold
        )
        engine.propose(s)
        engine.confirm(s.id, rationale=f"Rationale for {cat.value}")
        # Ensure confidence is high
        engine.store.set_confidence(s.id, 0.85)

    # Now run interview with dense confident corpus
    r_dense = client.post("/interview", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "role": "publisher"},
    })
    assert r_dense.status_code == 200
    # Zero friction: questions should be empty!
    assert len(r_dense.json()["questions"]) == 0, "dense confident corpus must fire 0 questions (E4 lazy)"


def test_dynamic_segments_audit_persistence(app):
    """Dynamic segments: driver proposes 3-5 segments from task_ctx; persisted per task for audit."""
    store = app.state.store
    task_ctx = {"id": "task_audit_123", "type": "write_doc", "role": "publisher"}

    segments = propose_segments(task_ctx, store=store, profile_id="owner")
    assert 3 <= len(segments) <= 5, "segments count must be between 3 and 5"

    # Verify persisted in store
    persisted = store.get_task_segments("task_audit_123")
    assert persisted == segments


def test_confidence_math_h5(app):
    """Confidence math: owner confirm -> +0.10; correction contradicting -> -0.25; decay."""
    engine = app.state.engine
    s = Standard(
        profile_id="owner",
        category=Category.DOCTRINE,
        polarity=Polarity.DO,
        directive="Initial doctrine",
        rationale="initial rationale",
        confidence=0.5,
    )
    engine.propose(s)
    initial_conf = engine.store.get_standard(s.id).confidence
    assert initial_conf == 0.5

    # Confirm adds +0.10
    engine.confirm(s.id)
    confirmed_conf = engine.store.get_standard(s.id).confidence
    assert confirmed_conf == pytest.approx(0.60, abs=1e-3)

    # Supersede: new correction contradicts old -> old decays by 0.25
    s_new = Standard(
        profile_id="owner",
        category=Category.DOCTRINE,
        polarity=Polarity.DO,
        directive="Better doctrine",
        rationale="superior rationale",
        confidence=0.5,
    )
    engine.supersede(s.id, s_new)
    superseded_conf = engine.store.get_standard(s.id).confidence
    assert superseded_conf == pytest.approx(0.35, abs=1e-3)

    # Apply 30d unused decay
    engine.apply_decay(s.id, days_unused=37)  # 30d + 1 week -> -0.05
    decayed_conf = engine.store.get_standard(s.id).confidence
    assert decayed_conf == pytest.approx(0.30, abs=1e-3)


def test_lancedb_versioning_and_snapshots(app):
    """LanceDB table versioned snapshots per cycle."""
    store = app.state.store
    v_start = store.version

    s = Standard(
        profile_id="owner",
        category=Category.AESTHETIC,
        polarity=Polarity.DO,
        directive="Clean typography",
        rationale="readability",
        confidence=0.7,
    )
    store.put_standard(s)
    assert store.version > v_start
    snapshots = store.list_snapshots()
    assert len(snapshots) >= 1


def test_judge_scorer_with_llm_model_set(client, monkeypatch):
    """When TASTE_JUDGE_MODEL is set, score uses LLM judge and produces valid float breakdown."""
    monkeypatch.setenv("TASTE_JUDGE_MODEL", "qwen-plus")
    from unittest.mock import patch

    mock_llm_resp = {
        "score": 0.88,
        "violations": ["minor verbosity"],
        "why": "Tone matches standard exemplars",
        "judge_conf": 0.95,
    }

    with patch("taste.score._call_llm_provider", return_value=mock_llm_resp):
        r = client.post("/score", json={
            "profile_id": "owner",
            "task_ctx": {"type": "write_doc"},
            "candidates": [{"label": "A", "content": "Shipped: robust telemetry verification."}],
        })
        assert r.status_code == 200
        res = r.json()
        assert len(res["scores"]) == 1
        s = res["scores"][0]
        assert s["score"] == 0.88
        assert s["violations"] == ["minor verbosity"]
        assert s["breakdown"]["positive"] == 0.88
        assert s["breakdown"]["negative"] == 1.0
        assert s["breakdown"]["judge_conf"] == 0.95


def test_judge_scorer_markdown_fenced_json(client, monkeypatch):
    """LLM outputs wrapped in markdown code fences are properly parsed."""
    monkeypatch.setenv("TASTE_JUDGE_MODEL", "qwen-plus")
    from taste.score import _parse_llm_json

    fenced_raw = "```json\n{\n  \"score\": 0.92,\n  \"violations\": [],\n  \"why\": \"clean output\",\n  \"judge_conf\": 0.98\n}\n```"
    parsed = _parse_llm_json(fenced_raw)
    assert parsed["score"] == 0.92
    assert parsed["why"] == "clean output"


def test_judge_scorer_fallback_on_llm_failure(client, monkeypatch):
    """When LLM provider raises an exception, score gracefully falls back to rules scorer."""
    monkeypatch.setenv("TASTE_JUDGE_MODEL", "qwen-plus")
    from unittest.mock import patch

    with patch("taste.score._call_llm_provider", side_effect=ConnectionError("provider timeout")):
        r = client.post("/score", json={
            "profile_id": "owner",
            "task_ctx": {"type": "write_doc"},
            "candidates": [{"label": "A", "content": "Short text."}],
        })
        assert r.status_code == 200
        res = r.json()
        assert len(res["scores"]) == 1
        assert "positive" in res["scores"][0]["breakdown"]
        assert "negative" in res["scores"][0]["breakdown"]


def test_elimination_filter_detects_never_rule_veto(client, app):
    """Elimination filter detects when task context asks for what an active never-rule forbids."""
    engine = app.state.engine
    s = Standard(
        profile_id="owner",
        category=Category.ANTI,
        polarity=Polarity.NEVER,
        directive="Never use generic openers in articles",
        rationale="authority and punchy delivery",
        confidence=0.85,
    )
    engine.propose(s)
    engine.confirm(s.id, rationale="authority and punchy delivery")

    # Task specifically mentions generic openers for articles
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {
            "type": "write_doc",
            "role": "publisher",
            "brief": "Write an article using generic openers for our newsletter audience",
        },
    })
    assert r.status_code == 200
    res = r.json()
    assert len(res["vetoes_hit"]) >= 1
    assert any("generic openers" in v for v in res["vetoes_hit"])


def test_interview_detects_short_rule_conflict():
    """Conflicting DO and NEVER rules with single substantive key terms are detected as conflict."""
    from taste.interview import _layers_conflict

    s_never = Standard(
        category=Category.ANTI,
        polarity=Polarity.NEVER,
        directive="Never use emojis",
        rationale="professionalism",
    )
    s_do = Standard(
        category=Category.AESTHETIC,
        polarity=Polarity.DO,
        directive="Always use emojis",
        rationale="expressiveness",
    )
    assert _layers_conflict([s_never, s_do]) is True


def test_packet_dynamic_segments_audit(client, app):
    """Packet compilation triggers dynamic segments decomposition and persists to LanceDB audit."""
    store = app.state.store
    task_ctx = {"id": "task_audit_pkt_789", "type": "code_review", "role": "engineer"}

    r = client.post("/packet", json={"profile_id": "owner", "task_ctx": task_ctx})
    assert r.status_code == 200
    pkt = r.json()

    assert "segments" in pkt["variant_meta"]
    segments = pkt["variant_meta"]["segments"]
    assert 3 <= len(segments) <= 5

    # One free degree per segment minimum
    assert len(pkt["free_degrees"]) == len(segments)

    # Verify audit persistence in LanceDB
    persisted = store.get_task_segments("task_audit_pkt_789")
    assert persisted == segments


def test_mcp_tools_surface(app):
    """MCP tools return the same JSON shapes as the HTTP routes."""
    import json

    from taste.mcp_server import create_tools

    tools = create_tools(app.state.store, app.state.engine)

    pkt = json.loads(tools["taste_packet"](task_ctx={"type": "write_doc"}))
    assert "directives" in pkt
    assert "free_degrees" in pkt
    assert len(pkt["free_degrees"]) >= 1

    scored = json.loads(tools["taste_score"](
        candidates=[{"label": "A", "content": "Sample output"}]))
    assert len(scored["scores"]) == 1
    assert "score" in scored["scores"][0]
    assert "breakdown" in scored["scores"][0]

    fb = json.loads(tools["taste_feedback"](
        pick="A", rationale="Clearly articulated verdict",
        task_ctx={"type": "write_doc"}))
    assert fb["accepted"] is True
    assert fb["record_id"]
    assert fb["extracted"]["candidates"] == 1

    recalled = json.loads(tools["taste_recall"](profile_id="owner"))
    assert recalled["count"] == len(recalled["standards"])
    assert any(row["state"] == "provisional" for row in recalled["standards"])


def test_hone_uses_llm_critique_revise_when_model_set(client, monkeypatch):
    from unittest.mock import patch
    monkeypatch.setenv("TASTE_JUDGE_MODEL", "mock-model")
    reply = {"draft": "We just really launched.", "critique": "hedges",
             "revised": "We launched."}
    with patch("taste.llm.chat_json", return_value=reply):
        r = client.post("/hone", json={"profile_id": "owner", "segment_ctx": {"segment": "intro"},
                                       "draft": "We just really launched."})
    assert r.status_code == 200
    contents = [c["content"] for c in r.json()["candidates"]]
    assert contents == ["We just really launched.", "We launched."]


def test_hone_falls_back_offline_when_llm_fails(client, monkeypatch):
    from unittest.mock import patch
    monkeypatch.setenv("TASTE_JUDGE_MODEL", "mock-model")
    with patch("taste.llm.chat_json", side_effect=ConnectionError("down")):
        r = client.post("/hone", json={"profile_id": "owner", "segment_ctx": {"segment": "intro"},
                                       "draft": "We just launched."})
    assert r.status_code == 200 and len(r.json()["candidates"]) == 2


def test_llm_requires_key(monkeypatch, tmp_path):
    import pytest

    from taste import llm
    for k in llm._KEY_NAMES:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "_ENV_FILE", tmp_path / "none.env")
    with pytest.raises(llm.LLMUnavailable):
        llm.chat_json("gpt-4o", "s", "p")


@pytest.mark.slow
def test_real_qwen3_embedder_is_semantic(monkeypatch):
    """C2 for real: Qwen3 must rank a paraphrase above an unrelated sentence."""
    import pytest
    monkeypatch.delenv("TASTE_EMBED", raising=False)
    from taste.embed import Qwen3Embedder, cosine
    try:
        e = Qwen3Embedder()
    except Exception as exc:
        pytest.skip(f"Qwen3 model unavailable: {exc}")
    a, b, c = e.embed(["Never use corporate jargon in headlines.",
                       "Avoid buzzwords when writing titles.",
                       "The tomato soup needs more salt."])
    assert len(a) == 1024 and cosine(a, b) > cosine(a, c)

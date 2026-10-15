"""Explicit edge case tests for taste-2:

- Empty corpus behavior for interview, mirror, packet, score, hone
- Candidate edge cases (empty candidates, whitespace, long text)
- Clamping of confidence boundaries [0.0, 1.0]
- Rationale validation edge cases (spaces-only, empty string)
- NEVER rules verification (provisional state, queued gate, free degrees >= 1, retirement retention)
"""

import pytest

from taste.models import Category, LayerState, Polarity, Standard


def test_empty_corpus_endpoints(client):
    """Empty corpus should produce valid responses without crashes."""
    # 1. standards
    r = client.get("/standards")
    assert r.status_code == 200
    assert r.json()["count"] == 0

    # 2. interview on empty corpus fires questions (thin corpus)
    r = client.post("/interview", json={"profile_id": "owner", "task_ctx": {"type": "write_doc"}})
    assert r.status_code == 200
    assert len(r.json()["questions"]) > 0

    # 3. mirror on empty corpus notes lack of coverage in assumptions
    r = client.post("/mirror", json={"profile_id": "owner", "task_ctx": {"type": "write_doc"}})
    assert r.status_code == 200
    assert any("first-principles" in a for a in r.json()["assumptions"])

    # 4. packet on empty corpus has 0 directives but has at least 1 free degree
    r = client.post("/packet", json={"profile_id": "owner", "task_ctx": {"type": "write_doc"}})
    assert r.status_code == 200
    pkt = r.json()
    assert pkt["directives"] == []
    assert len(pkt["free_degrees"]) >= 1

    # 5. score on empty corpus returns scores without crashing
    r = client.post("/score", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc"},
        "candidates": [{"label": "A", "content": "Just a normal draft"}],
    })
    assert r.status_code == 200
    scores = r.json()["scores"]
    assert len(scores) == 1
    assert scores[0]["score"] == 0.0

    # 6. hone on empty corpus returns candidates and pick
    r = client.post("/hone", json={
        "profile_id": "owner",
        "segment_ctx": {"type": "write_doc", "segment": "lead"},
        "draft": "Initial draft text.",
    })
    assert r.status_code == 200
    assert r.json()["pick"] in ["A", "B"]


def test_score_empty_and_whitespace_candidates(client):
    """Empty or whitespace candidate content should score gracefully."""
    r = client.post("/score", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc"},
        "candidates": [
            {"label": "A", "content": ""},
            {"label": "B", "content": "   \n\t  "},
        ],
    })
    assert r.status_code == 200
    scores = r.json()["scores"]
    assert len(scores) == 2
    assert scores[0]["score"] == 0.0
    assert scores[1]["score"] == 0.0


def test_feedback_whitespace_rationale_rejected_422(client):
    """Whitespace-only rationale must be rejected with 422."""
    r = client.post("/feedback", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc"},
        "pick": "A",
        "rationale": "   \t  \n  ",
    })
    assert r.status_code == 422, f"whitespace-only rationale must 422, got {r.status_code}"


def test_confidence_clamping_boundaries(app):
    """Confidence must never exceed 1.0 or drop below 0.0."""
    engine = app.state.engine
    s = Standard(
        profile_id="owner",
        category=Category.DOCTRINE,
        polarity=Polarity.DO,
        directive="Upper bound test",
        rationale="test",
        confidence=0.95,
    )
    engine.propose(s)
    # Confirm adds +0.10 -> 0.95 + 0.10 = 1.05 -> clamped to 1.0
    engine.confirm(s.id)
    assert engine.store.get_standard(s.id).confidence == 1.0

    # Large negative reinforcement
    engine.reinforce(s.id, -2.0)
    assert engine.store.get_standard(s.id).confidence == 0.0


def test_confirm_without_rationale_raises(app):
    """Activating without rationale must fail (contract NEVER rule)."""
    engine = app.state.engine
    s = Standard(
        profile_id="owner",
        category=Category.DOCTRINE,
        polarity=Polarity.DO,
        directive="No rationale directive",
        rationale="",
        confidence=0.5,
    )
    engine.propose(s)
    with pytest.raises(ValueError):
        engine.confirm(s.id, rationale="")


def test_decision_record_rationale_validation():
    """DecisionRecord must reject empty or whitespace rationale (NEVER rule 1)."""
    from pydantic import ValidationError

    from taste.models import DecisionRecord

    with pytest.raises(ValidationError):
        DecisionRecord(pick="A", rationale="")

    with pytest.raises(ValidationError):
        DecisionRecord(pick="A", rationale="   \n\t  ")


def test_standards_import_deduplication_in_batch(client):
    """Standards import must deduplicate identical items within the import batch."""
    s = {
        "id": "std_dedup_test",
        "profile_id": "owner",
        "category": "doctrine",
        "polarity": "do",
        "directive": "Always write unambiguous specifications",
        "rationale": "reduces defect surface",
        "confidence": 0.8,
        "provenance": "persona",
        "state": "active",
    }
    r = client.post("/standards/import", json={"profile_id": "owner", "standards": [s, s]})
    assert r.status_code == 200
    res = r.json()
    assert res["imported"] == 1
    assert res["skipped"] == 1


def test_supersede_nonexistent_returns_404(client):
    """Attempting to supersede a nonexistent standard must return 404."""
    new_s = {
        "profile_id": "owner",
        "category": "doctrine",
        "polarity": "do",
        "directive": "Fresh doctrine",
        "rationale": "better rationale",
    }
    r = client.post("/standards/nonexistent_sid/supersede", json=new_s)
    assert r.status_code == 404


def test_cli_standards_state_filter(tmp_path, monkeypatch, capsys):
    """taste standards CLI command properly filters by --state."""
    from taste.api import create_app
    from taste.cli import main
    from taste.models import Category, Polarity, Standard

    db_dir = tmp_path / "cli_test"
    monkeypatch.setenv("TASTE2_DB", str(db_dir))
    monkeypatch.setenv("TASTE_EMBED", "hash")

    app = create_app(db_dir)
    store = app.state.store
    s_active = Standard(
        id="std_cli_act",
        category=Category.DOCTRINE,
        polarity=Polarity.DO,
        directive="Active standard for CLI test",
        rationale="valid",
        state=LayerState.ACTIVE,
    )
    s_prov = Standard(
        id="std_cli_prv",
        category=Category.DOCTRINE,
        polarity=Polarity.DO,
        directive="Provisional standard for CLI test",
        rationale="valid",
        state=LayerState.PROVISIONAL,
    )
    store.put_standard(s_active)
    store.put_standard(s_prov)

    # 1. List active only
    main(["standards", "--state=active"])
    out = capsys.readouterr().out
    assert "std_cli_act" in out
    assert "std_cli_prv" not in out

    # 2. List provisional only
    main(["standards", "--state=provisional"])
    out = capsys.readouterr().out
    assert "std_cli_act" not in out
    assert "std_cli_prv" in out


def test_mcp_edge_cases(app):
    """Invalid category, empty candidates, and blank rationale stay envelopes."""
    import json

    from taste.mcp_server import create_tools

    tools = create_tools(app.state.store, app.state.engine)

    res = json.loads(tools["taste_recall"](category="nonexistent_category_type"))
    assert "invalid category" in res["error"]["message"]

    assert json.loads(tools["taste_score"]([])) == {"scores": []}

    feedback_res = json.loads(tools["taste_feedback"](pick="A", rationale="   "))
    assert feedback_res["error"]["code"] == "e422"
    assert "NEVER rule 1" in feedback_res["error"]["message"]



# CODE-001 — whitespace rationale must 422 for every blank shape ----------
@pytest.mark.parametrize("bad", ["", "   ", "\n\t"])
def test_feedback_blank_rationale_rejected_422(client, bad):
    r = client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {"type": "x"},
        "pick": "A", "rationale": bad})
    assert r.status_code == 422, f"rationale={bad!r} must 422"


# CODE-002 — never-rules come from active corpus, not retrieval hits -----
def test_never_rule_absent_from_retrieval_still_in_packet():
    from taste.inject import compile_packet
    from taste.models import PacketRequest

    never = Standard(
        profile_id="owner", category=Category.ANTI,
        polarity=Polarity.NEVER, state=LayerState.ACTIVE,
        directive="Never mention competitors by name",
        rationale="brand discipline")

    class _Store:
        def list_decisions(self, profile_id="owner"):
            return []

        def put_task_segments(self, *a, **kw):
            pass

    class _StubEngine:
        store = _Store()

        def retrieve(self, query, profile_id="owner", **kw):
            return []  # retrieval returns nothing at all

        def active(self, profile_id="owner"):
            return [never]

    pkt = compile_packet(_StubEngine(), PacketRequest(
        profile_id="owner", task_ctx={"type": "write_doc",
                                      "brief": "unrelated task"}))
    assert any(d.polarity == Polarity.NEVER and d.text == never.directive
               for d in pkt.directives)


# CODE-005 — RRF ties serialize deterministically ------------------------
def test_rrf_tie_order_is_deterministic(app):
    """Identical vectors + identical FTS rank = tied RRF; order by id."""
    engine = app.state.engine
    for sid in ("std_zzz", "std_aaa", "std_mmm"):
        s = Standard(
            id=sid, profile_id="owner", category=Category.DOCTRINE,
            polarity=Polarity.DO, state=LayerState.ACTIVE,
            directive="identical directive text for tie",
            rationale="same rationale everywhere")
        engine.propose(s)
        engine.confirm(s.id, rationale="same rationale everywhere")

    first = [s.id for s, _ in engine.retrieve("identical directive text")]
    for _ in range(3):
        again = [s.id for s, _ in engine.retrieve("identical directive text")]
        assert again == first
    tied = [sid for sid in first if sid in ("std_aaa", "std_mmm", "std_zzz")]
    assert tied == sorted(tied), "RRF ties must serialize by standard_id"


# Veto formula — long rule diluted by short task; short rule --------------
def _active_never(client, directive):
    r = client.post("/standards/import", json={
        "profile_id": "owner",
        "standards": [{
            "profile_id": "owner", "category": "anti", "polarity": "never",
            "directive": directive, "rationale": "because taste",
            "state": "active", "provenance": "persona"}]})
    assert r.status_code == 200, r.text
    sid = next(row["id"] for row in client.get("/standards").json()["standards"]
               if row["directive"] == directive)
    confirmed = client.post(f"/standards/{sid}/confirm",
                            json={"rationale": "because taste"})
    assert confirmed.status_code == 200, confirmed.text


def test_veto_fires_when_rule_terms_dominate_short_task(client):
    _active_never(client,
                  "Never use generic openers like in today's world as "
                  "widely known phrases in copywriting deliverables")
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "brief": "generic openers today"}})
    assert r.json()["vetoes_hit"], "task_cov clause must fire"


def test_veto_fires_on_short_single_term_rule(client):
    _active_never(client, "No emoji")
    r = client.post("/mirror", json={
        "profile_id": "owner",
        "task_ctx": {"type": "post", "brief": "add emoji everywhere"}})
    assert r.json()["vetoes_hit"], "single-term rule must fire"


def test_short_never_rule_fires_in_scorer(client):
    """CODE minor: 'No emoji' must produce a violation in the rules scorer."""
    _active_never(client, "No emoji")
    r = client.post("/score", json={
        "profile_id": "owner", "task_ctx": {"type": "post"},
        "candidates": [{"label": "A", "content": "great post emoji here"},
                       {"label": "B", "content": "clean plain text"}]})
    scores = {s["ref"]: s for s in r.json()["scores"]}
    assert scores["A"]["violations"], "single-term never-rule must fire"
    assert scores["B"]["violations"] == []


# API minor fixes ---------------------------------------------------------
def test_standards_invalid_state_returns_422_error_body(client):
    r = client.get("/standards", params={"state": "bad"})
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "invalid_state"


def test_hone_empty_driver_returns_empty_response(tmp_path, monkeypatch):
    monkeypatch.setenv("TASTE_EMBED", "hash")
    from taste.embed import get_embedder
    get_embedder.cache_clear()
    from taste.core import StandardsEngine
    from taste.hone import hone_segment
    from taste.models import HoneRequest
    from taste.store import Store

    engine = StandardsEngine(Store(tmp_path / "hone_lance"))
    resp = hone_segment(engine, HoneRequest(
        profile_id="owner", segment_ctx={"segment": "intro"}),
        driver=lambda draft, ctx, rules: ("", "", ""))
    assert resp.iterate is False

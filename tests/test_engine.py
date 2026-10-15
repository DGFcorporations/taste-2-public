"""taste-2 unit smoke — the loop end to end, in-process, offline.

Runs on the hashing embedder (TASTE_EMBED=hash via conftest) so it's
deterministic and fast.
"""

from taste.models import Category, LayerState, Polarity, Provenance, Standard

PERSONA = """# Persona: Owner
- Never use generic openers like "In today's world".
- Lead with the angle, then the structure, then the build.
- Prefer short declarative sentences because they carry authority. Example: "Shipped: evidence capture." Bad: "In today's world, shipping matters."
"""


def _confirm_all(client):
    rows = client.get("/standards").json()["standards"]
    for s in rows:
        if s["state"] == "provisional":
            client.post(f"/standards/{s['id']}/confirm")


def test_full_loop(client):
    # 1. ingest persona -> provisional candidates
    r = client.post("/ingest", json={
        "profile_id": "owner",
        "docs": [{"id": "p", "text": PERSONA, "format": "md"}]})
    assert r.status_code == 200
    assert len(r.json()["candidates"]) >= 3

    # Verify rich doc examples were extracted into variant_meta
    candidates = r.json()["candidates"]
    rich_cand = next((c for c in candidates if c.get("variant_meta", {}).get("examples")), None)
    assert rich_cand is not None, "rich doc example must be extracted"
    assert "Shipped: evidence capture" in rich_cand["variant_meta"]["examples"]

    # 2. owner confirms via comprehension gate
    _confirm_all(client)
    active = [s for s in client.get("/standards").json()["standards"]
              if s["state"] == "active"]
    assert len(active) >= 3

    # 3. packet compiles directives and exemplars from active taste
    pkt = client.post("/packet", json={
        "profile_id": "owner",
        "task_ctx": {"type": "write_doc", "brief": "ship notes"}}).json()
    never_texts = [d["text"] for d in pkt["directives"]
                   if d["polarity"] == "never"]
    assert any("generic openers" in t for t in never_texts)
    # I2: exemplars carried in packet
    assert len(pkt["exemplars"]) >= 1, "packet must carry exemplars as primary signal"

    # 4. scorer catches the violation taste forbids
    r = client.post("/score", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "candidates": [
            {"label": "A", "content": "In today's world, shipping matters. We shipped evidence capture; here's what changed."},
            {"label": "B", "content": "Shipped: evidence capture. What changed, why it matters."}]})
    scores = {s["ref"]: s for s in r.json()["scores"]}
    assert scores["A"]["violations"], "generic opener must be caught"
    assert scores["A"]["score"] < scores["B"]["score"]
    assert "positive" in scores["A"]["breakdown"]
    assert "negative" in scores["A"]["breakdown"]

    # 5. hone picks the better candidate using critique-revise
    h = client.post("/hone", json={
        "profile_id": "owner",
        "segment_ctx": {"type": "write_doc", "segment": "lead"},
        "draft": "In today's world, things are hard. We shipped a fix."}).json()
    assert h["pick"] and h["candidates"]
    assert len(h["candidates"]) == 2  # F2 critique-revise: initial (A) + revised (B)

    # 6. feedback lands + extracts a provisional why-layer
    before = client.get("/standards").json()["count"]
    r = client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"},
        "pick": "B", "rationale": "verdict-first beats throat-clearing"})
    assert r.status_code == 200
    assert client.get("/standards").json()["count"] == before + 1


def test_rationale_required_422(client):
    r = client.post("/feedback", json={
        "profile_id": "owner", "task_ctx": {}, "pick": "A", "rationale": ""})
    assert r.status_code == 422


def test_provisional_layers_never_apply(client):
    client.post("/ingest", json={
        "profile_id": "owner",
        "docs": [{"id": "p", "text": PERSONA}]})
    pkt = client.post("/packet", json={
        "profile_id": "owner", "task_ctx": {"type": "write_doc"}}).json()
    assert pkt["directives"] == [], \
        "provisional layers must not reach packets"


def test_supersede_keeps_history(app):
    from taste.embed import get_embedder
    get_embedder.cache_clear()
    s = Standard(category=Category.DOCTRINE, polarity=Polarity.DO,
                 directive="Lead with the angle", rationale="attention",
                 provenance=Provenance.PERSONA)
    app.state.engine.propose(s)
    app.state.engine.confirm(s.id)

    newer = Standard(category=Category.DOCTRINE, polarity=Polarity.DO,
                     directive="Lead with the verdict", rationale="faster",
                     provenance=Provenance.CORRECTION)
    app.state.engine.supersede(s.id, newer)

    old = app.state.store.get_standard(s.id)
    assert old.state == LayerState.SUPERSEDED  # NEVER deleted
    assert newer.supersedes == [s.id]

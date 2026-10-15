"""taste-2 API — the contract surface over UDS/Granian/FastAPI.

Every endpoint maps 1:1 to CONTRACT.md §3. create_app() is the conformance
entry point (TASTE_APP=taste.api:create_app + TASTE_APP_FACTORY=1).
"""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .core import StandardsEngine
from .feedback import record_feedback
from .hone import hone_segment
from .ingest import ingest_docs
from .inject import compile_packet
from .interview import run_interview
from .mirror import build_mirror
from .models import (
    FeedbackRequest,
    FeedbackResponse,
    HoneRequest,
    HoneResponse,
    ImportRequest,
    IngestRequest,
    IngestResponse,
    InterviewRequest,
    InterviewResponse,
    LayerState,
    MetaResponse,
    MirrorRequest,
    MirrorResponse,
    Packet,
    PacketRequest,
    Provenance,
    ScoreRequest,
    ScoreResult,
    Standard,
)
from .score import score_candidates
from .store import HoneBudgetExhausted, Store

PERMUTATION = {
    "schema": "A2-rich-docs",
    "retrieval": "B2-hybrid-rrf",
    "embed": "C2-Qwen3-0.6B",
    "scorer": "D2-llm-judge",
    "interview": "E4-lazy",
    "hone": "F2-critique-revise",
    "storage": "G2-LanceDB",
    "memory": "H5-confidence-math",
    "injection": "I2-few-shot-exemplars",
    "pipeline": "J2-dynamic",
}


def create_app(db_path: str | Path | None = None) -> FastAPI:
    store = Store(db_path) if db_path else Store()
    engine = StandardsEngine(store)

    app = FastAPI(title="taste-2", version="0.1.0")
    app.state.store = store
    app.state.engine = engine

    # Contract §1: {"error": {"code", "message"}}, never FastAPI's detail list.
    @app.exception_handler(HTTPException)
    async def _http_error(_request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": {"code": f"e{exc.status_code}",
                               "message": str(exc.detail)}})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_request: Request, exc: RequestValidationError):
        first = exc.errors()[:1]
        message = first[0].get("msg", "invalid request") if first else "invalid request"
        prefix = "Value error, "
        if message.startswith(prefix):
            message = message[len(prefix):]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_request", "message": message}})

    # ------------------------------------------------------------ meta & check
    @app.get("/meta", response_model=MetaResponse)
    def meta():
        return MetaResponse(
            name="taste-2",
            version="0.1.0",
            contract="v1",
            permutation=PERMUTATION,
        )

    @app.get("/check")
    def check(profile_id: str = "owner"):
        active = engine.active(profile_id)
        all_stds = store.list_standards(profile_id)
        cats: dict[str, int] = {}
        pols: dict[str, int] = {}
        for s in active:
            c = s.category.value if hasattr(s.category, "value") else str(s.category)
            p = s.polarity.value if hasattr(s.polarity, "value") else str(s.polarity)
            cats[c] = cats.get(c, 0) + 1
            pols[p] = pols.get(p, 0) + 1
        return {
            "status": "healthy",
            "name": "taste-2",
            "version": "0.1.0",
            "active_standards": len(active),
            "total_standards": len(all_stds),
            "categories": cats,
            "polarities": pols,
        }


    # ------------------------------------------------------------ task flow
    @app.post("/interview", response_model=InterviewResponse)
    def interview(req: InterviewRequest):
        return run_interview(engine, req.profile_id, req.task_ctx)

    @app.post("/mirror", response_model=MirrorResponse)
    def mirror(req: MirrorRequest):
        return build_mirror(engine, store, req)

    @app.post("/packet", response_model=Packet)
    def packet(req: PacketRequest):
        return compile_packet(engine, req)

    @app.post("/score", response_model=ScoreResult)
    def score(req: ScoreRequest):
        return ScoreResult(scores=score_candidates(
            engine, req.profile_id, req.task_ctx, req.candidates
        ))

    @app.post("/hone", response_model=HoneResponse)
    def hone(req: HoneRequest):
        try:
            return hone_segment(engine, req)
        except HoneBudgetExhausted as exc:
            raise HTTPException(429, str(exc)) from exc

    # ------------------------------------------------------------ learning
    @app.post("/feedback", response_model=FeedbackResponse)
    def feedback(req: FeedbackRequest):
        return record_feedback(engine, store, req)

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(req: IngestRequest):
        resp = ingest_docs(req.profile_id, req.docs)
        for cand in resp.candidates:
            engine.propose(cand)   # lands provisional — never auto-active
        return resp

    # ------------------------------------------------------------ corpus
    @app.get("/standards")
    def standards(profile_id: str = "owner",
                  state: str | None = None):
        st = None
        if state:
            try:
                st = LayerState(state)
            except ValueError:
                from fastapi.responses import JSONResponse
                return JSONResponse(
                    status_code=422,
                    content={"error": {
                        "code": "invalid_state",
                        "message": f"invalid state: {state!r} — expected one of "
                                   f"{[s.value for s in LayerState]}"}})
        rows = store.list_standards(profile_id, state=st)
        return {
            "standards": [s.model_dump(mode="json") for s in rows],
            "count": len(rows),
        }

    @app.post("/standards/import")
    def standards_import(req: ImportRequest):
        existing = {
            (s.profile_id, re.sub(r"\s+", " ", s.directive.lower()))
            for s in store.list_standards(req.profile_id)
        }
        imported = 0
        for s in req.standards:
            key = (s.profile_id, re.sub(r"\s+", " ", s.directive.lower()))
            if key in existing:
                continue
            existing.add(key)
            # Imported layers never silently activate if unconfirmed or without rationale
            if (s.provenance in (Provenance.AMENDMENT, Provenance.INFERENCE)
                    or not s.rationale.strip()):
                s.state = LayerState.PROVISIONAL
            # Declared state round-trips. It does not approve the row here.
            meta = dict(s.variant_meta)
            meta["local_approved"] = False
            s.variant_meta = meta
            vec_text = s.directive + " " + s.rationale
            from .embed import embed
            store.put_standard(s, embed([vec_text])[0])
            imported += 1
        return {"imported": imported, "skipped": len(req.standards) - imported}

    # ------------------------------------------------------ owner authority
    @app.post("/standards/{sid}/confirm")
    def confirm(sid: str, body: dict | None = None):
        rationale = (body or {}).get("rationale")
        try:
            ok = engine.confirm(sid, rationale=rationale)
        except ValueError as e:
            raise HTTPException(422, str(e)) from e
        if not ok:
            raise HTTPException(404, f"standard {sid} not found")
        return {"activated": sid}

    @app.post("/standards/{sid}/supersede")
    def supersede(sid: str, new: Standard):
        old = store.get_standard(sid)
        if not old:
            raise HTTPException(404, f"standard {sid} not found")
        return engine.supersede(sid, new)

    @app.post("/standards/{sid}/retire")
    def retire(sid: str):
        ok = engine.retire(sid)
        if not ok:
            raise HTTPException(404, f"standard {sid} not found")
        return {"retired": sid}

    @app.get("/review-queue")
    def review_queue(profile_id: str = "owner"):
        return {"pending": store.pending_inferred(profile_id)}

    @app.post("/review-queue/{iid}/resolve")
    def resolve(iid: str):
        store.resolve_inferred(iid)
        return {"resolved": iid}

    @app.get("/snapshots")
    def snapshots():
        return {"version": store.version, "snapshots": store.list_snapshots()}

    return app


app = create_app()

"""Storage — G2 permutation: LanceDB columnar table with vector column,
versioned snapshots per cycle, and columnar standards.

Mirrors the Store interface from taste-1 while backing state with LanceDB.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lancedb
import pyarrow as pa
from lancedb.index import FTS

from .models import (
    Category,
    DecisionRecord,
    LayerState,
    Polarity,
    Provenance,
    Standard,
)

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "lancedb"

logger = logging.getLogger("taste.store")

DIM = 1024


class HoneBudgetExhausted(Exception):
    """A task has used its hone calls. The cap is per task, across segments."""


def _standards_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("profile_id", pa.string()),
        pa.field("category", pa.string()),
        pa.field("polarity", pa.string()),
        pa.field("directive", pa.string()),
        pa.field("rationale", pa.string()),
        pa.field("content", pa.string()),
        pa.field("confidence", pa.float64()),
        pa.field("provenance", pa.string()),
        pa.field("state", pa.string()),
        pa.field("supersedes", pa.string()),
        pa.field("created", pa.string()),
        pa.field("updated", pa.string()),
        pa.field("variant_meta", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), DIM)),
    ])


def _decisions_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("profile_id", pa.string()),
        pa.field("task_ctx", pa.string()),
        pa.field("candidates", pa.string()),
        pa.field("pick", pa.string()),
        pa.field("rationale", pa.string()),
        pa.field("agent_role", pa.string()),
        pa.field("provenance", pa.string()),
        pa.field("ts", pa.string()),
        pa.field("variant_meta", pa.string()),
    ])


def _inferred_queue_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("profile_id", pa.string()),
        pa.field("kind", pa.string()),
        pa.field("payload", pa.string()),
        pa.field("ts", pa.string()),
        pa.field("resolved", pa.int64()),
    ])


def _hone_budget_schema() -> pa.Schema:
    return pa.schema([
        pa.field("id", pa.string()),
        pa.field("profile_id", pa.string()),
        pa.field("task_key", pa.string()),
        pa.field("reservations", pa.int64()),
    ])


def _task_segments_schema() -> pa.Schema:
    return pa.schema([
        pa.field("task_id", pa.string()),
        pa.field("profile_id", pa.string()),
        pa.field("segments", pa.string()),
        pa.field("ts", pa.string()),
    ])


class Store:
    def __init__(self, db_path: str | Path | None = None):
        path = Path(db_path or os.environ.get("TASTE2_DB", DEFAULT_DB))
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.db = lancedb.connect(str(path))
        self._lock = threading.Lock()

        # Initialize tables
        with self._lock:
            try:
                tables_res = self.db.list_tables()
                existing = set(tables_res.tables if hasattr(tables_res, "tables") else tables_res)
            except Exception as exc:
                logger.warning("list_tables() failed (%s); falling back to "
                               "table_names()", exc)
                existing = set(self.db.table_names())
            if "standards" in existing:
                self.standards_table = self.db.open_table("standards")
            else:
                self.standards_table = self.db.create_table(
                    "standards", schema=_standards_schema(), mode="overwrite"
                )
                try:
                    self.standards_table.create_index(
                        "content", config=FTS(), replace=True
                    )
                except Exception as exc:
                    logger.warning("FTS index creation on 'content' failed "
                                   "(%s); FTS lane will be unavailable", exc)

            if "decisions" in existing:
                self.decisions_table = self.db.open_table("decisions")
            else:
                self.decisions_table = self.db.create_table(
                    "decisions", schema=_decisions_schema(), mode="overwrite"
                )

            if "inferred_queue" in existing:
                self.queue_table = self.db.open_table("inferred_queue")
            else:
                self.queue_table = self.db.create_table(
                    "inferred_queue", schema=_inferred_queue_schema(), mode="overwrite"
                )

            if "task_segments" in existing:
                self.segments_table = self.db.open_table("task_segments")
            else:
                self.segments_table = self.db.create_table(
                    "task_segments", schema=_task_segments_schema(), mode="overwrite"
                )

            if "hone_budgets" in existing:
                self.hone_table = self.db.open_table("hone_budgets")
            else:
                self.hone_table = self.db.create_table(
                    "hone_budgets", schema=_hone_budget_schema(), mode="overwrite"
                )

    # ------------------------------------------------------------ standards
    def put_standard(self, s: Standard, embedding: list[float] | None = None):
        vec = embedding if embedding is not None else [0.0] * DIM
        content = f"{s.directive} {s.rationale}".strip()
        row = {
            "id": s.id,
            "profile_id": s.profile_id,
            "category": s.category.value,
            "polarity": s.polarity.value,
            "directive": s.directive,
            "rationale": s.rationale,
            "content": content,
            "confidence": float(s.confidence),
            "provenance": s.provenance.value,
            "state": s.state.value,
            "supersedes": json.dumps(s.supersedes),
            "created": s.created.isoformat(),
            "updated": s.updated.isoformat(),
            "variant_meta": json.dumps(s.variant_meta),
            "vector": vec,
        }
        with self._lock:
            self.standards_table.merge_insert("id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute([row])
            # Ensure FTS index stays active
            try:
                self.standards_table.create_index(
                    "content", config=FTS(), replace=True
                )
            except Exception as exc:
                logger.warning("FTS reindex after put_standard failed (%s); "
                               "FTS lane may be stale", exc)

    def get_standard(self, sid: str) -> Standard | None:
        safe_id = sid.replace("'", "''")
        rows = self.standards_table.search().where(f"id = '{safe_id}'").limit(1).to_list()
        return self._row_to_standard(rows[0]) if rows else None

    def list_standards(self, profile_id: str = "owner",
                       state: LayerState | None = None,
                       category: Category | None = None,
                       polarity: Polarity | None = None,
                       with_vectors: bool = False
                       ) -> list[Standard] | list[tuple[Standard, list[float] | None]]:
        safe_profile = profile_id.replace("'", "''")
        try:
            rows = self.standards_table.search().where(f"profile_id = '{safe_profile}'").to_list()
        except Exception as exc:
            logger.warning("filtered standards read failed (%s); falling back "
                           "to unfiltered scan + in-memory filter", exc)
            rows = self.standards_table.search().to_list()
            rows = [r for r in rows if r.get("profile_id") == profile_id]

        filtered = []
        for r in rows:
            if state and r.get("state") != state.value:
                continue
            if category and r.get("category") != category.value:
                continue
            if polarity and r.get("polarity") != polarity.value:
                continue
            std = self._row_to_standard(r)
            if with_vectors:
                raw_vec = r.get("vector")
                vec = list(raw_vec) if raw_vec is not None else None
                filtered.append((std, vec))
            else:
                filtered.append(std)
        return filtered

    def set_state(self, sid: str, state: LayerState):
        now_str = datetime.now(UTC).isoformat()
        safe_id = sid.replace("'", "''")
        with self._lock:
            self.standards_table.update(
                where=f"id = '{safe_id}'",
                values={"state": state.value, "updated": now_str}
            )

    def reserve_hone(self, profile_id: str, task_key: str, limit: int = 15) -> int:
        """Count one hone call. Raise once this task is past `limit`."""
        row_id = f"{profile_id}\x1f{task_key}"
        safe_id = row_id.replace("'", "''")
        with self._lock:
            hits = (self.hone_table.search()
                    .where(f"id = '{safe_id}'").limit(1).to_list())
            current = int(hits[0]["reservations"]) if hits else 0
            if current >= limit:
                raise HoneBudgetExhausted("The task hone budget is exhausted.")
            reservation = current + 1
            self.hone_table.merge_insert("id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute([{
                    "id": row_id,
                    "profile_id": profile_id,
                    "task_key": task_key,
                    "reservations": reservation,
                }])
            return reservation

    def set_variant_meta(self, sid: str, meta: dict):
        now_str = datetime.now(UTC).isoformat()
        safe_id = sid.replace("'", "''")
        with self._lock:
            self.standards_table.update(
                where=f"id = '{safe_id}'",
                values={"variant_meta": json.dumps(meta), "updated": now_str},
            )

    def set_rationale(self, sid: str, rationale: str):
        now_str = datetime.now(UTC).isoformat()
        safe_id = sid.replace("'", "''")
        s = self.get_standard(sid)
        directive = s.directive if s else ""
        content = f"{directive} {rationale}".strip()
        with self._lock:
            self.standards_table.update(
                where=f"id = '{safe_id}'",
                values={"rationale": rationale, "content": content, "updated": now_str}
            )
            try:
                self.standards_table.create_index(
                    "content", config=FTS(), replace=True
                )
            except Exception as exc:
                logger.warning("FTS reindex after set_rationale failed (%s); "
                               "FTS lane may be stale", exc)

    def set_confidence(self, sid: str, conf: float):
        now_str = datetime.now(UTC).isoformat()
        safe_id = sid.replace("'", "''")
        with self._lock:
            self.standards_table.update(
                where=f"id = '{safe_id}'",
                values={"confidence": float(conf), "updated": now_str}
            )

    def search_vector(self, query_vec: list[float], profile_id: str = "owner",
                      k: int = 20) -> list[tuple[Standard, float]]:
        """Vector lane search using LanceDB cosine distance."""
        try:
            safe_profile = profile_id.replace("'", "''")
            results = (self.standards_table.search(query_vec)
                       .where(f"profile_id = '{safe_profile}' AND state = 'active'")
                       .limit(k)
                       .to_list())
            out = []
            for r in results:
                dist = r.get("_distance", 0.0)
                # LanceDB cosine distance in [0, 2]; similarity = 1 - distance
                sim = 1.0 - float(dist)
                out.append((self._row_to_standard(r), sim))
            return out
        except Exception as exc:
            logger.warning("vector search failed (%s); returning empty lane "
                           "result — RRF will rely on the FTS lane only", exc)
            return []

    def search_fts(self, query_text: str, profile_id: str = "owner",
                   k: int = 20) -> list[tuple[Standard, float]]:
        """BM25 / FTS lane search using LanceDB full-text index."""
        if not query_text.strip():
            return []
        try:
            safe_profile = profile_id.replace("'", "''")
            results = (self.standards_table.search(query_text, query_type="fts")
                       .where(f"profile_id = '{safe_profile}' AND state = 'active'")
                       .limit(k)
                       .to_list())
            out = []
            for r in results:
                score = float(r.get("_score", 1.0))
                out.append((self._row_to_standard(r), score))
            return out
        except Exception as exc:
            logger.warning("FTS search failed (%s); returning empty lane "
                           "result — RRF will rely on the vector lane only",
                           exc)
            return []

    # ------------------------------------------------------------ versioning / snapshots
    @property
    def version(self) -> int:
        return self.standards_table.version

    def list_snapshots(self) -> list[dict[str, Any]]:
        return self.standards_table.list_versions()

    # ------------------------------------------------------------ row converter
    @staticmethod
    def _row_to_standard(r: dict[str, Any]) -> Standard:
        supersedes = r.get("supersedes")
        if isinstance(supersedes, str):
            try:
                supersedes_list = json.loads(supersedes)
            except Exception as exc:
                logger.warning("corrupt supersedes JSON for %s (%s); "
                               "treating as no lineage", r.get("id"), exc)
                supersedes_list = []
        elif isinstance(supersedes, list):
            supersedes_list = list(supersedes)
        else:
            supersedes_list = []

        variant_meta = r.get("variant_meta")
        if isinstance(variant_meta, str):
            try:
                meta_dict = json.loads(variant_meta)
            except Exception as exc:
                logger.warning("corrupt variant_meta JSON for %s (%s); "
                               "treating as empty", r.get("id"), exc)
                meta_dict = {}
        elif isinstance(variant_meta, dict):
            meta_dict = dict(variant_meta)
        else:
            meta_dict = {}

        created_val = r.get("created")
        if isinstance(created_val, str):
            try:
                created_dt = datetime.fromisoformat(created_val)
            except Exception as exc:
                logger.warning("unparseable created ts for %s (%s); "
                               "defaulting to now", r.get("id"), exc)
                created_dt = datetime.now(UTC)
        elif isinstance(created_val, datetime):
            created_dt = created_val
        else:
            created_dt = datetime.now(UTC)

        updated_val = r.get("updated")
        if isinstance(updated_val, str):
            try:
                updated_dt = datetime.fromisoformat(updated_val)
            except Exception as exc:
                logger.warning("unparseable updated ts for %s (%s); "
                               "defaulting to now", r.get("id"), exc)
                updated_dt = datetime.now(UTC)
        elif isinstance(updated_val, datetime):
            updated_dt = updated_val
        else:
            updated_dt = datetime.now(UTC)

        return Standard(
            id=str(r["id"]),
            profile_id=str(r.get("profile_id", "owner")),
            category=Category(str(r["category"])),
            polarity=Polarity(str(r["polarity"])),
            directive=str(r["directive"]),
            rationale=str(r.get("rationale", "")),
            confidence=float(r.get("confidence", 0.5)),
            provenance=Provenance(str(r.get("provenance", "persona"))),
            state=LayerState(str(r.get("state", "provisional"))),
            supersedes=supersedes_list,
            created=created_dt,
            updated=updated_dt,
            variant_meta=meta_dict,
        )

    # ------------------------------------------------------------ decisions
    def put_decision(self, d: DecisionRecord):
        row = {
            "id": d.id,
            "profile_id": d.profile_id,
            "task_ctx": json.dumps(d.task_ctx),
            "candidates": json.dumps([c.model_dump() for c in d.candidates]),
            "pick": d.pick,
            "rationale": d.rationale,
            "agent_role": d.agent_role,
            "provenance": d.provenance,
            "ts": d.ts.isoformat(),
            "variant_meta": json.dumps(d.variant_meta),
        }
        with self._lock:
            self.decisions_table.merge_insert("id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute([row])

    def list_decisions(self, profile_id: str = "owner") -> list[DecisionRecord]:
        safe_profile = profile_id.replace("'", "''")
        try:
            rows = self.decisions_table.search().where(f"profile_id = '{safe_profile}'").to_list()
        except Exception as exc:
            logger.warning("filtered decisions read failed (%s); falling back "
                           "to unfiltered scan + in-memory filter", exc)
            rows = self.decisions_table.search().to_list()
            rows = [r for r in rows if r.get("profile_id") == profile_id]

        # Sort by ts
        rows.sort(key=lambda r: str(r.get("ts", "")))
        out = []
        for r in rows:
            task_ctx = json.loads(r.get("task_ctx", "{}"))
            meta = json.loads(r.get("variant_meta", "{}"))
            ts_val = r.get("ts", "")
            try:
                ts_dt = datetime.fromisoformat(ts_val)
            except Exception as exc:
                logger.warning("unparseable decision ts for %s (%s); "
                               "defaulting to now", r.get("id"), exc)
                ts_dt = datetime.now(UTC)
            cands_raw = json.loads(r.get("candidates", "[]"))
            from .models import Candidate
            cands = [Candidate(**c) for c in cands_raw]
            out.append(DecisionRecord(
                id=str(r["id"]),
                profile_id=str(r["profile_id"]),
                task_ctx=task_ctx,
                candidates=cands,
                pick=str(r["pick"]),
                rationale=str(r["rationale"]),
                agent_role=str(r.get("agent_role", "")),
                provenance=str(r.get("provenance", "owner_pick")),
                ts=ts_dt,
                variant_meta=meta,
            ))
        return out

    # ------------------------------------------------------------ inferred queue
    def enqueue_inferred(self, iid: str, profile_id: str, kind: str, payload: dict):
        now_str = datetime.now(UTC).isoformat()
        row = {
            "id": iid,
            "profile_id": profile_id,
            "kind": kind,
            "payload": json.dumps(payload),
            "ts": now_str,
            "resolved": 0,
        }
        with self._lock:
            self.queue_table.merge_insert("id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute([row])

    def pending_inferred(self, profile_id: str = "owner") -> list[dict]:
        safe_profile = profile_id.replace("'", "''")
        try:
            rows = self.queue_table.search().where(
                f"profile_id = '{safe_profile}' AND resolved = 0"
            ).to_list()
        except Exception as exc:
            logger.warning("filtered inferred-queue read failed (%s); falling "
                           "back to unfiltered scan + in-memory filter", exc)
            rows = self.queue_table.search().to_list()
            rows = [r for r in rows if r.get("profile_id") == profile_id and r.get("resolved") == 0]

        rows.sort(key=lambda r: str(r.get("ts", "")))
        return [
            {
                "id": str(r["id"]),
                "kind": str(r["kind"]),
                "payload": json.loads(r.get("payload", "{}")),
                "ts": str(r["ts"]),
            }
            for r in rows
        ]

    def resolve_inferred(self, iid: str):
        safe_id = iid.replace("'", "''")
        with self._lock:
            self.queue_table.update(where=f"id = '{safe_id}'", values={"resolved": 1})

    # ------------------------------------------------------------ task segments (J2 dynamic audit)
    def put_task_segments(self, task_id: str, profile_id: str, segments: list[str]):
        now_str = datetime.now(UTC).isoformat()
        row = {
            "task_id": task_id,
            "profile_id": profile_id,
            "segments": json.dumps(segments),
            "ts": now_str,
        }
        with self._lock:
            self.segments_table.merge_insert("task_id") \
                .when_matched_update_all() \
                .when_not_matched_insert_all() \
                .execute([row])

    def get_task_segments(self, task_id: str) -> list[str] | None:
        safe_id = task_id.replace("'", "''")
        try:
            rows = self.segments_table.search().where(f"task_id = '{safe_id}'").limit(1).to_list()
            if rows:
                return json.loads(rows[0].get("segments", "[]"))
        except Exception as exc:
            logger.warning("get_task_segments(%s) failed (%s); returning None",
                           task_id, exc)
        return None

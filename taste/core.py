"""core — the standards engine (taste-2: A2 rich docs, B2 hybrid RRF, H5 confidence math).

State machine:
    provisional → active → superseded|retired
       (owner confirm)      (correction wins)   (never deleted — NEVER 5)

RRF retrieval (B2):
    v_hits = cosine_top(query, k=20)          # vector lane
    f_hits = fts_match(query, k=20)           # BM25 lane
    score(s) = 1/(60+v_rank(s)) + 1/(60+f_rank(s))   # absent lane = ∞ rank
    return top-12 by score

Confidence math (H5):
    owner confirm → +0.10; correction contradicting → −0.25; 30d unused → −0.05/week decay.
    Clamp [0.0, 1.0].
"""

from __future__ import annotations

from .embed import cosine, embed
from .models import (
    Category,
    LayerState,
    Standard,
)
from .store import Store

TOP_N = 12
CONFIDENCE_FLOOR = 0.3
RRF_K = 60


def locally_approved(standard: Standard) -> bool:
    """A portable ``state: active`` is not local owner approval.

    Import stamps ``variant_meta.local_approved = False``. Rows this
    engine activated itself omit the flag, so older confirmed corpora
    keep applying.
    """
    return standard.variant_meta.get("local_approved") is not False


class StandardsEngine:
    def __init__(self, store: Store):
        self.store = store

    # ---------------------------------------------------------- write path
    def propose(self, standard: Standard) -> Standard:
        """New layers ALWAYS arrive provisional — propose-never-apply."""
        standard.state = LayerState.PROVISIONAL
        vec = embed([standard.directive + " " + standard.rationale])[0]
        self.store.put_standard(standard, vec)
        return standard

    def confirm(self, sid: str, rationale: str | None = None) -> bool:
        """Owner confirms a restatement -> layer activates.
        H5 confidence math: owner confirm -> +0.10. Clamp [0, 1]."""
        s = self.store.get_standard(sid)
        if not s:
            return False
        if rationale is not None:
            s.rationale = rationale
            self.store.set_rationale(sid, rationale)
        if not s.rationale.strip():
            raise ValueError(f"{sid}: cannot activate without rationale")

        # H5: confirm -> +0.10 boost
        new_conf = min(1.0, max(0.0, s.confidence + 0.10))
        meta = dict(s.variant_meta)
        meta["local_approved"] = True
        self.store.set_variant_meta(sid, meta)
        self.store.set_confidence(sid, new_conf)
        self.store.set_state(sid, LayerState.ACTIVE)
        return True

    def supersede(self, old_id: str, new: Standard) -> Standard:
        """Correction wins: old layer marked superseded (kept forever),
        new layer links back via supersedes.
        H5 math: correction contradicting the layer -> -0.25 decay on old."""
        old = self.store.get_standard(old_id)
        if old:
            # Contradicted layer decays by 0.25
            old_conf = min(1.0, max(0.0, old.confidence - 0.25))
            self.store.set_confidence(old_id, old_conf)
            self.store.set_state(old_id, LayerState.SUPERSEDED)

        new.supersedes = [old_id]
        new.state = LayerState.PROVISIONAL
        self.propose(new)
        return new

    def retire(self, sid: str) -> bool:
        s = self.store.get_standard(sid)
        if not s:
            return False
        self.store.set_state(sid, LayerState.RETIRED)
        return True

    def reinforce(self, sid: str, delta: float):
        """H5 confidence reinforcement — clamped to [0.0, 1.0]."""
        s = self.store.get_standard(sid)
        if s:
            self.store.set_confidence(sid, min(1.0, max(0.0, s.confidence + delta)))

    def apply_decay(self, sid: str, days_unused: int = 30):
        """H5 decay: 30d unused -> -0.05/week decay. Clamp [0, 1]."""
        s = self.store.get_standard(sid)
        if not s:
            return
        if days_unused >= 30:
            weeks_beyond = (days_unused - 30) // 7
            if weeks_beyond > 0:
                decay = weeks_beyond * 0.05
                new_conf = min(1.0, max(0.0, s.confidence - decay))
                self.store.set_confidence(sid, new_conf)

    # ---------------------------------------------------------- read path (B2 RRF)
    def retrieve(self, query: str, profile_id: str = "owner",
                 category: str | None = None, top_n: int = TOP_N
                 ) -> list[tuple[Standard, float]]:
        """B2: hybrid RRF (BM25 + vector fusion).
        v_hits = cosine_top(query, k=20)
        f_hits = fts_match(query, k=20)
        score(s) = 1/(60+v_rank(s)) + 1/(60+f_rank(s))
        return top-12 by score
        """
        cat = Category(category) if category else None
        rows = self.store.list_standards(
            profile_id, state=LayerState.ACTIVE, category=cat, with_vectors=True
        )
        candidates = [(s, v) for s, v in rows
                      if v is not None and s.confidence >= CONFIDENCE_FLOOR
                      and locally_approved(s)]
        if not candidates:
            return []

        # 1. Vector lane (top 20)
        qv = embed([query])[0]
        v_scored = [(s, cosine(qv, v)) for s, v in candidates]
        # deterministic ranks: identical cosine scores tie-break by id,
        # not by storage scan order
        v_scored.sort(key=lambda t: (-t[1], t[0].id))
        v_hits = v_scored[:20]
        v_rank_map = {s.id: (rank + 1) for rank, (s, _) in enumerate(v_hits)}

        # 2. BM25 / FTS lane (top 20)
        fts_hits_raw = self.store.search_fts(query, profile_id, k=20)
        # Filter FTS hits to matching category & confidence floor
        f_hits = []
        for s, score in fts_hits_raw:
            if cat and s.category != cat:
                continue
            if s.confidence < CONFIDENCE_FLOOR or s.state != LayerState.ACTIVE:
                continue
            if not locally_approved(s):
                continue
            f_hits.append((s, score))
        # deterministic ranks: identical FTS scores tie-break by id
        f_hits.sort(key=lambda t: (-t[1], t[0].id))
        f_rank_map = {s.id: (rank + 1) for rank, (s, _) in enumerate(f_hits)}

        # Build candidate lookup
        std_lookup = {s.id: s for s, _ in candidates}
        for s, _ in f_hits:
            if s.id not in std_lookup:
                std_lookup[s.id] = s

        # 3. Hybrid RRF fusion
        all_ids = set(v_rank_map.keys()) | set(f_rank_map.keys())
        scored: list[tuple[Standard, float]] = []
        for sid in all_ids:
            s = std_lookup.get(sid)
            if not s:
                continue
            score = 0.0
            if sid in v_rank_map:
                score += 1.0 / (RRF_K + v_rank_map[sid])
            if sid in f_rank_map:
                score += 1.0 / (RRF_K + f_rank_map[sid])
            scored.append((s, score))

        # deterministic order: score desc, standard_id asc as tiebreak —
        # iterating a set makes tied scores hash-ordered otherwise
        scored.sort(key=lambda t: (-t[1], t[0].id))
        return scored[:top_n]

    def active(self, profile_id: str = "owner") -> list[Standard]:
        return [s for s in self.store.list_standards(
            profile_id, state=LayerState.ACTIVE) if locally_approved(s)]

    def coverage(self, profile_id: str, task_query: str) -> dict:
        """What taste already resolves for this task + where gaps are."""
        hits = self.retrieve(task_query, profile_id)
        by_cat: dict[str, int] = {}
        for s, rrf_score in hits:
            # In RRF, max score for top-1 in both lanes is 2/61 ~ 0.0328;
            # single lane rank 1 is 1/61 ~ 0.0164; rank 20 is 1/80 = 0.0125.
            if rrf_score >= 0.012:
                by_cat[s.category.value] = by_cat.get(s.category.value, 0) + 1

        all_cats = {"structure", "aesthetic", "procedural", "doctrine", "anti"}
        return {
            "resolved": sorted(by_cat.keys()),
            "gaps": sorted(all_cats - set(by_cat)),
        }

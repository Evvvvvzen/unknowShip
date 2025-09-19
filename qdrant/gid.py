# qdrant/gid.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from uuid import NAMESPACE_DNS, uuid5

from qdrant_client.http.models import FieldCondition, Filter, MatchValue, Range


# ---------------- 基礎工具 ----------------

def make_point_id(source_label: str, track_id: int, frame_idx: int, ts_ms: int) -> Tuple[str, str]:
    """產生穩定 UUID（v5）作為 Qdrant point_id，並回傳人類可讀 pid。"""
    pid_str = f"{source_label}|{int(track_id)}|f{int(frame_idx):06d}_t{int(ts_ms)}"
    return str(uuid5(NAMESPACE_DNS, pid_str)), pid_str


def _to_distance_from_score(score: float) -> float:
    """Qdrant score（COSINE 相似度 -1~1, 越大越像）→ 距離 d=1-score；異常回 1e9。"""
    try:
        s = float(score)
    except Exception:
        return 1e9
    if math.isnan(s) or math.isinf(s):
        return 1e9
    if -1.0 <= s <= 1.0:
        return 1.0 - s
    return s


def _l2_normalize(vec: Optional[np.ndarray], eps: float = 1e-12) -> Optional[np.ndarray]:
    """L2 normalize；零向量直接回傳。"""
    if vec is None:
        return None
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < eps:
        return v
    return (v / n).astype(np.float32)


# ---------------- 檢索結果容器 ----------------

@dataclass
class Hit:
    """簡化後的檢索結果容器"""
    score: float
    payload: Dict[str, Any]

    @property
    def distance(self) -> float:
        return _to_distance_from_score(self.score)


# ---------------- 決策器 ----------------

class GIDAssigner:
    """
    跨鏡頭 GID 決策器
    - 時間窗過濾（單點/區間/多點取 min-max）
    - 排除同來源
    - 排除 global_id=-1
    - 僅檢索 committed=True
    - 決策：加權投票 + tie-break（平均距離/最近幀/票數）
    """

    def __init__(
        self,
        store,
        reid_thresh: float,
        max_dt_ms: int,
        topk: int,
        seq_start: int = 1,
        debug: bool = False,
        debug_top_hits: int = 10,
        debug_dump_per_frame: bool = True,
    ):
        self.store = store
        self.reid_thresh = float(reid_thresh)
        self.max_dt_ms = int(max_dt_ms)
        self.topk = int(topk)
        self._seq = int(seq_start)
        # (source_label, local_track_id) -> gid
        self._local2global: Dict[Tuple[str, int], int] = {}
        # debug 設定
        self.debug = bool(debug)
        self.debug_top_hits = int(debug_top_hits)          # 每次搜尋列出前 N 命中
        self.debug_dump_per_frame = bool(debug_dump_per_frame)  # 是否逐幀列出 best 命中

    # ---------------- internal helpers ----------------

    def _build_time_filter(
        self,
        this_src_label: str,
        ts: Optional[Union[int, Tuple[int, int], Sequence[int]]],
    ) -> Filter:
        """
        ts 可為：
          - int: 建立 [ts-max_dt, ts+max_dt]
          - (start,end): 建立 [min-max 擴張 max_dt]
          - 序列: 取 min/max 後建立
        固定條件：
          - must: committed == True
          - must_not: source_label == this_src_label, global_id == -1
        """
        must = [FieldCondition(key="committed", match=MatchValue(value=True))]
        must_not = [
            FieldCondition(key="source_label", match=MatchValue(value=this_src_label)),
            FieldCondition(key="global_id", match=MatchValue(value=-1)),
        ]

        if ts is None or self.max_dt_ms <= 0:
            return Filter(must=must, must_not=must_not)

        if isinstance(ts, int):
            start = int(ts - self.max_dt_ms)
            end = int(ts + self.max_dt_ms)
        elif isinstance(ts, tuple) and len(ts) == 2:
            lo, hi = ts
            if lo is None or hi is None:
                return Filter(must=must, must_not=must_not)
            start = int(min(lo, hi) - self.max_dt_ms)
            end = int(max(lo, hi) + self.max_dt_ms)
        elif isinstance(ts, Sequence) and len(ts) > 0:
            vals = [int(v) for v in ts if v is not None]
            if not vals:
                return Filter(must=must, must_not=must_not)
            start = min(vals) - self.max_dt_ms
            end = max(vals) + self.max_dt_ms
        else:
            return Filter(must=must, must_not=must_not)

        must.append(FieldCondition(key="timestamp_ms", range=Range(gte=start, lte=end)))
        return Filter(must=must, must_not=must_not)

    def _normalize_vec(self, vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return _l2_normalize(vec)

    def _search_candidates(
        self,
        vec: np.ndarray,
        this_src_label: str,
        ts: Optional[Union[int, Tuple[int, int], Sequence[int]]],
    ) -> List[Hit]:
        qfilter = self._build_time_filter(this_src_label, ts)
        v = self._normalize_vec(vec)
        raw_hits = self.store.search(vector=v, topk=self.topk, query_filter=qfilter)
        hits: List[Hit] = []
        for h in raw_hits or []:
            sc = getattr(h, "score", None)
            pl = getattr(h, "payload", None)
            if sc is None or pl is None:
                if isinstance(h, dict):
                    sc = h.get("score")
                    pl = h.get("payload")
            if sc is None or pl is None:
                continue
            hits.append(Hit(score=float(sc), payload=dict(pl)))

        if self.debug:
            rng = None
            if isinstance(ts, int):
                rng = (int(ts - self.max_dt_ms), int(ts + self.max_dt_ms))
            print(f"[GID] search topk={self.topk}, time_win={rng}, not_src='{this_src_label}' → hits={len(hits)}")
            if self.debug_top_hits > 0 and hits:
                hs = sorted(hits, key=lambda h: h.distance)[: self.debug_top_hits]
                for i, h in enumerate(hs, 1):
                    pl = h.payload or {}
                    print(
                        f"    #{i}: gid={pl.get('global_id')} d={h.distance:.4f} ts={pl.get('timestamp_ms')} src={pl.get('source_label')}"
                    )
        return hits

    def _search_candidates_batch(
        self,
        vecs: Sequence[np.ndarray],
        this_src_label: str,
        ts: Optional[Union[int, Tuple[int, int], Sequence[int]]],
    ) -> List[List[Hit]]:
        vecs_norm = [self._normalize_vec(v) for v in vecs]
        hits_batch: List[List[Hit]] = []

        qfilter = self._build_time_filter(this_src_label, ts)

        if hasattr(self.store, "search_batch"):
            raw = self.store.search_batch(vectors=vecs_norm, topk=self.topk, query_filter=qfilter) or []
            for raw_hits in raw:
                cur: List[Hit] = []
                for h in raw_hits or []:
                    sc = getattr(h, "score", None)
                    pl = getattr(h, "payload", None)
                    if sc is None or pl is None:
                        if isinstance(h, dict):
                            sc = h.get("score")
                            pl = h.get("payload")
                    if sc is None or pl is None:
                        continue
                    cur.append(Hit(score=float(sc), payload=dict(pl)))
                hits_batch.append(cur)
        else:
            for v in vecs_norm:
                raw_hits = self.store.search(vector=v, topk=self.topk, query_filter=qfilter)
                cur: List[Hit] = []
                for h in raw_hits or []:
                    sc = getattr(h, "score", None)
                    pl = getattr(h, "payload", None)
                    if sc is None or pl is None:
                        if isinstance(h, dict):
                            sc = h.get("score")
                            pl = h.get("payload")
                    if sc is None or pl is None:
                        continue
                    cur.append(Hit(score=float(sc), payload=dict(pl)))
                hits_batch.append(cur)

        if self.debug:
            total = sum(len(hs) for hs in hits_batch)
            print(f"[GID] batch search n_vecs={len(vecs_norm)} total_hits={total}")
            if self.debug_top_hits > 0 and hits_batch and hits_batch[0]:
                hs = sorted(hits_batch[0], key=lambda h: h.distance)[: self.debug_top_hits]
                for i, h in enumerate(hs, 1):
                    pl = h.payload or {}
                    print(
                        f"    [first-vec] #{i}: gid={pl.get('global_id')} d={h.distance:.4f} ts={pl.get('timestamp_ms')} src={pl.get('source_label')}"
                    )
        return hits_batch

    def _pick_gid_from_hits(self, hits: Sequence[Hit]) -> Optional[int]:
        """挑出距離最小且 <= reid_thresh 的 gid（僅正 gid）。"""
        best_gid, best_d = None, 1e9
        for h in hits:
            pl = h.payload or {}
            gid = pl.get("global_id")
            if gid is None or int(gid) <= 0:
                continue
            d = h.distance
            if d <= self.reid_thresh and d < best_d:
                best_d, best_gid = d, int(gid)
        return best_gid

    def _issue_new_gid(self) -> int:
        gid = self._seq
        self._seq += 1
        return gid

    # ---------------- public APIs ----------------

    def assign_or_issue(
        self,
        feature_vec: np.ndarray,
        this_src_label: str,
        ts_ms: Optional[int],
        local_key: Tuple[str, int],
    ) -> int:
        if local_key in self._local2global:
            return int(self._local2global[local_key])

        if feature_vec is None:
            gid = self._issue_new_gid()
            self._local2global[local_key] = gid
            if self.debug:
                print("[GID] assign_or_issue: feature_vec is None → issue new", gid)
            return gid

        hits = self._search_candidates(feature_vec, this_src_label, ts_ms)
        gid = self._pick_gid_from_hits(hits)
        if gid is None:
            gid = self._issue_new_gid()

        self._local2global[local_key] = int(gid)
        return int(gid)

    def assign_by_aggregate(
        self,
        features: Sequence[np.ndarray],
        this_src_label: str,
        ts_ms: Optional[int],
        local_key: Tuple[str, int],
        vote_ratio: float = 1.5,
        min_votes: int = 10,
        *,
        weighted: bool = True,
        weight_tau: float = 0.2,
        min_votes_ratio: Optional[float] = None,
        tie_break: str = "mean_d",
    ) -> int:
        if local_key in self._local2global:
            return int(self._local2global[local_key])

        if not features:
            return self.assign_or_issue(
                feature_vec=None if not features else features[-1],
                this_src_label=this_src_label,
                ts_ms=ts_ms,
                local_key=local_key,
            )

        hits_batch = self._search_candidates_batch(features, this_src_label, ts_ms)

        votes: Dict[int, int] = {}
        weights: Dict[int, float] = {}
        d_stats: Dict[int, List[float]] = {}
        recency: Dict[int, int] = {}
        valid_votes = 0

        for idx, hits in enumerate(hits_batch):
            best_gid, best_d = None, 1e9
            for h in hits:
                gid = (h.payload or {}).get("global_id")
                if gid is None or int(gid) <= 0:
                    continue
                d = h.distance
                if d <= self.reid_thresh and d < best_d:
                    best_d, best_gid = d, int(gid)

            if best_gid is None:
                continue

            valid_votes += 1
            d_stats.setdefault(best_gid, []).append(best_d)

            if weighted:
                w = math.exp(-best_d / max(1e-6, weight_tau))
                weights[best_gid] = weights.get(best_gid, 0.0) + float(w)
            else:
                votes[best_gid] = votes.get(best_gid, 0) + 1

            if idx >= len(hits_batch) - 3:
                recency[best_gid] = recency.get(best_gid, 0) + 1

            if self.debug and self.debug_dump_per_frame:
                print(f"    frame#{idx:02d}: pick_gid={best_gid} d={best_d:.4f}")

        if valid_votes == 0:
            gid = self._issue_new_gid()
            self._local2global[local_key] = gid
            if self.debug:
                print("[GID] aggregate: no valid votes → new gid", gid)
            return gid

        if weighted:
            ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)
            top_val = ranked[0][1]
            second_val = ranked[1][1] if len(ranked) >= 2 else 0.0
        else:
            ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
            top_val = ranked[0][1]
            second_val = ranked[1][1] if len(ranked) >= 2 else 0.0

        top_gid = int(ranked[0][0])

        min_needed = float(min_votes)
        if min_votes_ratio is not None and 0.0 < float(min_votes_ratio) <= 1.0:
            dyn_needed = math.ceil(valid_votes * float(min_votes_ratio))
            min_needed = max(min_needed, dyn_needed)

        if self.debug:
            print("[GID] aggregate summary:")
            all_gids = set(d_stats.keys()) | set(weights.keys()) | set(votes.keys())
            for gid in sorted(all_gids):
                cnt = votes.get(gid, 0)
                wsum = weights.get(gid, 0.0)
                md = float(np.mean(d_stats.get(gid, [np.nan])))
                rc = recency.get(gid, 0)
                print(f"    gid={gid:>4} cnt={cnt:>3} wsum={wsum:7.3f} mean_d={md:6.4f} recent={rc}")
            print(f"    valid_votes={valid_votes}, reid_thresh={self.reid_thresh}, vote_ratio={vote_ratio}, min_votes={min_votes}")

        if second_val == 0 and top_val >= max(1.0, min_needed * 0.5):
            if top_gid > 0:
                gid = top_gid
                self._local2global[local_key] = gid
                if self.debug:
                    print(f"[GID] aggregate OK (no second): gid={gid}, top={top_val}, valid={valid_votes}")
                return gid

        pass_ratio = (top_val >= second_val * float(vote_ratio))
        pass_min = (top_val >= min_needed)

        if not (pass_ratio and pass_min):
            chosen_gid: Optional[int] = None
            if abs(top_val - second_val) <= 1e-6 and len(ranked) >= 2:
                a_gid, b_gid = int(ranked[0][0]), int(ranked[1][0])
                if tie_break == "mean_d":
                    a_md = float(np.mean(d_stats.get(a_gid, [1e9])))
                    b_md = float(np.mean(d_stats.get(b_gid, [1e9])))
                    chosen_gid = a_gid if a_md < b_md else b_gid
                elif tie_break == "recent":
                    a_r = recency.get(a_gid, 0)
                    b_r = recency.get(b_gid, 0)
                    chosen_gid = a_gid if a_r >= b_r else b_gid
                else:
                    chosen_gid = a_gid
            else:
                a_gid = top_gid
                a_md = float(np.mean(d_stats.get(a_gid, [1e9])))
                if a_md < self.reid_thresh * 0.7 and top_val >= max(1.0, min_needed * 0.8):
                    chosen_gid = a_gid

            if chosen_gid is not None and chosen_gid > 0:
                gid = int(chosen_gid)
                self._local2global[local_key] = gid
                if self.debug:
                    print(f"[GID] aggregate tie-break chosen gid={gid}, top={top_val}, second={second_val}")
                return gid

            gid = self._issue_new_gid()
            self._local2global[local_key] = gid
            if self.debug:
                print(f"[GID] aggregate fallback new gid={gid} (top={top_val}, second={second_val}, valid={valid_votes})")
            return gid

        gid = top_gid
        self._local2global[local_key] = gid
        if self.debug:
            print(f"[GID] aggregate OK: gid={gid}, top={top_val}, second={second_val}, valid={valid_votes}")
        return gid

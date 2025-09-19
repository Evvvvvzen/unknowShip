# tools/pairing.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Deque, Optional, List
from collections import deque
import threading
import numpy as np

@dataclass
class Candidate:
    """
    從『來源相機(send)』送進配對池的候選資料
    - gid: 已決策的全域 ID (>0)
    - class_id: 類別（同 class 才能配對；若 strict_class=True，class_id 必須 >= 0）
    - src_label: 來源攝影機標籤
    - ts_ms: 送入池時的時間戳（毫秒）
    - foot_xy: 送入時的 foot point（debug/距離衡量可用）
    - feat_avg: 可選；觀察期的平均/代表性特徵（ReID 比對時用）
    - ttl_ms: 壽命（毫秒），超過會被清掉
    - used: 一旦成功匹配後標記 used，後續 prune 會移除
    """
    gid: int
    class_id: int
    src_label: str
    ts_ms: int
    foot_xy: Tuple[int, int]
    feat_avg: Optional[np.ndarray]
    ttl_ms: int
    used: bool = False

def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - float((a @ b) / (na * nb))

class PairingHub:
    """
    跨執行緒安全的配對池（嚴格 class 分流）：
      key = (roi_id, class_id)  → deque[Candidate]
    """
    def __init__(self, max_per_key: int = 64, strict_class: bool = True):
        self.pool: Dict[Tuple[str, int], Deque[Candidate]] = {}
        self.lock = threading.Lock()
        self.max_per_key = int(max_per_key)
        self.strict_class = bool(strict_class)
        self._warned_unknown_class = False  # 避免重複洗版警告
        
        # 放在 PairingHub 類別內，其他方法之後都不用動
    def snapshot(self, roi_id: str, now_ms: int, max_per_class: int = 5):
        """
        回傳形如 {class_id: [Candidate, ...]} 的 dict，
        僅包含未過期且未 used 的候選；每個 class 最多取 max_per_class 筆（由新到舊）。
        """
        out = {}
        for (rid, cls_id), dq in getattr(self, "pool", {}).items():
            if rid != roi_id:
                continue
            items = []
            # dq 的尾端通常是最新；反向迭代，取最近的
            for c in reversed(dq):
                ts = int(getattr(c, "ts_ms", 0))
                ttl = int(getattr(c, "ttl_ms", 0))
                used = bool(getattr(c, "used", False))
                if used:
                    continue
                if ttl > 0 and now_ms - ts > ttl:
                    continue
                items.append(c)
                if len(items) >= max_per_class:
                    break
            if items:
                out[int(cls_id)] = items
            return out


    # ---- helpers ----
    def _class_ok(self, cls: int) -> bool:
        return (cls is not None) and (int(cls) >= 0)

    # ---- API ----
    def push(self, roi_id: str, cand: Candidate):
        """
        推入候選：
        - 若 strict_class=True 且 cand.class_id < 0 → 直接丟棄（不進池）
        - 依 (roi_id, class_id) 分桶；每桶保留上限 max_per_key
        """
        if self.strict_class and not self._class_ok(cand.class_id):
            if not self._warned_unknown_class:
                print("[PAIR][WARN] drop candidate with unknown class_id (<0). "
                      "Set strict_class=False if you want to keep them.")
                self._warned_unknown_class = True
            return

        key = (roi_id, int(cand.class_id))
        with self.lock:
            dq = self.pool.setdefault(key, deque())
            dq.append(cand)
            while len(dq) > self.max_per_key:
                dq.popleft()

    def candidate_count(self, roi_id: str, class_id: int, now_ms: Optional[int] = None) -> int:
        """
        回傳該 (roi_id, class_id) 桶內「未過期且未使用」的候選數。
        若 strict_class=True 且 class_id < 0 → 回 0。
        """
        if self.strict_class and not self._class_ok(class_id):
            return 0

        key = (roi_id, int(class_id))
        with self.lock:
            dq = self.pool.get(key, deque())
            if now_ms is None:
                # 所有未 used 的都算（不考慮過期）
                return sum(0 if c.used else 1 for c in dq)
            # 計算未過期＆未使用的
            n = 0
            for c in dq:
                if c.used:
                    continue
                if now_ms - c.ts_ms <= c.ttl_ms:
                    n += 1
            return n

    def _prune_key(self, key: Tuple[str, int], now_ms: int):
        dq = self.pool.get(key)
        if not dq:
            return
        kept = deque([c for c in dq if (not c.used) and (now_ms - c.ts_ms <= c.ttl_ms)])
        self.pool[key] = kept

    def prune(self, now_ms: int):
        with self.lock:
            for key in list(self.pool.keys()):
                self._prune_key(key, now_ms)

    def match(
        self,
        roi_id: str,
        class_id: int,
        now_ms: int,
        query_feat: Optional[np.ndarray],
        singleton_hint: bool = False,
        reid_thresh: float = 0.35,
    ) -> Optional[int]:
        """
        從池中找最適合的 gid，並標記該 candidate used。
        僅在 *同一個* (roi_id, class_id) 桶內進行。
        規則：
          1) 若 singleton_hint 且該桶僅 1 候選 → 直接使用（不比特徵）
          2) 若多候選：
               - 有 query_feat → 以 cosine 距離最小；若最小距 > reid_thresh → fallback
               - 沒 query_feat 或 A 失敗 → 用 |now - ts| 最小（最近時間）
        回傳 gid 或 None
        """
        if self.strict_class and not self._class_ok(class_id):
            return None

        key = (roi_id, int(class_id))
        with self.lock:
            dq = self.pool.get(key)
            if not dq:
                return None

            # 先剔除過期／used
            valid: List[int] = []
            for i, c in enumerate(dq):
                if c.used:
                    continue
                if now_ms - c.ts_ms <= c.ttl_ms:
                    valid.append(i)
            if not valid:
                return None

            if singleton_hint and len(valid) == 1:
                idx = valid[0]
                dq[idx].used = True
                return dq[idx].gid

            # 多候選
            # (A) 有特徵 → 以外觀距離排序
            if query_feat is not None:
                best_i, best_d = None, 1e9
                for i in valid:
                    c = dq[i]
                    if c.feat_avg is None:
                        continue
                    d = _cos_dist(query_feat, c.feat_avg)
                    if d < best_d:
                        best_d, best_i = d, i
                if best_i is not None and best_d <= reid_thresh:
                    dq[best_i].used = True
                    return dq[best_i].gid
                # 若特徵不可信（全部沒 feat 或距離過大）→ fallback 到時間最近

            # (B) 沒特徵或 A 失敗 → 時間最近
            best_i, best_gap = None, 1e18
            for i in valid:
                gap = abs(now_ms - dq[i].ts_ms)
                if gap < best_gap:
                    best_gap, best_i = gap, i
            if best_i is not None:
                dq[best_i].used = True
                return dq[best_i].gid

            return None

    def size_summary(self) -> Dict[str, int]:
        """
        debug 用：回傳各桶的候選數（未 used；不考慮過期）
        key 格式為 "roi_id|class_id"
        """
        out: Dict[str, int] = {}
        with self.lock:
            for (roi_id, cls_id), dq in self.pool.items():
                out[f"{roi_id}|{cls_id}"] = sum(0 if c.used else 1 for c in dq)
        return out

# pipeline/sync.py
import threading
from typing import List, Optional


class SyncState:
    """
    軟同步：限制各來源只能比最慢者快 tol_ms 以內。
    用法：
      - 每次拿到該來源當前 ts_ms，就呼叫 update_and_should_advance(src_idx, ts, tol)
      - 回傳 True 才前進讀下一幀；False 表示暫停一下再檢查
    """

    def __init__(self, n_sources: int):
        self.first_ts: List[Optional[int]] = [None] * n_sources
        self.adj_ts:   List[Optional[int]] = [None] * n_sources
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            n = len(self.first_ts)
            self.first_ts = [None] * n
            self.adj_ts = [None] * n

    def update_and_should_advance(self, src_idx: int, cur_ts_ms: int, tol_ms: int) -> bool:
        with self.lock:
            if self.first_ts[src_idx] is None:
                self.first_ts[src_idx] = cur_ts_ms
            cur_adj = cur_ts_ms - self.first_ts[src_idx]
            self.adj_ts[src_idx] = cur_adj

            vals = [v for v in self.adj_ts if v is not None]
            if len(vals) < 2:
                return True
            min_adj = min(vals)
            return cur_adj <= (min_adj + tol_ms)

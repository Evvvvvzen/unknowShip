# tools/obs_buffer.py
from __future__ import annotations
from typing import List
import numpy as np

class ObsBuffer:
    """
    觀察期收集與達標判斷（不決定 GID，只負責『是不是可以決策了』）。
    - obs_n：最多收集 N 幀就決策
    - timeout_s：收集時間超過多少秒也決策（但要有 obs_min 幀）
    - obs_min：timeout 決策時，至少需要的幀數
    """
    def __init__(self, obs_n: int, timeout_s: float, obs_min: int):
        self.obs_n = int(obs_n)
        self.timeout_s = float(timeout_s)
        self.obs_min = int(obs_min)

    def ready(self, feats: List[np.ndarray], start_ts_ms: int, now_ts_ms: int) -> bool:
        m = len(feats)
        if m >= self.obs_n:
            return True
        if (now_ts_ms - start_ts_ms) >= int(self.timeout_s * 1000) and m >= self.obs_min:
            return True
        return False

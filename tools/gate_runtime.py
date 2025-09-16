# tools/gate_runtime.py
from __future__ import annotations
from typing import Dict, Optional, Tuple

import cv2  # noqa: F401  # 這裡不直接用，但保留相依有助 IDE/型別提示
from .gate import State, draw_gate_segments, init_state_by_first, next_state

# 型別別名
Point = Tuple[int, int]
Segment = Tuple[Point, Point]
Segments = Tuple[Segment, Segment]  # (top_seg, bottom_seg)


class GateRuntime:
    """
    封裝 Gate 的『繪製＋狀態機更新』：
      - draw(frame): 若有 gate，就把 GL_top/GL_bottom 畫上去
      - update(track_id, foot_xy): 回傳該 track 最新 Gate 狀態（State）
    使用方式：
      gate_rt = GateRuntime(gate_segments, default_state=State.Inside)
      gate_rt.draw(frame)
      cur_state = gate_rt.update(track_id, (fx, fy))
    """
    def __init__(self, gate_segments: Optional[Segments], default_state: State = State.Inside):
        self.gate_segments: Optional[Segments] = gate_segments
        self.default_state: State = default_state
        self._track_state: Dict[int, State] = {}

        # 容錯：若 gate_segments 不完整，當作沒有 gate
        if self.gate_segments is not None:
            top, bottom = self.gate_segments
            if top is None or bottom is None:
                self.gate_segments = None

    def draw(self, frame) -> None:
        """把 Gate 畫在原始影像上（若有設定 gate）。"""
        if self.gate_segments is None:
            return
        top_seg, bot_seg = self.gate_segments
        draw_gate_segments(frame, top_seg, bot_seg)

    def update(self, track_id: int, foot_xy: Point) -> State:
        """
        用 foot point 更新指定 track 的 Gate 狀態並回傳。
        - 沒有 gate 時：始終回傳 default_state（例如 Inside）
        - 有 gate 時：使用 gate.py 的 init_state_by_first / next_state
        """
        if self.gate_segments is None:
            # 無 gate：固定回預設狀態
            self._track_state[track_id] = self.default_state
            return self.default_state

        top_seg, bot_seg = self.gate_segments
        prev = self._track_state.get(track_id)
        if prev is None:
            prev = init_state_by_first(foot_xy, top_seg)
        cur = next_state(prev, foot_xy, top_seg, bot_seg)
        self._track_state[track_id] = cur
        return cur

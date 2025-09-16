# tools/gate.py
from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
import json
import cv2

# ====== 可調參數 ======
# 允許在做「以 y 插值求 x」時，y 超出線段端點的放寬量（像素）
Y_EXTEND = 12
# 邊界容差（像素）：靠近線邊時的抖動緩衝
DX_EPS = 3
DY_EPS = 3

# 幾何型別別名
Point = Tuple[int, int]
Segment = Tuple[Point, Point]


class State(str, Enum):
    Inside = "Inside"              # 在 GL_bottom 右側
    PreparingExit = "PreparingExit"  # 從 Inside 進入「走廊」
    Outside = "Outside"            # 在 GL_top 左側
    PreparingEnter = "PreparingEnter"  # 從 Outside 進入「走廊」

# ---------------- I/O 與繪製 ----------------


def load_gate_segments(path: str | Path) -> Tuple[Segment, Segment]:
    """
    從 JSON 讀 gate，格式需含：
      gate.GL_top.p1, gate.GL_top.p2, gate.GL_bottom.p1, gate.GL_bottom.p2
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        g = json.load(f)["gate"]

    def tup(v): return (int(v[0]), int(v[1]))
    top = (tup(g["GL_top"]["p1"]), tup(g["GL_top"]["p2"]))
    bottom = (tup(g["GL_bottom"]["p1"]), tup(g["GL_bottom"]["p2"]))
    return top, bottom


def draw_gate_segments(im, top: Segment, bottom: Segment) -> None:
    # 畫線與端點
    cv2.line(im, top[0], top[1], (0, 0, 255), 2)
    cv2.circle(im, top[0], 5, (0, 0, 255), -1)
    cv2.circle(im, top[1], 5, (0, 0, 255), -1)

    cv2.line(im, bottom[0], bottom[1], (0, 255, 255), 2)
    cv2.circle(im, bottom[0], 5, (0, 255, 255), -1)
    cv2.circle(im, bottom[1], 5, (0, 255, 255), -1)

    # ---- 標籤：放在該線段「最小 y 的端點」上方一點 ----
    def min_y_point(seg: Segment):
        return seg[0] if seg[0][1] <= seg[1][1] else seg[1]

    # 位置微調（像素）
    LABEL_DX = 6   # 往右一點
    LABEL_DY = 6   # 往上（y 要減少）

    # GL_top 標籤位置
    tx, ty = min_y_point(top)
    tx = max(0, tx + LABEL_DX)
    ty = max(0, ty - LABEL_DY)
    cv2.putText(
        im, "GL_top", (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
    )

    # GL_bottom 標籤位置
    bx, by = min_y_point(bottom)
    bx = max(0, bx + LABEL_DX)
    by = max(0, by - LABEL_DY)
    cv2.putText(
        im, "GL_bottom", (bx, by),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )


# ---------------- 幾何工具：支援垂直/近垂直線 ----------------


def _ymin_of_segment(seg: Segment) -> int:
    return min(seg[0][1], seg[1][1])


def _y_cut(top: Segment, bottom: Segment) -> int:
    """
    兩條 Gate 的「最上端」y（四個端點的最小 y）。
    """
    return min(top[0][1], top[1][1], bottom[0][1], bottom[1][1])


def interp_x_on_segment(seg: Segment, y: int, extend: int = Y_EXTEND) -> Optional[float]:
    """
    在線段 seg=((x1,y1),(x2,y2)) 的 y 範圍內（含 extend），以 y 線性插值得到 x。
    水平線段或 y 超界回傳 None（但近垂直線會正常工作）。
    """
    (x1, y1), (x2, y2) = seg
    if y1 == y2:
        return None  # 水平線無法用 y 當自變數
    ymin, ymax = sorted([y1, y2])
    if y < ymin - extend or y > ymax + extend:
        return None
    t = (y - y1) / float(y2 - y1)
    return x1 + t * (x2 - x1)


def _fallback_x(seg: Segment) -> float:
    """當插值失敗時，退而求其次用該段兩端點 x 的平均。對近垂直線很接近真值。"""
    return (seg[0][0] + seg[1][0]) / 2.0


def _classify_region(foot_xy: Point, top: Segment, bottom: Segment) -> Optional[str]:
    """
    區域分類（僅在 y >= y_cut 才生效）：
      - 'Outside'  : 在 GL_top 左側
      - 'Corridor' : 在 GL_top 與 GL_bottom 之間
      - 'Inside'   : 在 GL_bottom 右側
    回傳 None 表示 y 尚未達到門檻（外海上方）：呼叫端應忽略、不更新狀態。
    """
    x, y = foot_xy
    y_cut = _y_cut(top, bottom)

    if y < y_cut:      # 嚴格依你需求，不加容差
        return "Outside"  # 直接視為港外（不看 x，也不進入走廊判定）

    # 以「y -> x」插值，對垂直或近垂直線穩定
    x_top = interp_x_on_segment(top, y)
    x_bottom = interp_x_on_segment(bottom, y)

    if x_top is None:
        x_top = _fallback_x(top)
    if x_bottom is None:
        x_bottom = _fallback_x(bottom)

    # 允許 GL_top、GL_bottom 在局部高度處出現交錯（使用排序保險）
    left, right = sorted([x_top, x_bottom])

    if x <= left - DX_EPS:
        return "Outside"
    elif x >= right + DX_EPS:
        return "Inside"
    else:
        return "Corridor"

# ---------------- 狀態機（符合你定義） ----------------


def init_state_by_first(foot_xy: Point, top_seg: Segment) -> State:
    """
    初始狀態：若第一筆點位於 GL_bottom 右側（且 y 過門檻），視為 Inside；否則 Outside。
    （簡化：初始不進入「準備」狀態）
    """
    # 這裡需要 bottom，因此用 next_state 的分類邏輯會更健壯；
    # 但為了維持簽名，我們以 top_seg 的 x 為基準近似：
    x, y = foot_xy
    # 盡量用 y->x 插值；失敗則平均 x
    x_top = interp_x_on_segment(top_seg, y) or _fallback_x(top_seg)
    # 初始不知道 bottom，因此用「右側=Inside / 左或中=Outside」的保守近似
    return State.Inside if x >= x_top + DX_EPS else State.Outside


def next_state(prev: State, foot_xy: Point, top_seg: Segment, bottom_seg: Segment) -> State:
    """
    只有在 y >= y_cut（兩線端點中最小 y）時才評估：
      - Outside   : 在 GL_top 左側
      - Corridor  : 在兩線之間（只做「準備進/離港」）
      - Inside    : 在 GL_bottom 右側
    """
    region = _classify_region(foot_xy, top_seg, bottom_seg)

    # 外海上方：不更新，維持前狀態
    if region is None:
        return prev

    if prev == State.Inside:
        if region == "Corridor":
            return State.PreparingExit
        # 仍在右側或回到右側，都維持 Inside
        return State.Inside if region == "Inside" else State.PreparingExit

    if prev == State.PreparingExit:
        if region == "Outside":
            return State.Outside            # 完成離港
        elif region == "Inside":
            return State.Inside             # 取消離港
        else:
            return State.PreparingExit      # 繼續在走廊

    if prev == State.Outside:
        if region == "Corridor":
            return State.PreparingEnter
        # 仍在左側或回到左側，都維持 Outside
        return State.Outside if region == "Outside" else State.PreparingEnter

    if prev == State.PreparingEnter:
        if region == "Inside":
            return State.Inside             # 完成入港
        elif region == "Outside":
            return State.Outside            # 取消入港
        else:
            return State.PreparingEnter     # 繼續在走廊

    return prev

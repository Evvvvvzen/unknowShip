# tools/roi.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional
import json
import cv2
import numpy as np

Point = Tuple[int, int]
Polygon = List[Point]

@dataclass
class ROI:
    """
    一塊配對區域（多邊形）
    id: 唯一識別（不同相機對應的同一物理區域用同 id）
    poly: 多邊形座標（影像像素座標，順/逆時針皆可）
    """
    id: str
    poly: Polygon

# ---------------- 既有：多邊形 ROI ----------------

def _point_in_poly(pt: Point, poly: Polygon) -> bool:
    """射線法：測試點是否在多邊形內（含邊界）"""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = ((y1 > y) != (y2 > y))
        if cond:
            xinters = (x2 - x1) * (y - y1) / float((y2 - y1) + 1e-12) + x1
            if x <= xinters:
                inside = not inside
    return inside

def point_in_any_rois(pt: Point, rois: List[ROI]) -> List[str]:
    """回傳所有包含該點的 ROI id"""
    return [roi.id for roi in rois if _point_in_poly(pt, roi.poly)]

def draw_rois(frame, rois: List[ROI], color=(255, 0, 255), thickness=2):
    """在 frame 上把每個多邊形 ROI 畫出來，並在多邊形第一點附近寫 id"""
    for roi in rois:
        contour = [tuple(map(int, p)) for p in roi.poly]
        if len(contour) >= 3:
            arr = np.array(contour, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [arr], isClosed=True, color=color, thickness=thickness)
            px, py = contour[0]
            cv2.putText(frame, f"ROI:{roi.id}", (int(px), int(py) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

def load_roi_config(path: str) -> Dict[str, Dict[str, Any]]:
    """
    載入 ROI 設定檔（JSON），回傳：
    {
      "<camera_label>": {
        "role": "send" | "recv" | "none",
        "rois": [ROI(...), ...]
      },
      ...
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    parsed: Dict[str, Dict[str, Any]] = {}
    for cam_label, conf in raw.items():
        role = conf.get("pairing_role", "none")
        rois = []
        for r in conf.get("rois", []):
            rid = r.get("id", "roi")
            poly = [(int(x), int(y)) for x, y in r.get("poly", [])]
            if len(poly) >= 3:
                rois.append(ROI(id=rid, poly=poly))
        parsed[cam_label] = {"role": role, "rois": rois}
    return parsed

# ---------------- 新增：邊緣 ROI（畫面寬度的 x% 條帶） ----------------

def edge_x(frame_shape, side: str, ratio: float) -> int:
    """
    邊界線的 x 位置：
      side='left'  → x = round(W * ratio)
      side='right' → x = round(W * (1 - ratio))
    """
    h, w = frame_shape[:2]
    ratio = float(max(0.0, min(1.0, ratio)))
    if side.lower() == "left":
        return int(round(w * ratio))
    else:
        return int(round(w * (1.0 - ratio)))

def in_edge_roi(pt: Point, frame_shape, side: str, ratio: float) -> bool:
    """
    測試點是否在邊緣 ROI（垂直條帶）裡：
      side='left'  → x <= w*ratio
      side='right' → x >= w*(1-ratio)
    """
    h, w = frame_shape[:2]
    x, _ = pt
    bound = edge_x(frame_shape, side, ratio)
    if side.lower() == "left":
        return x <= bound
    else:
        return x >= bound

def draw_roi_edge_line(frame, side: str, ratio: float, color=(0, 255, 0), thickness=2):
    """
    在 ROI 邊界畫『綠色直線』，滿足你的需求。ratio=0.1 表示寬度 10%。
    """
    h, w = frame.shape[:2]
    x = edge_x(frame.shape, side, ratio)
    cv2.line(frame, (x, 0), (x, h-1), color, thickness)
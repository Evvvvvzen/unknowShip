from typing import Tuple


def xyxy_to_xywh(x1: int, y1: int, x2: int, y2: int):
    return [int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))]


def bbox_foot_point(l: int, t: int, r: int, b: int) -> Tuple[int, int]:
    return ((l + r) // 2, b)


def bnorm(cy: int, H: int) -> float:
    return float(cy) / float(H) if H > 0 else 0.0

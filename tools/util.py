import sys
from pathlib import Path

YOLOV7_DIR = Path("/home/aiotlabserver/yolov7")
if YOLOV7_DIR.exists() and str(YOLOV7_DIR) not in sys.path:
    sys.path.insert(0, str(YOLOV7_DIR))

try:
    from utils.torch_utils import select_device
except Exception as e:
    raise ImportError(
        "找不到 yolov7 的 `utils.torch_utils.select_device`。"
        "請確認在入口程式先把 YOLOV7_DIR 加進 sys.path，或調整 tools/util.py 的 YOLOV7_DIR。"
    ) from e
    

from typing import Union
import numpy as np
import cv2
import torch
from utils.torch_utils import select_device  


def safe_select(device_str: str):
    try:
        return select_device(device_str)
    except AssertionError:
        print("[WARN] 指定 GPU 無效，改用 CPU")
        return select_device("cpu")

def src_label(src: Union[str, int, Path]) -> str:
    if isinstance(src, (str, Path)):
        try:
            return Path(src).stem
        except Exception:
            return str(src)
    return f"cam{int(src)}"

def open_source(src: Union[str, int, Path]):
    if isinstance(src, (str, Path)):
        cap = cv2.VideoCapture(str(src))
    else:
        cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(src))
    return cap

def to_distance_from_score(score: float) -> float:
    try:
        s = float(score)
    except Exception:
        return 1.0
    # 若是 cosine 相似度（-1~1），轉成距離（越小越像）
    if -1.0 <= s <= 1.0:
        return 1.0 - s
    return s

def l2norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

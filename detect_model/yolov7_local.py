import sys
from pathlib import Path
from typing import List, Optional
import numpy as np
import torch
import torch.serialization as ts

# 讓本地 yolov7 可 import
YOLOV7_DIR = Path("/home/aiotlabserver/yolov7")
assert YOLOV7_DIR.exists(), f"YOLOv7 專案路徑不存在：{YOLOV7_DIR}"
sys.path.append(str(YOLOV7_DIR))

from models.yolo import Model
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords

from detect_model.base import Detector
from tools.types import Detection
from tools.util import safe_select

try:
    ts.add_safe_globals([Model])
except Exception:
    pass

class YOLOv7LocalDetector(Detector):
    def __init__(self, weights: Path, device_str: str,
                 img_size: int, conf_thres: float, iou_thres: float,
                 class_filter: Optional[set]):
        self.device = safe_select(device_str)
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.class_filter = class_filter

        assert weights.exists(), f"找不到權重：{weights}"
        print("[INFO] loading YOLOv7 (local, offline) ...")
        ckpt = torch.load(str(weights), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.model = ckpt["model"]
        else:
            raise RuntimeError("權重檔不含 'model' 物件；請用官方 yolov7*.pt")
        try:
            self.model.float().fuse()
        except Exception:
            self.model.float()
        self.model.eval().to(self.device)

    @torch.no_grad()
    def detect(self, frame: np.ndarray) -> List[Detection]:
        img = letterbox(frame, self.img_size, stride=32, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(self.device).float() / 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        pred = self.model(im)[0]
        try:
            det = non_max_suppression(pred, conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                      classes=None, agnostic=False, max_det=300)[0]
        except TypeError:
            det = non_max_suppression(pred, self.conf_thres, self.iou_thres, None, False, False)[0]

        out: List[Detection] = []
        if det is not None and len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                c = int(cls.item())
                if (self.class_filter is None) or (c in self.class_filter):
                    x1, y1, x2, y2 = map(int, xyxy)
                    out.append(Detection(bbox=(x1, y1, x2, y2), conf=float(conf.item()), cls_id=c))
        return out

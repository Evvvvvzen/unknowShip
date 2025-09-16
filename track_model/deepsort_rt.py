from typing import List, Optional
import numpy as np
from tools.types import Detection, TrackInfo
from tools.geometry import xyxy_to_xywh


class DeepSortRTTracker:
    """
    包裝 deep-sort-realtime。若 TorchReID 會碰 GPU，請在呼叫端用 gpu_lock 序列化。
    """

    def __init__(self, max_age: int, n_init: int, max_cos_dist: float, nn_budget: int, reid_model_name: str, reid_on_gpu: bool):
        from deep_sort_realtime.deepsort_tracker import DeepSort
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cos_dist,
            nn_budget=nn_budget,
            nms_max_overlap=1.0,
            embedder="torchreid",
            embedder_model_name=reid_model_name,
            embedder_gpu=reid_on_gpu,
            half=False,
            bgr=True
        )

    def update(self, dets: List[Detection], frame: np.ndarray) -> List[TrackInfo]:
        ds_dets = []
        for d in dets:
            x1, y1, x2, y2 = d.bbox
            ds_dets.append([xyxy_to_xywh(x1, y1, x2, y2),
                           float(d.conf), int(d.cls_id)])
        tracks = self._tracker.update_tracks(ds_dets, frame=frame)

        out: List[TrackInfo] = []
        for t in tracks:
            confirmed = bool(t.is_confirmed())
            tsu = int(getattr(t, "time_since_update", 1e9))
            l, tt, r, b = map(int, t.to_ltrb())
            cls_id = int(getattr(t, "det_class", -1))
            feat = None
            if hasattr(t, "features"):
                feats = getattr(t, "features")
                if feats and len(feats) > 0:
                    feat = np.asarray(feats[-1], dtype=np.float32)
            out.append(TrackInfo(
                track_id=int(t.track_id),
                bbox=(l, tt, r, b),
                cls_id=cls_id,
                confirmed=confirmed,
                time_since_update=tsu,
                feature=feat
            ))
        return out

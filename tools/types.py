from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Any
import numpy as np

BBox = Tuple[int, int, int, int]  # l, t, r, b

@dataclass
class Detection:
    bbox: BBox
    conf: float
    cls_id: int

@dataclass
class TrackInfo:
    track_id: int
    bbox: BBox
    cls_id: int
    confirmed: bool
    time_since_update: int
    feature: Optional[np.ndarray] = None  # (D,)

@dataclass
class FramePacket:
    src_idx: int
    src_label: str
    frame: Any
    ts_ms: int
    frame_idx: int

@dataclass
class DetPacket(FramePacket):
    dets: List[Detection]

@dataclass
class DBItem:
    vector: np.ndarray
    payload: dict
    point_id: str
    
@dataclass
class TrackState:
    start_ts: int
    obs_feats: List[np.ndarray] = field(default_factory=list)
    obs_pids: List[str] = field(default_factory=list)
    committed: bool = False
    gid: Optional[int] = None
    # optional: drift/ambiguity hooks（先保留欄位）
    ambiguous: bool = False
    drift_cnt: int = 0
    anchor_sum: Optional[np.ndarray] = None
    anchor_n: int = 0
    last_bbox: Optional[BBox] = None
    # Debug / 連續性訊號
    prev_feat: Optional[np.ndarray] = None
    prev_foot: Optional[Tuple[int,int]] = None
    prev_ts: Optional[int] = None
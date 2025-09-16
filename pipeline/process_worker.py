# pipeline/process_worker.py
import time
import threading
from queue import Queue, Empty
from typing import Any, Dict, Tuple, List, Optional

import cv2  # type: ignore
import numpy as np  # type: ignore

from track_model.deepsort_rt import DeepSortRTTracker
from qdrant.gid import GIDAssigner, make_point_id
from qdrant.store import QdrantStore
from tools.types import DetPacket, DBItem, TrackState
from tools.geometry import bbox_foot_point
from tools.util import src_label
from tools.viz import (
    draw_box, draw_foot, draw_gid, draw_fps, draw_id,
    draw_status_line, status_text_and_color,
)
from tools.gate import State  # 只保留 State（其餘改由 GateRuntime 內部使用）
from tools.gate_runtime import GateRuntime        # <<< Gate 封裝
from tools.obs_buffer import ObsBuffer            # 已抽離的觀察期判斷

# <<< ROI / Pairing >>>
from tools.roi import in_edge_roi, draw_roi_edge_line
from tools.pairing import PairingHub, Candidate

STOP = object()

class ProcessThread(threading.Thread):
    """
    單路處理：
      DeepSORT → 觀察期蒐集/決策（GID:? → GID:K）→ Gate 狀態（可選）→ ROI/配對（可選）→ 繪圖 → display_queue & db_queue
    """

    def __init__(
        self,
        src_idx: int,
        src: Any,
        store: QdrantStore,
        gidm: GIDAssigner,
        process_queue: Queue,
        display_queue: Queue,
        db_queue: Queue,
        gpu_lock: threading.Lock,
        show_size: Tuple[int, int],
        window_name_prefix: str,
        ds_params: Dict[str, Any],
        # ==== 觀察期參數 ====
        obs_n: int,
        obs_timeout_s: float,
        obs_min: int,
        vote_ratio: float,
        gid_unset: int,
        # ==== （可選）綁架/歧義檢測參數（預留） ====
        enable_drift: bool = False,
        anchor_n: int = 5,
        drift_d_th: float = 0.35,
        area_jump_trig: float = 1.6,
        drift_consec: int = 3,
        # ==== Gate（可選）：(top_seg, bottom_seg) ====
        gate_segments: Optional[
            Tuple[Tuple[Tuple[int, int], Tuple[int, int]],
                  Tuple[Tuple[int, int], Tuple[int, int]]]
        ] = None,
        # ==== 無 Gate 時的預設狀態（例如非第 0 支鏡頭一律 Inside） ====
        default_state_no_gate: State = State.Inside,

        # ==== 配對/ROI（邊緣條帶） ====
        pairing_hub: Optional[PairingHub] = None,
        pairing_role: str = "both",          # "send" | "recv" | "both" | "none"
        pairing_ttl_ms: int = 6000,
        pairing_reid_thresh: float = 0.35,
        auto_edge_roi: bool = True,          # 此版只用邊緣 ROI
        edge_roi_ratio: float = 0.10,        # 10%
        edge_side: Optional[str] = None,     # "left" or "right"（由 main 決定）
    ):
        super().__init__(daemon=True)
        self.src_idx = src_idx
        self.src = src
        self.store = store
        self.gidm = gidm
        self.process_queue = process_queue
        self.display_queue = display_queue
        self.db_queue = db_queue
        self.gpu_lock = gpu_lock

        self.label = src_label(src)
        self.wname = f"{window_name_prefix} [{self.label}]"
        self.show_size = show_size

        # DeepSORT
        self.tracker = DeepSortRTTracker(**ds_params)

        # 觀察期設定
        self.obs_n = int(obs_n)
        self.obs_timeout_s = float(obs_timeout_s)
        self.obs_min = int(obs_min)
        self.vote_ratio = float(vote_ratio)
        self.gid_unset = int(gid_unset)

        # 抽離的觀察期 buffer
        self.obs = ObsBuffer(obs_n=self.obs_n, timeout_s=self.obs_timeout_s, obs_min=self.obs_min)

        # 綁架/歧義檢測設定（目前預留，預設關閉）
        self.enable_drift = bool(enable_drift)
        self.anchor_n = int(anchor_n)
        self.drift_d_th = float(drift_d_th)
        self.area_jump_trig = float(area_jump_trig)
        self.drift_consec = int(drift_consec)

        # 每條 track 的觀察期/狀態
        self.state: Dict[int, Dict[str, Any]] = {}

        # Gate：封裝（含繪製與 per-track 狀態）
        self.gate_rt = GateRuntime(gate_segments=gate_segments, default_state=default_state_no_gate)

        # ROI / Pairing
        self.pairing_hub = pairing_hub
        self.pairing_role = pairing_role.lower()
        self.pairing_ttl_ms = int(pairing_ttl_ms)
        self.pairing_reid_thresh = float(pairing_reid_thresh)
        self.auto_edge_roi = bool(auto_edge_roi)
        self.edge_roi_ratio = float(edge_roi_ratio)
        self.edge_side = (edge_side or "right").lower()      # 由 main 明確指定
        self.edge_roi_id = "handoff_corridor"                # 兩端共用相同 id
        self._last_prune_ts = 0
        
        # --- DEBUG 狀態：ROI 進出偵測 / 池摘要列印節流 ---
        self.prev_in_roi: Dict[int, bool] = {}     # track_id -> 上一幀是否在 ROI
        self._last_pool_log_ts = 0                 # pairing_hub 摘要每秒列印一次

        self.frames = 0
        self.t0 = time.time()
        self.proc_fps = 0.0

        cv2.namedWindow(self.wname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.wname, *self.show_size)

    # ========== 幫手：幾何 & 相似度 ==========
    @staticmethod
    def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, np.float32).ravel()
        b = np.asarray(b, np.float32).ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - float((a @ b) / (na * nb))

    @staticmethod
    def _bbox_area(x1, y1, x2, y2):
        return max(1, int(x2 - x1)) * max(1, int(y2 - y1))

    @staticmethod
    def _center(x1, y1, x2, y2):
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    # ========== 觀察期狀態初始化 ==========
    def _ensure_track_state(self, track_id: int, now_ms: int, bbox: Tuple[int, int, int, int]):
        st = self.state.get(track_id)
        if st is None:
            st = TrackState(start_ts=now_ms, last_bbox=bbox)
            self.state[track_id] = st
        return st

    # ========== 觀察期 commit 條件 ==========
    def _should_commit(self, st: Dict[str, Any], now_ms: int) -> bool:
        """
        委派給 tools/obs_buffer.ObsBuffer 來判斷是否達標。
        兼容 TrackState（dataclass）或 dict 的兩種取值方式。
        """
        try:
            feats = st["obs_feats"]          # 若 st 是 dict
            start_ts = st["start_ts"]
        except Exception:
            feats = getattr(st, "obs_feats", [])
            start_ts = getattr(st, "start_ts", 0)
        return self.obs.ready(feats, start_ts, now_ms)

    # ========== 視覺化：在畫面中間最上方，顯示 ROI 內船舶摘要 ==========
    def _draw_roi_banner(self, frame, items: List[str]):
        """
        items: 每個元素像 "tid=3 gid=? cls=0"
        顯示格式：
          ROI[right 10%] 內船舶: N ｜ tid=.. gid=.. cls=.. ; ...
        """
        h, w = frame.shape[:2]
        max_list = 6
        shown = items[:max_list]
        suffix = " …" if len(items) > max_list else ""
        header = f"ROI[{self.edge_side} {int(self.edge_roi_ratio*100)}%] inside ship: {len(items)}"
        detail = " ｜ " + " ; ".join(shown) + suffix if shown else ""
        text = header + detail

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(6, (w - tw) // 2)
        y = max(28, int(0.05 * h))

        # 半透明底
        pad = 10
        x1, y1 = x - pad, y - th - pad // 2
        x2, y2 = x + tw + pad, y + pad // 2
        x1 = max(0, x1); y1 = max(0, y1); x2 = min(w-1, x2); y2 = min(h-1, y2)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        alpha = 0.35
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # 置中描邊字
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness+2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, (50, 220, 50), thickness, cv2.LINE_AA)

    # ========== 主流程：處理每幀的 tracks ==========
    def _handle_tracks(self, item: DetPacket, tracks):
        frame = item.frame

        # 1) Gate 繪製（若有）
        self.gate_rt.draw(frame)

        # 2) ROI 綠線（固定 10% 邊緣，依 edge_side）
        if self.auto_edge_roi:
            draw_roi_edge_line(frame, self.edge_side, self.edge_roi_ratio, color=(0, 255, 0), thickness=2)
        # elif self.auto_edge_roi and self.edge_side == "left":
        #     draw_roi_edge_line(frame, self.edge_side, float(0.5), color=(0, 255, 0), thickness=2)

        # 3) 每秒清一次配對池 + 池摘要列印（只讓 cam0 列印，避免洗版）
        if self.pairing_hub is not None:
            now = int(item.ts_ms)
            if now - self._last_prune_ts >= 1000:
                self.pairing_hub.prune(now)
                self._last_prune_ts = now
                if self.src_idx == 0 and now - self._last_pool_log_ts >= 1000:
                    print("[PAIR][POOL] summary:", self.pairing_hub.size_summary())
                    self._last_pool_log_ts = now

        # 4) 先統計：當前畫面 ROI 內的 track 數（做 singleton 用）與 foot map
        in_roi_count = 0
        tmp_foots: Dict[int, Tuple[int, int]] = {}
        for t in tracks:
            if (not t.confirmed) or (t.time_since_update > 0):
                continue
            l, tt, r, bb = t.bbox
            fx, fy = bbox_foot_point(l, tt, r, bb)
            tmp_foots[t.track_id] = (fx, fy)
            if self.auto_edge_roi and in_edge_roi((fx, fy), frame.shape, self.edge_side, self.edge_roi_ratio):
                in_roi_count += 1

        # 為 ROI 橫幅準備的清單
        roi_items_for_banner: List[str] = []

        # 5) 逐條處理
        for t in tracks:
            if (not t.confirmed) or (t.time_since_update > 0):
                continue

            l, tt, r, bb = t.bbox
            track_id = t.track_id
            cls_id = t.cls_id

            # 畫框 & local ID
            draw_box(frame, l, tt, r, bb)
            draw_id(frame, l, tt, track_id)

            # foot point（用預先算好的）
            fx, fy = tmp_foots.get(track_id, bbox_foot_point(l, tt, r, bb))
            draw_foot(frame, fx, fy)

            # Gate 狀態
            gate_state = self.gate_rt.update(track_id, (fx, fy))
            gate_txt = gate_state.value  # Inside / Outside / ...

            # ReID 特徵
            feat = t.feature
            if feat is None:
                continue

            # 狀態存取與觀察期蒐集
            st = self._ensure_track_state(track_id, now_ms=int(item.ts_ms), bbox=(l, tt, r, bb))
            try:
                committed = st["committed"]
                ambiguous = st.get("ambiguous", False)
                if not committed and not ambiguous:
                    st["obs_feats"].append(np.asarray(feat, np.float32).ravel())
            except Exception:
                if (not getattr(st, "committed", False)) and (not getattr(st, "ambiguous", False)):
                    st.obs_feats.append(np.asarray(feat, np.float32).ravel())

            # === SEND：Inside & 已決 & 在 ROI → 丟入配對池 ===
            in_roi = self.auto_edge_roi and in_edge_roi((fx, fy), frame.shape, self.edge_side, self.edge_roi_ratio)
            is_committed = getattr(st, "committed", st.get("committed", False)) if isinstance(st, dict) else getattr(st, "committed", False)
            if (
                self.pairing_hub is not None and
                self.pairing_role in ("send", "both") and
                gate_txt == "Inside" and
                in_roi and
                is_committed
            ):
                try:
                    feats = st["obs_feats"]
                except Exception:
                    feats = st.obs_feats
                feat_avg = (np.mean(np.stack(feats), axis=0).astype(np.float32)
                            if feats else np.asarray(feat, np.float32).ravel())
                try:
                    gid_val = int(st["gid"])
                except Exception:
                    gid_val = int(getattr(st, "gid", -1))
                self.pairing_hub.push(self.edge_roi_id, Candidate(
                    gid=gid_val,
                    class_id=int(cls_id) if isinstance(cls_id, (int, np.integer)) else -1,
                    src_label=self.label,
                    ts_ms=int(item.ts_ms),
                    foot_xy=(fx, fy),
                    feat_avg=feat_avg,
                    ttl_ms=self.pairing_ttl_ms
                ))
                # DEBUG：推送成功後印出該 key 當前候選數
                cur_cnt = self.pairing_hub.candidate_count(
                    self.edge_roi_id,
                    int(cls_id) if isinstance(cls_id, (int, np.integer)) else -1,
                    now_ms=int(item.ts_ms)
                )
                print(f"[PAIR][PUSH] cam={self.src_idx}({self.label}) tid={track_id} gid={gid_val} "
                      f"cls={int(cls_id) if isinstance(cls_id,(int,np.integer)) else -1} "
                      f"roi={self.edge_roi_id} count={cur_cnt}")

            # --- DEBUG：偵測 ROI 進/出事件並列印 ---
            was_in = self.prev_in_roi.get(track_id, False)
            if in_roi and not was_in:
                print(f"[ROI][ENTER] cam={self.src_idx}({self.label}) tid={track_id} "
                      f"cls={int(cls_id) if isinstance(cls_id,(int,np.integer)) else -1} "
                      f"gate={gate_txt} xy=({fx},{fy}) side={self.edge_side} ratio={self.edge_roi_ratio}")
            elif (not in_roi) and was_in:
                print(f"[ROI][LEAVE] cam={self.src_idx}({self.label}) tid={track_id} "
                      f"cls={int(cls_id) if isinstance(cls_id,(int,np.integer)) else -1} "
                      f"gate={gate_txt} xy=({fx},{fy}) side={self.edge_side}")
            self.prev_in_roi[track_id] = in_roi

            # === RECV：Inside & 未決 & 在 ROI → 嘗試快速配對 ===
            if (
                self.pairing_hub is not None and
                self.pairing_role in ("recv", "both") and
                gate_txt == "Inside" and
                in_roi and
                (not is_committed)
            ):
                cand_cnt = self.pairing_hub.candidate_count(self.edge_roi_id, int(cls_id), now_ms=int(item.ts_ms))
                roi_singleton = (in_roi_count == 1) and (cand_cnt == 1)
                gid = self.pairing_hub.match(
                    roi_id=self.edge_roi_id,
                    class_id=int(cls_id) if isinstance(cls_id, (int, np.integer)) else -1,
                    now_ms=int(item.ts_ms),
                    query_feat=None if roi_singleton else np.asarray(feat, np.float32).ravel(),
                    singleton_hint=roi_singleton,
                    reid_thresh=self.pairing_reid_thresh
                )
                if gid is not None:
                    try:
                        st["gid"] = int(gid); st["committed"] = True
                    except Exception:
                        st.gid = int(gid); st.committed = True
                    print(f"[PAIR][MATCH] cam={self.src_idx}({self.label}) tid={track_id} "
                          f"gid={gid} cls={int(cls_id) if isinstance(cls_id,(int,np.integer)) else -1} "
                          f"roi={self.edge_roi_id} rule={'singleton' if roi_singleton else 'reid/temporal'}")

            # === 觀察期決策（若還沒被 ROI 快配） ===
            is_committed = getattr(st, "committed", st.get("committed", False)) if isinstance(st, dict) else getattr(st, "committed", False)
            if (not is_committed) and self._should_commit(st, now_ms=int(item.ts_ms)):
                try:
                    feats = st["obs_feats"]
                except Exception:
                    feats = st.obs_feats
                gid = self.gidm.assign_by_aggregate(
                    feats,
                    this_src_label=self.label,
                    ts_ms=int(item.ts_ms),
                    local_key=(self.label, track_id),
                    vote_ratio=self.vote_ratio,
                    min_votes=max(10, self.obs_min // 2),
                )
                try:
                    st["gid"] = int(gid); st["committed"] = True
                    m = len(st["obs_feats"])
                except Exception:
                    st.gid = int(gid); st.committed = True
                    m = len(st.obs_feats)
                print(f"[OBS] commit gid={int(gid)} (m={m}) src={self.label} tid={track_id}")

            # 6) 繪製 GID
            try:
                gid_to_draw = st["gid"] if st.get("committed") else "?"
                gid_for_list = (st["gid"] if st.get("committed") else "?")
            except Exception:
                gid_to_draw = st.gid if getattr(st, "committed", False) else "?"
                gid_for_list = (st.gid if getattr(st, "committed", False) else "?")
            draw_gid(frame, l, tt, gid_to_draw)

            # 7) 左下角畫『Gate 狀態 | 觀察期狀態』
            try:
                status_dict = {
                    "committed": st["committed"],
                    "obs_feats": st["obs_feats"],
                    "ambiguous": st.get("ambiguous", False),
                }
            except Exception:
                status_dict = {
                    "committed": getattr(st, "committed", False),
                    "obs_feats": getattr(st, "obs_feats", []),
                    "ambiguous": getattr(st, "ambiguous", False),
                }
            obs_txt, obs_color = status_text_and_color(status_dict, self.obs_n)
            status_line = f"{gate_txt} | {obs_txt}" if gate_txt is not None else obs_txt
            draw_status_line(frame, l, bb, status_line, color=obs_color)

            # 收集 ROI 橫幅清單
            if in_roi:
                cls_val = int(cls_id) if isinstance(cls_id, (int, np.integer)) else -1
                roi_items_for_banner.append(f"tid={track_id} gid={gid_for_list} cls={cls_val}")

            # 8) 寫 DB
            try:
                gid_payload = st["gid"] if st.get("committed") else self.gid_unset
            except Exception:
                gid_payload = st.gid if getattr(st, "committed", False) else self.gid_unset

            point_id, pid_str = make_point_id(self.label, track_id, item.frame_idx, item.ts_ms)
            payload = {
                "source": str(self.src),
                "source_label": self.label,
                "track_id": int(track_id),
                "global_id": int(gid_payload),
                "cls": int(cls_id) if isinstance(cls_id, (int, np.integer)) else -1,
                "timestamp_ms": int(item.ts_ms),
                "frame_idx": int(item.frame_idx),
                "bbox": [int(l), int(tt), int(r), int(bb)],
                "pid": pid_str
            }
            try:
                self.db_queue.put(DBItem(
                    vector=np.asarray(feat, np.float32),
                    payload=payload, point_id=point_id
                ), timeout=0.2)
            except Exception:
                pass

        # 在畫面最上方中間，畫出本幀 ROI 摘要
        self._draw_roi_banner(frame, roi_items_for_banner)

        # FPS & display
        self.frames += 1
        if time.time() - self.t0 >= 1.0:
            self.proc_fps = self.frames / (time.time() - self.t0)
            self.frames = 0
            self.t0 = time.time()
        draw_fps(frame, self.proc_fps)

        try:
            while not self.display_queue.empty():
                self.display_queue.get_nowait()
            self.display_queue.put({
                "type": "FRAME", "wname": self.wname,
                "frame": cv2.resize(frame, self.show_size)
            }, timeout=0.2)
        except Exception:
            pass

    def run(self):
        while True:
            try:
                item = self.process_queue.get(timeout=0.1)
            except Empty:
                continue
            if isinstance(item, dict) and item.get("type") == "STOP":
                self.display_queue.put({"type": "STOP", "wname": self.wname})
                break

            assert isinstance(item, DetPacket)
            with self.gpu_lock:
                tracks = self.tracker.update(item.dets, frame=item.frame)

            self._handle_tracks(item, tracks)

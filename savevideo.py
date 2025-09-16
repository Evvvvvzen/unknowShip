#savevideo.py
import time
import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, Tuple, Any

import cv2
import numpy as np
import torch


from tools.geometry import bbox_foot_point
from tools.types import FramePacket, DetPacket, DBItem, Detection
from qdrant.gid import GIDAssigner, make_point_id
from qdrant.store import QdrantStore
from track_model.deepsort_rt import DeepSortRTTracker
from detect_model.yolov7_local import YOLOv7LocalDetector
from tools.util import src_label, open_source, safe_select

# ==== 你的可調參數（保留原值）====
YOLOV7_DIR = Path("/home/aiotlabserver/yolov7")
YOLOV7_WEIGHTS = YOLOV7_DIR / "ship_weights/best.pt"

SOURCES = [
    "./video/left_fast.mp4",
    "./video/right_fast.mp4",
]

IMG_SIZE = 1280
CLASS_FILTER = None
CONF_THRES = 0.5
IOU_THRES = 0.45
DET_DEVICE = "0" if torch.cuda.is_available() else "cpu"

# DeepSORT（TorchReID）
DS_MAX_AGE = 30
DS_N_INIT = 15
DS_MAX_COS_DIST = 0.25
DS_NN_BUDGET = 100
REID_MODEL_NAME = "osnet_ibn_x1_0"
REID_ON_GPU = torch.cuda.is_available()

# Qdrant
QDRANT_URL = "http://127.0.0.1:6333"
QDRANT_COLLECTION = "reid_embeddings"
QDRANT_TCP_TIMEOUT = 10.0

# GLOBAL ID（跨鏡）
REID_THRESH = 0.23
MAX_DT_MS = 1500
TOPK = 10
GLOBAL_ID_SEQ_START = 1

# 視窗
WINDOW_NAME = "YOLOv7 + DeepSORT (multi-cam)"
SHOW_SIZE = (1280, 960)

# --- 軟同步設定（新增） ---
SYNC_ENABLE = True      # 要關閉同步，改成 False
SYNC_TOL_MS = 50      # 容忍跨來源相對時間差（毫秒）

# --- 錄影設定 ---
SAVE_DISPLAY_STREAM = True
SAVE_ORIGINAL_STREAM = True
OUTPUT_DIR = Path("./output_video")

DISPLAY_SUFFIX = "_display.mp4"   # ← 顯示流（SHOW_SIZE）
ORIG_SUFFIX    = "_orig.mp4"      # ← 原尺寸流
OUTPUT_CODEC   = "mp4v"           # 可試 "avc1"
OUTPUT_FPS     = 30.0



# ==== 匯入我們的模組 ====

# ====== 全域鎖與佇列 ======
gpu_lock = threading.Lock()     # YOLO 推論、DeepSORT ReID 共用
qdrant_lock = threading.Lock()

inference_queue: Queue = Queue(maxsize=32)  # Capture -> GPU
process_queues: Dict[int, Queue] = {}       # GPU -> per-source
display_queues: Dict[int, Queue] = {}       # per-source -> GUI
db_queue: Queue = Queue(maxsize=128)        # per-source -> DB

STOP = object()


def _open_writer_orig(label: str, frame: np.ndarray):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    h, w = frame.shape[:2]
    out_path = OUTPUT_DIR / f"{label}{ORIG_SUFFIX}"   # ← 用 ORIG_SUFFIX
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(OUTPUT_FPS), (int(w), int(h)))
    if not writer.isOpened():
        raise RuntimeError(f"無法開啟輸出檔案：{out_path}")
    return writer, out_path

def _open_writer_for_display(wname: str, frame: np.ndarray) -> tuple[cv2.VideoWriter, Path]: # type: ignore
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_name_from_wname(wname)
    out_path = OUTPUT_DIR / f"{safe_name}{DISPLAY_SUFFIX}"  # ← 用 DISPLAY_SUFFIX
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(OUTPUT_FPS), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"無法開啟輸出檔案：{out_path}")
    return writer, out_path



def _safe_name_from_wname(wname: str) -> str:
    return (
        wname.replace(" ", "_")
             .replace("[", "_")
             .replace("]", "_")
             .replace("|", "_")
    )

def _open_writer_for_display(wname: str, frame: np.ndarray) -> tuple[cv2.VideoWriter, Path]:
    """
    依照「顯示畫面」大小開一個 VideoWriter。
    會用你實際顯示的幀尺寸（通常是 SHOW_SIZE 的結果）。
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_name_from_wname(wname)
    out_path = OUTPUT_DIR / f"{safe_name}{DISPLAY_SUFFIX}"
    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*OUTPUT_CODEC)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(OUTPUT_FPS), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"無法開啟輸出檔案：{out_path}")
    return writer, out_path



# ====== 同步狀態（新增） ======
class SyncState:
    """
    控制「是否允許某來源讀下一幀」的軟同步。
    原理：每路維護「相對時間」adj_ts = ts_ms - first_ts；
          只有在 adj_ts <= (所有來源最小 adj_ts + 容忍) 才允許前進。
    """

    def __init__(self, n_sources: int):
        self.first_ts = [None] * n_sources   # 各來源首幀時間
        self.adj_ts = [None] * n_sources   # 各來源目前相對時間
        self.lock = threading.Lock()

    def update_and_should_advance(self, src_idx: int, cur_ts_ms: int, tol_ms: int) -> bool:
        with self.lock:
            if self.first_ts[src_idx] is None:
                self.first_ts[src_idx] = cur_ts_ms
            cur_adj = cur_ts_ms - self.first_ts[src_idx]
            self.adj_ts[src_idx] = cur_adj

            vals = [v for v in self.adj_ts if v is not None]
            if len(vals) < 2:
                # 只有一個來源有值（另一邊還沒開始），暫時放行
                return True
            min_adj = min(vals)
            # 只有沒超過最慢者 + 容忍，才允許前進
            return cur_adj <= (min_adj + tol_ms)

# ====== 執行緒 ======


class CaptureThread(threading.Thread):
    """
    來源擷取執行緒（含軟同步 gating）
    - 若本路「超前」太多，暫停讀取（不再 cap.read），等待其他路追上
    - 為了節省算力，暫停時不會不斷丟同一幀到推論（避免重複推論同幀）
    """

    def __init__(self, src_idx: int, src: Any, sync: SyncState):
        super().__init__(daemon=True)
        self.src_idx = src_idx
        self.src = src
        self.sync = sync
        self.cap = open_source(src)
        self.label = src_label(src)
        self.active = self.cap.isOpened()
        # 暫存上一幀（用來做同步判斷 / 顯示可不更新）
        self.buf_frame = None
        self.buf_ts = None
        self.buf_fidx = None

    def run(self):
        if not self.active:
            print(f"[ERROR] 無法開啟來源：{self.src}")
            inference_queue.put({"type": "STOP", "src_idx": self.src_idx})
            return

        # 先讀第一幀（一定放行）
        ok, frame = self.cap.read()
        if not ok or frame is None:
            print(f"[WARN] 來源結束（開頭失敗）：{self.src}")
            self.active = False
        else:
            self.buf_ts = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.buf_fidx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.buf_frame = frame
            if SYNC_ENABLE:
                _ = self.sync.update_and_should_advance(
                    self.src_idx, self.buf_ts, SYNC_TOL_MS)
            try:
                inference_queue.put(
                    {"type": "FRAME", "src_idx": self.src_idx, "label": self.label,
                     "frame": self.buf_frame, "ts_ms": self.buf_ts, "frame_idx": self.buf_fidx},
                    timeout=0.5
                )
            except:
                pass

        while self.active:
            # 判斷是否允許前進到「下一幀」
            allow_next = True
            if SYNC_ENABLE and self.buf_ts is not None:
                allow_next = self.sync.update_and_should_advance(
                    self.src_idx, self.buf_ts, SYNC_TOL_MS)

            if not allow_next:
                # 超前：暫停讀取，什麼都不送（避免重複推論同幀）
                time.sleep(0.002)
                continue

            # 允許前進 → 讀下一幀
            ok, frame = self.cap.read()
            if not ok or frame is None:
                print(f"[WARN] 來源結束：{self.src}")
                self.active = False
                break

            ts_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            fidx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            # 更新暫存與同步狀態
            self.buf_frame, self.buf_ts, self.buf_fidx = frame, ts_ms, fidx
            if SYNC_ENABLE:
                _ = self.sync.update_and_should_advance(
                    self.src_idx, self.buf_ts, SYNC_TOL_MS)

            # 丟這幀去推論
            try:
                inference_queue.put(
                    {"type": "FRAME", "src_idx": self.src_idx, "label": self.label,
                     "frame": frame, "ts_ms": ts_ms, "frame_idx": fidx},
                    timeout=0.5
                )
            except:
                pass

        inference_queue.put({"type": "STOP", "src_idx": self.src_idx})
        self.cap.release()


class GPUWorker(threading.Thread):
    def __init__(self, detector: YOLOv7LocalDetector):
        super().__init__(daemon=True)
        self.detector = detector
        self.stop_marks = set()

    def run(self):
        total_sources = set(range(len(SOURCES)))
        while True:
            try:
                item = inference_queue.get(timeout=0.5)
            except Empty:
                if self.stop_marks == total_sources:
                    break
                continue
            if item["type"] == "STOP":
                self.stop_marks.add(item["src_idx"])
                process_queues[item["src_idx"]].put({"type": "STOP"})
                continue
            if item["type"] != "FRAME":
                continue
            frame = item["frame"]
            # YOLO 推論（持 gpu_lock）
            with gpu_lock:
                dets = self.detector.detect(frame)
            # 丟給該來源處理線
            process_queues[item["src_idx"]].put(DetPacket(
                src_idx=item["src_idx"], src_label=item["label"], frame=frame,
                ts_ms=item["ts_ms"], frame_idx=item["frame_idx"], dets=dets
            ))


class ProcessThread(threading.Thread):
    def __init__(self, src_idx: int, src: Any, store: QdrantStore, gidm: GIDAssigner):
        super().__init__(daemon=True)
        self.src_idx = src_idx
        self.src = src
        self.store = store
        self.gidm = gidm
        self.label = src_label(src)
        self.wname = f"{WINDOW_NAME} [{self.label}]"
        self.tracker = DeepSortRTTracker(
            max_age=DS_MAX_AGE, n_init=DS_N_INIT, max_cos_dist=DS_MAX_COS_DIST,
            nn_budget=DS_NN_BUDGET, reid_model_name=REID_MODEL_NAME, reid_on_gpu=REID_ON_GPU
        )
        self.frames = 0
        self.t0 = time.time()
        self.proc_fps = 0.0

        # 原尺寸錄影 writer（延遲建立）
        self.writer = None
        self.out_path = None

        cv2.namedWindow(self.wname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.wname, *SHOW_SIZE)

    def _ensure_writer(self, frame: np.ndarray):
        if not SAVE_ORIGINAL_STREAM:
            return
        if self.writer is not None:
            return
        try:
            self.writer, self.out_path = _open_writer_orig(self.label, frame)
            print(f"[REC] {self.label} → {self.out_path}")
        except Exception as e:
            print(f"[REC] 開啟 writer 失敗：{e}")
            self.writer, self.out_path = None, None

    def _write_orig(self, frame: np.ndarray):
        if SAVE_ORIGINAL_STREAM and self.writer is not None:
            try:
                self.writer.write(frame)
            except Exception as e:
                print(f"[REC] 寫入失敗（{self.label}）：{e}")

    def _close_writer(self):
        if self.writer is not None:
            try:
                self.writer.release()
            except Exception:
                pass
            print(f"[SAVE] 關閉輸出：{self.out_path}")
            self.writer, self.out_path = None, None

    def run(self):
        while True:
            try:
                item = process_queues[self.src_idx].get(timeout=1.0)
            except Empty:
                continue
            if isinstance(item, dict) and item.get("type") == "STOP":
                # 結束前關閉 writer
                self._close_writer()
                display_queues[self.src_idx].put({"type": "STOP", "wname": self.wname})
                break

            assert isinstance(item, DetPacket)
            frame = item.frame                # ← 這是「原尺寸」影像
            H, W = frame.shape[:2]

            # 第一次看到 frame 才建立 writer（原尺寸）
            self._ensure_writer(frame)

            # DeepSORT 更新（可能會用到 GPU → 用 gpu_lock 序列化）
            with gpu_lock:
                tracks = self.tracker.update(item.dets, frame=frame)

            # 先把所有標註畫在『原尺寸 frame』上
            for t in tracks:
                if (not t.confirmed) or (t.time_since_update > 0):
                    continue
                l, tt, r, bb = t.bbox
                track_id = t.track_id
                cls_id = t.cls_id

                # 畫框 & local ID（畫在原尺寸 frame）
                cv2.rectangle(frame, (l, tt), (r, bb), (0, 200, 0), 2)
                cv2.putText(frame, f"ID:{track_id}", (l, max(0, tt - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)

                # 船底中心點（原尺寸）
                fx, fy = bbox_foot_point(l, tt, r, bb)
                cv2.circle(frame, (fx, fy), 5, (0, 0, 255), -1)

                # 特徵
                feat = t.feature
                if feat is None:
                    continue

                # GID 決策
                gid = self.gidm.assign_or_issue(
                    feat, this_src_label=self.label, ts_ms=item.ts_ms,
                    local_key=(self.label, track_id)
                )
                cv2.putText(frame, f"GID:{gid}", (l, max(0, tt - 28)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Qdrant upsert（交給 DB）
                point_id, pid_str = make_point_id(self.label, track_id, item.frame_idx, item.ts_ms)
                payload = {
                    "source": str(self.src),
                    "source_label": self.label,
                    "track_id": int(track_id),
                    "global_id": int(gid),
                    "cls": int(cls_id) if isinstance(cls_id, (int, np.integer)) else -1,
                    "timestamp_ms": int(item.ts_ms),
                    "frame_idx": int(item.frame_idx),
                    "bbox": [int(l), int(tt), int(r), int(bb)],
                    "pid": pid_str
                }
                try:
                    db_queue.put(DBItem(vector=np.asarray(feat, np.float32),
                                        payload=payload, point_id=point_id), timeout=0.2)
                except:
                    pass

            # 👉 在送去顯示前，先把「原尺寸 frame」寫檔
            self._write_orig(frame)

            # FPS 顯示（畫在原尺寸 frame 上）
            self.frames += 1
            if time.time() - self.t0 >= 1.0:
                self.proc_fps = self.frames / (time.time() - self.t0)
                self.frames = 0
                self.t0 = time.time()
            cv2.putText(frame, f"Proc:{self.proc_fps:.1f}", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # 再把原尺寸 frame 縮放成 SHOW_SIZE，送到 GUI 顯示
            try:
                while not display_queues[self.src_idx].empty():
                    display_queues[self.src_idx].get_nowait()
                display_queues[self.src_idx].put({
                    "type": "FRAME",
                    "wname": self.wname,
                    "frame": cv2.resize(frame, SHOW_SIZE)
                }, timeout=0.2)
            except:
                pass



class DBWorker(threading.Thread):
    def __init__(self, store: QdrantStore):
        super().__init__(daemon=True)
        self.store = store

    def run(self):
        while True:
            item = db_queue.get()
            if item is STOP:
                break
            assert hasattr(item, "vector")
            try:
                self.store.upsert(item.vector, item.payload,
                                  point_id=item.point_id)
            except Exception as e:
                print(f"[DBWorker] upsert 失敗：{e}")

# ====== 入口 ======


def main():
    # 建立偵測器
    detector = YOLOv7LocalDetector(
        weights=YOLOV7_WEIGHTS, device_str=DET_DEVICE,
        img_size=IMG_SIZE, conf_thres=CONF_THRES, iou_thres=IOU_THRES,
        class_filter=CLASS_FILTER
    )
    # 建立向量庫與 GID 決策器
    store = QdrantStore(url=QDRANT_URL, collection=QDRANT_COLLECTION,
                        timeout=QDRANT_TCP_TIMEOUT, lock=qdrant_lock)
    gidm = GIDAssigner(store, reid_thresh=REID_THRESH, max_dt_ms=MAX_DT_MS,
                       topk=TOPK, seq_start=GLOBAL_ID_SEQ_START)

    # 佇列
    for i, _ in enumerate(SOURCES):
        process_queues[i] = Queue(maxsize=4)
        display_queues[i] = Queue(maxsize=2)

    # 同步狀態（新增）
    sync = SyncState(len(SOURCES))

    gpuw = GPUWorker(detector)
    gpuw.start()
    dbw = DBWorker(store)
    dbw.start()

    # 每來源擷取與處理
    caps, procs = [], []
    for i, src in enumerate(SOURCES):
        ct = CaptureThread(i, src, sync)
        ct.start()
        caps.append(ct)
        pt = ProcessThread(i, src, store, gidm)
        pt.start()
        procs.append(pt)

    # GUI 主迴圈
    active = [True] * len(SOURCES)

    # 每個視窗名（wname）對應一個 writer
    writers: Dict[str, cv2.VideoWriter] = {}
    out_paths: Dict[str, Path] = {}

    while any(active):
        for i in range(len(SOURCES)):
            try:
                item = display_queues[i].get(timeout=0.01)
            except Empty:
                continue

            if item.get("type") == "STOP":
                active[i] = False
                wname = item.get("wname", f"{WINDOW_NAME}[{i}]")

                # 這一路結束 → 收掉 writer
                if SAVE_DISPLAY_STREAM and wname in writers:
                    try:
                        writers[wname].release()
                    except Exception:
                        pass
                    p = out_paths.get(wname)
                    if p is not None:
                        print(f"[SAVE] 關閉輸出：{p}")
                    writers.pop(wname, None)
                    out_paths.pop(wname, None)

                try:
                    cv2.destroyWindow(wname)
                except Exception:
                    pass
                continue

            if item.get("type") != "FRAME":
                continue

            wname = item["wname"]
            disp_frame = item["frame"]   # 這就是送去 imshow 的「顯示幀」（已 resize/已繪製）

            # 先存檔（若啟用）
            if SAVE_DISPLAY_STREAM:
                if wname not in writers:
                    try:
                        writer, out_path = _open_writer_for_display(
                            wname, disp_frame)
                        writers[wname] = writer
                        out_paths[wname] = out_path
                        print(f"[REC] {wname} → {out_path}")
                    except Exception as e:
                        print(f"[REC] 開啟 writer 失敗：{e}")
                w = writers.get(wname)
                if w is not None:
                    try:
                        w.write(disp_frame)
                    except Exception as e:
                        print(f"[REC] 寫入失敗（{wname}）：{e}")

            # 再顯示
            cv2.imshow(wname, disp_frame)

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    # 收尾：確保所有 writer 關閉
    if SAVE_DISPLAY_STREAM:
        for wname, w in list(writers.items()):
            try:
                w.release()
            except Exception:
                pass
            p = out_paths.get(wname)
            if p is not None:
                print(f"[SAVE] 關閉輸出：{p}")
        writers.clear()
        out_paths.clear()

    db_queue.put(STOP)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

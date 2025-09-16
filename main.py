# main.py
from pipeline import SyncState, CaptureThread, GPUWorker, ProcessThread, DBWorker
from qdrant.gid import GIDAssigner
from qdrant.store import QdrantStore
from detect_model.yolov7_local import YOLOv7LocalDetector

import threading
from queue import Queue, Empty
from pathlib import Path
from typing import Dict, Optional, Tuple
from tools.gate import load_gate_segments, State
from tools.pairing import PairingHub   

import cv2
import torch

# ── YOLOv7 權重與專案位置 ─────────────────────────────────────────────────────
YOLOV7_DIR = Path("/home/aiotlabserver/yolov7")
YOLOV7_WEIGHTS = YOLOV7_DIR / "ship_weights/best.pt"

# ── 輸入來源（可多路） ───────────────────────────────────────────────────────
SOURCES = [
    "./video/left_fast.mp4",   # index=0（左邊）：這路啟用 Gate
    "./video/right_fast.mp4",
]

# ── YOLO 推論參數 ────────────────────────────────────────────────────────────
IMG_SIZE = 1280             # YOLOv7 輸入影像大小（會 resize 成正方形）
CLASS_FILTER = None          # 若指定 class index，僅保留該類別；None = 不過濾
CONF_THRES = 0.45             # 物件偵測的信心分數閾值 (confidence threshold)
IOU_THRES = 0.5              # NMS 的 IoU 閾值（重疊過大則刪除）
DET_DEVICE = "0" if torch.cuda.is_available() else "cpu" # 偵測裝置：GPU id=0，如果沒有 GPU 就用 CPU

# ── DeepSORT（TorchReID） ───────────────────────────────────────────────────
DS_MAX_AGE = 150              # track 最多幾幀沒匹配就刪除（避免 ghost）
DS_N_INIT = 10               # track 至少要連續匹配 N 幀才確定成為有效目標
DS_MAX_COS_DIST = 0.3       # ReID embedding 最大 cosine distance（越小越嚴格）
DS_NN_BUDGET = 100           # ReID 特徵向量緩衝區大小（記憶多少歷史特徵）
REID_MODEL_NAME = "osnet_ibn_x1_0"   # TorchReID 模型名稱（特徵抽取 backbone）
REID_ON_GPU = torch.cuda.is_available() # ReID 是否跑在 GPU（可提升效能）

# ── Qdrant ──────────────────────────────────────────────────────────────────
QDRANT_URL = "http://127.0.0.1:6333" # Qdrant 伺服器 URL
QDRANT_COLLECTION = "test"        # collection 名稱（類似資料表）
QDRANT_TCP_TIMEOUT = 10.0            # TCP timeout（秒），避免連線卡死

# ── GID（跨鏡頭 ID）策略參數 ────────────────────────────────────────────────
REID_THRESH = 0.2           # 全域比對閾值：embedding similarity 要低於才算同人/同船
MAX_DT_MS = 2000             # 允許的最大時間差（毫秒）→ 跨鏡頭比對時間窗
TOPK = 20                    # 從 Qdrant 中取前 K 個候選做比對
GLOBAL_ID_SEQ_START = 1      # global_id 從哪個數字開始（避免跟 -1 混淆）

# ── GUI 視窗設定 ────────────────────────────────────────────────────────────
WINDOW_NAME = "YOLOv7 + DeepSORT (multi-cam)"   # OpenCV 視窗名稱
SHOW_SIZE = (1280, 960)       # 視窗顯示大小（寬, 高），不影響模型輸入

# ── 軟同步（多來源時間對齊） ───────────────────────────────────────────────
SYNC_ENABLE = True            # 是否啟用多攝影機的時間同步
SYNC_TOL_MS = 50              # 容許的時間誤差（毫秒），超過就不算同步

# ── Gate 設定：只給 index=0 那路使用 ───────────────────────────────────────
GATE_CONFIG = "./tools/gate_config.json"   
# Gate JSON 檔案路徑（座標以原始畫面像素為基準）

# ===== 觀察期參數（GID: ? → 聚合後再決策）=====
OBS_N = 25                    # 最多收集 N 幀就要決策（避免無限等待）
OBS_TIMEOUT_S = 3            # 最長觀察秒數，超過就強制決策
OBS_MIN = 5                   # 強制決策時至少需要的最少幀數
VOTE_RATIO = 0.4              # 最高票的票數需 ≥ 次高票 * 0.3 才成立
GID_UNSET = -1                # 未決時寫入 DB 的 global_id 值（暫時占位）

# === 共享佇列與鎖 ===
gpu_lock = threading.Lock()
qdrant_lock = threading.Lock()
inference_queue: Queue = Queue(maxsize=32)   # Capture → GPU
process_queues: Dict[int, Queue] = {}        # GPU → per-source
display_queues: Dict[int, Queue] = {}        # per-source → GUI
db_queue: Queue = Queue(maxsize=128)         # per-source → DB
STOP = object()

pairing_hub = PairingHub(max_per_key=128)

def main():
    # 偵測器
    detector = YOLOv7LocalDetector(weights=YOLOV7_WEIGHTS,
                                    device_str=DET_DEVICE,
                                    img_size=IMG_SIZE,
                                    conf_thres=CONF_THRES,
                                    iou_thres=IOU_THRES,
                                    class_filter=CLASS_FILTER
                                )

    # 向量庫與 GID
    store = QdrantStore(url=QDRANT_URL,
                        collection=QDRANT_COLLECTION,
                        timeout=QDRANT_TCP_TIMEOUT,
                        lock=qdrant_lock
                    )

    gidm = GIDAssigner(store, reid_thresh=REID_THRESH,
                       max_dt_ms=MAX_DT_MS,
                       topk=TOPK,
                       seq_start=GLOBAL_ID_SEQ_START,
                      )
    # 佇列
    for i, _ in enumerate(SOURCES):
        process_queues[i] = Queue(maxsize=16)
        display_queues[i] = Queue(maxsize=8)

    # 同步狀態
    sync = SyncState(len(SOURCES))

    # 啟動 GPU worker
    gpuw = GPUWorker(detector,
                     inference_queue,
                     process_queues,
                     gpu_lock,
                     num_sources=len(SOURCES))
    gpuw.start()

    # 啟動 DB worker（這個類別提供 run()；用 thread 包起來執行）
    dbw = DBWorker(store, db_queue)
    threading.Thread(target=dbw.run, daemon=True).start()

    # 載入 Gate（僅 camera 0）
    gate_segments_cam0 = None
    try:
        gate_segments_cam0 = load_gate_segments(GATE_CONFIG)
        print(f"[Gate] loaded from {GATE_CONFIG} for source index 0")
    except Exception as e:
        print(f"[Gate] skip ({e})")
        gate_segments_cam0 = None

    # === 啟動擷取執行緒（很重要！把 frame 丟進 inference_queue） ===
    captures = []
    for i, src in enumerate(SOURCES):
        ct = CaptureThread(i, 
                           src, 
                           inference_queue, 
                           sync,
                           SYNC_ENABLE, 
                           SYNC_TOL_MS
        )
        ct.start()
        captures.append(ct)
    
    # === 啟動每路處理執行緒 ===
    procs = []
    for i, src in enumerate(SOURCES):
        
        # cam[0]（左鏡頭）→ 右邊 10%；cam[1]（右鏡頭）→ 左邊 10%
        edge_side = "right" if i == 0 else "left"
        
        pt = ProcessThread(
            src_idx = i, 
            src = src, 
            store = store, 
            gidm = gidm,
            process_queue = process_queues[i], 
            display_queue = display_queues[i], 
            db_queue = db_queue,
            gpu_lock = gpu_lock, 
            show_size = SHOW_SIZE, 
            window_name_prefix = WINDOW_NAME,
            ds_params = dict(
                            max_age = DS_MAX_AGE, 
                            n_init = DS_N_INIT, 
                            max_cos_dist = DS_MAX_COS_DIST,
                            nn_budget = DS_NN_BUDGET, 
                            reid_model_name = REID_MODEL_NAME, 
                            reid_on_gpu = REID_ON_GPU
                        ),
            obs_n = OBS_N,
            obs_timeout_s = OBS_TIMEOUT_S,
            obs_min = OBS_MIN,
            vote_ratio = VOTE_RATIO,
            gid_unset = GID_UNSET,
            gate_segments = gate_segments_cam0 if i == 0 else None, # Gate（只有 camera 0 用 gate；其他鏡頭一律視為港內）
            default_state_no_gate = State.Inside,
            
            # <<< 新增：配對/ROI 參數 >>>
            pairing_hub=pairing_hub,
            pairing_role="both",
            pairing_ttl_ms=6000,
            pairing_reid_thresh=0.35,
            auto_edge_roi=True,
            edge_roi_ratio=0.10,      # 10%
            edge_side=edge_side,       # index 決定：0→right、1→left
        )
        pt.start()
        procs.append(pt)

    # GUI 迴圈
    active = [True] * len(SOURCES)
    while any(active):
        for i in range(len(SOURCES)):
            try:
                item = display_queues[i].get(timeout = 0.1)
            except Empty:
                continue
            if item.get("type") == "STOP":
                active[i] = False
                try:
                    cv2.destroyWindow(item.get("wname", f"{WINDOW_NAME}[{i}]"))
                except:
                    pass
                continue
            if item.get("type") != "FRAME":
                continue
            cv2.imshow(item["wname"], item["frame"])
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    db_queue.put(STOP)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

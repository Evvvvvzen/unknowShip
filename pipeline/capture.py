# pipeline/capture.py
import time
import threading
from queue import Queue
from typing import Any, Callable

import cv2

from .sync import SyncState
from tools.util import open_source, src_label

STOP = object()

class CaptureThread(threading.Thread):
    """
    來源擷取執行緒（可選軟同步）。
    負責：cap.read() → inference_queue.put({"type":"FRAME", ...})
    """
    def __init__(
        self,
        src_idx: int,
        src: Any,
        inference_queue: Queue,
        sync: SyncState,
        sync_enable: bool,
        sync_tol_ms: int,
    ):
        super().__init__(daemon=True)
        self.src_idx = src_idx
        self.src = src
        self.sync = sync
        self.sync_enable = bool(sync_enable)
        self.sync_tol_ms = int(sync_tol_ms)

        self.inference_queue = inference_queue
        self.cap = open_source(src)
        self.label = src_label(src)
        self.active = self.cap.isOpened()

        self.buf_frame = None
        self.buf_ts = None
        self.buf_fidx = None

    def run(self):
        if not self.active:
            print(f"[Capture] 無法開啟來源：{self.src}")
            self.inference_queue.put({"type": "STOP", "src_idx": self.src_idx})
            return

        ok, frame = self.cap.read()
        if not ok or frame is None:
            print(f"[Capture] 來源結束（開頭失敗）：{self.src}")
            self.active = False
        else:
            self.buf_ts = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            self.buf_fidx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.buf_frame = frame
            if self.sync_enable:
                _ = self.sync.update_and_should_advance(self.src_idx, self.buf_ts, self.sync_tol_ms)
            try:
                self.inference_queue.put(
                    {"type": "FRAME", "src_idx": self.src_idx, "label": self.label,
                     "frame": self.buf_frame, "ts_ms": self.buf_ts, "frame_idx": self.buf_fidx},
                    timeout=0.5
                )
            except:
                pass

        while self.active:
            allow_next = True
            if self.sync_enable and self.buf_ts is not None:
                allow_next = self.sync.update_and_should_advance(self.src_idx, self.buf_ts, self.sync_tol_ms)

            if not allow_next:
                time.sleep(0.002)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                print(f"[Capture] 來源結束：{self.src}")
                self.active = False
                break

            ts_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            fidx  = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.buf_frame, self.buf_ts, self.buf_fidx = frame, ts_ms, fidx
            if self.sync_enable:
                _ = self.sync.update_and_should_advance(self.src_idx, self.buf_ts, self.sync_tol_ms)

            try:
                self.inference_queue.put(
                    {"type": "FRAME", "src_idx": self.src_idx, "label": self.label,
                     "frame": frame, "ts_ms": ts_ms, "frame_idx": fidx},
                    timeout=0.5
                )
            except:
                pass

        self.inference_queue.put({"type": "STOP", "src_idx": self.src_idx})
        self.cap.release()

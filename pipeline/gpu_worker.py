# pipeline/gpu_worker.py
import threading
from queue import Queue, Empty
from typing import Dict

from detect_model.yolov7_local import YOLOv7LocalDetector
from tools.types import DetPacket

STOP = object()


class GPUWorker(threading.Thread):
    """
    把 Capture 丟進來的 frame 做 YOLO 偵測，送到 per-source process_queue。
    """

    def __init__(
        self,
        detector: YOLOv7LocalDetector,
        inference_queue: Queue,
        process_queues: Dict[int, Queue],
        gpu_lock: threading.Lock,
        num_sources: int,
    ):
        super().__init__(daemon=True)
        self.detector = detector
        self.inference_queue = inference_queue
        self.process_queues = process_queues
        self.gpu_lock = gpu_lock
        self.stop_marks = set()
        self.total_sources = set(range(num_sources))

    def run(self):
        while True:
            try:
                item = self.inference_queue.get(timeout=0.5)
            except Empty:
                if self.stop_marks == self.total_sources:
                    break
                continue
            if item["type"] == "STOP":
                self.stop_marks.add(item["src_idx"])
                self.process_queues[item["src_idx"]].put({"type": "STOP"})
                continue
            if item["type"] != "FRAME":
                continue

            frame = item["frame"]
            with self.gpu_lock:
                dets = self.detector.detect(frame)

            self.process_queues[item["src_idx"]].put(DetPacket(
                src_idx=item["src_idx"], src_label=item["label"], frame=frame,
                ts_ms=item["ts_ms"], frame_idx=item["frame_idx"], dets=dets
            ))

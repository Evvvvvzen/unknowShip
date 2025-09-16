# pipeline/db_worker.py
from __future__ import annotations

from qdrant.store import QdrantStore
from tools.types import DBItem

class DBWorker:
    """
    支援兩類工作：
    - DBItem（向量 upsert）
    - {"op":"set_payload", "points":[...], "payload":{...}}（批次回填）
    """
    def __init__(self, store: QdrantStore, db_queue):
        self.store = store
        self.db_queue = db_queue

    def run(self):
        while True:
            item = self.db_queue.get()
            if item is None or (isinstance(item, dict) and item.get("type") == "STOP"):
                break

            # set_payload 工作
            if isinstance(item, dict) and item.get("op") == "set_payload":
                try:
                    self.store.set_payload_by_ids(
                        points=item["points"], payload=item["payload"]
                    )
                except Exception as e:
                    print(f"[DBWorker] set_payload 失敗：{e}")
                continue

            # 一般 upsert
            try:
                assert hasattr(item, "vector")
                # ---- 未決樣本消毒（非常重要）----
                payload = dict(item.payload) if hasattr(item, "payload") else {}
                gid = payload.get("global_id", None)

                if gid is None or int(gid) <= 0:
                    # 未決：不要把 global_id 寫入，並標 committed=False
                    payload.pop("global_id", None)
                    payload["committed"] = False
                else:
                    # 已決：寫入 global_id，並標 committed=True
                    payload["global_id"] = int(gid)
                    payload["committed"] = True

                # （可選）保底：若缺少必要欄位可以補一下型別/格式
                # if "timestamp_ms" in payload: payload["timestamp_ms"] = int(payload["timestamp_ms"])

                self.store.upsert(item.vector, payload, point_id=item.point_id)
            except Exception as e:
                print(f"[DBWorker] upsert 失敗：{e}")


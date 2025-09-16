# qdrant/store.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Union
import numpy as np
from threading import Lock

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range

PointId = Union[int, str]

@dataclass
class QdrantStore:
    url: str
    collection: str
    timeout: float = 10.0
    lock: Optional[Lock] = None
    
    def __init__(self, url: str, collection: str, timeout: float = 10.0, lock=None, distance: Distance = Distance.COSINE):
        self.client = QdrantClient(url=url, timeout=timeout)
        self.collection = collection
        self.lock = lock
        self.distance = distance
        self._vector_size: Optional[int] = None

    def __post_init__(self):
        self.client = QdrantClient(url=self.url, timeout=self.timeout)
        self._vector_size: Optional[int] = None

    def _ensure(self, vec: np.ndarray):
        v = np.asarray(vec, np.float32).ravel()
        if self._vector_size is None:
            size = len(v)
            if self.client.collection_exists(self.collection):
                info = self.client.get_collection(self.collection)
                current = getattr(getattr(info, "vectors", None), "size", None)
                if current is not None and current != size:
                    raise RuntimeError(f"[Qdrant] collection '{self.collection}' dim={current} != {size}")
            else:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=len(v), distance=Distance.COSINE)
                )
            self._vector_size = len(v)
        return v

    def upsert(self, vector: np.ndarray, payload: Dict[str, Any], point_id: Optional[PointId] = None):
        v = self._ensure(vector)
        ps = PointStruct(id=point_id, vector=v.tolist(), payload=payload)
        if self.lock:
            with self.lock:
                self.client.upsert(self.collection, [ps])
        else:
            self.client.upsert(self.collection, [ps])

    def set_payload_by_ids(self, points: List[PointId], payload: Dict[str, Any]):
        """批量更新既有點的 payload。"""
        if not points:
            return
        if self.lock:
            with self.lock:
                self.client.set_payload(collection_name=self.collection, payload=payload, points=points)
        else:
            self.client.set_payload(collection_name=self.collection, payload=payload, points=points)

    # 簡單包一個「查其他來源、時間窗」的 search，給 GID 決策用
    def search_other_sources(self, query_vec: np.ndarray, this_src_label: str, ts_ms: Optional[int], topk: int, max_dt_ms: Optional[int]):

        v = np.asarray(query_vec, np.float32).ravel().tolist()
        must = []
        must_not = [FieldCondition(key="source_label", match=MatchValue(value=this_src_label))]
        if ts_ms is not None and max_dt_ms and max_dt_ms > 0:
            must.append(FieldCondition(
                key="timestamp_ms",
                range=Range(gte=int(ts_ms - max_dt_ms), lte=int(ts_ms + max_dt_ms))
            ))
        qf = Filter(must=must, must_not=must_not)
        if self.lock:
            with self.lock:
                return self.client.search(collection_name=self.collection, query_vector=v, limit=topk, query_filter=qf)
        else:
            return self.client.search(collection_name=self.collection, query_vector=v, limit=topk, query_filter=qf)

    def search(self, vector: np.ndarray, topk: int, query_filter: Optional[Filter] = None):
        """
        封裝 Qdrant search，允許把外部建立好的 Filter 丟進來（支援時間窗/同來源排除）。
        """
        vec = np.asarray(vector, np.float32).ravel().tolist()
        with (self.lock or _DummyLock()):
            return self.client.search(
                collection_name = self.collection,
                query_vector = vec,
                limit = int(topk),
                query_filter = query_filter
            )
            
class _DummyLock:
    def __enter__(self): return None
    def __exit__(self, *args): return False
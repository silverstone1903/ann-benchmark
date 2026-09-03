"""FAISS adapter — runs in-process, unlike the other engines which are client-server systems
fronted by their own docker-compose stack. FAISS is a library, not a server: wrapping it in an
HTTP microservice would add artificial network/serialization overhead to its latency numbers
that the other engines don't have to pay, and wouldn't reflect how FAISS is actually used in
practice (embedded, not client-server). No CPU/mem docker limits apply to it either — it runs
with whatever the host process has, same as the benchmark harness itself.
"""

import os
import tempfile
import time

import faiss
import numpy as np

from benchmark.common.base_adapter import BaseAdapter, QueryResult


class Adapter(BaseAdapter):
    name = "faiss"

    def __init__(self) -> None:
        self.index: faiss.IndexHNSWFlat | None = None
        self.ids: list[str] = []

    def connect(self) -> None:
        pass  # nothing to connect to

    def build_index(self, ids: list[str], vectors: np.ndarray, index_params: dict) -> float:
        dim = vectors.shape[1]
        m = int(index_params.get("m", 16))
        ef_construction = int(index_params.get("ef_construction", 200))

        start = time.perf_counter()
        # Vectors are pre-normalized, so inner product == cosine similarity.
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = ef_construction
        index.add(np.ascontiguousarray(vectors, dtype=np.float32))

        self.index = index
        self.ids = ids
        return time.perf_counter() - start

    def set_search_params(self, search_params: dict) -> None:
        self.index.hnsw.efSearch = int(search_params.get("ef_search", 40))

    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        start = time.perf_counter()
        _, idx = self.index.search(vec, k)
        latency_ms = (time.perf_counter() - start) * 1000
        ids_out = [self.ids[i] for i in idx[0] if i != -1]
        return QueryResult(ids=ids_out, latency_ms=latency_ms)

    def index_size_bytes(self) -> int:
        fd, path = tempfile.mkstemp(suffix=".faiss")
        os.close(fd)
        try:
            faiss.write_index(self.index, path)
            return os.path.getsize(path)
        finally:
            os.remove(path)

    def close(self) -> None:
        self.index = None

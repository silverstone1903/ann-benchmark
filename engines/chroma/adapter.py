"""Chroma adapter — HNSW (hnswlib) collection accessed over its HTTP server.

Verified against the exact installed client version (chromadb==1.5.9, checked against its
GitHub tag). `create_collection`/`Collection.modify()` both take a plain
`configuration={"hnsw": {...}}` dict (no typed config objects needed, unlike 0.6.3 which this
adapter was briefly written against before the user upgraded — see project memory).

Empirically confirmed (via a diagnostic that re-fetched the collection after each modify()) that
`modify(configuration={"hnsw": {"ef_search": ...}})` *does* persist the new value server-side —
but it has no effect on actual query results: recall stayed bit-identical across the whole
ef_search sweep (10->160) despite the stored config correctly showing each value in turn. The
public client also has no way to pass ef_search per-query as an override. So the live-updating
in-place tuning that pgvector/redis/elasticsearch/qdrant/weaviate all use doesn't work for this
engine/version — most likely the server's loaded HNSW index segment doesn't hot-reload search
params from a metadata-only update. The only way to actually get different ef_search behavior is
to delete and recreate the collection (forcing a fresh index load) for every sweep point, which
is what `set_search_params()` does here — slower than the other engines, but it's what produces
real, varying data instead of a flat line that misrepresents this engine.
"""

import time

import chromadb
import numpy as np

from benchmark.common.base_adapter import BaseAdapter, QueryResult
from benchmark.common.docker_utils import container_dir_size_bytes

HOST = "localhost"
PORT = 8000
CONTAINER_NAME = "vecbench-chroma"
COLLECTION_NAME = "items"
BATCH_SIZE = 1000


class Adapter(BaseAdapter):
    name = "chroma"

    def __init__(self) -> None:
        self.client: chromadb.HttpClient | None = None
        self.collection = None
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._m = 16
        self._ef_construction = 200

    def connect(self) -> None:
        last_err: Exception | None = None
        for _ in range(30):
            try:
                client = chromadb.HttpClient(host=HOST, port=PORT)
                client.heartbeat()
                self.client = client
                return
            except Exception as e:  # noqa: BLE001 - broad on purpose during startup polling
                last_err = e
                time.sleep(1)
        raise RuntimeError(f"could not connect to chroma: {last_err}")

    def build_index(self, ids: list[str], vectors: np.ndarray, index_params: dict) -> float:
        self._ids = ids
        self._vectors = vectors
        self._m = int(index_params.get("m", 16))
        self._ef_construction = int(index_params.get("ef_construction", 200))
        return self._rebuild(ef_search=100)

    def set_search_params(self, search_params: dict) -> None:
        self._rebuild(ef_search=int(search_params.get("search_ef", 40)))

    def _rebuild(self, ef_search: int) -> float:
        start = time.perf_counter()
        try:
            self.client.delete_collection(COLLECTION_NAME)
        except Exception:  # noqa: BLE001 - collection may not exist yet
            pass

        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            configuration={
                "hnsw": {
                    "space": "cosine",
                    "ef_construction": self._ef_construction,
                    "ef_search": ef_search,
                    "max_neighbors": self._m,
                }
            },
        )

        for i in range(0, len(self._ids), BATCH_SIZE):
            self.collection.add(
                ids=self._ids[i : i + BATCH_SIZE],
                embeddings=self._vectors[i : i + BATCH_SIZE].tolist(),
            )

        return time.perf_counter() - start

    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        start = time.perf_counter()
        result = self.collection.query(query_embeddings=[vector.tolist()], n_results=k, include=[])
        latency_ms = (time.perf_counter() - start) * 1000
        return QueryResult(ids=result["ids"][0], latency_ms=latency_ms)

    def index_size_bytes(self) -> int:
        # chromadb 1.x's server is Rust-based and persists to /data by default (the older
        # Python-based server used /chroma/chroma, which no longer exists in this image).
        return container_dir_size_bytes(CONTAINER_NAME, "/data")

    def close(self) -> None:
        pass

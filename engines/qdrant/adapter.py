"""Qdrant adapter — purpose-built vector DB, HNSW index, cosine distance."""

import time

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import Distance, HnswConfigDiff, PointStruct, SearchParams, VectorParams

from benchmark.common.base_adapter import BaseAdapter, QueryResult
from benchmark.common.docker_utils import container_dir_size_bytes

HOST = "localhost"
PORT = 6333
CONTAINER_NAME = "vecbench-qdrant"
COLLECTION = "items"
BATCH_SIZE = 256


class Adapter(BaseAdapter):
    name = "qdrant"

    def __init__(self) -> None:
        self.client: QdrantClient | None = None
        self._id_map: dict[int, str] = {}
        self._hnsw_ef = 40

    def connect(self) -> None:
        last_err: Exception | None = None
        for _ in range(30):
            try:
                client = QdrantClient(host=HOST, port=PORT)
                client.get_collections()
                self.client = client
                return
            except Exception as e:  # noqa: BLE001 - broad on purpose during startup polling
                last_err = e
                time.sleep(1)
        raise RuntimeError(f"could not connect to qdrant: {last_err}")

    def build_index(self, ids: list[str], vectors: np.ndarray, index_params: dict) -> float:
        dim = vectors.shape[1]
        m = int(index_params.get("m", 16))
        ef_construction = int(index_params.get("ef_construction", 200))

        start = time.perf_counter()
        try:
            self.client.delete_collection(COLLECTION)
        except UnexpectedResponse:
            pass

        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=m, ef_construct=ef_construction),
        )

        self._id_map = dict(enumerate(ids))
        points = [PointStruct(id=i, vector=vec.tolist()) for i, vec in enumerate(vectors)]
        for i in range(0, len(points), BATCH_SIZE):
            self.client.upsert(collection_name=COLLECTION, points=points[i : i + BATCH_SIZE], wait=True)

        return time.perf_counter() - start

    def set_search_params(self, search_params: dict) -> None:
        self._hnsw_ef = int(search_params.get("hnsw_ef", 40))

    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        start = time.perf_counter()
        # .search() was removed in recent qdrant-client versions in favor of .query_points()
        response = self.client.query_points(
            collection_name=COLLECTION,
            query=vector.tolist(),
            limit=k,
            search_params=SearchParams(hnsw_ef=self._hnsw_ef),
        )
        latency_ms = (time.perf_counter() - start) * 1000
        ids_out = [self._id_map[point.id] for point in response.points]
        return QueryResult(ids=ids_out, latency_ms=latency_ms)

    def index_size_bytes(self) -> int:
        return container_dir_size_bytes(CONTAINER_NAME, "/qdrant/storage")

    def close(self) -> None:
        if self.client:
            self.client.close()

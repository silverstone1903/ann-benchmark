"""Weaviate adapter — purpose-built vector DB, HNSW index, cosine distance.

We supply our own vectors (no built-in vectorizer). Objects are keyed by a UUID derived
deterministically from our string id (generate_uuid5) so query results can be mapped back to
the original id via a local dict, without needing an extra stored property.

Riskiest part: `ef` is one of the few HNSW params Weaviate allows updating live (without an
index rebuild) via collection.config.update(...) — used here to sweep search-time recall vs.
latency the same way the other engines' set_search_params() does.
"""

import time

import numpy as np
import weaviate
from weaviate.classes.config import Configure, Reconfigure, VectorDistances
from weaviate.classes.data import DataObject
from weaviate.util import generate_uuid5

from benchmark.common.base_adapter import BaseAdapter, QueryResult
from benchmark.common.docker_utils import container_dir_size_bytes

CONTAINER_NAME = "vecbench-weaviate"
COLLECTION = "Items"
BATCH_SIZE = 200


class Adapter(BaseAdapter):
    name = "weaviate"

    def __init__(self) -> None:
        self.client: weaviate.WeaviateClient | None = None
        self.collection = None
        self._id_map: dict[str, str] = {}
        self._ef = 40

    def connect(self) -> None:
        last_err: Exception | None = None
        for _ in range(60):
            try:
                client = weaviate.connect_to_local()
                if client.is_ready():
                    self.client = client
                    return
            except Exception as e:  # noqa: BLE001 - broad on purpose during startup polling
                last_err = e
            time.sleep(1)
        raise RuntimeError(f"could not connect to weaviate: {last_err}")

    def build_index(self, ids: list[str], vectors: np.ndarray, index_params: dict) -> float:
        m = int(index_params.get("m", 16))
        ef_construction = int(index_params.get("ef_construction", 200))

        start = time.perf_counter()
        if self.client.collections.exists(COLLECTION):
            self.client.collections.delete(COLLECTION)

        self.client.collections.create(
            COLLECTION,
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE,
                max_connections=m,
                ef_construction=ef_construction,
            ),
            properties=[],
        )
        self.collection = self.client.collections.get(COLLECTION)

        self._id_map = {}
        objects = []
        for doc_id, vec in zip(ids, vectors):
            uid = generate_uuid5(doc_id)
            self._id_map[uid] = doc_id
            objects.append(DataObject(properties={}, vector=vec.tolist(), uuid=uid))

        for i in range(0, len(objects), BATCH_SIZE):
            self.collection.data.insert_many(objects[i : i + BATCH_SIZE])

        return time.perf_counter() - start

    def set_search_params(self, search_params: dict) -> None:
        self._ef = int(search_params.get("ef", 40))
        self.collection.config.update(vector_index_config=Reconfigure.VectorIndex.hnsw(ef=self._ef))

    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        start = time.perf_counter()
        result = self.collection.query.near_vector(near_vector=vector.tolist(), limit=k)
        latency_ms = (time.perf_counter() - start) * 1000
        ids_out = [self._id_map.get(str(obj.uuid), str(obj.uuid)) for obj in result.objects]
        return QueryResult(ids=ids_out, latency_ms=latency_ms)

    def index_size_bytes(self) -> int:
        return container_dir_size_bytes(CONTAINER_NAME, "/var/lib/weaviate")

    def close(self) -> None:
        if self.client:
            self.client.close()

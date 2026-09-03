"""Redis Stack (RediSearch) adapter — HNSW vector index over a hash-per-document collection."""

import time

import numpy as np
import redis
from redis.commands.search.field import VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from benchmark.common.base_adapter import BaseAdapter, QueryResult

HOST = "localhost"
PORT = 6379
INDEX_NAME = "idx:items"
PREFIX = "item:"


class Adapter(BaseAdapter):
    name = "redis"

    def __init__(self) -> None:
        self.client: redis.Redis | None = None
        self._ef_runtime = 40

    def connect(self) -> None:
        last_err: Exception | None = None
        for _ in range(30):
            try:
                client = redis.Redis(host=HOST, port=PORT)
                client.ping()
                self.client = client
                return
            except redis.exceptions.ConnectionError as e:
                last_err = e
                time.sleep(1)
        raise RuntimeError(f"could not connect to redis: {last_err}")

    def build_index(
        self, ids: list[str], vectors: np.ndarray, index_params: dict
    ) -> float:
        dim = vectors.shape[1]
        m = int(index_params.get("m", 16))
        ef_construction = int(index_params.get("ef_construction", 200))

        start = time.perf_counter()
        try:
            self.client.ft(INDEX_NAME).dropindex(delete_documents=True)
        except redis.exceptions.ResponseError:
            pass  # index didn't exist yet

        schema = (
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": dim,
                    "DISTANCE_METRIC": "COSINE",
                    "M": m,
                    "EF_CONSTRUCTION": ef_construction,
                },
            ),
        )
        self.client.ft(INDEX_NAME).create_index(
            schema,
            definition=IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH),
        )

        pipe = self.client.pipeline(transaction=False)
        for doc_id, vec in zip(ids, vectors):
            pipe.hset(
                f"{PREFIX}{doc_id}",
                mapping={"embedding": np.asarray(vec, dtype=np.float32).tobytes()},
            )
        pipe.execute()

        return time.perf_counter() - start

    def set_search_params(self, search_params: dict) -> None:
        self._ef_runtime = int(search_params.get("ef_runtime", 40))

    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        vec_bytes = np.asarray(vector, dtype=np.float32).tobytes()
        q = (
            Query(
                f"*=>[KNN {k} @embedding $vec EF_RUNTIME {self._ef_runtime} AS score]"
            )
            .sort_by("score")
            .return_fields("score")
            .paging(0, k)
            .dialect(2)
        )
        start = time.perf_counter()
        res = self.client.ft(INDEX_NAME).search(q, query_params={"vec": vec_bytes})
        latency_ms = (time.perf_counter() - start) * 1000
        ids_out = [doc.id[len(PREFIX) :] for doc in res.docs]
        return QueryResult(ids=ids_out, latency_ms=latency_ms)

    def index_size_bytes(self) -> int:
        info = self.client.ft(INDEX_NAME).info()
        size_mb = float(info.get("vector_index_sz_mb", 0))
        return int(size_mb * 1024 * 1024)

    def close(self) -> None:
        if self.client:
            self.client.close()

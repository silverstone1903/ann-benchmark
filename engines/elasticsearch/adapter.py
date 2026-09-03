"""Elasticsearch adapter — dense_vector field with HNSW index_options, kNN query API."""

import time

import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

from benchmark.common.base_adapter import BaseAdapter, QueryResult

URL = "http://localhost:9200"
INDEX_NAME = "items"


class Adapter(BaseAdapter):
    name = "elasticsearch"

    def __init__(self) -> None:
        self.client: Elasticsearch | None = None
        self._num_candidates = 40

    def connect(self) -> None:
        client = Elasticsearch(URL, request_timeout=30)
        last_err: Exception | None = None
        for _ in range(60):
            try:
                if client.ping():
                    self.client = client
                    return
            except Exception as e:  # noqa: BLE001 - broad on purpose during startup polling
                last_err = e
            time.sleep(1)
        raise RuntimeError(f"could not connect to elasticsearch: {last_err}")

    def build_index(self, ids: list[str], vectors: np.ndarray, index_params: dict) -> float:
        dim = vectors.shape[1]
        m = int(index_params.get("m", 16))
        ef_construction = int(index_params.get("ef_construction", 200))

        start = time.perf_counter()
        if self.client.indices.exists(index=INDEX_NAME):
            self.client.indices.delete(index=INDEX_NAME)

        self.client.indices.create(
            index=INDEX_NAME,
            mappings={
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": dim,
                        "index": True,
                        "similarity": "cosine",
                        "index_options": {"type": "hnsw", "m": m, "ef_construction": ef_construction},
                    }
                }
            },
        )

        actions = (
            {"_index": INDEX_NAME, "_id": doc_id, "_source": {"embedding": vec.tolist()}}
            for doc_id, vec in zip(ids, vectors)
        )
        bulk(self.client, actions, chunk_size=500)
        self.client.indices.refresh(index=INDEX_NAME)

        return time.perf_counter() - start

    def set_search_params(self, search_params: dict) -> None:
        self._num_candidates = int(search_params.get("num_candidates", 40))

    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        start = time.perf_counter()
        resp = self.client.search(
            index=INDEX_NAME,
            knn={
                "field": "embedding",
                "query_vector": vector.tolist(),
                "k": k,
                "num_candidates": max(self._num_candidates, k),
            },
            size=k,
            source=False,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        ids_out = [hit["_id"] for hit in resp["hits"]["hits"]]
        return QueryResult(ids=ids_out, latency_ms=latency_ms)

    def index_size_bytes(self) -> int:
        stats = self.client.indices.stats(index=INDEX_NAME)
        return stats["indices"][INDEX_NAME]["total"]["store"]["size_in_bytes"]

    def close(self) -> None:
        if self.client:
            self.client.close()

"""Annoy adapter — runs in-process, same reasoning as engines/faiss/adapter.py.

Vectors are pre-normalized, so Annoy's "angular" metric ranks results identically to cosine
similarity.
"""

import os
import tempfile
import time

import numpy as np
from annoy import AnnoyIndex

from benchmark.common.base_adapter import BaseAdapter, QueryResult


class Adapter(BaseAdapter):
    name = "annoy"

    def __init__(self) -> None:
        self.index: AnnoyIndex | None = None
        self.ids: list[str] = []
        self._search_k = -1
        self._index_path: str | None = None

    def connect(self) -> None:
        pass  # nothing to connect to

    def build_index(self, ids: list[str], vectors: np.ndarray, index_params: dict) -> float:
        dim = vectors.shape[1]
        n_trees = int(index_params.get("n_trees", 50))

        start = time.perf_counter()
        index = AnnoyIndex(dim, "angular")
        for i, vec in enumerate(vectors):
            index.add_item(i, vec.tolist())
        index.build(n_trees)

        # save() switches this index to reading from an mmap of `path`, so the file must stay
        # on disk (and stay open) for the rest of the run — save once here, at build time
        # (consistent with the other engines' build time including a persist-to-disk step),
        # and reuse the same path for index_size_bytes() instead of a save-then-delete cycle,
        # which fails on Windows with a PermissionError while the mmap is still held open.
        fd, path = tempfile.mkstemp(suffix=".ann")
        os.close(fd)
        index.save(path)

        self.index = index
        self.ids = ids
        self._index_path = path
        return time.perf_counter() - start

    def set_search_params(self, search_params: dict) -> None:
        self._search_k = int(search_params.get("search_k", -1))

    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        start = time.perf_counter()
        idx = self.index.get_nns_by_vector(vector.tolist(), k, search_k=self._search_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return QueryResult(ids=[self.ids[i] for i in idx], latency_ms=latency_ms)

    def index_size_bytes(self) -> int:
        return os.path.getsize(self._index_path)

    def close(self) -> None:
        self.index = None
        if self._index_path and os.path.exists(self._index_path):
            try:
                os.remove(self._index_path)
            except OSError:
                pass  # Windows: annoy may still hold the file mmap'ed open

"""Common interface every engine adapter implements.

An adapter owns one engine's index for one benchmark run: connect -> build_index ->
(search-param sweep) -> query -> stats -> close. The runner (runner.py) drives this
lifecycle identically for every engine so the resulting metrics are comparable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class QueryResult:
    ids: list[str]  # ranked result doc ids, best match first
    latency_ms: float


class BaseAdapter(ABC):
    name: str

    @abstractmethod
    def connect(self) -> None:
        """Open a client connection to the already-running engine container."""

    @abstractmethod
    def build_index(self, ids: list[str], vectors: np.ndarray, index_params: dict) -> float:
        """Create the collection/table/index and load all vectors. Returns build time (s)."""

    @abstractmethod
    def set_search_params(self, search_params: dict) -> None:
        """Apply a search-time knob (ef_search, nprobe, search_k, ...) before querying."""

    @abstractmethod
    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        """Run a single top-k query, return ranked ids + latency in ms."""

    @abstractmethod
    def index_size_bytes(self) -> int:
        """On-disk size of the index/collection, in bytes."""

    @abstractmethod
    def close(self) -> None:
        """Release the client connection (does not tear down the container)."""

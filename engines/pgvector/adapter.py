"""pgvector adapter — the reference engine used to validate the benchmark harness end-to-end
before replicating the same interface for the other 7 engines.

A single psycopg connection is not safe to use concurrently from multiple threads, and the
benchmark runner measures QPS under concurrency (ThreadPoolExecutor). So `query()` uses a
lazily-created thread-local connection; `self.conn` (opened in connect()) is reserved for the
single-threaded build/admin operations.
"""

import threading
import time

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from benchmark.common.base_adapter import BaseAdapter, QueryResult

DSN = "host=localhost port=5432 dbname=vecbench user=bench password=bench"
TABLE = "items"


class Adapter(BaseAdapter):
    name = "pgvector"

    def __init__(self) -> None:
        self.conn: psycopg.Connection | None = None
        self._local = threading.local()
        self._ef_search = 40

    def _new_connection(self, register: bool = True) -> psycopg.Connection:
        # `register_vector` looks up the `vector` type in pg_type, which only exists once
        # `CREATE EXTENSION vector` has run — so the very first connection (made before the
        # extension exists) must skip it; build_index() registers it on self.conn right after
        # creating the extension, and every connection opened after that can register eagerly.
        last_err: Exception | None = None
        for _ in range(30):
            try:
                conn = psycopg.connect(DSN, autocommit=True)
                if register:
                    register_vector(conn)
                return conn
            except psycopg.OperationalError as e:
                last_err = e
                time.sleep(1)
        raise RuntimeError(f"could not connect to pgvector: {last_err}")

    def _thread_conn(self) -> psycopg.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = self._new_connection()
            with conn.cursor() as cur:
                cur.execute(f"SET hnsw.ef_search = {self._ef_search}")
            self._local.conn = conn
        return conn

    def connect(self) -> None:
        self.conn = self._new_connection(register=False)

    def build_index(self, ids: list[str], vectors: np.ndarray, index_params: dict) -> float:
        dim = vectors.shape[1]
        m = index_params.get("m", 16)
        ef_construction = index_params.get("ef_construction", 200)

        start = time.perf_counter()
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(self.conn)
        with self.conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
            cur.execute(f"CREATE TABLE {TABLE} (id text PRIMARY KEY, embedding vector({dim}))")

            rows = list(zip(ids, vectors))
            cur.executemany(f"INSERT INTO {TABLE} (id, embedding) VALUES (%s, %s)", rows)

            cur.execute(
                f"CREATE INDEX ON {TABLE} USING hnsw (embedding vector_cosine_ops) "
                f"WITH (m = {int(m)}, ef_construction = {int(ef_construction)})"
            )
        return time.perf_counter() - start

    def set_search_params(self, search_params: dict) -> None:
        self._ef_search = int(search_params.get("ef_search", 40))
        with self.conn.cursor() as cur:
            cur.execute(f"SET hnsw.ef_search = {self._ef_search}")
        # Drop cached per-thread connections so they're lazily recreated (and re-SET
        # with the new ef_search) on next use instead of querying with a stale value.
        self._local = threading.local()

    def query(self, vector: np.ndarray, k: int) -> QueryResult:
        conn = self._thread_conn()
        start = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id FROM {TABLE} ORDER BY embedding <=> %s LIMIT %s",
                (vector, k),
            )
            rows = cur.fetchall()
        latency_ms = (time.perf_counter() - start) * 1000
        return QueryResult(ids=[r[0] for r in rows], latency_ms=latency_ms)

    def index_size_bytes(self) -> int:
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT pg_total_relation_size('{TABLE}')")
            return cur.fetchone()[0]

    def close(self) -> None:
        if self.conn:
            self.conn.close()

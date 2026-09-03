"""Orchestrates one engine's full benchmark run: compose up -> connect -> build_index ->
warm-up -> search-param sweep (recall/precision/nDCG + latency + QPS at each point) ->
compose down -> write results/<engine>_<ts>.json.

Identical for every engine — only the adapter (engines/<name>/adapter.py) and config
(benchmark/configs/<name>.yaml) differ, which is what keeps the comparison fair.
"""

import importlib
import json
import pathlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.common.base_adapter import BaseAdapter  # noqa: E402
from benchmark.common.docker_utils import compose_down, compose_up, wait_tcp  # noqa: E402
from benchmark.common.metrics import (  # noqa: E402
    latency_percentiles,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config(engine: str) -> dict:
    with open(ROOT / "benchmark" / "configs" / f"{engine}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data():
    emb_dir = ROOT / "embeddings"
    with open(emb_dir / "ids.json", encoding="utf-8") as f:
        ids = json.load(f)
    corpus_vecs = np.load(emb_dir / "corpus.npy")
    query_vecs = np.load(emb_dir / "queries.npy")

    with open(ROOT / "ground_truth" / "exact_knn.json", encoding="utf-8") as f:
        exact_knn = json.load(f)
    with open(ROOT / "ground_truth" / "labels.json", encoding="utf-8") as f:
        labels = json.load(f)

    return corpus_vecs, ids["corpus"], query_vecs, ids["query"], exact_knn, labels


def measure_qps(adapter: BaseAdapter, query_vecs: np.ndarray, k: int, concurrency: int, n_queries: int) -> float:
    sample = query_vecs[:n_queries]
    start = time.perf_counter()
    if concurrency == 1:
        for qv in sample:
            adapter.query(qv, k)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(lambda qv: adapter.query(qv, k), sample))
    elapsed = time.perf_counter() - start
    return len(sample) / elapsed if elapsed > 0 else float("inf")


def run(
    engine: str,
    k: int = 10,
    warmup_queries: int = 20,
    concurrency: tuple[int, ...] = (1, 8),
    qps_sample: int = 200,
) -> pathlib.Path:
    config = load_config(engine)
    compose_file = ROOT / "engines" / engine / "docker-compose.yml"
    # faiss/annoy are libraries, not client-server systems, so they run in-process with no
    # docker-compose stack and no health check — everyone else (pgvector, redis, elasticsearch,
    # qdrant, weaviate, chroma) is fronted by its own container.
    has_docker = compose_file.exists()

    adapter_module = importlib.import_module(f"engines.{engine}.adapter")
    adapter: BaseAdapter = adapter_module.Adapter()

    corpus_vecs, corpus_ids, query_vecs, query_ids, exact_knn, labels = load_data()

    relevant_by_label: dict[int, set[str]] = {}
    for doc_id, label in labels.items():
        if doc_id.startswith("corpus_"):
            relevant_by_label.setdefault(label, set()).add(doc_id)

    if has_docker:
        compose_up(compose_file)
    try:
        if has_docker and config.get("health_check"):
            wait_tcp(**config["health_check"])
        adapter.connect()
        build_time_s = adapter.build_index(corpus_ids, corpus_vecs, config["index"])
        index_size = adapter.index_size_bytes()

        sweep_results = []
        param_name = config["search_sweep"]["param_name"]
        for search_val in config["search_sweep"]["values"]:
            adapter.set_search_params({param_name: search_val})

            for qv in query_vecs[:warmup_queries]:
                adapter.query(qv, k)

            recalls, precisions, ndcgs, latencies = [], [], [], []
            for qid, qv in zip(query_ids, query_vecs):
                result = adapter.query(qv, k)
                latencies.append(result.latency_ms)
                relevant = relevant_by_label.get(labels[qid], set())
                recalls.append(recall_at_k(result.ids, exact_knn[qid], k))
                precisions.append(precision_at_k(result.ids, relevant, k))
                ndcgs.append(ndcg_at_k(result.ids, relevant, k))

            qps_by_concurrency = {
                f"qps_c{c}": measure_qps(adapter, query_vecs, k, c, qps_sample) for c in concurrency
            }

            sweep_results.append(
                {
                    "search_param": {param_name: search_val},
                    "recall_at_k": float(np.mean(recalls)),
                    "precision_at_k": float(np.mean(precisions)),
                    "ndcg_at_k": float(np.mean(ndcgs)),
                    **latency_percentiles(latencies),
                    **qps_by_concurrency,
                }
            )

        results = {
            "engine": engine,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "k": k,
            "n_corpus": len(corpus_ids),
            "n_queries": len(query_ids),
            "build_time_s": build_time_s,
            "index_size_bytes": index_size,
            "index_params": config["index"],
            "sweep": sweep_results,
        }
    finally:
        adapter.close()
        if has_docker:
            compose_down(compose_file)

    out_path = RESULTS_DIR / f"{engine}_{int(time.time())}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return out_path

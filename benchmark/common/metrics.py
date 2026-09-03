"""Retrieval-quality and latency metrics shared by every engine's benchmark run."""

import math

import numpy as np


def recall_at_k(retrieved: list[str], ground_truth: list[str], k: int) -> float:
    """Fraction of the top-k exact (brute-force) neighbors also found in retrieved top-k."""
    gt_set = set(ground_truth[:k])
    if not gt_set:
        return 0.0
    hit = len(set(retrieved[:k]) & gt_set)
    return hit / len(gt_set)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved docs that share the query's label (relevance proxy)."""
    if k == 0:
        return 0.0
    hit = sum(1 for doc_id in retrieved[:k] if doc_id in relevant)
    return hit / k


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, doc_id in enumerate(retrieved[:k]) if doc_id in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def latency_percentiles(latencies_ms: list[float]) -> dict:
    arr = np.array(latencies_ms)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(np.mean(arr)),
    }

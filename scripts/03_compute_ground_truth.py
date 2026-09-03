"""Compute exact (brute-force) nearest-neighbor ground truth and label lookup.

Since embeddings are L2-normalized, inner product == cosine similarity, so a flat
IndexFlatIP index over the (small, ~8K) corpus gives exact cosine top-K trivially.

Output:
  ground_truth/exact_knn.json  {query_id: [corpus_id, ...]}  top-100 by cosine similarity, ranked
  ground_truth/labels.json     {id: label}  for both corpus and query docs (relevance proxy)
"""

import json
import pathlib

import faiss
import numpy as np
import pandas as pd

TOP_K = 100

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_PARQUET = ROOT / "data" / "processed" / "corpus.parquet"
EMB_DIR = ROOT / "embeddings"
OUT_DIR = ROOT / "ground_truth"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_parquet(CORPUS_PARQUET)
    with open(EMB_DIR / "ids.json", encoding="utf-8") as f:
        ids = json.load(f)

    corpus_vecs = np.load(EMB_DIR / "corpus.npy")
    query_vecs = np.load(EMB_DIR / "queries.npy")
    corpus_ids = ids["corpus"]
    query_ids = ids["query"]

    assert corpus_vecs.shape[0] == len(corpus_ids)
    assert query_vecs.shape[0] == len(query_ids)

    dim = corpus_vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(corpus_vecs)

    k = min(TOP_K, len(corpus_ids))
    _, neighbor_idx = index.search(query_vecs, k)

    exact_knn = {
        query_ids[i]: [corpus_ids[j] for j in neighbor_idx[i]] for i in range(len(query_ids))
    }
    with open(OUT_DIR / "exact_knn.json", "w", encoding="utf-8") as f:
        json.dump(exact_knn, f)

    labels = dict(zip(df["id"], df["label"].astype(int)))
    with open(OUT_DIR / "labels.json", "w", encoding="utf-8") as f:
        json.dump(labels, f)

    print(f"exact_knn: {len(exact_knn)} queries x top-{k} -> {OUT_DIR / 'exact_knn.json'}")
    print(f"labels: {len(labels)} ids -> {OUT_DIR / 'labels.json'}")


if __name__ == "__main__":
    main()

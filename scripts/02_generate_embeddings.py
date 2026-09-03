"""Generate embeddings for corpus + query texts via a local Ollama server.

Uses `nomic-embed-text`, which expects task-prefixed input for best retrieval quality:
  - documents: "search_document: <text>"
  - queries:   "search_query: <text>"
(see https://ollama.com/library/nomic-embed-text)

Output:
  embeddings/corpus.npy   float32 (N_corpus, 768), L2-normalized
  embeddings/queries.npy  float32 (N_query, 768), L2-normalized
  embeddings/ids.json     {"corpus": [id, ...], "query": [id, ...]} — row order matches the .npy files
"""

import json
import pathlib

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

OLLAMA_URL = "http://localhost:11434/api/embed"
MODEL = "nomic-embed-text"
BATCH_SIZE = 32

ROOT = pathlib.Path(__file__).resolve().parent.parent
CORPUS_PARQUET = ROOT / "data" / "processed" / "corpus.parquet"
OUT_DIR = ROOT / "embeddings"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def embed_batch(texts: list[str]) -> np.ndarray:
    resp = requests.post(OLLAMA_URL, json={"model": MODEL, "input": texts}, timeout=120)
    resp.raise_for_status()
    embeddings = resp.json()["embeddings"]
    return np.array(embeddings, dtype=np.float32)


def embed_all(texts: list[str], prefix: str) -> np.ndarray:
    prefixed = [f"{prefix}{t}" for t in texts]
    chunks = []
    for i in tqdm(range(0, len(prefixed), BATCH_SIZE), desc=f"embedding ({prefix.strip(': ')})"):
        batch = prefixed[i : i + BATCH_SIZE]
        chunks.append(embed_batch(batch))
    vecs = np.concatenate(chunks, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


def main() -> None:
    df = pd.read_parquet(CORPUS_PARQUET)

    corpus_df = df[df["split"] == "corpus"].reset_index(drop=True)
    query_df = df[df["split"] == "query"].reset_index(drop=True)

    corpus_vecs = embed_all(corpus_df["text"].tolist(), prefix="search_document: ")
    query_vecs = embed_all(query_df["text"].tolist(), prefix="search_query: ")

    np.save(OUT_DIR / "corpus.npy", corpus_vecs)
    np.save(OUT_DIR / "queries.npy", query_vecs)

    ids = {
        "corpus": corpus_df["id"].tolist(),
        "query": query_df["id"].tolist(),
    }
    with open(OUT_DIR / "ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)

    print(f"corpus embeddings: {corpus_vecs.shape} -> {OUT_DIR / 'corpus.npy'}")
    print(f"query embeddings:  {query_vecs.shape} -> {OUT_DIR / 'queries.npy'}")


if __name__ == "__main__":
    main()

# Vector Search Benchmark

A small, hands-on comparison of 8 vector search engines (pgvector, Redis Stack, Elasticsearch, Qdrant, Weaviate, Chroma, FAISS, Annoy) under the same conditions: same embeddings, same hardware, same metrics.

## Folder Structure

- `scripts/` : dataset prep, embedding generation, ground truth, the benchmark runner, result
  aggregation and plots
- `engines/<name>/` : one adapter plus a `docker-compose.yml` per engine (FAISS and Annoy have no
  compose file, they're libraries and run in-process)
- `benchmark/common/` : the shared adapter interface and runner that every engine plugs into
- `results/` : one JSON per run, a combined `summary.csv`, and plots under `results/plots/`

## Requirements

- Docker
- Python 3.10+
- Ollama, with `nomic-embed-text` pulled

## Running it

Each engine is independent and can be run on its own, no need to spin everything up at once.

```bash
pip install -r requirements.txt
python scripts/01_prepare_dataset.py
python scripts/02_generate_embeddings.py
python scripts/03_compute_ground_truth.py
python scripts/04_run_benchmark.py --engine pgvector
python scripts/04_run_benchmark.py --engine redis
python scripts/04_run_benchmark.py --engine elasticsearch
python scripts/04_run_benchmark.py --engine qdrant
python scripts/04_run_benchmark.py --engine weaviate
python scripts/04_run_benchmark.py --engine chroma
python scripts/04_run_benchmark.py --engine faiss
python scripts/04_run_benchmark.py --engine annoy
```

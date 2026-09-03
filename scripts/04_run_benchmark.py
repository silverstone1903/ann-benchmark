"""CLI entry point: run the full benchmark pipeline for one engine.

Usage:
    python scripts/04_run_benchmark.py --engine pgvector
"""

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark.common.runner import run  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True, help="engine name, e.g. pgvector")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--warmup-queries", type=int, default=20)
    args = parser.parse_args()

    out_path = run(args.engine, k=args.k, warmup_queries=args.warmup_queries)
    print(f"results -> {out_path}")


if __name__ == "__main__":
    main()

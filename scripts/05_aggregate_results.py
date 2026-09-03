"""Merge every engine's results/<engine>_<ts>.json into one comparison table, and add the
columns that unify what's comparable across engines despite their differing raw sweep knobs:

- sweep_param_name / sweep_param_value: each engine's own search-time knob (ef_search,
  num_candidates, search_k, ...) collapsed from N sparse per-engine columns into two dense ones.
- sweep_rank: 1..5, this row's ordinal position within its engine's own sweep (ascending). Raw
  knob values aren't comparable across engines (annoy's search_k=2000 and qdrant's hnsw_ef=160
  aren't the same kind of thing) — rank is, and gives every engine a common x-axis.

If an engine has more than one results/<engine>_*.json (re-runs while debugging), only the
most recent run's 5 sweep points are kept — older runs would otherwise duplicate rows.
"""

import json
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"

SWEEP_PARAM_COLUMNS = [
    "ef_search",
    "ef_runtime",
    "num_candidates",
    "hnsw_ef",
    "ef",
    "search_ef",
    "search_k",
]


def main() -> None:
    rows = []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for point in data["sweep"]:
            rows.append(
                {
                    "engine": data["engine"],
                    "timestamp": data["timestamp"],
                    "k": data["k"],
                    "build_time_s": data["build_time_s"],
                    "index_size_bytes": data["index_size_bytes"],
                    **point["search_param"],
                    "recall_at_k": point["recall_at_k"],
                    "precision_at_k": point["precision_at_k"],
                    "ndcg_at_k": point["ndcg_at_k"],
                    "p50_ms": point["p50_ms"],
                    "p95_ms": point["p95_ms"],
                    "p99_ms": point["p99_ms"],
                    "qps_c1": point.get("qps_c1"),
                    "qps_c8": point.get("qps_c8"),
                }
            )

    if not rows:
        print(f"no results found under {RESULTS_DIR}/*.json")
        return

    df = pd.DataFrame(rows)

    latest_ts = df.groupby("engine")["timestamp"].transform("max")
    df = df[df["timestamp"] == latest_ts].reset_index(drop=True)

    sweep_cols = [c for c in SWEEP_PARAM_COLUMNS if c in df.columns]
    df["sweep_param_name"] = df[sweep_cols].notna().idxmax(axis=1)
    df["sweep_param_value"] = df[sweep_cols].apply(lambda row: row.dropna().iloc[0], axis=1)
    df["sweep_rank"] = df.groupby("engine")["sweep_param_value"].rank(method="dense").astype(int)

    out_path = RESULTS_DIR / "summary.csv"
    df.to_csv(out_path, index=False)

    print(df.to_string(index=False))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()

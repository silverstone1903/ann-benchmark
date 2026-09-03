"""Download AG News and subsample a small labeled corpus + held-out query set.

Output: data/processed/corpus.parquet with columns id, text, label, split
  - split == "corpus": ~8,000 docs (2,000 per class) used as the indexed collection
  - split == "query":  ~500 docs (125 per class), drawn from AG News' separate test split,
                        used as benchmark queries (same-label docs treated as relevant)
"""

import pathlib

import pandas as pd
from datasets import load_dataset

SEED = 2026
PER_CLASS_CORPUS = 2000
PER_CLASS_QUERY = 125

OUT_DIR = pathlib.Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sample_per_class(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda g: g.sample(n=min(n_per_class, len(g)), random_state=seed))
        .reset_index(drop=True)
    )


def main() -> None:
    # HF retired the old script-based "ag_news" loader; try the current parquet-based repo
    # first and fall back to the legacy name in case the environment still resolves it.
    try:
        ds = load_dataset("fancyzhx/ag_news")
    except Exception:
        ds = load_dataset("ag_news")

    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    corpus_df = sample_per_class(train_df, PER_CLASS_CORPUS, SEED)
    query_df = sample_per_class(test_df, PER_CLASS_QUERY, SEED)

    corpus_df = corpus_df.assign(
        id=[f"corpus_{i:05d}" for i in range(len(corpus_df))],
        split="corpus",
    )[["id", "text", "label", "split"]]

    query_df = query_df.assign(
        id=[f"query_{i:05d}" for i in range(len(query_df))],
        split="query",
    )[["id", "text", "label", "split"]]

    out_df = pd.concat([corpus_df, query_df], ignore_index=True)
    out_path = OUT_DIR / "corpus.parquet"
    out_df.to_parquet(out_path, index=False)

    print(f"corpus: {len(corpus_df)} docs, query: {len(query_df)} docs")
    print(
        f"label distribution (corpus):\n{corpus_df['label'].value_counts().sort_index()}"
    )
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()

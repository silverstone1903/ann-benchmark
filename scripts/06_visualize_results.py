"""Visualize results/summary.csv (produced by 05_aggregate_results.py) with plotnine.

Six focused charts rather than one crowded dashboard:
  00_snapshot.png             - simple bar-chart snapshot per engine (recall/p50/QPS), the
                                 grounding "at a glance" chart before the more layered ones below
  01_recall_vs_latency.png   - recall/latency Pareto curve per engine (the headline chart)
  02_recall_vs_throughput.png - recall/QPS Pareto curve per engine
  03_recall_by_effort_rank.png - all engines on a common "effort rank" axis (see sweep_rank)
  04_build_cost.png          - build time vs on-disk index size, one point per engine
  05_concurrency_scaling.png - QPS at 1 vs 8 concurrent clients, per engine

Colors: a fixed 8-hue categorical order (one hex per engine, never reassigned or cycled), from
a palette pre-validated for colorblind-safe adjacent contrast — see the dataviz skill.
"""

import pathlib

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_wrap,
    geom_col,
    geom_line,
    geom_point,
    geom_text,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
    scale_x_log10,
    scale_y_log10,
    theme,
    theme_minimal,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

# Fixed engine -> color assignment (never cycled, never reassigned by chart or by rank) so the
# same engine reads as the same color in every chart, including ones that only show a subset.
ENGINE_COLORS = {
    "pgvector": "#2a78d6",
    "redis": "#eb6834",
    "elasticsearch": "#1baf7a",
    "qdrant": "#eda100",
    "weaviate": "#e87ba4",
    "chroma": "#008300",
    "faiss": "#4a3aa7",
    "annoy": "#e34948",
}
ENGINE_ORDER = list(ENGINE_COLORS)

BASE_THEME = theme_minimal() + theme(
    figure_size=(8, 5.5),
    plot_title=element_text(weight="bold", size=13),
    plot_subtitle=element_text(size=9, color="#52514e"),
)


def _load() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_DIR / "summary.csv")
    df["engine"] = pd.Categorical(df["engine"], categories=ENGINE_ORDER, ordered=True)
    return df


def plot_snapshot(df: pd.DataFrame) -> None:
    """Simplest possible chart: recall/p50/QPS per engine as plain bars, at one shared effort
    point (the middle of each engine's own sweep). No Pareto curves, no rank axis, just "what did
    each engine do at a middling setting" — meant to ground the reader before the more layered
    charts below, which was missing when this was 5 charts and all of them already assumed the
    reader had a mental model of the recall/latency tradeoff.
    """
    mid_rank = int(df["sweep_rank"].median())
    snap = df[df["sweep_rank"] == mid_rank][["engine", "recall_at_k", "p50_ms", "qps_c1"]].copy()
    long = snap.melt(id_vars="engine", var_name="metric", value_name="value")

    metric_labels = {
        "recall_at_k": "Recall@10",
        "p50_ms": "p50 latency, ms",
        "qps_c1": "QPS (1 client)",
    }
    long["metric"] = pd.Categorical(
        long["metric"].map(metric_labels), categories=list(metric_labels.values()), ordered=True
    )

    p = (
        ggplot(long, aes(x="engine", y="value", fill="engine"))
        + geom_col(width=0.7, show_legend=False)
        + facet_wrap("~metric", scales="free_y")
        + scale_fill_manual(values=ENGINE_COLORS, limits=ENGINE_ORDER)
        + labs(
            title="Engines at a glance",
            subtitle=f"Recall@10, p50 latency, and single-thread QPS at one shared effort point (rank {mid_rank} of 5)",
            x="",
            y="",
        )
        + BASE_THEME
        + theme(axis_text_x=element_text(rotation=40, ha="right"), figure_size=(10, 4.5))
    )
    p.save(PLOTS_DIR / "00_snapshot.png", dpi=200, verbose=False)


def plot_recall_vs_latency(df: pd.DataFrame) -> None:
    p = (
        ggplot(df, aes(x="p50_ms", y="recall_at_k", color="engine", group="engine"))
        + geom_line(size=0.8)
        + geom_point(size=2.3)
        + scale_x_log10(name="p50 query latency, ms (log scale)")
        + scale_color_manual(values=ENGINE_COLORS, limits=ENGINE_ORDER, name="Engine")
        + labs(
            title="Recall–latency tradeoff",
            subtitle="Recall@10 vs p50 latency across each engine's own search-effort sweep — up-and-left is better",
            y="Recall@10",
        )
        + BASE_THEME
    )
    p.save(PLOTS_DIR / "01_recall_vs_latency.png", dpi=200, verbose=False)


def plot_recall_vs_throughput(df: pd.DataFrame) -> None:
    p = (
        ggplot(df, aes(x="qps_c1", y="recall_at_k", color="engine", group="engine"))
        + geom_line(size=0.8)
        + geom_point(size=2.3)
        + scale_x_log10(name="single-thread QPS (log scale)")
        + scale_color_manual(values=ENGINE_COLORS, limits=ENGINE_ORDER, name="Engine")
        + labs(
            title="Recall–throughput tradeoff",
            subtitle="Recall@10 vs single-thread queries/sec — up-and-right is better",
            y="Recall@10",
        )
        + BASE_THEME
    )
    p.save(PLOTS_DIR / "02_recall_vs_throughput.png", dpi=200, verbose=False)


def plot_recall_by_effort_rank(df: pd.DataFrame) -> None:
    p = (
        ggplot(df, aes(x="sweep_rank", y="recall_at_k", color="engine", group="engine"))
        + geom_line(size=0.8)
        + geom_point(size=2.3)
        + scale_x_continuous(breaks=[1, 2, 3, 4, 5], name="search-effort rank (1=lowest, 5=highest, own sweep)")
        + scale_color_manual(values=ENGINE_COLORS, limits=ENGINE_ORDER, name="Engine")
        + labs(
            title="Recall saturation by effort level",
            subtitle="Every engine's 5-point sweep on one common ordinal axis — raw knob values aren't comparable, rank order is",
            y="Recall@10",
        )
        + BASE_THEME
    )
    p.save(PLOTS_DIR / "03_recall_by_effort_rank.png", dpi=200, verbose=False)


def plot_build_cost(df: pd.DataFrame) -> None:
    build_df = df.drop_duplicates("engine")[["engine", "build_time_s", "index_size_bytes"]].copy()
    build_df["index_size_mb"] = build_df["index_size_bytes"] / (1024 * 1024)

    p = (
        ggplot(build_df, aes(x="index_size_mb", y="build_time_s", color="engine", label="engine"))
        + geom_point(size=4)
        + geom_text(nudge_y=0.06, size=9, ha="left", format_string=" {}", show_legend=False)
        + scale_x_log10(name="on-disk index size, MB (log scale)")
        + scale_color_manual(values=ENGINE_COLORS, limits=ENGINE_ORDER)
        + labs(
            title="Indexing cost",
            subtitle="Build time vs on-disk index size for the same ~8K vectors (768-dim) — directly labeled, no legend needed",
            y="Build time, s",
        )
        + BASE_THEME
        + theme(legend_position="none")
    )
    p.save(PLOTS_DIR / "04_build_cost.png", dpi=200, verbose=False)


def plot_concurrency_scaling(df: pd.DataFrame) -> None:
    mid_rank = int(df["sweep_rank"].median())
    mid = df[df["sweep_rank"] == mid_rank][["engine", "qps_c1", "qps_c8"]]
    long = mid.melt(id_vars="engine", var_name="concurrency", value_name="qps")
    long["concurrency"] = long["concurrency"].map({"qps_c1": "1 client", "qps_c8": "8 clients"})
    long["concurrency"] = pd.Categorical(long["concurrency"], categories=["1 client", "8 clients"], ordered=True)

    # Concurrency is an ordinal progression (low->high), not a categorical identity, so it gets
    # a single sequential hue stepped light->dark rather than two arbitrary categorical colors.
    p = (
        ggplot(long, aes(x="engine", y="qps", fill="concurrency"))
        + geom_col(position="dodge", width=0.7)
        + scale_y_log10(name="QPS (log scale)")
        + scale_fill_manual(values=["#86b6ef", "#184f95"], name="Concurrency")
        + labs(
            title="Concurrency scaling",
            subtitle=f"QPS at 1 vs 8 concurrent clients, effort rank {mid_rank} of 5",
            x="",
        )
        + BASE_THEME
        + theme(axis_text_x=element_text(rotation=30, ha="right"))
    )
    p.save(PLOTS_DIR / "05_concurrency_scaling.png", dpi=200, verbose=False)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df = _load()

    plot_snapshot(df)
    plot_recall_vs_latency(df)
    plot_recall_vs_throughput(df)
    plot_recall_by_effort_rank(df)
    plot_build_cost(df)
    plot_concurrency_scaling(df)

    print(f"saved 6 charts -> {PLOTS_DIR}")


if __name__ == "__main__":
    main()

"""Plot the degradation benchmark from `benchmark_results/summary.csv`.

Two views, both a grid of line charts over the two sweeps (separation,
x reversed easy -> no structure; random-points contamination):

- `--view best` (default): rows = best distance-based / best LCA solution,
  one line per clustering validity index. The original headline view.
- `--view models`: rows = validity index, one line per algorithm (k-means,
  AHC, HDBSCAN, LCA — each at its own best gap-selected configuration for
  that index), to compare how the algorithms degrade.

`--metric k` (default) plots the number of identified clusters excluding
singletons, fixed 0-20 axis with unit ticks and a reference line at the true
k; `--metric ari` plots the adjusted Rand index against the true classes.

With several seeds, lines show the mean across seeds and translucent dots the
individual seed values.

Usage:
    python plot_benchmark.py [--view best|models] [--metric k|ari]
                             [--summary benchmark_results/summary.csv]
                             [--out <png>]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

# Validated categorical palette (dataviz reference, light mode) — fixed order.
INDEXES = {
    "silhouette": ("Silhouette", "#2a78d6"),
    "calinski_harabasz": ("Calinski-Harabasz", "#1baf7a"),
    "davies_bouldin": ("Davies-Bouldin", "#eda100"),
    "dunn": ("Dunn (GD43)", "#008300"),
}
MODELS = {
    "kmeans": ("K-means", "#2a78d6"),
    "AHC": ("AHC", "#1baf7a"),
    "HDBSCAN": ("HDBSCAN", "#eda100"),
    "latent": ("LCA", "#008300"),
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

TRUE_K = 4
# Fixed y scale so one extreme run cannot distort the whole grid; values
# above the cap are clipped at the axis edge.
METRICS = {
    "k": dict(col="n_clust_effective", label="clusters identified (≥ 2 members)",
              lim=(0, 20), tick=1, ref=TRUE_K, ref_label=f"true k = {TRUE_K}"),
    "ari": dict(col="ari", label="ARI vs true classes",
                lim=(-0.1, 1.0), tick=0.2, ref=None, ref_label=None),
}

POOLS = [("distance", "Best distance-based solution"), ("LCA", "Best LCA solution")]
SWEEPS = [
    ("separation", "class_sep", "Class separation (easy → no structure)", True),
    ("random", "noise_prop", "Proportion of uniform random points", False),
]


def draw_panel(ax, sub, xcol, reverse_x, series_col, series_map, m):
    """One panel: metric `m` vs `xcol`, one line per value of `series_col`."""
    multi_seed = sub["seed"].nunique() > 1 if len(sub) else False
    lo, hi = m["lim"]
    for key, (label, color) in series_map.items():
        s = sub[sub[series_col] == key].dropna(subset=[m["col"]])
        if s.empty:
            continue
        mean = s.groupby(xcol)[m["col"]].mean().sort_index()
        # Semi-transparent marks so exactly juxtaposed lines (several series
        # flat at the same k) stay individually visible where they overlap.
        ax.plot(
            mean.index, mean.values.clip(lo, hi),
            color=color, lw=2, alpha=0.7,
            solid_capstyle="round", solid_joinstyle="round",
            marker="o", ms=8, mec=SURFACE, mew=2, label=label,
        )
        if multi_seed:
            ax.scatter(
                s[xcol], s[m["col"]].clip(lo, hi),
                color=color, s=14, alpha=0.35, linewidths=0, zorder=1,
            )

    if m["ref"] is not None:
        ax.axhline(m["ref"], color=AXIS, lw=1, zorder=0)
    if reverse_x:
        ax.invert_xaxis()

    ax.yaxis.set_major_locator(MultipleLocator(m["tick"]))
    ax.set_ylim(lo, hi)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_facecolor(SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=["best", "models"], default="best")
    ap.add_argument("--metric", choices=list(METRICS), default="k")
    ap.add_argument(
        "--summary", type=Path, default=Path("benchmark_results/summary.csv")
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.summary)
    if "rank" not in df.columns:  # summary written before the per-model schema
        df["rank"] = 0
    m = METRICS[args.metric]

    # Rows of the grid: (row filter, row label, series to draw as lines).
    if args.view == "best":
        rows = [
            (df["pool"].eq(pool) & df["rank"].eq(0), pool_label, "index", INDEXES)
            for pool, pool_label in POOLS
        ]
        title = "Best solution per validity index"
    else:
        rows = [
            (df["index"].eq(col), label, "model", MODELS)
            for col, (label, _) in INDEXES.items()
        ]
        title = "Each algorithm's best configuration, by validity index"

    fig, axes = plt.subplots(
        len(rows), len(SWEEPS),
        figsize=(10, 3.4 * len(rows)), sharey="row", facecolor=SURFACE,
        squeeze=False,
    )
    for i, (row_mask, row_label, series_col, series_map) in enumerate(rows):
        for j, (sweep, xcol, xlabel, reverse_x) in enumerate(SWEEPS):
            ax = axes[i, j]
            sub = df[row_mask & df["sweep"].eq(sweep)]
            draw_panel(ax, sub, xcol, reverse_x, series_col, series_map, m)
            if i == 0:
                ax.set_title(
                    "Separation sweep (no contamination)" if sweep == "separation"
                    else f"Random-points sweep (class_sep = {sub['class_sep'].iloc[0] if len(sub) else '—'})",
                    fontsize=10, color=INK,
                )
            if i == len(rows) - 1:
                ax.set_xlabel(xlabel, fontsize=9, color=INK_2)
            if j == 0:
                ax.set_ylabel(f"{row_label}\n{m['label']}", fontsize=9, color=INK_2)

    # Proxy handles at full opacity (the plotted marks are semi-transparent).
    series_map = INDEXES if args.view == "best" else MODELS
    handles = [
        plt.Line2D([], [], color=color, lw=2, marker="o", ms=8,
                   mec=SURFACE, mew=2)
        for _, color in series_map.values()
    ]
    labels = [label for label, _ in series_map.values()]
    if m["ref"] is not None:
        handles.append(plt.Line2D([], [], color=AXIS, lw=1))
        labels.append(m["ref_label"])
    fig.legend(
        handles, labels, loc="lower center", ncol=len(handles), frameon=False,
        fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.005),
    )
    fig.suptitle(title, fontsize=12, color=INK, y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 1))

    out = args.out or Path(f"benchmark_results/benchmark_{args.view}_{args.metric}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()

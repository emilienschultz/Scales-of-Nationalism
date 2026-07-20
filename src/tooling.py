"""Reporting helpers built on top of the clustering pipeline output.

`model_comparison_table` turns `process_dataset(...)` output into the Table 2
layout: per validity index, the best and second-best distance-based
configurations (k-means / AHC / HDBSCAN) plus the best LCA model, shown apart.

Selection mirrors `5_clustering_results_replic_830.ipynb`:
- k-means / AHC / latent are restricted to the **gap-statistic-selected**
  candidates (`candidate_models` rows where `<index>_gap == 1`);
- HDBSCAN (no gap statistic) is taken from `all_models` directly;
- the distance pool keeps **one best row per algorithm** (the notebook holds
  exactly one k-means / AHC / HDBSCAN row each), so best and second-best are
  always different algorithms.

Sorting raw `all_models` by the CVI instead — as an earlier version did —
breaks the LCA row in particular: the raw silhouette/CH/etc. over latent models
with k=1..16 picks whichever k maximises the index (usually a degenerate
2-class solution), not the gap-selected k the working paper reports.
"""

import pandas as pd

# Internal model name -> display label
ALGO_NAMES = {
    "HDBSCAN": "HDBSCAN",
    "kmeans": "K-means",
    "AHC": "AHC",
    "latent": "LCA",
}

# (column, display label, ascending?) -- ascending=True means a lower score is better
INDEX_SPEC = [
    ("silhouette", "Silhouette (SL)", False),
    ("calinski_harabasz", "Calinski-Harab. (CH)", False),
    ("davies_bouldin", "Davies-Bouldin (DB)", True),
    ("dunn", "Generalized Dunn (GD43)", False),
]


def model_comparison_table(
    all_models, candidate_models, n_samples, unbalanced_ratio=0.9, decimal=","
):
    """Build the Table 2 model-comparison frame from `process_dataset` output.

    For each validity index, returns the best and second-best distance-based
    configurations (k-means / AHC / HDBSCAN) and the best LCA model, shown
    separately. Selection matches `5_clustering_results_replic_830.ipynb`:
    k-means / AHC / latent are taken only from the gap-statistic-selected
    candidates (`candidate_models` rows flagged `<index>_gap == 1`), while
    HDBSCAN comes from `all_models` (it has no gap statistic).

    A "configuration" is one (algorithm, n_clust) pair; among duplicate
    parameter grids that yield the same (algorithm, n_clust) the best-scoring
    one is kept. A `*` flags unbalanced solutions where the largest cluster
    holds at least `unbalanced_ratio` of all individuals.

    Parameters
    ----------
    all_models : pd.DataFrame
        The `all_models` frame returned by `process_dataset` (used for HDBSCAN).
    candidate_models : pd.DataFrame
        The `candidate_models` frame returned by `process_dataset` — the
        gap-selected k-means / AHC / latent models, with `<index>_gap` flags.
    n_samples : int
        Number of observations (used for the unbalanced-cluster flag).
    unbalanced_ratio : float
        Largest-cluster share above which a model is flagged with `*`.
    decimal : str
        Decimal separator for the formatted scores (',' for French style).

    Returns
    -------
    pd.DataFrame with columns Section / Clustering Validity Index / Algorithm /
    Clusters Nb / Score.
    """
    candidate_models = candidate_models.copy()
    candidate_models["unbalanced"] = (
        candidate_models["max_clust_size"] >= unbalanced_ratio * n_samples
    )
    hdbscan = all_models[all_models["model"] == "HDBSCAN"].copy()
    hdbscan["unbalanced"] = hdbscan["max_clust_size"] >= unbalanced_ratio * n_samples

    def fmt_n(row):
        return f"{int(row['n_clust'])}{'*' if row['unbalanced'] else ''}"

    def fmt_score(v):
        return f"{v:.2f}".replace(".", decimal)

    def ranked(sub, col, asc, by=("model", "n_clust")):
        return (
            sub.dropna(subset=[col])
            .sort_values(col, ascending=asc)
            .drop_duplicates(subset=list(by))
            .reset_index(drop=True)
        )

    # Per index, build the gap-selected distance pool (k-means / AHC + HDBSCAN)
    # and the gap-selected LCA pool, then rank each by the index.
    #
    # The distance pool is deduplicated by `model` (one best row per algorithm)
    # so the best and second-best configurations are necessarily *different*
    # algorithms — mirroring `5_clustering_results_replic_830.ipynb`, which
    # keeps exactly one k-means / AHC / HDBSCAN row each. Deduplicating by
    # (model, n_clust) instead lets one algorithm — typically HDBSCAN, with its
    # many degenerate small-k splits — occupy both slots and push the others out.
    dist_by_index = {}
    lca_by_index = {}
    for col, _, asc in INDEX_SPEC:
        gap_sel = candidate_models[candidate_models[f"{col}_gap"] == 1]
        dist_pool = pd.concat(
            [gap_sel[gap_sel["model"].isin(["kmeans", "AHC"])], hdbscan],
            ignore_index=True,
        )
        dist_by_index[col] = ranked(dist_pool, col, asc, by=["model"])
        lca_by_index[col] = ranked(gap_sel[gap_sel["model"] == "latent"], col, asc)

    rows = []
    sections = [
        ("Best configurations", dist_by_index, 0),
        ("Second-best configurations", dist_by_index, 1),
        ("Best LCA models", lca_by_index, 0),
    ]
    for section, pool, rank in sections:
        for col, label, _ in INDEX_SPEC:
            source = pool[col]
            if len(source) <= rank:
                continue
            r = source.iloc[rank]
            rows.append(
                {
                    "Section": section,
                    "Clustering Validity Index": label,
                    "Algorithm": ALGO_NAMES[r["model"]],
                    "Clusters Nb": fmt_n(r),
                    "Score": fmt_score(r[col]),
                }
            )
    return pd.DataFrame(rows)

"""Partition-similarity (average AMI) step run after the gap-statistic selection,
and the best-fits grid (`best_fits_nmi`, pairwise NMI of the top fits per index
and the true classes; see its docstring — the conventions below describe
`partition_similarity`).

The comparison table reports the best partition per clustering validity index,
but says nothing about whether those "best" partitions agree with each other.
`partition_similarity` measures that agreement: it pools the per-CVI best
distance-based solutions (through the shared `selected_pools` helper, so the
pool matches the table and the app dropdown), computes the pairwise Adjusted
Mutual Information between their stored label vectors, and scores every
combination of `n_combo` partitions by its average pairwise AMI. The
highest-scoring combination identifies the most homogeneous subset of
indicators.

Conventions:
- AMI rather than NMI: chance-corrected, so partitions with many clusters get
  no accidental-overlap advantage. 1 = identical, ~0 = chance-level agreement,
  slightly negative = worse than chance;
- AMI is computed on the stored `pred_clust` vectors (no re-fit), so it scores
  exactly the partitions the CVIs were computed on;
- HDBSCAN noise points (label -1) are kept and count as one extra cluster;
- deduplication operates on the partitions themselves, not the configurations:
  two parameter settings that converge to the same labeling (up to cluster
  renumbering) are merged into one pool entry, so a duplicated partition
  cannot inflate the average AMI with 1.0 self-agreement pairs. Merged
  configurations are recorded in the entry's `duplicates` field;
- each combination reports both the mean and the minimum of its pairwise
  AMIs: the minimum is the weakest link, which a mean over three pairs can
  otherwise hide.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score

from src.model_eval import ch_score, db_score, dunn_score, sil_score
from src.tooling import (
    ALGO_NAMES,
    INDEX_SPEC,
    MAX_NOISE,
    MIN_CLUSTER_SHARE,
    eligible_fits,
    selected_pools,
)


def _lookup_labels(all_models, model, params, n_clust):
    """Recover the stored label vector of one (model, params, n_clust) fit."""
    row_id = (
        (all_models["model"] == model)
        & (all_models["params"] == params)
        & (all_models["n_clust"] == n_clust)
    )
    matches = all_models.loc[row_id, "pred_clust"]
    if matches.empty:
        raise ValueError(f"no stored fit for {model} {params} n={n_clust}")
    return np.asarray(matches.iloc[0])


def _canonical(labels):
    """Renumber `labels` by order of first appearance.

    Two label vectors describe the same partition iff their canonical forms
    are equal; -1 (HDBSCAN noise) is renumbered like any other label, matching
    the noise-is-one-extra-cluster convention used for the AMI itself.
    """
    order = {}
    for v in labels:
        if v not in order:
            order[v] = len(order)
    return tuple(order[v] for v in labels)


def partition_similarity(
    all_models,
    candidate_models,
    n_combo=3,
    max_noise=MAX_NOISE,
    min_share=MIN_CLUSTER_SHARE,
):
    """Average-AMI homogeneity of the per-CVI best distance-based partitions.

    Pools the rank-0 distance solution of each validity index (via
    `selected_pools`, the same selection the comparison table and the app
    dropdown use), merges entries whose stored labelings are identical up to
    cluster renumbering (whether picked by several indices or reached from
    different parameter settings), then scores every combination of `n_combo`
    distinct partitions by its mean and minimum pairwise AMI (chance-corrected
    mutual information). When fewer than `n_combo` distinct partitions exist,
    the single combination of the whole pool is scored instead and flagged as
    partial.

    Parameters
    ----------
    all_models, candidate_models : pd.DataFrame
        The frames returned by `process_dataset` (labels are read from
        `all_models['pred_clust']`, so no re-fit is needed).
    n_combo : int
        Number of partitions per combination.
    max_noise : float or None
        HDBSCAN noise cap in percent, passed to `selected_pools`.
    min_share : float or None
        Smallest admissible cluster as a share of the sample, passed to
        `selected_pools`.

    Returns
    -------
    dict with keys:
    - 'pool': one row per distinct partition (label, indices, model, params,
      n_clust, duplicates — the labels of same-partition configurations
      merged into the row);
    - 'pairwise_ami': symmetric DataFrame of pairwise AMIs, indexed by the
      pool labels;
    - 'combos': one row per combination (members, indices, ami, ami_min),
      best-first by mean then minimum pairwise AMI;
    - 'best': the top row of 'combos' as a dict, or None if the pool is empty;
    - 'n_combo', 'partial': the requested size and whether the pool was too
      small to honour it.
    """
    dist_by_index, _ = selected_pools(
        all_models, candidate_models, max_noise=max_noise, min_share=min_share
    )

    # Best distance solution per index, deduplicated across indices.
    pool = {}
    for col, label, _ in INDEX_SPEC:
        ranked = dist_by_index[col]
        if ranked.empty:
            continue
        r = ranked.iloc[0]
        key = (r["model"], str(r["params"]), int(r["n_clust"]))
        if key not in pool:
            pool[key] = {
                "label": f"{ALGO_NAMES[r['model']]} (n={int(r['n_clust'])})",
                "indices": [],
                "model": r["model"],
                "params": r["params"],
                "n_clust": int(r["n_clust"]),
            }
        pool[key]["indices"].append(label)

    # Merge configurations that produced the same partition (identical
    # labelings up to cluster renumbering): keeping them apart would seed the
    # combinations with pairwise-AMI-1.0 pairs and inflate the average.
    merged = {}
    for p in pool.values():
        p["pred"] = _lookup_labels(all_models, p["model"], p["params"], p["n_clust"])
        key = _canonical(p["pred"])
        if key in merged:
            kept = merged[key]
            kept["indices"].extend(
                idx for idx in p["indices"] if idx not in kept["indices"]
            )
            kept["duplicates"].append(f"{p['label']} with {p['params']}")
        else:
            p["duplicates"] = []
            merged[key] = p
    partitions = list(merged.values())

    def pool_frame():
        return pd.DataFrame(
            [{k: v for k, v in p.items() if k != "pred"} for p in partitions]
        )

    empty = {
        "pool": pool_frame(),
        "pairwise_ami": pd.DataFrame(),
        "combos": pd.DataFrame(columns=["members", "indices", "ami", "ami_min"]),
        "best": None,
        "n_combo": n_combo,
        "partial": len(partitions) < n_combo,
    }
    if len(partitions) < 2:
        return empty

    # Duplicate labels can happen (same algorithm and n, different partitions);
    # disambiguate so the AMI matrix index stays unique.
    seen = {}
    for p in partitions:
        seen[p["label"]] = seen.get(p["label"], 0) + 1
        if seen[p["label"]] > 1:
            p["label"] = f"{p['label']} #{seen[p['label']]}"

    labels_vec = [p["pred"] for p in partitions]
    names = [p["label"] for p in partitions]

    ami = pd.DataFrame(np.eye(len(partitions)), index=names, columns=names)
    for i, j in combinations(range(len(partitions)), 2):
        v = adjusted_mutual_info_score(labels_vec[i], labels_vec[j])
        ami.iloc[i, j] = ami.iloc[j, i] = v

    size = min(n_combo, len(partitions))
    rows = []
    for combo in combinations(range(len(partitions)), size):
        pairs = [ami.iloc[i, j] for i, j in combinations(combo, 2)]
        rows.append(
            {
                "members": " + ".join(names[i] for i in combo),
                "indices": " + ".join(
                    idx for i in combo for idx in partitions[i]["indices"]
                ),
                "ami": float(np.mean(pairs)),
                "ami_min": float(np.min(pairs)),
            }
        )
    combos = (
        pd.DataFrame(rows)
        .sort_values(["ami", "ami_min"], ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    return {
        "pool": pool_frame(),
        "pairwise_ami": ami,
        "combos": combos,
        "best": combos.iloc[0].to_dict(),
        "n_combo": n_combo,
        "partial": len(partitions) < n_combo,
    }


# Short index codes for compact fit labels.
INDEX_SHORT = {
    "silhouette": "SL",
    "calinski_harabasz": "CH",
    "davies_bouldin": "DB",
    "dunn": "GD43",
}
TRUTH_LABEL = "True classes"


_INDEX_FUNCS = {
    "silhouette": sil_score,
    "calinski_harabasz": ch_score,
    "davies_bouldin": db_score,
    "dunn": dunn_score,
}


def _rescore_latent(pools, all_models, data_n):
    """Re-rank `eligible_fits` pools with LCA scored on the standardized data.

    The pipeline computes the validity indices of categorical LCA fits on the
    raw data it was fitted on, and those of k-means / AHC / HDBSCAN on the
    standardized data. The values are not on the same scale (raw-data values
    are typically higher), so a common ranking would favour LCA. Here each
    latent row's index value is recomputed from its stored labels on
    `data_n`, the matrix the other algorithms were scored on; the gap
    selection of the LCA fits themselves is left untouched.
    """
    data_n = np.asarray(data_n, dtype=float)
    cache = {}
    out = {}
    for col, _, asc in INDEX_SPEC:
        pool = pools[col].copy()
        for i in pool.index[pool["model"] == "latent"]:
            r = pool.loc[i]
            key = (str(r["params"]), int(r["n_clust"]), col)
            if key not in cache:
                pred = _lookup_labels(all_models, "latent", r["params"], int(r["n_clust"]))
                cache[key] = float(_INDEX_FUNCS[col](data_n, pred))
            pool.loc[i, col] = cache[key]
        out[col] = (
            pool.dropna(subset=[col])
            .sort_values(col, ascending=asc, kind="stable")
            .reset_index(drop=True)
        )
    return out


def best_fits_nmi(
    all_models,
    candidate_models,
    y_true=None,
    n_best=4,
    max_noise=MAX_NOISE,
    data_n=None,
    min_share=MIN_CLUSTER_SHARE,
):
    """Pairwise NMI between the `n_best` best fits of each validity index.

    For every validity index, ranks all eligible fits together, whatever the
    algorithm (`eligible_fits`: gap-selected k-means / AHC / LCA and HDBSCAN
    under the `max_noise` cap, the same eligibility as `selected_pools`), and keeps
    the `n_best` best *distinct partitions*: up to 4 indices x 4 = 16 fits.
    Many parameter settings converge to the same labeling (up to cluster
    renumbering) and tie on the index; they count as one fit, represented by
    the first setting in rank order, with the number of settings merged in
    `n_configs` and the algorithms that reach it in `algorithms`. Without this the top of every index would be one partition
    repeated. When `y_true` is given, the true partition is added as a first
    entry, so the matrix also shows how close each fit is to the ground truth.

    Partitions are only merged within an index: the grid is read index by
    index, so a partition ranked by two indices appears twice (with a pairwise
    NMI of 1). Conventions as in `partition_similarity`: stored `pred_clust`
    vectors (no re-fit), noise (-1, in the fits or in `y_true`) counted as one
    extra cluster. An index with fewer than `n_best` distinct eligible
    partitions contributes fewer fits.

    NMI (arithmetic normalization, sklearn default) runs from 0 to 1 and,
    unlike the AMI used by `partition_similarity`, is not corrected for
    chance: unrelated partitions score above 0, the more so the more clusters
    they have.

    `data_n` is the standardized data the distance-based models were scored
    on. When given, LCA fits are ranked on index values recomputed on it from
    their stored labels (`_rescore_latent`), so the four algorithms compete on
    the same scale; without it LCA keeps the pipeline values, computed on the
    raw categorical data and not comparable with the others.

    Returns
    -------
    dict with keys:
    - 'fits': one row per fit (label, index, index_key, rank, model, params,
      n_clust, score, noise — the share (%) of points labelled -1 —, n_configs, algorithms, and `nmi_truth` when `y_true` is given, plus
      `nmi_truth_classes` — the same NMI restricted to the points that have a
      true class — when `y_true` contains noise points), in grid order;
    - 'nmi': symmetric DataFrame of pairwise NMIs indexed by the labels, with
      `TRUTH_LABEL` first when `y_true` is given.
    """
    pools = eligible_fits(
        all_models, candidate_models, max_noise=max_noise, min_share=min_share
    )
    if data_n is not None:
        pools = _rescore_latent(pools, all_models, data_n)

    fits, preds = [], []
    for col, index_label, _ in INDEX_SPEC:
        kept = {}  # canonical partition -> fit entry, in rank order
        for _, r in pools[col].iterrows():
            n_clust = int(r["n_clust"])
            pred = _lookup_labels(all_models, r["model"], r["params"], n_clust)
            key = _canonical(pred)
            if key in kept:
                kept[key]["n_configs"] += 1
                if ALGO_NAMES[r["model"]] not in kept[key]["algorithms"]:
                    kept[key]["algorithms"].append(ALGO_NAMES[r["model"]])
                continue
            if len(kept) == n_best:
                continue  # keep scanning only to complete the `n_configs` counts
            rank = len(kept) + 1
            kept[key] = {
                "label": (
                    f"{INDEX_SHORT[col]} #{rank} · "
                    f"{ALGO_NAMES[r['model']]} (n={n_clust})"
                ),
                "index": index_label,
                "index_key": col,
                "rank": rank,
                "model": r["model"],
                "params": r["params"],
                "n_clust": n_clust,
                "score": float(r[col]),
                "noise": 100 * float(np.mean(pred == -1)),
                "n_configs": 1,
                "algorithms": [ALGO_NAMES[r["model"]]],
            }
            preds.append(pred)
        fits.extend(kept.values())

    names = [f["label"] for f in fits]
    if y_true is not None:
        names = [TRUTH_LABEL] + names
        preds = [np.asarray(y_true)] + preds

    nmi = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, j in combinations(range(len(names)), 2):
        v = normalized_mutual_info_score(preds[i], preds[j])
        nmi.iloc[i, j] = nmi.iloc[j, i] = v

    fits = pd.DataFrame(
        fits,
        columns=[
            "label", "index", "index_key", "rank", "model", "params",
            "n_clust", "score", "noise", "n_configs", "algorithms",
        ],
    )
    if y_true is not None:
        fits["nmi_truth"] = nmi.loc[fits["label"], TRUTH_LABEL].to_numpy()
        # k-means / AHC / LCA cannot output noise, so with random points in
        # the scenario (y_true == -1) they cannot reach 1 on `nmi_truth` even
        # when every real class is recovered. Also score the fits on the
        # points that do have a true class.
        has_class = preds[0] != -1
        if not has_class.all() and has_class.any():
            fits["nmi_truth_classes"] = [
                normalized_mutual_info_score(preds[0][has_class], p[has_class])
                for p in preds[1:]
            ]
    return {"fits": fits, "nmi": nmi}

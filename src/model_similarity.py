"""Partition-similarity (average AMI) step run after the gap-statistic selection.

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
- HDBSCAN noise points (label -1) are kept and count as one extra cluster.
"""

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score

from src.tooling import ALGO_NAMES, INDEX_SPEC, selected_pools


def _lookup_labels(all_models, model, params, n_clust):
    """Recover the stored label vector of one (model, params, n_clust) fit."""
    row_id = (
        (all_models["model"] == model)
        & (all_models["params"] == params)
        & (all_models["n_clust"] == n_clust)
    )
    return np.asarray(all_models.loc[row_id, "pred_clust"].iloc[0])


def partition_similarity(all_models, candidate_models, n_combo=3):
    """Average-AMI homogeneity of the per-CVI best distance-based partitions.

    Pools the rank-0 distance solution of each validity index (via
    `selected_pools`, the same selection the comparison table and the app
    dropdown use), deduplicates partitions picked by several indices, then
    scores every combination of `n_combo` partitions by its mean pairwise
    AMI (chance-corrected mutual information). When fewer than `n_combo`
    distinct partitions exist, the single combination of the whole pool is
    scored instead and flagged as partial.

    Parameters
    ----------
    all_models, candidate_models : pd.DataFrame
        The frames returned by `process_dataset` (labels are read from
        `all_models['pred_clust']`, so no re-fit is needed).
    n_combo : int
        Number of partitions per combination.

    Returns
    -------
    dict with keys:
    - 'pool': one row per distinct partition (label, indices, model, params,
      n_clust);
    - 'pairwise_ami': symmetric DataFrame of pairwise AMIs, indexed by the
      pool labels;
    - 'combos': one row per combination (members, indices, ami), best-first;
    - 'best': the top row of 'combos' as a dict, or None if the pool is empty;
    - 'n_combo', 'partial': the requested size and whether the pool was too
      small to honour it.
    """
    dist_by_index, _ = selected_pools(all_models, candidate_models)

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

    partitions = list(pool.values())
    empty = {
        "pool": pd.DataFrame(partitions),
        "pairwise_ami": pd.DataFrame(),
        "combos": pd.DataFrame(columns=["members", "indices", "ami"]),
        "best": None,
        "n_combo": n_combo,
        "partial": len(partitions) < n_combo,
    }
    if len(partitions) < 2:
        return empty

    # Duplicate labels can happen (same algorithm and n from different params);
    # disambiguate so the AMI matrix index stays unique.
    seen = {}
    for p in partitions:
        seen[p["label"]] = seen.get(p["label"], 0) + 1
        if seen[p["label"]] > 1:
            p["label"] = f"{p['label']} #{seen[p['label']]}"

    labels_vec = [
        _lookup_labels(all_models, p["model"], p["params"], p["n_clust"])
        for p in partitions
    ]
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
            }
        )
    combos = (
        pd.DataFrame(rows).sort_values("ami", ascending=False).reset_index(drop=True)
    )

    return {
        "pool": pd.DataFrame(partitions),
        "pairwise_ami": ami,
        "combos": combos,
        "best": combos.iloc[0].to_dict(),
        "n_combo": n_combo,
        "partial": len(partitions) < n_combo,
    }

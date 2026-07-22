"""Run the full clustering robustness pipeline on a single dataframe.

Mirrors the procedure in `4_clustering.ipynb`: fits latent (StepMix), k-means,
AHC and HDBSCAN models across their parameter grids, computes the gap
statistic for latent / k-means / AHC to identify the optimal number of
clusters per (model, params) combination, and returns the resulting metrics
tables. Intended to test framework robustness on synthetic datasets produced
by `Generate_clusters.ipynb`.

Usage:
    from process_dataset import process_dataset
    result = process_dataset(df)             # df: pd.DataFrame
    result['all_models']                     # every fit + CVIs
    result['candidate_models']               # gap-selected (model, params, n_clust)
    result['gap_values']                     # per-k gap statistics

Defaults mirror `4_clustering.ipynb` so the two can be compared directly:
`msrt='categorical'` with `subtract_one=True` (reproducing the notebook's
`data2004[var_list_n] - 1`), `standardize=True`, `run_hdbscan=True`. Override
these for lighter or synthetic (Gaussian) runs.

Differences from `4_clustering.ipynb`
--------------------------------------
Kept here so the divergence from the notebook can be tracked over time.

Structural (by design — this runs on an arbitrary numeric frame, not the GSS):
- No data prep / I/O. Takes a DataFrame in, returns dicts out (the notebook
  reads `data2004_*.parquet` and writes `output/models/*.csv`).
- Covariates and sample weights are not wired in (`covar='without'`,
  `controls=None`). This matches the notebook's gap-statistic path, which
  also ignores them.
"""

from itertools import product

import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from src.model_fit import do_AHC, do_hdbscan, do_kmeans, do_StepMix
from src.model_select import bootstrap_gap, compute_gap, get_gap
from src.model_similarity import partition_similarity

CVI = ["silhouette", "calinski_harabasz", "davies_bouldin", "dunn"]


def process_dataset(
    data,
    max_clust=16,
    gap_iters=500,
    n_jobs=8,
    msrt="categorical",
    standardize=True,
    run_hdbscan=True,
    subtract_one=True,
    n_combo=3,
    verbose=True,
):
    """Run the clustering robustness pipeline on a single dataframe.

    Parameters
    ----------
    data : pd.DataFrame
        One row per observation, one column per feature.
    max_clust : int
        Upper bound of the cluster-count sweep for latent / k-means / AHC.
    gap_iters : int
        Bootstrap iterations for the gap statistic. Set lower (e.g. 50) for
        quick robustness checks; the headline replication uses 500.
    n_jobs : int
        Workers passed to `joblib.Parallel`.
    msrt : str
        StepMix `measurement` ('categorical' or 'continuous'). Defaults to
        'categorical' to match the notebook; use 'continuous' for Gaussian
        features (e.g. raw `make_classification` output).
    standardize : bool
        Standardize features before fitting distance-based models (and
        continuous latent models).
    run_hdbscan : bool
        Include the HDBSCAN parameter sweep (5 x 14 x 15 = 1050 fits).
    subtract_one : bool
        When `msrt='categorical'`, subtract 1 from the data so columns start at
        0, as StepMix expects (reproducing the notebook's
        `data2004[var_list_n] - 1`). Set False if the data is already 0-indexed.
        Ignored when `msrt='continuous'`.
    n_combo : int
        Size of the partition combinations scored by average pairwise AMI in
        the similarity step (see `src.model_similarity.partition_similarity`).
    verbose : bool
        Display tqdm progress bars.

    Returns
    -------
    dict with keys 'all_models', 'candidate_models', 'gap_values', 'similarity'.
    """
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    if standardize:
        scaler = StandardScaler()
        data_n = pd.DataFrame(
            scaler.fit_transform(data), columns=data.columns, index=data.index
        )
    else:
        data_n = data.copy()

    # Latent models eat 0-indexed integers when categorical, standardized
    # floats when continuous. `subtract_one` reproduces the notebook's
    # `data2004[var_list_n] - 1` so 1..K Likert columns start at 0.
    if msrt == "categorical":
        data_latent = data - 1 if subtract_one else data
    else:
        data_latent = data_n

    # ---------- 1. Fit models across parameter grids ----------

    # 1.1 Latent
    latent_params = [(msrt, "without")]
    latent_grid = list(product(range(1, max_clust + 1), latent_params))
    latent_results = Parallel(n_jobs=n_jobs)(
        delayed(do_StepMix)(data_latent, None, n, m, c)
        for n, (m, c) in tqdm(
            latent_grid, desc="Fitting latent models", disable=not verbose
        )
    )
    latent_all = pd.DataFrame(latent_results)
    latent_all["params"] = latent_all["params"].apply(
        lambda d: {k: v for k, v in d.items() if k not in ["NAs", "wgt"]}
    )

    # 1.2 k-means
    kmeans_params = list(
        product(["euclidean", "manhattan", "chebyshev"], ["mean", "median", "medoid"])
    )
    kmeans_grid = list(product(range(2, max_clust + 1), kmeans_params))
    kmeans_results = Parallel(n_jobs=n_jobs)(
        delayed(do_kmeans)(data_n, n, dist, link)
        for n, (dist, link) in tqdm(
            kmeans_grid, desc="Fitting KMeans models", disable=not verbose
        )
    )
    kmeans_all = pd.DataFrame(kmeans_results)

    # 1.3 AHC
    ahc_params = [
        *product(
            ["manhattan", "euclidean", "chebyshev", "hamming"],
            ["single", "average", "complete"],
        ),
        ("euclidean", "ward"),
    ]
    ahc_grid = list(product(range(1, max_clust + 1), ahc_params))
    ahc_results = Parallel(n_jobs=n_jobs)(
        delayed(do_AHC)(data_n, n, dist, link)
        for n, (dist, link) in tqdm(
            ahc_grid, desc="Fitting AHC models", disable=not verbose
        )
    )
    ahc_all = pd.DataFrame(ahc_results)

    # 1.4 HDBSCAN (no gap statistic — algorithm picks n itself)
    if run_hdbscan:
        hdb_params = list(
            product(
                ["manhattan", "euclidean", "chebyshev", "mahalanobis", "hamming"],
                range(2, 16),
                range(1, 16),
            )
        )
        hdbscan_results = Parallel(n_jobs=n_jobs)(
            delayed(do_hdbscan)(data_n, dist, mc, ms)
            for dist, mc, ms in tqdm(
                hdb_params, desc="Fitting HDBSCAN models", disable=not verbose
            )
        )
        hdbscan_all = pd.DataFrame(hdbscan_results)
    else:
        hdbscan_all = pd.DataFrame()

    all_models = pd.concat([latent_all, kmeans_all, ahc_all, hdbscan_all]).reset_index(
        drop=True
    )

    # ---------- 2. Gap statistic for latent / k-means / AHC ----------

    gap_models = pd.concat([latent_all, kmeans_all, ahc_all]).reset_index(drop=True)

    params = {"kmeans": kmeans_params, "AHC": ahc_params, "latent": latent_params}
    param_names = {
        "kmeans": ["dist", "link"],
        "AHC": ["dist", "link"],
        "latent": ["msrt", "covar"],
    }
    models = ["kmeans", "AHC", "latent"]

    bootstrap_grid = [
        (model, dict(zip(param_names[model], vals)), n_val, n_iter)
        for model in models
        for vals in params[model]
        for n_val in (
            range(1, max_clust + 1) if model == "latent" else range(2, max_clust + 1)
        )
        for n_iter in range(gap_iters)
    ]

    model_grid = [
        (model, dict(zip(param_names[model], vals)))
        for model in models
        for vals in params[model]
    ]

    boot_results = Parallel(n_jobs=n_jobs)(
        delayed(bootstrap_gap)(
            data=data_latent if model == "latent" else data_n,
            controls=None,
            n=n_val,
            model=model,
            params=config,
            iter_num=n_iter,
        )
        for model, config, n_val, n_iter in tqdm(
            bootstrap_grid, desc="Bootstrapping CVIs", disable=not verbose
        )
    )
    bootstrap_results = pd.concat(boot_results).reset_index(drop=True)
    # The gap statistic only consumes the CVI columns; carrying one label
    # vector per bootstrap fit (grid size x gap_iters rows) wastes memory.
    bootstrap_results = bootstrap_results.drop(columns=["pred_clust"])
    bootstrap_results["params"] = bootstrap_results["params"].apply(
        lambda d: {k: v for k, v in d.items() if k not in ["NAs", "wgt"]}
    )

    gap_values = []
    for model, config in model_grid:
        rows_id = (bootstrap_results["model"] == model) & (
            bootstrap_results["params"] == config
        )
        gap_values.append(
            compute_gap(bootstrap_results[rows_id], gap_models, model, config, CVI)
        )
    gap_values = pd.concat(gap_values, ignore_index=True)

    # ---------- 3. Identify the optimal n per (model, params) ----------

    cols = ["model", "params", "n_clust"] + list(CVI) + [f"{idx}_gap" for idx in CVI]
    candidate_models = pd.DataFrame(columns=cols)
    candidate_models["model"] = candidate_models["model"].astype("object")
    candidate_models["params"] = candidate_models["params"].astype("object")

    for model, config in model_grid:
        for index in CVI:
            best_n = get_gap(gap_values, model, config, index)
            if best_n == "none":
                continue

            row_id = (
                (candidate_models["model"] == model)
                & (candidate_models["params"] == config)
                & (candidate_models["n_clust"] == best_n)
            )

            if candidate_models[row_id].empty:
                model_id = (
                    (gap_models["model"] == model)
                    & (gap_models["params"] == config)
                    & (gap_models["n_clust"] == best_n)
                )
                new_row = {
                    "model": model,
                    "params": config,
                    "n_clust": best_n,
                    "min_clust_size": gap_models.loc[model_id, "min_clust_size"].values[
                        0
                    ],
                    "max_clust_size": gap_models.loc[model_id, "max_clust_size"].values[
                        0
                    ],
                    "silhouette": gap_models.loc[model_id, "silhouette"].values[0],
                    "calinski_harabasz": gap_models.loc[
                        model_id, "calinski_harabasz"
                    ].values[0],
                    "davies_bouldin": gap_models.loc[model_id, "davies_bouldin"].values[
                        0
                    ],
                    "dunn": gap_models.loc[model_id, "dunn"].values[0],
                    f"{index}_gap": 1,
                }
                candidate_models = pd.concat(
                    [candidate_models, pd.DataFrame([new_row])], ignore_index=True
                )
            else:
                candidate_models.loc[row_id, f"{index}_gap"] = 1

    # ---------- 4. AMI similarity of the per-CVI best distance partitions ----------

    similarity = partition_similarity(all_models, candidate_models, n_combo=n_combo)

    return {
        "all_models": all_models,
        "candidate_models": candidate_models,
        "gap_values": gap_values,
        "similarity": similarity,
    }

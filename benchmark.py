"""Benchmark: stability of the best clustering solution under degradation.

Two sweeps over synthetic scenarios (4 true classes, 23 features, all
informative), reusing `process_dataset` for the full robustness pipeline:

- **separation**: class_sep degrades from easy to (almost) no structure, with
  no contamination;
- **random**: the proportion of points replaced by uniform background noise
  (no true class, y_true = -1) grows from 0 to 0.5 at the baseline separation.
  (An earlier flip_y sweep was dropped: make_classification's flip_y only
  reassigns labels, the features are untouched, so clustering never sees it.)

For each scenario and each clustering validity index, the best distance-based
solution and the best LCA solution (same gap-statistic selection as the app,
via `selected_pools`) are looked up by their stored labels (`pred_clust` in
`all_models` — the exact partition the scores describe), and the number of
identified clusters is counted **excluding singleton clusters** (< 2 members)
and HDBSCAN noise. The question: is the recovered k stable near the baseline,
and where does it diverge as structure degrades?

Every scenario run is pickled to `--out-dir` keyed by its config hash (n_jobs
excluded), so re-runs are free and partial sweeps can be resumed. A flat
`summary.csv` is (re)built from all cached runs at the end of every invocation.

Usage
-----
    python benchmark.py --list                     # show the scenario grid
    python benchmark.py --n-jobs 16                # run everything locally
    python benchmark.py --task-id 3                # run scenario 3 only
    python benchmark.py --summarize-only           # rebuild summary.csv

HPC (SLURM job array, one scenario per task — n tasks = `--list` count):

    #!/bin/bash
    #SBATCH --array=0-20
    #SBATCH --cpus-per-task=16
    #SBATCH --mem=8G
    export OMP_NUM_THREADS=1     # joblib workers own the parallelism
    python benchmark.py --task-id $SLURM_ARRAY_TASK_ID

(`--task-id` falls back to $SLURM_ARRAY_TASK_ID and `--n-jobs` to
$SLURM_CPUS_PER_TASK automatically, so the two flags can be omitted.)

Then aggregate and plot:

    python benchmark.py --summarize-only
    python plot_benchmark.py
"""

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from process_dataset import process_dataset
from src.app_viz import refit_labels
from src.data_gen import SimConfig, generate_clusters
from src.tooling import INDEX_SPEC, selected_pools

# --------------------------- scenario grid ---------------------------

# Fixed data design requested for this benchmark: 4 true classes, 23 features,
# all informative (no redundant columns), Likert-binned like the replic_830 app.
# flip_y stays at 0 everywhere: it only relabels points, the features (and so
# the clustering problem) are unchanged.
BASE_DATA_CFG = dict(
    n_samples=830,
    n_features=23,
    n_informative=23,
    n_redundant=0,
    n_classes=4,
    n_clusters_per_class=1,
    class_balance=None,
    flip_y=0.0,
    likert=True,
)

# Sweep 1: separation degrades, no contamination.
SEP_GRID = [10, 7, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1]
# Sweep 2: proportion of uniform random points grows at the baseline separation.
RANDOM_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
BASE_SEP = 5.0

# Clusters smaller than this do not count as "identified".
MIN_CLUSTER_SIZE = 2


def build_scenarios(seeds):
    """Full scenario list: one dict per (sweep point, seed)."""
    scenarios = []
    for seed in seeds:
        for sep in SEP_GRID:
            scenarios.append(
                dict(sweep="separation", class_sep=sep, noise_prop=0.0, seed=seed)
            )
        for prop in RANDOM_GRID:
            scenarios.append(
                dict(sweep="random", class_sep=BASE_SEP, noise_prop=prop, seed=seed)
            )
    return scenarios


def scenario_data_cfg(scn):
    return dict(
        BASE_DATA_CFG,
        class_sep=scn["class_sep"],
        noise_prop=scn["noise_prop"],
        random_state=scn["seed"],
    )


def make_pipe_cfg(args):
    return dict(
        max_clust=args.max_clust,
        gap_iters=args.gap_iters,
        msrt="categorical",
        standardize=True,
        run_hdbscan=True,
        subtract_one=True,
    )


def config_key(data_cfg, pipe_cfg):
    """Stable short hash of the config. n_jobs is kept out on purpose: it
    changes the schedule, not the result, so runs cache across machines."""
    blob = json.dumps({"data": data_cfg, "pipeline": pipe_cfg}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


# --------------------------- per-scenario summary ---------------------------


def effective_n_clusters(labels, min_size=MIN_CLUSTER_SIZE):
    """Clusters with at least `min_size` members; HDBSCAN noise (-1) excluded."""
    labels = np.asarray(labels)
    _, counts = np.unique(labels[labels >= 0], return_counts=True)
    return int((counts >= min_size).sum()), int((counts < min_size).sum())


def solution_labels(df, all_models, row, pipe_cfg):
    """Label vector of the scored fit behind a `selected_pools` row.

    The pipeline stores each fit's partition in `all_models['pred_clust']`
    (k-means and StepMix are unseeded, so a re-fit could land on a different
    local optimum than the one the recorded CVI scores describe). Look the
    winning (model, params, n_clust) up there; fall back to `refit_labels`
    only for payloads cached before `pred_clust` was persisted.
    """
    if "pred_clust" not in all_models.columns:
        return refit_labels(
            df,
            row["model"],
            row["params"],
            int(row["n_clust"]),
            standardize=pipe_cfg["standardize"],
            msrt=pipe_cfg["msrt"],
            subtract_one=pipe_cfg["subtract_one"],
        )
    hit = all_models[
        (all_models["model"] == row["model"])
        & (all_models["n_clust"] == row["n_clust"])
    ]
    hit = hit[hit["params"].apply(lambda p: p == row["params"])]
    return np.asarray(hit.iloc[0]["pred_clust"])


# Bump when the summary schema or logic changes: cached payloads carry their
# version, and stale summaries are recomputed from the stored labels (cheap —
# no refit) instead of being silently reused.
SUMMARY_VERSION = 2


def summarize_run(df, y_true, result, pipe_cfg):
    """One row per (validity index, algorithm), so models can be compared.

    The distance pool from `selected_pools` is already deduplicated to each
    algorithm's best gap-selected configuration (k-means / AHC / HDBSCAN),
    ranked best-first: rank 0 within the pool is the old "best solution". The
    LCA pool contributes its single best latent row. Labels come from the
    scored fit itself (see `solution_labels`), so counts and ARI describe
    exactly the partition behind each recorded score.
    """
    all_models = result["all_models"]
    dist_by_index, lca_by_index = selected_pools(all_models, result["candidate_models"])

    rows = []
    for col, label, _ in INDEX_SPEC:
        for pool_name, pool in [("distance", dist_by_index), ("LCA", lca_by_index)]:
            top = pool[col]
            if pool_name == "LCA":
                top = top.head(1)
            if len(top) == 0:
                rows.append(
                    dict(index=col, index_label=label, pool=pool_name,
                         rank=0, model=None)
                )
                continue
            for rank in range(len(top)):
                r = top.iloc[rank]
                labels_pred = solution_labels(df, all_models, r, pipe_cfg)
                n_eff, n_singleton = effective_n_clusters(labels_pred)
                rows.append(
                    dict(
                        index=col,
                        index_label=label,
                        pool=pool_name,
                        rank=rank,
                        model=r["model"],
                        params=str(r["params"]),
                        n_clust=int(r["n_clust"]),
                        n_clust_effective=n_eff,
                        n_singleton=n_singleton,
                        n_noise=int((labels_pred == -1).sum()),
                        score=float(r[col]),
                        ari=float(adjusted_rand_score(y_true, labels_pred)),
                    )
                )
    return rows


def ensure_summary(payload, path):
    """Recompute and persist the payload's summary if its version is stale."""
    if payload.get("summary_version") != SUMMARY_VERSION:
        payload["summary"] = summarize_run(
            payload["df"], payload["y_true"], payload["result"], payload["pipe_cfg"]
        )
        payload["summary_version"] = SUMMARY_VERSION
        with open(path, "wb") as fh:
            pickle.dump(payload, fh)
    return payload


# --------------------------- run / cache ---------------------------


def run_scenario(scn, pipe_cfg, out_dir, n_jobs):
    """Run one scenario (or load it from cache); returns its payload."""
    data_cfg = scenario_data_cfg(scn)
    key = config_key(data_cfg, pipe_cfg)
    path = out_dir / f"{key}.pkl"

    if path.exists():
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        payload = ensure_summary(payload, path)
        print(f"[cached] {key}  {scn}")
        return payload

    print(f"[run]    {key}  {scn}")
    t0 = time.time()
    df, y_true = generate_clusters(SimConfig(**data_cfg))
    result = process_dataset(df, verbose=False, n_jobs=n_jobs, **pipe_cfg)
    payload = dict(
        key=key,
        data_cfg=data_cfg,
        pipe_cfg=pipe_cfg,
        df=df,
        y_true=y_true,
        result=result,
        summary=summarize_run(df, y_true, result, pipe_cfg),
        summary_version=SUMMARY_VERSION,
        timestamp=time.time(),
    )
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    print(f"         done in {time.time() - t0:.0f}s -> {path}")
    return payload


def build_summary(scenarios, pipe_cfg, out_dir):
    """Aggregate the cached summaries of `scenarios` into one flat frame.

    Scenarios whose cache file is missing are skipped (useful while a job
    array is still filling the cache). A scenario shared by both sweeps
    (sep=5, flip=0) contributes one row per sweep.
    """
    rows, missing = [], 0
    for scn in scenarios:
        data_cfg = scenario_data_cfg(scn)
        path = out_dir / f"{config_key(data_cfg, pipe_cfg)}.pkl"
        if not path.exists():
            missing += 1
            continue
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        payload = ensure_summary(payload, path)
        for row in payload["summary"]:
            rows.append(dict(scn, key=payload["key"], **row))
    if missing:
        print(f"[summary] {missing}/{len(scenarios)} scenarios not computed yet")
    return pd.DataFrame(rows)


# --------------------------- CLI ---------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sweep", choices=["separation", "random", "all"], default="all")
    ap.add_argument(
        "--task-id",
        type=int,
        default=int(os.environ.get("SLURM_ARRAY_TASK_ID", -1)),
        help="Run a single scenario by index (for SLURM job arrays); "
        "-1 runs every scenario sequentially.",
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", -1)),
        help="joblib workers inside process_dataset (-1 = all cores).",
    )
    ap.add_argument("--max-clust", type=int, default=10)
    ap.add_argument(
        "--gap-iters",
        type=int,
        default=50,
        help="Gap-statistic bootstrap iterations (500 = headline replication).",
    )
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out-dir", type=Path, default=Path("benchmark_results"))
    ap.add_argument("--list", action="store_true", help="Print scenarios and exit.")
    ap.add_argument(
        "--summarize-only",
        action="store_true",
        help="Rebuild summary.csv from cached runs without computing anything.",
    )
    args = ap.parse_args()

    scenarios = build_scenarios(args.seeds)
    if args.sweep != "all":
        scenarios = [s for s in scenarios if s["sweep"] == args.sweep]
    pipe_cfg = make_pipe_cfg(args)

    if args.list:
        for i, scn in enumerate(scenarios):
            key = config_key(scenario_data_cfg(scn), pipe_cfg)
            print(f"{i:3d}  {key}  {scn}")
        print(f"\n{len(scenarios)} scenarios (SLURM: --array=0-{len(scenarios) - 1})")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.summarize_only:
        todo = scenarios if args.task_id < 0 else [scenarios[args.task_id]]
        for scn in todo:
            run_scenario(scn, pipe_cfg, args.out_dir, args.n_jobs)

    summary = build_summary(scenarios, pipe_cfg, args.out_dir)
    if summary.empty:
        print("[summary] nothing cached yet — summary.csv not written")
        return
    out_csv = args.out_dir / "summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"[summary] {len(summary)} rows -> {out_csv}")


if __name__ == "__main__":
    main()

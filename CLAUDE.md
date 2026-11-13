# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Clustering robustness study: fit many clustering models (LCA via StepMix, k-means variants, AHC, HDBSCAN) across parameter grids on a dataset, select the optimal cluster count per configuration with a bootstrap gap statistic, and compare the best solutions per clustering validity index (silhouette, Calinski-Harabasz, Davies-Bouldin, generalized Dunn). Work runs both on synthetic Likert-style data with known ground truth and on the real GSS-2004 data (`../data/data2004_830.parquet`, outside the repo; the `replic_830` baseline: 830 respondents × 23 Likert items — the `*_n` columns — 4 latent classes). The code replicates and stress-tests the procedure from the original notebooks `4_clustering.ipynb` / `5_clustering_results_replic_830.ipynb` (in this repo, they import the same `src/` modules; their saved outputs predate the gap-statistic fixes); docstrings track divergences from them.

## Commands

```bash
pip install -r requirements.txt   # no packaging; plain requirements file
streamlit run app.py              # interactive robustness explorer
python benchmark.py --list        # degradation benchmark (separation / noise sweeps)
python benchmark.py --task-id N   # one scenario (SLURM array); then plot_benchmark.py
```

There are no tests or linter configured.

Quick programmatic run (mirrors the notebook's "light" settings — full replication uses `max_clust=16`, `gap_iters=500`):

```python
from process_dataset import process_dataset
result = process_dataset(df, max_clust=10, gap_iters=50, run_hdbscan=False)
# result keys: 'all_models', 'candidate_models', 'gap_values', 'similarity'
```

## Architecture

Two entry points share the `src/` library:

- **`process_dataset.py`** — the pipeline. Four stages: (1) fit all models across parameter grids in parallel (joblib); (2) bootstrap the gap statistic for latent/k-means/AHC (HDBSCAN picks its own n, so it is skipped); (3) mark gap-selected `(model, params, n_clust)` rows as `candidate_models`; (4) score the similarity of the per-CVI best distance solutions with average pairwise AMI (chance-corrected mutual information, over every combination of `n_combo` partitions, default 3) into the `'similarity'` result key. Preprocessing is model-dependent and must stay consistent everywhere: distance-based models get standardized data (`data_n`); categorical latent models get raw data minus 1 (`subtract_one=True` reproduces the notebook's `data2004 - 1` so 1..5 Likert columns start at 0, as StepMix expects).
- **`app.py`** — Streamlit UI. Generates a synthetic scenario (`SimConfig` sidebar), runs the pipeline, shows PCA scatter plots, the model-comparison table, and a "Partition similarity" tab (highest-average-AMI combination of best partitions, all combination scores, pairwise AMI matrix; always recomputed from the stored labels so cached runs follow the current selection rule). Runs are pickled to `cache/<md5-of-config>.pkl`; identical configs reload instead of recomputing. Pipeline params are hardcoded in `PIPE_CFG` (light settings, HDBSCAN forced on) — edit there to change the treatment.
- **`benchmark.py` / `plot_benchmark.py`** — degradation benchmark: sweeps class separation (easy→no structure) and the proportion of uniform random points (0→0.5 at baseline separation) on a fixed design (4 classes, 23/23 informative features), records the gap-selected best distance / best LCA solution per CVI with its stored labels, and counts identified clusters excluding singletons (plus ARI vs truth; noise points carry y_true = -1). `make_classification`'s flip_y is deliberately not swept: it only relabels points, the features are unchanged. Runs cache to `benchmark_results/<hash>.pkl` (hash excludes `n_jobs`); designed for SLURM job arrays via `--task-id` with joblib parallelism inside each scenario.
- **`../clustering/Generate_clusters.ipynb`** (sibling repo, not here) — sandbox notebook for the synthetic data generator and end-to-end pipeline tests, including reproducing the real-data results. `../clustering/BUG_REPORT.md` documents the bugs found in the original gap statistic / CVI code.

`src/` modules:
- `model_fit.py` — one `do_*` function per algorithm (`do_StepMix`, `do_kmeans`, `do_AHC`, `do_hdbscan`); each returns a dict of fit metadata + CVIs (via `model_eval.get_metrics`).
- `model_select.py` — gap statistic: `bootstrap_gap` (fit on uniform-random data), `compute_gap`, `get_gap` (best n per index).
- `model_eval.py` — CVI computations, each wrapped to return NaN instead of raising when undefined.
- `tooling.py` — `model_comparison_table` / `selected_pools` build the "Table 2" report. Selection rule: k-means/AHC/latent are restricted to gap-selected candidates; HDBSCAN comes from `all_models` directly, restricted to fits with at most 50% noise (`MAX_NOISE`); the distance pool keeps one best row per algorithm. Do not rank raw `all_models` by CVI instead — that breaks the LCA row (picks a degenerate k rather than the gap-selected one; see the module docstring).
- `model_similarity.py` — `partition_similarity`: average-AMI homogeneity of the per-CVI best distance partitions (pool built through `selected_pools`; labels read from `all_models['pred_clust']`, never re-fit).
- `app_viz.py` — `pca_scatter` and `refit_labels` for the app. The app plots the stored `pred_clust` of the scored fit; `refit_labels` is only a fallback for runs cached before labels were stored, and must reproduce the pipeline's per-model preprocessing exactly.
- `data_gen.py` — `SimConfig` + `generate_clusters` (wraps `make_classification`, optional Likert binning).
- `hopkins.py`, `model_plot.py` — clusterability score and standalone plotting (notebook-side helpers).

## Consistency invariants

- The app's "Solution to visualize" dropdown (`best_solutions` in `app.py`), the comparison table (`model_comparison_table`), and the AMI similarity step (`partition_similarity`) must select from the same pools — all three go through `selected_pools`. Keep them in sync if selection logic changes.
- In the AMI computations HDBSCAN noise points (label -1) count as one extra cluster (unlike the CVIs, which exclude them); AMI runs on the stored `pred_clust` vectors so it scores exactly the partitions the CVIs were computed on. AMI is chance-corrected (~0 = chance-level agreement, can be slightly negative), unlike the NMI used in an earlier version.
- `app.py` caches keyed on the full data + pipeline config; any change to pipeline semantics that doesn't change the config dict will silently serve stale cached results — clear `cache/` when that happens.
- Silhouette uses `metric='manhattan'` and Dunn uses `cityblock`; keep metrics consistent with the notebooks when comparing results.

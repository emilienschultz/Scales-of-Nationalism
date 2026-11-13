# Documenting changes

State on 2026-11-12. Baseline = Maël Dieudonné's code, last commit `b6387fd` (2025-12-22).
My changes = 9 commits, `2424948` → `d05b85a` (20–28 July 2026). These are committed; the fixes of section 3 are not yet.

## 1. Changes to the existing code

Only three existing modules were touched (`src/hopkins.py`, `src/model_plot.py` and the original notebooks are unchanged).

### `src/model_select.py` — gap statistic (3 behavioural fixes)

| Change | Before | Now |
|---|---|---|
| Selection rule (`get_gap`) | `argmin` of the margin `gap(k) - gap(k+1) + s(k+1)`, which is biased towards large k | **Smallest k** satisfying `Gap(k) >= Gap(k+1) - s(k+1)` (Tibshirani et al. 2001) |
| Gap orientation (`compute_gap`) | `log(rand) - log(model)` for every index | Kept for Davies-Bouldin (lower is better); **flipped** for silhouette, CH and Dunn, so a larger gap always means more structure than the null |
| Null data (`bootstrap_gap`) | Uniform on `[min, max + 1)` for all models | `+1` only for categorical latent models; continuous data (k-means, AHC, continuous latent) uses `[min, max)` |

These three fixes change which `n_clust` is selected, so results differ from the original notebooks.

### `src/model_eval.py` — CVIs

- CVIs and min/max cluster sizes are now computed on **denoised** data (HDBSCAN `-1` points excluded). Before, the noise mask was computed but not used. k-means, AHC and latent are unaffected.
- `get_metrics` now stores the label vector `pred_clust` in each result row, so partitions are reused instead of re-fitted (fits are unseeded).
- `clust_size` returns NaN for an all-noise fit instead of raising.

### `src/model_fit.py`

- `do_StepMix`: `n` coerced to a Python `int` (StepMix rejects `numpy.int64`).

### `requirements.txt`

- Added `streamlit`.

## 2. New code built on top

| File | Role |
|---|---|
| `process_dataset.py` | The notebook procedure as one function: fit all models, bootstrap gap, flag gap-selected candidates, score partition similarity |
| `src/tooling.py` | `selected_pools` (single source for "best" solutions) and `model_comparison_table` ("Table 2") |
| `src/model_similarity.py` | Mean and minimum pairwise AMI between the per-CVI best distance partitions; identical partitions are merged before scoring (last commit) |
| `app.py`, `src/app_viz.py`, `src/data_gen.py` | Streamlit explorer on synthetic Likert data (PCA plots, comparison table, similarity tab) |
| `benchmark.py`, `plot_benchmark.py` | Degradation benchmark (class separation and noise sweeps), SLURM-ready |
| `1_prepare.ipynb`, `2_desc_PCA.ipynb` | Notebook versions added next to the existing `.Rmd` files |

## 3. Review of the changes (2026-11-12)

Each change was re-read and tested on a cached synthetic run (4 true classes) and on the real data (`../data/data2004_830.parquet`, light settings: `max_clust=10`, `gap_iters=50`).

| Change | Verdict | Evidence |
|---|---|---|
| Gap orientation + smallest-k rule | Relevant, kept | On the synthetic run, 35% of the 92 (configuration, index) selections find the true k=4, against 3% with the original code; each fix alone gives 4% and 11% |
| `+1` only for categorical latent | Relevant, kept | StepMix truncates floats, so `[min, max+1)` gives uniform categories; standardized data must not get the `+1` |
| CVIs on denoised data | Coherent, kept, but with a side effect (see below) | HDBSCAN `n_clust` already excluded noise |
| Stored `pred_clust`, `int(n)`, NaN sizes | Correct, kept | Pipeline and app run end to end |

Bugs fixed during this review (uncommitted):

| File | Fix |
|---|---|
| `src/model_select.py` | The null data of categorical latent models is now floored to integers, so the null CVIs are computed on the data StepMix actually fits (before: model on integers, CVIs on floats) |
| `src/model_select.py`, `process_dataset.py` | `pred_clust` is dropped inside `bootstrap_gap` instead of after collecting all results (about 1 GB of label vectors at full settings, and never dropped in `4_clustering.ipynb`) |
| `src/tooling.py` | The `*` flag also fires when the HDBSCAN noise pool holds ≥ 90% of the sample. Denoising had disabled it for these solutions |
| `app.py` | The "Clusters" tab plots the stored partition of the scored fit instead of a new unseeded re-fit (re-fit kept as fallback for old caches) |
| `CLAUDE.md` | Stale statements corrected (notebooks 4/5 are in the repo, data in `../data`, `Generate_clusters.ipynb` in `../clustering`) |

## 4. Open points (not changed, your decision)

- **HDBSCAN solutions that are almost all noise win the ranking.** With noise excluded from the CVIs, a few tiny clusters score very well: on the real data the second-best Davies-Bouldin solution is 2 clusters of 2 and 4 points with 99.3% noise; on the synthetic run the silhouette and Dunn winner (also the Davies-Bouldin runner-up) has 90.8% noise. **Resolved (2026-11-13):** HDBSCAN fits with more than 50% noise are excluded from the pools (`MAX_NOISE` in `src/tooling.py`, applied in `selected_pools`, so the table, app, similarity step and benchmark all follow; benchmark summaries are recomputed via `SUMMARY_VERSION = 3`).
- **Real data gives k=2 everywhere** at light settings: LCA selects 2 classes for all four indices (not the 4 of the paper), and 75 of 92 selections are k=2. To confirm at full settings (`max_clust=16`, `gap_iters=500`).
- **Runs are not reproducible**: nothing is seeded. Two identical real-data runs disagree on 7 of 92 gap selections (k-means and AHC).
- **Gap uses `log(mean)` but `s` uses `std(log)`** (original code, Tibshirani uses the mean of logs). Effect expected to be very small; left as is.
- **Original notebooks not re-run**: `4_clustering.ipynb` and `5_clustering_results_*.ipynb` still show outputs from the old gap rule and import the modified `src/`.
- **App cache**: `cache/5dfa72050261.pkl` predates the latent null fix; clear `cache/` to recompute.
- **No automated tests.**

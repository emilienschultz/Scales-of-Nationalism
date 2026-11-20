# Tracking

Log of the changes made to this project, newest first. Entries are added with the `/track` skill.

## 2026-11-20 10:42 — New gap selection rule after the artifact review

Summary: Review of the synthetic-data chain for artifacts, followed by three fixes to the gap selection. A k is now selected only if its gap beats the uniform null (gap > s) and no cluster holds less than 1% of the sample; among those, the smallest k within one s of the highest gap replaces the first-local-maximum rule, which stopped at k=2. Partitions isolating one outlier (829 / 1) no longer win Davies-Bouldin and Dunn. The default scenario is now 23 informative features, no redundant ones.

- `src/model_select.py`: new rule in `get_gap`; `select_candidates` (stage 3, recomputable from stored `gap_values`)
- `src/tooling.py`: `MIN_CLUSTER_SHARE = 0.01`, `min_cluster_size`; `min_share` argument on every selection function, applied to HDBSCAN too
- `process_dataset.py`, `src/model_similarity.py`, `benchmark.py` (`SUMMARY_VERSION = 4`): use the new selection
- `app.py`: selection recomputed at display time; slider "Minimum cluster size (% of sample)"; warning when nothing beats the null (not tested in the browser)
- `src/data_gen.py`: defaults `n_informative=23`, `n_redundant=0`; `CLAUDE.md` updated

Impact: no re-fit needed, cached app runs follow the new rule; rebuild benchmark summaries with `python benchmark.py --summarize-only`; `4_clustering.ipynb` will give different selections if re-run. Known limits: structureless data still gets a k on 3 to 12 of 23 configurations (the box null, not the threshold); HDBSCAN takes most "best distance" slots, often with 20–50% noise; Dunn still prefers k=2; Hamming AHC null is degenerate.

Commit: `9e9a67a` (+ uncommitted changes)

## 2026-11-13 18:44 — NMI matrix of best fits vs true classes

Summary: State of the changes to build a matrix to compare NMI: the app has a new tab showing, for each validity index, the four best distinct partitions across all algorithms, their pairwise NMI and their NMI with the synthetic true classes. The HDBSCAN noise cap is now a sidebar setting (default 50%). **Careful: the code has not been entirely properly tested. Especially the change regarding LCA should be rechecked** (in the matrix, LCA is ranked on indices recomputed on the standardized data, not the raw-data values of the pipeline).

- `src/model_similarity.py`: `best_fits_nmi` (top-4 distinct partitions per index, NMI matrix, NMI with truth with and without random points, noise share) and `_rescore_latent` (LCA re-scoring, to recheck)
- `src/tooling.py`: `eligible_fits` (all algorithms in one ranking); `max_noise` argument on every selection function
- `src/app_viz.py`: `nmi_heatmap`
- `app.py`: tab "Best fits vs true classes"; slider "HDBSCAN maximum noise (%)" applied at display time to all tabs; similarity tab guarded for old cached runs
- `CLAUDE.md`: documentation updated

Impact: no re-run needed (everything is computed from stored partitions). The comparison table, similarity tab and benchmark are unchanged at the default cap. Known limits: near-duplicate HDBSCAN fits often fill the four slots.

Commit: `62f29c0` (+ uncommitted changes)

## 2026-11-13 14:16 — HDBSCAN noise cap and review fixes

Summary: Putting a cap in HDBSCAN to avoid unsignificant clusters: fits leaving more than 50% of the sample as noise can no longer be picked as a best solution. Recorded with it, the fixes from the review of the July changes. First entry: covers only the work since commit `d05b85a`.

- `src/tooling.py`: `MAX_NOISE = 50` applied in `selected_pools` (table, app, similarity and benchmark follow); the `*` flag also fires when noise holds ≥ 90% of the sample
- `src/model_select.py`, `process_dataset.py`: null data of categorical latent models floored to integers; label vectors dropped inside `bootstrap_gap` (memory)
- `app.py`: "Clusters" tab plots the stored partition, not a re-fit; similarity tab always recomputed with the current selection
- `benchmark.py`: `SUMMARY_VERSION = 3` so cached summaries are rebuilt with the cap
- `CLAUDE.md`, `documenting_changes.md`, `.claude/skills/track/`: documentation updated, `/track` skill added

Impact: app cache cleared (runs recompute); rebuild benchmark summaries with `python benchmark.py --summarize-only`; the outputs of notebooks 4 and 5 predate all of this.

Commit: `d05b85a` (+ uncommitted changes)

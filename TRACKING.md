# Tracking

Log of the changes made to this project, newest first. Entries are added with the `/track` skill.

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

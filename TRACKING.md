# Tracking

Log of the changes made to this project, newest first. Entries are added with the `/track` skill.

## 2026-11-13 14:16 — HDBSCAN noise cap and review fixes

Summary: Putting a cap in HDBSCAN to avoid unsignificant clusters: fits leaving more than 50% of the sample as noise can no longer be picked as a best solution. Recorded with it, the fixes from the review of the July changes. First entry: covers only the work since commit `d05b85a`.

- `src/tooling.py`: `MAX_NOISE = 50` applied in `selected_pools` (table, app, similarity and benchmark follow); the `*` flag also fires when noise holds ≥ 90% of the sample
- `src/model_select.py`, `process_dataset.py`: null data of categorical latent models floored to integers; label vectors dropped inside `bootstrap_gap` (memory)
- `app.py`: "Clusters" tab plots the stored partition, not a re-fit; similarity tab always recomputed with the current selection
- `benchmark.py`: `SUMMARY_VERSION = 3` so cached summaries are rebuilt with the cap
- `CLAUDE.md`, `documenting_changes.md`, `.claude/skills/track/`: documentation updated, `/track` skill added

Impact: app cache cleared (runs recompute); rebuild benchmark summaries with `python benchmark.py --summarize-only`; the outputs of notebooks 4 and 5 predate all of this.

Commit: `d05b85a` (+ uncommitted changes)

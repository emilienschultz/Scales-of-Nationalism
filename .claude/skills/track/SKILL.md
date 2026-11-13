---
name: track
description: Record the changes made to this project in TRACKING.md, with the date and time and a short summary. Use only when the user asks to record, log or track the changes (for example "/track", "record this", "log the changes"). Do not run it on your own after ordinary edits.
---

# Track project changes

Add one entry to `TRACKING.md` at the repository root describing what changed in the project since the last recorded entry. If the user passed text with the command, treat it as their own note about the change and include it in the summary.

## Steps

1. **Read the last entry.** Open `TRACKING.md` if it exists and find the most recent entry: its timestamp and its `Commit:` line. If the file does not exist, this is the first entry.

2. **Work out what changed since then.** Use, in this order:
   - what was done in the current conversation, if the changes were made here;
   - `git log --stat <last recorded commit>..HEAD` for commits made since the last entry (for a first entry, do not go through the whole history: cover only the current work and say so);
   - `git status --short` and `git diff HEAD --stat` for uncommitted work, reading the actual diff of a file when the stat line is not enough to describe the change.

   Ignore `__pycache__/`, `cache/`, `benchmark_results/`, `.DS_Store` and `TRACKING.md` itself. If nothing changed since the last entry, tell the user and do not write an entry.

3. **Get the time and commit.** Run `date "+%Y-%m-%d %H:%M"` and `git rev-parse --short HEAD`. Never guess the time.

4. **Write the entry** directly under the file header, so the newest entry comes first. Never edit or delete earlier entries.

## Entry format

```markdown
## 2026-11-13 14:30 — Short title of the change

Summary: one to three sentences on what changed and why, in plain words.

- `path/to/file.py`: what changed in this file
- `other/file.py`: what changed in this file

Commit: `abc1234` (+ uncommitted changes)
```

Rules for the entry:

- The title is a few words naming the change, not a list of files.
- The summary says what the change does to the results or behaviour of the project (for example "HDBSCAN fits with more than 50% noise are no longer eligible as best solutions"), not how the code was edited.
- One bullet per file that matters; group trivial edits (docs, comments) in a single bullet.
- If the change alters results already produced (tables, cached runs, benchmark summaries), add a line `Impact: ...` saying what must be re-run.
- The `Commit:` line gives the current `HEAD`; add `(+ uncommitted changes)` when the working tree is not clean, since the next entry starts from this point.
- Keep the whole entry under about 12 lines.

## File header

When creating `TRACKING.md`, start it with:

```markdown
# Tracking

Log of the changes made to this project, newest first. Entries are added with the `/track` skill.
```

## After writing

Tell the user in one or two sentences what was recorded. Do not commit `TRACKING.md` unless asked.

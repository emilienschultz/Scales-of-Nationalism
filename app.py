"""Streamlit app: run the clustering robustness pipeline on synthetic datasets.

Set a cluster scenario in the sidebar, run the treatment, and inspect the
resulting clusters and best/second-best model comparison. Each run is cached to
disk keyed by its config, so re-launching an identical configuration loads
instantly instead of recomputing.

Run with:
    streamlit run app.py
"""

import ast
import hashlib
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from process_dataset import process_dataset
from src.app_viz import pca_scatter, refit_labels
from src.data_gen import SimConfig, generate_clusters
from src.model_similarity import partition_similarity
from src.tooling import INDEX_SPEC, model_comparison_table, selected_pools

CACHE_DIR = Path(__file__).parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# Pipeline parameters are fixed for now (sidebar section hidden). Edit here to
# change the treatment; HDBSCAN is forced on.
PIPE_CFG = dict(
    max_clust=10,
    gap_iters=50,
    n_jobs=15,
    msrt="categorical",
    standardize=True,
    run_hdbscan=True,
    subtract_one=True,
    n_combo=3,
)

st.set_page_config(page_title="Clustering robustness", layout="wide")


# --------------------------- caching ---------------------------


def config_key(data_cfg: dict, pipe_cfg: dict) -> str:
    """Stable short hash of the full (data + pipeline) configuration."""
    blob = json.dumps({"data": data_cfg, "pipeline": pipe_cfg}, sort_keys=True)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def run_or_load(data_cfg: dict, pipe_cfg: dict, preview=None):
    """Return the pipeline output for this config, computing it only if needed.

    On a cache miss the dataset is generated first and handed to `preview(df,
    y_true)` (so its PCA can be shown live while the treatment runs), then the
    pipeline runs and the result is persisted to `cache/<key>.pkl`.
    """
    key = config_key(data_cfg, pipe_cfg)
    path = CACHE_DIR / f"{key}.pkl"

    if path.exists():
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        payload["from_cache"] = True
        return payload

    df, y_true = generate_clusters(SimConfig(**data_cfg))
    if preview is not None:
        preview(df, y_true)
    result = process_dataset(df, verbose=False, **pipe_cfg)

    payload = {
        "key": key,
        "data_cfg": data_cfg,
        "pipe_cfg": pipe_cfg,
        "df": df,
        "y_true": y_true,
        "result": result,
        "timestamp": time.time(),
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)
    payload["from_cache"] = False
    return payload


def list_cached():
    """List cached runs as payload dicts (newest first)."""
    runs = []
    for path in CACHE_DIR.glob("*.pkl"):
        try:
            with open(path, "rb") as fh:
                runs.append(pickle.load(fh))
        except Exception:
            continue
    return sorted(runs, key=lambda p: p.get("timestamp", 0), reverse=True)


# --------------------------- solution selection ---------------------------


def best_solutions(all_models, candidate_models):
    """Recover the best / second-best / best-LCA rows behind the comparison table.

    Uses the same gap-statistic-based selection as `model_comparison_table` (via
    the shared `selected_pools` helper) but keeps the underlying (model, params,
    n_clust) so each solution can be re-fit and plotted. This guarantees the
    "Solution to visualize" dropdown offers exactly the solutions shown in the
    comparison panel. Returns a deduplicated DataFrame of distance-based best
    (rank 0) and second-best (rank 1) solutions plus the best LCA, across all
    validity indices.
    """
    dist_by_index, lca_by_index = selected_pools(all_models, candidate_models)

    rows = []
    for col, label, _ in INDEX_SPEC:
        top = dist_by_index[col]
        for rank, tag in [(0, "Best"), (1, "Second-best")]:
            if len(top) > rank:
                r = top.iloc[rank]
                rows.append(
                    {
                        "tag": tag,
                        "index": label,
                        "model": r["model"],
                        "params": r["params"],
                        "n_clust": int(r["n_clust"]),
                    }
                )
        top_lca = lca_by_index[col]
        if len(top_lca) > 0:
            r = top_lca.iloc[0]
            rows.append(
                {
                    "tag": "Best LCA",
                    "index": label,
                    "model": r["model"],
                    "params": r["params"],
                    "n_clust": int(r["n_clust"]),
                }
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["params_str"] = out["params"].astype(str)
    return out.drop_duplicates(subset=["model", "params_str", "n_clust"]).reset_index(
        drop=True
    )


@st.cache_data(show_spinner=False)
def cached_refit(run_key, model, params_str, n_clust, _df):
    """Cache re-fits per (run, solution) so reselecting is instant."""
    params = ast.literal_eval(params_str)
    return refit_labels(
        _df,
        model,
        params,
        n_clust,
        standardize=PIPE_CFG["standardize"],
        msrt=PIPE_CFG["msrt"],
        subtract_one=PIPE_CFG["subtract_one"],
    )


# --------------------------- sidebar form ---------------------------

st.title("Clustering robustness explorer")
st.caption(
    "Generate a synthetic cluster scenario, run the full robustness pipeline "
    "(LCA · K-means · AHC · HDBSCAN), and compare the best solutions."
)

defaults = SimConfig()

with st.sidebar:
    st.header("Cluster scenario")
    n_samples = st.number_input("Samples", 50, 5000, defaults.n_samples, step=10)
    n_features = st.number_input("Features", 2, 100, defaults.n_features, step=1)
    n_classes = st.number_input("True classes", 2, 12, defaults.n_classes, step=1)
    n_informative = st.number_input(
        "Informative features", 1, 50, defaults.n_informative, step=1
    )
    n_redundant = st.number_input(
        "Redundant features", 0, 50, defaults.n_redundant, step=1
    )
    n_clusters_per_class = st.number_input(
        "Clusters per class", 1, 5, defaults.n_clusters_per_class, step=1
    )
    class_sep = st.slider("Class separation", 0.1, 10.0, defaults.class_sep, step=0.1)
    # flip_y is not exposed: it only relabels points (features unchanged), so
    # the clustering problem is identical — contamination is the real knob.
    noise_prop = st.slider(
        "Random points proportion",
        0.0, 0.5, defaults.noise_prop, step=0.05,
        help="Share of samples replaced by uniform background points "
        "(no true class, shown as noise in the ground-truth plot).",
    )
    likert = st.checkbox("Bin to Likert {1..5}", value=defaults.likert)
    random_state = st.number_input(
        "Random seed", 0, 9999, defaults.random_state, step=1
    )

    run_clicked = st.button("▶ Run treatment", type="primary", use_container_width=True)

data_cfg = dict(
    n_samples=int(n_samples),
    n_features=int(n_features),
    n_informative=int(n_informative),
    n_redundant=int(n_redundant),
    n_classes=int(n_classes),
    n_clusters_per_class=int(n_clusters_per_class),
    class_balance=None,
    flip_y=0.0,
    class_sep=float(class_sep),
    noise_prop=float(noise_prop),
    random_state=int(random_state),
    likert=bool(likert),
)


# --------------------------- run / load ---------------------------

if run_clicked:
    key = config_key(data_cfg, PIPE_CFG)
    cached = (CACHE_DIR / f"{key}.pkl").exists()
    if cached:
        with st.spinner("Loading cached run…"):
            st.session_state["payload"] = run_or_load(data_cfg, PIPE_CFG)
    else:
        live = st.container()

        def preview(df, y_true):
            with live:
                st.caption(
                    "Generated cluster scenario (true classes) — treatment running…"
                )
                fig, _ = pca_scatter(df, y_true, title="Ground-truth classes")
                st.pyplot(fig)

        with st.spinner("Running treatment (this can take a while)…"):
            st.session_state["payload"] = run_or_load(
                data_cfg, PIPE_CFG, preview=preview
            )

# Sidebar: browse previously cached runs
with st.sidebar:
    st.header("Cached runs")
    runs = list_cached()
    if runs:
        labels = {
            f"{r['data_cfg']['n_classes']} cls · "
            f"{r['data_cfg']['n_samples']}×{r['data_cfg']['n_features']} · "
            f"sep={r['data_cfg']['class_sep']} · "
            f"rnd={r['data_cfg'].get('noise_prop', 0.0)} · [{r['key']}]": r["key"]
            for r in runs
        }
        pick = st.selectbox("Reload a run", ["—"] + list(labels.keys()))
        if pick != "—" and st.button("Load selected", use_container_width=True):
            chosen = labels[pick]
            st.session_state["payload"] = next(r for r in runs if r["key"] == chosen)
            st.session_state["payload"]["from_cache"] = True
        if st.button("🗑 Clear all cache", use_container_width=True):
            for p in CACHE_DIR.glob("*.pkl"):
                p.unlink()
            st.session_state.pop("payload", None)
            st.rerun()
    else:
        st.caption("No cached runs yet.")


# --------------------------- display ---------------------------

payload = st.session_state.get("payload")

if payload is None:
    st.info("Set a scenario in the sidebar, then click **Run treatment**.")
    st.stop()

df = payload["df"]
y_true = payload["y_true"]
result = payload["result"]
all_models = result["all_models"]

src_tag = "loaded from cache" if payload.get("from_cache") else "freshly computed"
st.success(
    f"Run `{payload['key']}` — {src_tag}. Dataset: {df.shape[0]} × {df.shape[1]}."
)

tab_clusters, tab_table, tab_similarity = st.tabs(
    ["Clusters", "Best-solution comparison", "Partition similarity"]
)

with tab_clusters:
    left, right = st.columns(2)
    with left:
        st.subheader("True configuration")
        fig_true, pca = pca_scatter(
            df, y_true, title="Ground-truth classes", figsize=(5, 3.75)
        )
        st.pyplot(fig_true, use_container_width=False)

    with right:
        st.subheader("Best / second-best solution")
        sols = best_solutions(all_models, result["candidate_models"])
        if sols.empty:
            st.caption("No gap-selectable solutions for this run.")
        else:
            sols["label"] = sols.apply(
                lambda r: f"{r['tag']} · {r['index']} · {r['model']} (n={r['n_clust']})",
                axis=1,
            )
            choice = st.selectbox("Solution to visualize", sols["label"].tolist())
            row = sols[sols["label"] == choice].iloc[0]
            with st.spinner("Re-fitting selected solution…"):
                labels = cached_refit(
                    payload["key"], row["model"], row["params_str"], row["n_clust"], df
                )
            fig_pred, _ = pca_scatter(
                df, labels, title=f"{row['model']} (n={int(row['n_clust'])})", pca=pca
            )
            st.pyplot(fig_pred)

with tab_table:
    st.subheader("Model comparison — best & second-best by validity index")
    table = model_comparison_table(
        all_models, result["candidate_models"], n_samples=len(df)
    )
    st.dataframe(
        table.set_index(["Section", "Clustering Validity Index"]),
        use_container_width=True,
    )
    st.download_button(
        "Download table (CSV)",
        table.to_csv(index=False).encode(),
        file_name=f"table_{payload['key']}.csv",
        mime="text/csv",
    )

with tab_similarity:
    st.subheader("Partition similarity — average AMI of the best solutions")
    # Recompute on the fly (cheap — a handful of stored label vectors, no
    # re-fit) for runs cached before the similarity step existed, or cached
    # with the old NMI-based version of it.
    similarity = result.get("similarity")
    if similarity is None or "pairwise_ami" not in similarity:
        similarity = partition_similarity(
            all_models, result["candidate_models"], n_combo=PIPE_CFG["n_combo"]
        )

    if similarity["best"] is None:
        st.caption("Fewer than two distinct best partitions — nothing to compare.")
    else:
        best = similarity["best"]
        n_shown = min(similarity["n_combo"], len(similarity["pool"]))
        if similarity["partial"]:
            st.warning(
                f"Only {len(similarity['pool'])} distinct best partitions available "
                f"(fewer than the requested combination size of "
                f"{similarity['n_combo']}); the average AMI is computed over all "
                f"of them."
            )
        st.metric(
            f"Highest average AMI ({n_shown}-partition combination)",
            f"{best['ami']:.3f}",
        )
        st.markdown(
            f"**Indicators kept:** {best['indices']}  \n"
            f"**Solutions:** {best['members']}"
        )

        st.markdown(f"**All {n_shown}-partition combinations**")
        st.dataframe(
            similarity["combos"].rename(
                columns={
                    "members": "Solutions",
                    "indices": "Validity indices",
                    "ami": "Average AMI",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("**Pairwise AMI between best partitions**")
        st.dataframe(similarity["pairwise_ami"].round(3), use_container_width=True)

    st.caption(
        "Pool: the best distance-based solution (k-means / AHC / HDBSCAN) per "
        "validity index, deduplicated across indices. AMI (chance-corrected: "
        "1 = identical, ~0 = chance-level, <0 = worse than chance) is computed "
        "on the stored pipeline labels; HDBSCAN noise counts as its own cluster."
    )

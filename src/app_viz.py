"""Visualisation and re-fit helpers for the Streamlit robustness app (`app.py`).

Two functions:

- `pca_scatter` projects a feature frame to 2-D with PCA and plots it coloured
  by a label vector, returning the figure *and* the fitted PCA so a prediction
  can be drawn in the exact same projection as the ground truth. Noise points
  (label `-1`, from HDBSCAN) are drawn as grey stars.
- `refit_labels` re-runs a single (model, params, n_clust) solution to recover
  its cluster assignments. It reproduces the per-model preprocessing of
  `process_dataset` (standardisation for distance models, `-1` shift for
  categorical latent models) so the labels match what the pipeline scored.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.model_fit import do_AHC, do_StepMix, do_hdbscan, do_kmeans


def _draw_cluster_shells(ax, reduced, labels, cmap, norm):
    """Outline each cluster with its convex hull to make groups easier to read.

    For every non-noise cluster with at least three non-collinear points, draw
    the convex hull as a soft filled polygon coloured to match the cluster's
    points. Clusters too small (or degenerate) to form a hull are skipped.
    """
    for k in np.unique(labels):
        if k == -1:
            continue
        pts = reduced[labels == k]
        if len(pts) < 3:
            continue
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Collinear / degenerate point set — no 2-D hull.
            continue
        verts = pts[hull.vertices]
        color = cmap(norm(k))
        ax.fill(
            verts[:, 0], verts[:, 1],
            facecolor=color, edgecolor=color,
            alpha=0.12, linewidth=1.5, zorder=0,
        )


def pca_scatter(data, labels, title=None, pca=None, figsize=(8, 6)):
    """Scatter `data` in 2-D PCA space, coloured by `labels`.

    Parameters
    ----------
    data : pd.DataFrame or array-like
        Feature matrix (one row per observation).
    labels : array-like
        Cluster / class assignment per row. `-1` is treated as noise.
    title : str, optional
        Axis title.
    pca : sklearn.decomposition.PCA, optional
        A PCA already fitted on a reference frame. When given, `data` is
        projected with `transform` instead of refitting, so two plots share the
        same axes (e.g. ground truth vs. predicted labels).

    Returns
    -------
    (matplotlib.figure.Figure, PCA)
        The figure and the PCA used (fitted here if none was passed in).
    """
    labels = np.asarray(labels)

    if pca is None:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(data)
    else:
        reduced = pca.transform(data)
    explained = pca.explained_variance_ratio_ * 100

    fig, ax = plt.subplots(figsize=figsize)

    mask_noise = labels == -1
    mask_clusters = ~mask_noise

    # Shared colormap + normalisation so cluster points and their convex-hull
    # shells get the exact same colour.
    cmap = plt.get_cmap("tab10")
    cluster_labels = labels[mask_clusters]
    if cluster_labels.size:
        norm = Normalize(vmin=cluster_labels.min(), vmax=cluster_labels.max())
    else:
        norm = Normalize(vmin=0, vmax=1)

    # External shell first, then noise, then cluster points on top.
    _draw_cluster_shells(ax, reduced, labels, cmap, norm)

    if np.any(mask_noise):
        ax.scatter(
            reduced[mask_noise, 0], reduced[mask_noise, 1],
            c="gray", marker="*", s=50, label="Noise",
        )

    scatter = ax.scatter(
        reduced[mask_clusters, 0], reduced[mask_clusters, 1],
        c=cluster_labels, cmap=cmap, norm=norm, s=20,
        edgecolors="k", linewidths=0.3,
    )

    ax.axhline(0, color="grey", linestyle="dashed", linewidth=1)
    ax.axvline(0, color="grey", linestyle="dashed", linewidth=1)
    ax.set_xlabel(f"PC1 ({explained[0]:.2f}%)")
    ax.set_ylabel(f"PC2 ({explained[1]:.2f}%)")
    if title:
        ax.set_title(title)

    # Legend labelled with cluster sizes, mirroring the notebook / model_plot.
    sizes = pd.Series(labels[mask_clusters]).value_counts().sort_index()
    handles, _ = scatter.legend_elements()
    leg_labels = [f"Clst. {int(k) + 1} | n={v}" for k, v in sizes.items()]
    if np.any(mask_noise):
        handles.append(
            plt.Line2D([0], [0], marker="*", color="gray",
                       linestyle="None", markersize=10)
        )
        leg_labels.append(f"Noise | n={int(mask_noise.sum())}")
    ax.legend(handles, leg_labels, title="")

    fig.tight_layout()
    return fig, pca


def refit_labels(
    data, model, params, n_clust,
    standardize=True, msrt="categorical", subtract_one=True,
):
    """Recover the cluster labels of a single pipeline solution.

    Reproduces `process_dataset`'s preprocessing so the assignments match the
    scored run: distance models (k-means / AHC / HDBSCAN) see standardised
    features, while a categorical latent model sees the `-1` shifted frame.

    Parameters
    ----------
    data : pd.DataFrame
        Raw feature frame (the same one passed to `process_dataset`).
    model : str
        One of 'latent', 'kmeans', 'AHC', 'HDBSCAN'.
    params : dict
        The solution's parameter dict, as stored in `all_models['params']`.
    n_clust : int
        Number of clusters to fit (ignored by HDBSCAN, which picks its own).
    standardize, msrt, subtract_one
        Must match the values used for the original `process_dataset` run.

    Returns
    -------
    np.ndarray
        Cluster label per row of `data` (`-1` = noise for HDBSCAN).
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

    if msrt == "categorical":
        data_latent = data - 1 if subtract_one else data
    else:
        data_latent = data_n

    if model == "latent":
        labels = do_StepMix(
            data_latent, None, n_clust,
            params.get("msrt", msrt), params.get("covar", "without"),
            refit=True,
        )
    elif model == "kmeans":
        labels = do_kmeans(data_n, n_clust, params["dist"], params["link"], refit=True)
    elif model == "AHC":
        labels = do_AHC(data_n, n_clust, params["dist"], params["link"], refit=True)
    elif model == "HDBSCAN":
        labels = do_hdbscan(
            data_n, params["dist"], params["min_clust"], params["min_smpl"],
            refit=True,
        )
    else:
        raise ValueError(f"Unknown model: {model!r}")

    return np.asarray(labels)

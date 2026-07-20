"""Synthetic cluster-scenario generator for the robustness app (`app.py`).

`SimConfig` holds one scenario's parameters (defaults match the `replic_830`
baseline) and `generate_clusters` wraps `sklearn.datasets.make_classification`,
optionally binning the output to a {1..5} Likert scale. Extracted from
`Generate_clusters.ipynb`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


@dataclass
class SimConfig:
    """Parameters for one synthetic-data scenario.

    Defaults match the `replic_830` baseline.
    """
    n_samples: int = 830            # respondents
    n_features: int = 23            # Likert items
    n_informative: int = 4          # features carrying real cluster signal
    n_redundant: int = 4            # linear combos of the informative features
    n_classes: int = 4              # B&D headline = 4 latent classes
    n_clusters_per_class: int = 1   # one Gaussian blob per class
    class_balance: tuple | None = None  # relative cluster sizes; None = equal.
                                        # proportions (0.4, 0.2, 0.2, 0.2) or
                                        # counts (400, 150, 150, 130) -- both
                                        # normalised internally. Length must
                                        # equal n_classes.
    flip_y: float = 0.01            # fraction of labels randomly flipped (Y noise
                                    # only: features are unchanged, so clustering
                                    # is unaffected — it only degrades y_true)
    class_sep: float = 5.0          # larger = easier problem
    noise_prop: float = 0.0         # proportion of the n_samples replaced by
                                    # uniform random points over the feature
                                    # range (no true class: y_true = -1)
    random_state: int = 0
    likert: bool = True             # set True to bin to {1..5}


def generate_clusters(cfg):
    """Generate one scenario from `cfg`, returning `(df, y_true)`."""
    # Class balance: None -> equal sizes; otherwise normalise the given
    # proportions / counts to weights that sum to 1 (as make_classification
    # expects). Length must match n_classes.
    weights = None
    if cfg.class_balance is not None:
        if len(cfg.class_balance) != cfg.n_classes:
            raise ValueError(
                f"class_balance has {len(cfg.class_balance)} entries "
                f"but n_classes={cfg.n_classes}"
            )
        w = np.asarray(cfg.class_balance, dtype=float)
        weights = (w / w.sum()).tolist()

    # Split the sample between clustered points and uniform background noise.
    # Noise points are drawn over the clustered points' feature range and get
    # y_true = -1 (they belong to no class), then everything is shuffled so
    # the Likert binning and any preview see one homogeneous frame.
    n_noise = int(round(cfg.n_samples * cfg.noise_prop))
    n_clustered = cfg.n_samples - n_noise

    X, y_true = make_classification(
        n_samples=n_clustered,
        n_features=cfg.n_features,
        n_informative=cfg.n_informative,
        n_redundant=cfg.n_redundant,
        n_repeated=0,
        n_classes=cfg.n_classes,
        n_clusters_per_class=cfg.n_clusters_per_class,
        weights=weights,
        flip_y=cfg.flip_y,
        class_sep=cfg.class_sep,
        random_state=cfg.random_state,
    )
    if n_noise > 0:
        rng = np.random.default_rng(cfg.random_state)
        X_noise = rng.uniform(
            X.min(axis=0), X.max(axis=0), size=(n_noise, cfg.n_features)
        )
        X = np.vstack([X, X_noise])
        y_true = np.concatenate([y_true, np.full(n_noise, -1)])
        perm = rng.permutation(cfg.n_samples)
        X, y_true = X[perm], y_true[perm]
    if cfg.likert:
        col_min = X.min(axis=0)
        col_max = X.max(axis=0)
        scaled = (X - col_min) / np.where(col_max - col_min == 0, 1, col_max - col_min)
        X = np.clip(np.round(scaled * 4) + 1, 1, 5).astype(int)
    df = pd.DataFrame(X, columns=[f"V{i + 1}" for i in range(cfg.n_features)])
    return df, y_true

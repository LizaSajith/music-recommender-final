"""Shared data helpers for Member 3. Uses Member 1's cleaning/normalisation
(content_based.py) so everyone works on the identical 8-feature matrix."""
import os
import sys

import numpy as np
import pandas as pd

# Use Member 1's code from the repo root instead of keeping a copy here.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(_ROOT, "content_based"))
from content_based import load_and_clean, build_feature_matrix, FEATURES

DATA_PATH = os.path.join(_ROOT, "data", "dataset.csv")


def load_features(csv_path=DATA_PATH):
    """Return (df, X_norm). Same cleaning + z-scoring as Member 1."""
    df = load_and_clean(csv_path)
    return df, build_feature_matrix(df)


def synthetic_ratings(genres, n_users=300, n_factors=6, density=0.10, seed=1):
    """STAND-IN for Member 2's user x song matrix (np.nan = unrated).
    Low-rank taste model: each genre loads on `n_factors` hidden taste factors,
    each user has weights on those factors -> true rank = n_factors.
    Replace with Member 2's real/synthetic matrix when available."""
    rng = np.random.default_rng(seed)
    genres = np.asarray(genres)
    uniq, g_idx = np.unique(genres, return_inverse=True)
    G = rng.normal(0, 1, (len(uniq), n_factors))        # genre -> factors
    U = rng.normal(0, 1, (n_users, n_factors))          # user  -> factors
    latent = U @ G[g_idx].T
    R = np.clip(np.round(3 + 0.8 * latent + rng.normal(0, 0.5, latent.shape)), 1, 5)
    R[rng.random(R.shape) >= density] = np.nan
    return R

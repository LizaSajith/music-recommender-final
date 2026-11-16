"""Hybrid recommender = content-based (cosine similarity of audio features)
+ collaborative filtering (truncated SVD of the user x song rating matrix).

    hybrid_score = alpha * content_score + (1 - alpha) * collab_score

Both scores are on a 0-1 scale before they are blended.

Candidate pool:
  * "Content-only" mode ranks all ~81k songs.
  * With a user profile, only the rating catalog (the songs that appear in
    interactions.csv) is ranked, because only those songs have collaborative
    scores. Songs the user already rated are excluded.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import K
from content_based.content_based import load_and_clean, build_feature_matrix
from collab_filtering.collab_filtering import (
    build_interaction_matrix,
    fill_missing_with_user_means,
    perform_svd,
    reconstruct_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def minmax_scale(values):
    """Scale scores to [0, 1], safely handling equal scores."""
    values = np.asarray(values, dtype=float)
    low, high = values.min(), values.max()

    if np.isclose(high, low):
        return np.zeros_like(values)

    return (values - low) / (high - low)


@lru_cache(maxsize=1)
def build_hybrid_recommender():
    """Load data and fit both models once (cached for the whole session)."""
    # Content side: cleaned songs, z-scored features, unit-length rows
    songs = load_and_clean(DATA_DIR / "dataset.csv")
    X = build_feature_matrix(songs)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X_unit = np.divide(
        X, norms,
        out=np.zeros_like(X, dtype=float),
        where=norms != 0
    )

    # Collaborative side: user x song matrix -> mean-filled -> truncated SVD
    interactions = pd.read_csv(DATA_DIR / "interactions.csv")
    matrix_df, R = build_interaction_matrix(interactions)

    if not set(matrix_df.columns) & set(songs["track_id"]):
        raise ValueError(
            "No track_id in interactions.csv matches the song table. "
            "Re-run generate_interactions.py."
        )

    R_filled = fill_missing_with_user_means(R)
    U, S, VT = perform_svd(R_filled)
    k = min(K, len(S))
    R_pred = reconstruct_matrix(U, S, VT, k)

    return songs, X_unit, interactions, matrix_df, R_pred


def recommend_hybrid(
    song_name,
    user_id=None,
    n=10,
    alpha=0.5
):
    songs, X_unit, interactions, matrix_df, R_pred = (
        build_hybrid_recommender()
    )

    matches = songs.index[
        songs["track_name"].str.casefold()
        == song_name.casefold()
    ].to_numpy()

    if len(matches) == 0:
        raise ValueError(f"Song not found: {song_name}")

    query_idx = int(matches[0])

    result = songs.copy()
    result["content_score"] = X_unit @ X_unit[query_idx]
    result["collab_score"] = 0.0

    # The selected song is never recommended to itself
    result = result.drop(index=query_idx)

    if user_id is not None:
        if user_id not in matrix_df.index:
            raise ValueError(
                f"Unknown user {user_id}. Available users: "
                f"{', '.join(map(str, matrix_df.index[:10]))}"
            )

        # Only catalog songs have collaborative scores
        result = result[result["track_id"].isin(matrix_df.columns)]

        # Do not recommend songs this user already rated
        seen = interactions.loc[
            interactions["user_id"] == user_id, "track_id"
        ]
        result = result[~result["track_id"].isin(seen)].copy()

        # Predicted ratings (1-5) -> 0-1 collaborative score
        user_idx = matrix_df.index.get_loc(user_id)
        predicted = pd.Series(R_pred[user_idx], index=matrix_df.columns)
        raw = result["track_id"].map(predicted)
        result["collab_score"] = (raw.clip(1, 5) - 1) / 4

    # Content scores rescaled to 0-1 across the candidates being ranked
    result["content_score"] = minmax_scale(
        result["content_score"].to_numpy()
    )

    if user_id is None:
        result["hybrid_score"] = result["content_score"]
    else:
        result["hybrid_score"] = (
            alpha * result["content_score"]
            + (1 - alpha) * result["collab_score"]
        )

    return result.sort_values(
        "hybrid_score", ascending=False
    )[
        [
            "track_id", "track_name", "artists",
            "track_genre", "content_score",
            "collab_score", "hybrid_score"
        ]
    ].head(n).reset_index(drop=True)


if __name__ == "__main__":
    recommendations = recommend_hybrid(
        song_name="Comedy",
        user_id="U01",
        n=10,
        alpha=0.5
    )

    print("\nTop 10 hybrid recommendations\n")
    print(recommendations.to_string(index=False))

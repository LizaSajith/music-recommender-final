import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Make the repo root importable so this file works from anywhere.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")

from config import K   # shared SVD rank (see config.py)


# ============================================================
# 1. LOAD INTERACTION DATA
# ============================================================

def load_interactions(csv_path):

    df = pd.read_csv(csv_path)

    print("Interaction data loaded.")
    print("Users:", df["user_id"].nunique())
    print("Songs:", df["track_id"].nunique())
    print("Interactions:", len(df))

    return df


# ============================================================
# 2. BUILD USER × SONG MATRIX
# ============================================================

def build_interaction_matrix(df):

    R_df = df.pivot_table(
        index="user_id",
        columns="track_id",
        values="rating",
        fill_value=0
    )

    R = R_df.to_numpy(dtype=float)

    return R_df, R


# ============================================================
# 2b. FILL UNRATED ENTRIES WITH EACH USER'S MEAN RATING
# ============================================================
# Filling unrated cells with 0 tells SVD "the user hates these songs",
# which drags every prediction towards 0. Filling with the user's own
# average rating is a neutral guess instead (same method as Member 3's
# held-out RMSE evaluation).

def fill_missing_with_user_means(R):

    observed = R > 0

    counts = observed.sum(axis=1, keepdims=True)
    sums = (R * observed).sum(axis=1, keepdims=True)

    global_mean = R[observed].mean()

    user_means = np.where(
        counts > 0,
        sums / np.maximum(counts, 1),
        global_mean
    )

    return np.where(observed, R, user_means)


# ============================================================
# 3. MATRIX RANK
# ============================================================

def check_matrix_rank(R):

    return np.linalg.matrix_rank(R)


# ============================================================
# 4. SVD
# ============================================================

def perform_svd(R):

    U, S, VT = np.linalg.svd(
        R,
        full_matrices=False
    )

    return U, S, VT


# ============================================================
# 5. TRUNCATED SVD RECONSTRUCTION
# ============================================================

def reconstruct_matrix(U, S, VT, k):

    Uk = U[:, :k]
    Sk = S[:k]
    VTk = VT[:k, :]

    Rk = Uk @ np.diag(Sk) @ VTk

    return Rk


# ============================================================
# 6. RECONSTRUCTION RMSE
# ============================================================

def reconstruction_rmse(R, Rk):

    error = R - Rk

    mse = np.mean(error ** 2)

    rmse = np.sqrt(mse)

    return rmse


# ============================================================
# 7. FIND TOP-N RECOMMENDATIONS
# ============================================================

def recommend_songs(
    R_df,
    Rk,
    user_id,
    n=5
):

    # Find the row corresponding to the selected user
    user_index = list(R_df.index).index(user_id)

    # Original interactions
    original_ratings = R_df.iloc[user_index].to_numpy()

    # Predicted ratings from reconstructed matrix
    predicted_ratings = Rk[user_index]

    # Only recommend songs the user has NOT interacted with
    unseen = original_ratings == 0

    # Copy predicted scores
    scores = predicted_ratings.copy()

    # Ignore songs already interacted with
    scores[~unseen] = -np.inf

    # Get indices of highest predicted scores
    top_indices = np.argsort(scores)[::-1][:n]

    # Track IDs
    track_ids = R_df.columns.to_numpy()

    recommendations = pd.DataFrame({
        "track_id": track_ids[top_indices],
        "predicted_score": predicted_ratings[top_indices]
    })

    return recommendations


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    df = load_interactions(
        os.path.join(DATA_DIR, "interactions.csv")
    )

    # --------------------------------------------------------
    # Build interaction matrix
    # --------------------------------------------------------

    R_df, R = build_interaction_matrix(df)

    print("\nInteraction matrix shape:")
    print(R.shape)

    # --------------------------------------------------------
    # Matrix rank
    # --------------------------------------------------------

    rank = check_matrix_rank(R)

    print("\nMatrix rank:")
    print(rank)

    # --------------------------------------------------------
    # SVD
    # --------------------------------------------------------

    # Unrated entries -> user's mean rating, then SVD of the filled matrix
    R_filled = fill_missing_with_user_means(R)

    U, S, VT = perform_svd(R_filled)

    print("\nSVD completed.")

    print("U shape:", U.shape)
    print("Number of singular values:", len(S))
    print("VT shape:", VT.shape)

    print("\nSingular values:")
    print(S)

    # --------------------------------------------------------
    # Test different k values
    # --------------------------------------------------------

    k_values = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]

    errors = []

    print("\nReconstruction error:")
    print("---------------------")

    for k in k_values:

        Rk = reconstruct_matrix(
            U,
            S,
            VT,
            k
        )

        # error measured on the ratings that really exist
        observed = R > 0

        rmse = reconstruction_rmse(
            R[observed],
            Rk[observed]
        )

        errors.append(rmse)

        print(
            f"k = {k:2d}  RMSE = {rmse:.4f}"
        )

    # --------------------------------------------------------
    # Plot RMSE vs k
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        k_values,
        errors,
        marker="o"
    )

    plt.xlabel("Number of latent dimensions (k)")
    plt.ylabel("Reconstruction RMSE")
    plt.title("Reconstruction Error vs SVD Rank")

    plt.grid(True)

    plt.savefig(
        os.path.join(DATA_DIR, "reconstruction_error_vs_rank.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(
        "\nGraph saved as:"
        " ../data/reconstruction_error_vs_rank.png"
    )

    # --------------------------------------------------------
    # Choose k
    # --------------------------------------------------------

    chosen_k = K   # from config.py

    print("\nChosen k:", chosen_k)

    Rk = reconstruct_matrix(
        U,
        S,
        VT,
        chosen_k
    )

    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    user_id = "U01"

    recommendations = recommend_songs(
        R_df,
        Rk,
        user_id,
        n=5
    )

    # --------------------------------------------------------
    # Add song names
    # --------------------------------------------------------

   # Load original song information
       # --------------------------------------------------------
    # Add song names
    # --------------------------------------------------------

    # Load original song information
    song_info = pd.read_csv(
        os.path.join(DATA_DIR, "dataset.csv")
    )

    # Remove unnecessary index column if present
    if "Unnamed: 0" in song_info.columns:
        song_info = song_info.drop(
            columns=["Unnamed: 0"]
        )

    # Keep one row per track
    song_info = song_info[
        ["track_id", "track_name", "artists"]
    ].drop_duplicates(
        "track_id"
    )

    recommendations = recommendations.merge(
        song_info,
        on="track_id",
        how="left"
    )

    recommendations = recommendations[
        [
            "track_id",
            "track_name",
            "artists",
            "predicted_score"
        ]
    ]

    print("\nTop 5 Collaborative Recommendations")
    print("------------------------------------")

    print(
        recommendations.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save recommendations
    # --------------------------------------------------------

    recommendations.to_csv(
        os.path.join(DATA_DIR, "recommendations_U01.csv"),
        index=False
    )

    print(
        "\nRecommendations saved as:"
        " ../data/recommendations_U01.csv"
    )
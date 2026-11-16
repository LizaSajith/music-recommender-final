import os
import sys

import pandas as pd
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")

from config import K
from collab_filtering.collab_filtering import fill_missing_with_user_means


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

interactions = pd.read_csv(
    os.path.join(DATA_DIR, "interactions.csv")
)

songs = pd.read_csv(
    os.path.join(DATA_DIR, "dataset.csv")
)

if "Unnamed: 0" in songs.columns:
    songs = songs.drop(
        columns=["Unnamed: 0"]
    )


# ------------------------------------------------------------
# BUILD USER × SONG MATRIX
# ------------------------------------------------------------

R_df = interactions.pivot_table(
    index="user_id",
    columns="track_id",
    values="rating",
    fill_value=0
)

R = R_df.to_numpy(
    dtype=float
)


# ------------------------------------------------------------
# SVD
# ------------------------------------------------------------

R_filled = fill_missing_with_user_means(R)

U, S, VT = np.linalg.svd(
    R_filled,
    full_matrices=False
)


# ------------------------------------------------------------
# CHOSEN RANK
# ------------------------------------------------------------

k = K   # from config.py

Uk = U[:, :k]
Sk = S[:k]
VTk = VT[:k, :]

Rk = Uk @ np.diag(Sk) @ VTk


# ------------------------------------------------------------
# FUNCTION TO RECOMMEND SONGS
# ------------------------------------------------------------

def recommend_for_user(
    user_id,
    n=5
):

    user_index = list(
        R_df.index
    ).index(user_id)

    original = R_df.iloc[
        user_index
    ].to_numpy()

    predicted = Rk[
        user_index
    ]

    # Only songs the user has NOT interacted with
    unseen = original == 0

    scores = predicted.copy()

    scores[~unseen] = -np.inf

    top_indices = np.argsort(
        scores
    )[::-1][:n]

    result = pd.DataFrame({
        "track_id":
            R_df.columns.to_numpy()[top_indices],

        "predicted_score":
            predicted[top_indices]
    })

    result = result.merge(
        songs[
            [
                "track_id",
                "track_name",
                "artists",
                "track_genre"
            ]
        ].drop_duplicates(
            "track_id"
        ),
        on="track_id",
        how="left"
    )

    return result[
        [
            "track_id",
            "track_name",
            "artists",
            "track_genre",
            "predicted_score"
        ]
    ]


# ------------------------------------------------------------
# EVALUATE THREE USERS
# ------------------------------------------------------------

test_users = [
    "U01",
    "U02",
    "U03"
]


for user in test_users:

    print("\n")
    print("=" * 70)
    print("USER:", user)
    print("=" * 70)

    # --------------------------------------------------------
    # Actual highly-rated songs
    # --------------------------------------------------------

    actual = interactions[
        (interactions["user_id"] == user)
        &
        (interactions["rating"] >= 4)
    ]

    actual = actual.merge(
        songs[
            [
                "track_id",
                "track_name",
                "artists",
                "track_genre"
            ]
        ].drop_duplicates(
            "track_id"
        ),
        on="track_id",
        how="left"
    )

    print("\nSongs actually rated 4 or 5:")

    if len(actual) == 0:

        print("No high-rated songs.")

    else:

        print(
            actual[
                [
                    "track_name",
                    "artists",
                    "track_genre",
                    "rating"
                ]
            ].head(10).to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Recommended songs
    # --------------------------------------------------------

    recommendations = recommend_for_user(
        user,
        n=5
    )

    print("\nTop-5 SVD recommendations:")

    print(
        recommendations.to_string(
            index=False
        )
    )


print("\n")
print("=" * 70)
print("EVALUATION COMPLETE")
print("=" * 70)
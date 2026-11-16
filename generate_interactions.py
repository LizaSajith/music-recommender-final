"""Generate synthetic user-song ratings.

Why a small catalog: collaborative filtering needs users to rate the SAME songs.
Sampling 80 songs per user from all ~81k songs gave almost no overlap (1,168 of
1,184 songs had a single rater), so SVD had nothing to learn from. Here we pick a
500-song catalog first, then every user rates songs from that catalog only
(about 3 raters per song).

Run from the repo root:  python generate_interactions.py
Writes: data/interactions.csv   (long format: user_id, track_id, rating)
        data/ratings_matrix.csv (users x songs, blank = unrated; for member3/run_all.py)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import (RANDOM_SEED, NUM_USERS, CATALOG_PER_GROUP,
                    RATINGS_PER_USER, PREFERRED_FRACTION)
from content_based.content_based import load_and_clean

DATA_DIR = ROOT / "data"

TASTE_GROUPS = {
    "pop_dance": ["pop", "dance", "dancehall", "edm", "electronic", "house",
                  "disco", "synth-pop", "electro", "techno", "trance",
                  "progressive-house"],
    "rock_metal": ["rock", "alt-rock", "alternative", "hard-rock", "punk",
                   "punk-rock", "metal", "heavy-metal", "black-metal",
                   "metalcore", "grunge", "hardcore", "industrial"],
    "classical": ["classical", "opera", "piano", "romance", "new-age",
                  "ambient", "study"],
    "hiphop_soul": ["hip-hop", "r-n-b", "soul", "funk", "jazz", "trip-hop",
                    "groove", "indie", "indie-pop"],
    "world_latin": ["latin", "latino", "reggaeton", "reggae", "salsa", "samba",
                    "tango", "spanish", "world-music", "brazil", "country",
                    "folk", "indian", "malay", "mandopop"],
}


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    # Same cleaning as the rest of the project (both dedup passes), so every
    # rated track_id is guaranteed to exist in the song table.
    songs = load_and_clean(DATA_DIR / "dataset.csv")
    print("Clean songs:", len(songs))

    # ---- 1. catalog: CATALOG_PER_GROUP songs per taste group --------------
    catalog = {}
    for group, genres in TASTE_GROUPS.items():
        pool = songs.loc[songs["track_genre"].isin(genres), "track_id"].to_numpy()
        n = min(CATALOG_PER_GROUP, len(pool))
        catalog[group] = rng.choice(pool, size=n, replace=False)
        print(f"  {group:12s} catalog songs: {n}")
    all_catalog = np.concatenate(list(catalog.values()))

    # ---- 2. users: 4 per taste group ---------------------------------------
    groups = list(TASTE_GROUPS)
    per_group = NUM_USERS // len(groups)
    rows = []
    for i in range(NUM_USERS):
        user_id = f"U{i + 1:02d}"
        group = groups[min(i // per_group, len(groups) - 1)]

        own = catalog[group]
        other = np.setdiff1d(all_catalog, own)
        n_pref = min(int(RATINGS_PER_USER * PREFERRED_FRACTION), len(own))
        n_other = min(RATINGS_PER_USER - n_pref, len(other))

        liked = rng.choice(own, size=n_pref, replace=False)
        rest = rng.choice(other, size=n_other, replace=False)

        for tid in liked:   # mostly high, a little noise
            rows.append((user_id, tid, int(rng.choice([3, 4, 5], p=[0.15, 0.35, 0.50]))))
        for tid in rest:    # mostly low, a little noise
            rows.append((user_id, tid, int(rng.choice([1, 2, 3, 4], p=[0.35, 0.35, 0.20, 0.10]))))

    interactions = pd.DataFrame(rows, columns=["user_id", "track_id", "rating"])
    interactions = interactions.drop_duplicates(["user_id", "track_id"]).reset_index(drop=True)
    interactions.to_csv(DATA_DIR / "interactions.csv", index=False)

    # Wide version for Member 3's evaluation (blank = unrated)
    wide = interactions.pivot(index="user_id", columns="track_id", values="rating")
    wide.to_csv(DATA_DIR / "ratings_matrix.csv")

    # ---- 3. report ---------------------------------------------------------
    raters = interactions.groupby("track_id")["user_id"].nunique()
    print("\nUsers:", interactions["user_id"].nunique(),
          "| songs:", interactions["track_id"].nunique(),
          "| ratings:", len(interactions))
    print("Matrix density: %.1f%%" % (100 * len(interactions) / wide.size))
    print("Raters per song  min/mean/max: %d / %.1f / %d"
          % (raters.min(), raters.mean(), raters.max()))
    print("Songs rated by 1 user only:", int((raters == 1).sum()))
    print("All rated ids in clean song table:",
          interactions["track_id"].isin(songs["track_id"]).all())
    print("Saved data/interactions.csv and data/ratings_matrix.csv")


if __name__ == "__main__":
    main()

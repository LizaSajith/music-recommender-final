"""Shared settings for the whole project. Change a value here, not in each file."""

# Rank of the truncated SVD used for collaborative filtering.
# Chosen by Member 3's held-out RMSE sweep (member3/evaluation.py).
K = 5

# Synthetic ratings (generate_interactions.py)
RANDOM_SEED = 42
NUM_USERS = 20                # 5 taste groups x 4 users
CATALOG_PER_GROUP = 100       # 5 groups x 100 = 500-song rating catalog
RATINGS_PER_USER = 80         # each user rates 80 of the 500 catalog songs
PREFERRED_FRACTION = 0.6      # share of a user's ratings from their own taste group

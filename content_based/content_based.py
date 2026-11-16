import pandas as pd
import numpy as np

FEATURES = ["danceability", "energy", "tempo", "valence",
            "acousticness", "instrumentalness", "loudness", "speechiness"]


def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Unnamed: 0"])
    df = df.dropna()
    df = df.drop_duplicates(subset="track_id")
    df = df.drop_duplicates(subset=["track_name", "artists"]).reset_index(drop=True)
    return df

def build_feature_matrix(df):
    X = df[FEATURES].to_numpy()
    X_norm = (X - X.mean(axis=0)) / X.std(axis=0)
    return X_norm


def check_rank(X_norm):
    return np.linalg.matrix_rank(X_norm)


def build_similarity_basis(X_norm):
    row_norms = np.linalg.norm(X_norm, axis=1, keepdims=True)
    return X_norm / row_norms


def recommend_similar(df, X_unit, song_name, k=5):
    matches = df.index[df["track_name"] == song_name]
    if len(matches) == 0:
        raise ValueError(f"'{song_name}' not found in dataset")
    idx = matches[0]
    query_vec = X_unit[idx]
    scores = X_unit @ query_vec
    top_k = np.argsort(scores)[::-1][1:k + 1]
    return df.iloc[top_k][["track_name", "artists"]]

def recommend_similar_euclidean(df, X_norm, song_name, k=5):
    matches = df.index[df["track_name"] == song_name]
    if len(matches) == 0:
        raise ValueError(f"'{song_name}' not found in dataset")
    idx = matches[0]
    query_vec = X_norm[idx]
    distances = np.linalg.norm(X_norm - query_vec, axis=1)
    top_k = np.argsort(distances)[1:k + 1]
    return df.iloc[top_k][["track_name", "artists"]]


if __name__ == "__main__":
    df = load_and_clean("../data/dataset.csv")
    X_norm = build_feature_matrix(df)
    print("rank:", check_rank(X_norm), "/", X_norm.shape[1])
    X_unit = build_similarity_basis(X_norm)
    print(recommend_similar(df, X_unit, "Comedy", k=5))
    print(recommend_similar_euclidean(df, X_norm, "Comedy", k=5))


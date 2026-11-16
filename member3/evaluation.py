"""Member 3 - Part C: evaluation.
 1. Truncated-SVD reconstruction error (RMSE) vs rank k  -> choose k
 2. Spot-check: 3 query songs, top-5 cosine neighbours, do they look reasonable?
Cosine similarity is implemented by hand (dot / norms), same as Member 1's branch.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_utils import load_features, synthetic_ratings


# ---------- SVD branch (minimal version of Member 2's method, for evaluation) ----------
def fill_user_mean(R):
    mu = np.nanmean(R, axis=1, keepdims=True)
    mu = np.where(np.isnan(mu), np.nanmean(R), mu)
    return np.where(np.isnan(R), mu, R), mu


def truncated_svd(R_filled, k):
    if k == 0:
        return R_filled                                   # baseline: user-mean prediction
    U, s, Vt = np.linalg.svd(R_filled, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]          # R_k = U_k S_k V_k^T


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def rank_sweep(R, ks, holdout_frac=0.2, seed=0):
    rng = np.random.default_rng(seed)
    obs = np.argwhere(~np.isnan(R))
    test_idx = obs[rng.random(len(obs)) < holdout_frac]
    R_train = R.copy()
    R_train[test_idx[:, 0], test_idx[:, 1]] = np.nan
    train_mask = ~np.isnan(R_train)
    filled, _ = fill_user_mean(R_train)
    out = []
    for k in ks:
        Rk = truncated_svd(filled, k)
        tr = rmse(Rk[train_mask], R[train_mask])
        te = rmse(Rk[test_idx[:, 0], test_idx[:, 1]], R[test_idx[:, 0], test_idx[:, 1]])
        out.append((k, tr, te))
    return np.array(out), len(test_idx)


def euclid_top_k(Xz, q, k=5):
    d = np.linalg.norm(Xz - Xz[q], axis=1); d[q] = np.inf
    return np.argsort(d)[:k]


# ---------- content-based spot check ----------
def cosine_sim_to_all(Xz, q):
    return (Xz @ Xz[q]) / (np.linalg.norm(Xz, axis=1) * np.linalg.norm(Xz[q]) + 1e-12)


def top_k_neighbours(Xz, q, k=5):
    s = cosine_sim_to_all(Xz, q); s[q] = -np.inf
    idx = np.argsort(s)[::-1][:k]
    return idx, cosine_sim_to_all(Xz, q)[idx]


SPOT_SONGS = ["Shape of You", "Bohemian Rhapsody - Remastered 2011", "Enter Sandman"]


def run(ratings=None, out="outputs", n_rating_songs=3000):
    """ratings: Member 2's user x song matrix (np.nan = unrated). If None, a
    synthetic stand-in is built on a random subsample of songs."""
    df, Xz = load_features()
    lines = []
    P = lambda *a: (print(*a), lines.append(" ".join(str(x) for x in a)))

    # 1. rank sweep
    if ratings is None:
        rng = np.random.default_rng(0)
        sub = rng.choice(len(df), n_rating_songs, replace=False)
        R = synthetic_ratings(df["track_genre"].to_numpy()[sub])
        P("[NOTE] using SYNTHETIC ratings (stand-in for Member 2's matrix)")
    else:
        R = ratings
    ks = list(range(0, 21))
    res, n_test = rank_sweep(R, ks)
    best = res[1:, 2].min()
    k_star = int(res[1:][np.argmax(res[1:, 2] <= best * 1.01), 0])
    P(f"Ratings matrix {R.shape}, observed {np.sum(~np.isnan(R))}, held-out entries {n_test}")
    P(f"{'k':>3} {'train RMSE':>11} {'test RMSE':>10}")
    for k, tr, te in res:
        P(f"{int(k):>3} {tr:>11.4f} {te:>10.4f}{'   <-- chosen k' if int(k)==k_star else ''}")
    P(f"\nChosen rank k = {k_star} (smallest k within 1% of best held-out RMSE {best:.4f})")
    # baseline: predicting each user's mean
    filled, mu = fill_user_mean(R)
    P(f"Baseline (k=0, predict each user's mean): held-out RMSE {res[0,2]:.4f}; at k={k_star}: {res[k_star,2]:.4f} ({1-res[k_star,2]/res[0,2]:.1%} lower)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(res[:, 0], res[:, 1], "o-", label="train RMSE", color="#4aa89a")
    ax.plot(res[:, 0], res[:, 2], "s-", label="held-out RMSE", color="#d9825b")
    ax.axvline(k_star, ls="--", color="gray"); ax.text(k_star + .3, ax.get_ylim()[1]*.95, f"k={k_star}")
    ax.set_xlabel("rank k"); ax.set_ylabel("RMSE (rating points)")
    ax.set_title("Reconstruction error vs. rank"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{out}/rmse_vs_rank.png", dpi=150); plt.close(fig)

    s = np.linalg.svd(filled, compute_uv=False)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(range(1, 21), s[:20], "o-", color="#4aa89a"); ax.set_yscale("log")
    ax.set_xlabel("index"); ax.set_ylabel("singular value (log)")
    ax.set_title("Singular value spectrum of ratings matrix"); fig.tight_layout()
    fig.savefig(f"{out}/singular_values.png", dpi=150); plt.close(fig)

    # 2. spot-check on real songs
    P("\n=== Spot-check: top-5 cosine neighbours for 3 query songs ===")
    rows = []
    for name in SPOT_SONGS:
        q = int(df.index[df["track_name"] == name][0])
        idx, sims = top_k_neighbours(Xz, q, 5)
        P(f"\nQuery: {name} - {df['artists'][q]}  [{df['track_genre'][q]}]")
        for j, sm in zip(idx, sims):
            P(f"   {df['track_name'][j][:38]:<38} {df['artists'][j][:26]:<26} [{df['track_genre'][j]:<12}] cos={sm:.3f}")
            rows.append((name, df['track_name'][j], df['artists'][j], df['track_genre'][j], round(float(sm), 3)))

    # 3. quantitative sanity check on 2000 random songs
    rng = np.random.default_rng(1)
    qs = rng.choice(len(df), 2000, replace=False)
    genres = df["track_genre"].to_numpy()
    prec = np.mean([np.mean(genres[top_k_neighbours(Xz, q, 5)[0]] == genres[q]) for q in qs])
    overlap = np.mean([len(set(top_k_neighbours(Xz, q, 5)[0]) & set(euclid_top_k(Xz, q, 5))) / 5 for q in qs])
    P(f"\nGenre precision@5 (2000 random queries): {prec:.3f}   random baseline ~ {1/len(np.unique(genres)):.3f}")
    P(f"Cosine vs Euclidean top-5 overlap: {overlap:.2f}")

    open(f"{out}/evaluation_results.txt", "w").write("\n".join(lines))
    import csv
    with open(f"{out}/spot_check.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["query", "neighbour", "artists", "genre", "cosine"]); w.writerows(rows)
    return k_star


if __name__ == "__main__":
    run()

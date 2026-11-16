"""Member 3 - Part A: covariance -> eigendecomposition -> PCA projection."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from data_utils import load_features, FEATURES

SHOW_GENRES = ["classical", "hip-hop", "heavy-metal", "edm", "country", "ambient"]


def covariance_matrix(Xz):
    """C = Xc^T Xc / (n-1), written by hand."""
    Xc = Xz - Xz.mean(axis=0)
    return Xc.T @ Xc / (Xc.shape[0] - 1)


def pca_eig(Xz, n_components=3):
    C = covariance_matrix(Xz)
    assert np.allclose(C, np.cov(Xz, rowvar=False))
    vals, vecs = np.linalg.eig(C)
    vals, vecs = vals.real, vecs.real
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    W = vecs[:, :n_components]
    Z = (Xz - Xz.mean(axis=0)) @ W
    return C, vals, vecs, W, Z


def run(out="outputs"):
    df, Xz = load_features()
    n = len(df)
    C, vals, vecs, W, Z = pca_eig(Xz, 3)
    evr = vals / vals.sum()
    log = []
    P = lambda *a: (print(*a), log.append(" ".join(map(str, a))))
    P(f"Songs after cleaning: {n}, features: {Xz.shape[1]}")
    P("Rank of normalised feature matrix:", np.linalg.matrix_rank(Xz), "/", Xz.shape[1])
    P("Eigenvalues:", np.round(vals, 3))
    P("Explained variance ratio:", np.round(evr, 3))
    P("Cumulative top-2 / top-3:", round(evr[:2].sum(), 3), "/", round(evr[:3].sum(), 3))
    # cross-check: eigenvalues of C == s^2/(n-1) from the SVD of centred X
    s = np.linalg.svd(Xz - Xz.mean(0), compute_uv=False)
    P("Matches SVD singular values^2/(n-1):", np.allclose(np.sort(vals), np.sort(s**2 / (n - 1))))
    P("\nLoadings (rows=features):")
    for f, row in zip(FEATURES, W):
        P(f"  {f:<17}" + "  ".join(f"{v:+.3f}" for v in row))

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    x = range(1, len(vals) + 1)
    ax.bar(x, evr, color="#4aa89a", label="per component")
    ax.plot(x, evr.cumsum(), "o-", color="#d9825b", label="cumulative")
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio")
    ax.set_title("Scree plot"); ax.legend(); fig.tight_layout()
    fig.savefig(f"{out}/pca_scree.png", dpi=150); plt.close(fig)

    genres = df["track_genre"].to_numpy()
    rng = np.random.default_rng(0)
    sel = {g: rng.choice(np.where(genres == g)[0], min(600, (genres == g).sum()), replace=False)
           for g in SHOW_GENRES}
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(Z[:, 0][::40], Z[:, 1][::40], s=4, c="lightgray", label="all (1 in 40)")
    for g, idx in sel.items():
        ax.scatter(Z[idx, 0], Z[idx, 1], s=9, alpha=.7, label=g)
    ax.set_xlabel(f"PC1 ({evr[0]:.0%} var)"); ax.set_ylabel(f"PC2 ({evr[1]:.0%} var)")
    ax.set_title("Songs on top-2 principal components"); ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/pca_clusters_2d.png", dpi=150); plt.close(fig)

    fig = plt.figure(figsize=(7.5, 6)); ax = fig.add_subplot(111, projection="3d")
    for g, idx in sel.items():
        ax.scatter(Z[idx, 0], Z[idx, 1], Z[idx, 2], s=6, alpha=.6, label=g)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("PC3"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{out}/pca_clusters_3d.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(W, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_yticks(range(len(FEATURES))); ax.set_yticklabels(FEATURES)
    ax.set_xticks(range(3)); ax.set_xticklabels(["PC1", "PC2", "PC3"])
    for i in range(W.shape[0]):
        for j in range(3): ax.text(j, i, f"{W[i,j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im); ax.set_title("Feature loadings"); fig.tight_layout()
    fig.savefig(f"{out}/pca_loadings.png", dpi=150); plt.close(fig)
    open(f"{out}/pca_results.txt", "w").write("\n".join(log))
    return Z, vals


if __name__ == "__main__":
    run()

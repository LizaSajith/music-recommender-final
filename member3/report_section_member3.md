# Member 3 — Dimensionality Reduction & Evaluation

Data: Spotify audio-features dataset, 81,343 tracks after Member 1's cleaning; 8 z-scored features. The normalised matrix has full column rank (8/8), so no feature needed to be dropped.

## 1. PCA via eigendecomposition (eigenvalues / eigenvectors)
1. Covariance matrix C = XcᵀXc/(n−1) (by hand; verified against `np.cov`).
2. `numpy.linalg.eig(C)`, eigenvalues sorted descending. C is symmetric so the eigenvectors are orthogonal.
3. Projection Z = Xc·W using the top-k eigenvectors.
4. Cross-check: the eigenvalues equal σᵢ²/(n−1) from the SVD of Xc — linking PCA to the SVD used in collaborative filtering.

**Results.** Eigenvalues: 2.87, 1.42, 1.00, 0.89, 0.85, 0.50, 0.33, 0.14. Explained variance: PC1 35.9%, PC2 17.8%, PC3 12.5% (top-2 = 53.7%, top-3 = 66.2%).
- **PC1** (energy +0.50, loudness +0.52 vs acousticness −0.44): a quiet/acoustic ↔ loud/energetic axis. Classical and ambient sit at the far left, metal/EDM/hip-hop at the right.
- **PC2** (danceability +0.53, valence +0.52 vs instrumentalness −0.35): a "danceable/happy" axis; hip-hop high, heavy-metal low.
- **PC3** is almost entirely speechiness (−0.94).
Genres separate along PC1/PC2 but overlap heavily in the loud/energetic region — audio features alone only partly identify genre, so only ~2/3 of variance survives in 3 dimensions.

## 2. Worked example: RREF (rank / linear dependence)
Toy 4×3 matrix with loudness = danceability + 2·energy. `sympy.rref()` finds pivots in columns 1, 2 only → rank 2 < 3; null-space vector (−1, −2, 1) is the exact dependency; `numpy.linalg.matrix_rank` agrees. A rank-deficient A makes AᵀA singular, so Ax = b has no unique solution — hence the rank check before similarity. On the real data the check returns 8/8, so the features are independent.

## 3. Evaluation
**Reconstruction error (SVD branch).** 20% of observed ratings held out, missing entries filled with user means, R_k = U_kΣ_kV_kᵀ, RMSE for k = 0…20 (k = 0 = user-mean baseline).
- Train RMSE decreases monotonically with k (1.36 at k=1 down to 0 at k=20, where the matrix is reproduced exactly); held-out RMSE reaches its minimum at k = 5 and then rises (overfitting).
- **Chosen k = 5** (smallest k within 1% of best held-out RMSE). Held-out RMSE 1.361 vs 1.442 for the user-mean baseline, i.e. about 5.6% lower. The gain is modest: the ratings matrix is small (20 users x 493 songs, 16% observed, 326 held-out entries) and synthetic, so the figures show the method working rather than real-world accuracy.
- Data: Member 2's synthetic matrix from `generate_interactions.py` (500-song catalog, 80 ratings per user, about 3 raters per song).

**Spot-check (3 query songs, top-5 cosine neighbours).** Shape of You, Bohemian Rhapsody, Enter Sandman. Neighbours match the query's audio profile (cos 0.97–0.99) but often not its genre or style — e.g. Shape of You returns dancehall, salsa and kids' songs; Enter Sandman returns Ozzy Osbourne (hard-rock), Stone Temple Pilots (grunge) and Visions of Atlantis (goth), which are reasonable, alongside a few oddities.
**Quantitative check (2,000 random queries).** Genre precision@5 = 13.6% vs ≈0.9% for random choice over 114 genres (~15× lift). Cosine and Euclidean top-5 lists overlap 63%.

**Takeaway.** Audio features alone give plausible "sounds similar" results, not "same style" results; this motivates the hybrid merge with collaborative filtering.

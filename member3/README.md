# Member 3 — Dimensionality Reduction & Evaluation
Setup:  pip install numpy pandas matplotlib sympy
Run (from inside the member3/ folder):
        python run_all.py ../data/ratings_matrix.csv   (Member 2's ratings; made by generate_interactions.py)
        python run_all.py                              (synthetic stand-in ratings)

Files
- data_utils.py      loads data via Member 1's code; synthetic ratings stand-in
- pca_projection.py  covariance -> numpy.linalg.eig -> projection, plots
- rref_example.py    sympy.rref() toy example
- evaluation.py      SVD RMSE-vs-k sweep, chosen k, spot-check, precision@5
- report_section_member3.md   text for report/slides
- outputs/           plots + result text files

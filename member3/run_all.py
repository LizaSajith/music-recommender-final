"""Run all of Member 3's work:  python run_all.py
Dataset expected at data/dataset.csv. Member 2's ratings (optional):
  python run_all.py ratings.csv   (users x songs, blanks = unrated; must align with cleaned song order)"""
import sys, os, subprocess
import numpy as np, pandas as pd
os.makedirs("outputs", exist_ok=True)
import pca_projection, evaluation
print("=== PCA ==="); pca_projection.run()
print("\n=== RREF ==="); subprocess.run([sys.executable, "rref_example.py"], check=True)
print("\n=== EVALUATION ===")
R = pd.read_csv(sys.argv[1], index_col=0).to_numpy(float) if len(sys.argv) > 1 else None
evaluation.run(R)

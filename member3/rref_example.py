"""Member 3 - Part B: small worked RREF example (toy matrix, NOT the real pipeline).
Shows how rank/dependency is read off by hand, matching numpy.linalg.matrix_rank."""
import numpy as np
import sympy as sp

# 4 songs x 3 features (danceability, energy, loudness); loudness = danceability + 2*energy
A = sp.Matrix([
    [1, 2, 5],
    [2, 1, 4],
    [0, 3, 6],
    [3, 3, 9],
])

rref, pivots = A.rref()
print("A =");        sp.pprint(A)
print("\nRREF(A) =");  sp.pprint(rref)
print("\nPivot columns:", pivots, "-> rank =", len(pivots))
print("numpy.linalg.matrix_rank:", np.linalg.matrix_rank(np.array(A.tolist(), float)))

null = A.nullspace()[0]
print("\nNull-space vector (the dependency c with A c = 0):"); sp.pprint(null.T)
print("\nReading: column3 = 1*col1 + 2*col2 -> the third column is redundant,")
print("so A has rank 2 < 3 columns, A^T A is singular, and Ax=b has no unique solution.")
print("This is why the real pipeline checks matrix_rank before dropping collinear features.")

with open("outputs/rref_example_output.txt", "w") as f:
    f.write(f"A =\n{A}\n\nRREF =\n{rref}\n\nPivot columns: {pivots}, rank = {len(pivots)}\n"
            f"Null-space vector: {list(null)}\n"
            "Interpretation: col3 = col1 + 2*col2, so rank 2 < 3.\n")

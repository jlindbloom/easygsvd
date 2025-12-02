import numpy as np

from easygsvd.gsvd import gsvd


def _random_full_column_rank_pair(N, seed, max_tries=10):
    rng = np.random.default_rng(seed)
    for _ in range(max_tries):
        M = int(rng.integers(80, 131))
        K = int(rng.integers(80, 131))
        A = rng.standard_normal((M, N))
        L = rng.standard_normal((K, N))
        if np.linalg.matrix_rank(np.vstack([A, L])) == N:
            return A, L
    raise RuntimeError("Failed to generate full-column-rank stacked matrix within max_tries.")


def test_gsvd_diagonalizes_normal_equations():
    diag_tol = 1e-10
    eye_tol = 1e-12
    N = 100

    for seed in range(50):
        A, L = _random_full_column_rank_pair(N=N, seed=seed)
        G = gsvd(A, L, full_matrices=False)

        ATA = A.T @ A
        LTL = L.T @ L
        M = ATA + LTL

        mat_ATA = G.X.T @ ATA @ G.X
        mat_LTL = G.X.T @ LTL @ G.X
        mat_M = G.X.T @ M @ G.X

        # Diagonal structure
        assert np.allclose(mat_ATA, np.diag(np.diag(mat_ATA)), atol=diag_tol)
        assert np.allclose(mat_LTL, np.diag(np.diag(mat_LTL)), atol=diag_tol)

        # Expected diagonals
        assert np.allclose(np.diag(mat_ATA), G.c**2, atol=diag_tol, rtol=diag_tol)
        assert np.allclose(np.diag(mat_LTL), G.s**2, atol=diag_tol, rtol=diag_tol)

        # Sum to identity
        assert np.allclose(mat_M, np.eye(N), atol=eye_tol)

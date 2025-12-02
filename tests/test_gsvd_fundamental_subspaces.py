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


def _proj(B, v):
    # Orthogonal projection onto col(B) via pseudoinverse
    return B @ np.linalg.pinv(B) @ v


def test_fundamental_subspaces_from_gsvd():
    N = 100
    tol = 1e-10
    rng = np.random.default_rng(0)

    for seed in range(50):
        A, L = _random_full_column_rank_pair(N=N, seed=seed)
        G = gsvd(A, L, full_matrices=False)

        # Nullspaces
        if G.X3.size > 0:
            assert np.linalg.norm(A @ G.X3, ord='fro') <= tol
        if G.X1.size > 0:
            assert np.linalg.norm(L @ G.X1, ord='fro') <= tol

        # Column spaces: test random vectors projected onto col(Uhat) and col(Vhat)
        y = rng.standard_normal((N,))
        Ay = A @ y
        Ly = L @ y
        proj_Ay = _proj(G.Uhat, Ay)
        proj_Ly = _proj(G.Vhat, Ly)
        denom_A = max(1.0, np.linalg.norm(Ay))
        denom_L = max(1.0, np.linalg.norm(Ly))
        assert np.linalg.norm(Ay - proj_Ay) / denom_A <= tol
        assert np.linalg.norm(Ly - proj_Ly) / denom_L <= tol

        # Row spaces via Y blocks
        zA = rng.standard_normal((A.shape[0],))
        zL = rng.standard_normal((L.shape[0],))
        Atz = A.T @ zA
        Ltz = L.T @ zL

        YA_basis = np.hstack([G.Y1, G.Y2])
        YL_basis = np.hstack([G.Y2, G.Y3])

        proj_Atz = _proj(YA_basis, Atz)
        proj_Ltz = _proj(YL_basis, Ltz)
        denom_At = max(1.0, np.linalg.norm(Atz))
        denom_Lt = max(1.0, np.linalg.norm(Ltz))
        assert np.linalg.norm(Atz - proj_Atz) / denom_At <= tol
        assert np.linalg.norm(Ltz - proj_Ltz) / denom_Lt <= tol

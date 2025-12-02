import numpy as np

from easygsvd.gsvd import gsvd
from easygsvd.util import colspaces_equal


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


def _reconstruct_A_L(res):
    A_rec = (res.U1 @ res.Y1.T) + (res.U2 @ (np.diag(res.c_check) @ res.Y2.T))
    L_rec = (res.V2 @ (np.diag(res.s_check) @ res.Y2.T)) + (res.V3 @ res.Y3.T)
    return A_rec, L_rec


def test_incremental_append_matches_fresh_gsvd():
    N = 100
    tol = 1e-10

    c_tol = 1e-7
    for seed in range(50):
        rng = np.random.default_rng(seed)
        A, L = _random_full_column_rank_pair(N=N, seed=seed)

        # Random number of columns to append
        P = int(rng.integers(0, 26))
        if P == 0:
            A_new = None
            L_new = None
            A_aug = A
            L_aug = L
        else:
            A_new = rng.standard_normal((A.shape[0], P))
            L_new = rng.standard_normal((L.shape[0], P))
            A_aug = np.hstack([A, A_new])
            L_aug = np.hstack([L, L_new])

        # Baseline GSVD and incremental append
        G_base = gsvd(A, L, full_matrices=False)
        assert hasattr(G_base, "append_column"), "GSVDResult is missing append_column; ensure updated gsvd.py is in use."
        G_inc = G_base.append_column(A_new, L_new)

        # Fresh GSVD on augmented matrices
        G_fresh = gsvd(A_aug, L_aug, full_matrices=False)

        # Generalized singular values should match (order may differ slightly)
        assert np.allclose(np.sort(G_inc.c), np.sort(G_fresh.c), atol=c_tol, rtol=c_tol)
        assert np.allclose(np.sort(G_inc.s), np.sort(G_fresh.s), atol=1e-7, rtol=1e-7)
        assert np.allclose(np.sort(G_inc.gamma_check), np.sort(G_fresh.gamma_check), atol=1e-7, rtol=1e-7)

        # Column spaces of Uhat and Vhat should coincide
        assert colspaces_equal(G_inc.Uhat, G_fresh.Uhat, tol=1e-8)
        assert colspaces_equal(G_inc.Vhat, G_fresh.Vhat, tol=1e-8)

        # Reconstructed A and L should match between incremental and fresh GSVD
        A_inc, L_inc = _reconstruct_A_L(G_inc)
        A_fresh, L_fresh = _reconstruct_A_L(G_fresh)
        assert np.allclose(A_inc, A_fresh, atol=1e-10)
        assert np.allclose(L_inc, L_fresh, atol=1e-10)

        # And both should reproduce the augmented matrices
        assert np.allclose(A_inc, A_aug, atol=1e-10)
        assert np.allclose(L_inc, L_aug, atol=1e-10)

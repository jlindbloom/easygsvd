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


def test_gsvd_reconstructs_A_and_L():
    N = 100
    tol_full = 1e-9
    tol_econ = 1e-9

    for seed in range(50):
        A, L = _random_full_column_rank_pair(N=N, seed=seed)

        # Full decomposition
        G_full = gsvd(A, L, full_matrices=True)
        Y12 = np.hstack([G_full.Y1, G_full.Y2])  # shape N x r_A
        Y23 = np.hstack([G_full.Y2, G_full.Y3])  # shape N x r_L

        A_recon_full = G_full.Uhat @ (np.diag(G_full.c_hat) @ Y12.T)
        L_recon_full = G_full.Vhat @ (np.diag(G_full.s_hat) @ Y23.T)

        assert np.allclose(A_recon_full, A, atol=tol_full)
        assert np.allclose(L_recon_full, L, atol=tol_full)

        # Economic form reconstruction
        G = gsvd(A, L, full_matrices=False)
        A_recon = (G.U1 @ G.Y1.T) + (G.U2 @ (np.diag(G.c_check) @ G.Y2.T))
        L_recon = (G.V2 @ (np.diag(G.s_check) @ G.Y2.T)) + (G.V3 @ G.Y3.T)

        assert np.allclose(A_recon, A, atol=tol_econ)
        assert np.allclose(L_recon, L, atol=tol_econ)

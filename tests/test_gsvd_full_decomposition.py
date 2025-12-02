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


def test_full_decomposition_block_forms():
    N = 100
    tol = 1e-8

    for seed in range(20):
        A, L = _random_full_column_rank_pair(N=N, seed=seed)
        G = gsvd(A, L, full_matrices=True)

        M, K = A.shape[0], L.shape[0]
        r_A = G.r_A
        r_L = G.r_L
        n_A = G.n_A
        n_L = G.n_L

        # U^T A X block structure
        UTAX = G.U.T @ (A @ G.X)
        expected_A = np.zeros((M, N))
        expected_A[:r_A, :r_A] = np.diag(G.c_hat)
        assert np.allclose(UTAX, expected_A, atol=tol)

        # V^T L X block structure
        VTLX = G.V.T @ (L @ G.X)
        expected_L = np.zeros((K, N))
        expected_L[:r_L, n_L:n_L + r_L] = np.diag(G.s_hat)
        assert np.allclose(VTLX, expected_L, atol=tol)

        # Reconstruct A and L via full forms
        Xinv = np.linalg.inv(G.X)
        A_recon = G.U @ expected_A @ Xinv
        L_recon = G.V @ expected_L @ Xinv
        assert np.allclose(A_recon, A, atol=tol)
        assert np.allclose(L_recon, L, atol=tol)


def test_full_decomposition_after_incremental_append():
    N = 100
    tol = 1e-7

    for seed in range(10):
        rng = np.random.default_rng(seed)
        A, L = _random_full_column_rank_pair(N=N, seed=seed)

        # Append random number of columns
        P = int(rng.integers(1, 6))
        A_new = rng.standard_normal((A.shape[0], P))
        L_new = rng.standard_normal((L.shape[0], P))
        A_aug = np.hstack([A, A_new])
        L_aug = np.hstack([L, L_new])

        G_base = gsvd(A, L, full_matrices=True)
        G_inc = G_base.append_column(A_new, L_new)
        G_fresh = gsvd(A_aug, L_aug, full_matrices=True)

        # Compare full block structures for the incremental result
        def check_full_blocks(G, A_mat, L_mat):
            M, K = A_mat.shape[0], L_mat.shape[0]
            r_A = G.r_A
            r_L = G.r_L
            n_L = G.n_L

            UTAX = G.U.T @ (A_mat @ G.X)
            expected_A = np.zeros((M, G.X.shape[0]))
            expected_A[:r_A, :r_A] = np.diag(G.c_hat)

            VTLX = G.V.T @ (L_mat @ G.X)
            expected_L = np.zeros((K, G.X.shape[0]))
            expected_L[:r_L, n_L:n_L + r_L] = np.diag(G.s_hat)

            assert np.allclose(UTAX, expected_A, atol=tol)
            assert np.allclose(VTLX, expected_L, atol=tol)

            Xinv = np.linalg.inv(G.X)
            A_rec = G.U @ expected_A @ Xinv
            L_rec = G.V @ expected_L @ Xinv
            assert np.allclose(A_rec, A_mat, atol=tol)
            assert np.allclose(L_rec, L_mat, atol=tol)

        check_full_blocks(G_inc, A_aug, L_aug)
        check_full_blocks(G_fresh, A_aug, L_aug)

import numpy as np

from easygsvd.gsvd import gsvd


def _rank_deficient_matrix(m, n, rank, rng):
    """Construct an m x n matrix with exact rank via SVD factors."""
    U, _ = np.linalg.qr(rng.standard_normal((m, rank)))
    V, _ = np.linalg.qr(rng.standard_normal((n, rank)))
    s = rng.uniform(0.5, 2.0, size=rank)
    return U @ (s[:, None] * V.T)


def test_extreme_cases_rank_profiles():
    rng = np.random.default_rng(0)
    tol = 1e-8
    N = 100

    # Case 1: A full rank, L rank-deficient by 1 (n_A = 0, n_L = 1)
    A_full = _rank_deficient_matrix(m=N + 5, n=N, rank=N, rng=rng)
    L_rank_def = _rank_deficient_matrix(m=N + 3, n=N, rank=N - 1, rng=rng)
    G1 = gsvd(A_full, L_rank_def, full_matrices=True)

    assert G1.n_A == 0
    assert G1.n_L == 1
    # Reconstruct and compare
    A1_rec = (G1.U1 @ G1.Y1.T) + (G1.U2 @ (np.diag(G1.c_check) @ G1.Y2.T))
    L1_rec = (G1.V2 @ (np.diag(G1.s_check) @ G1.Y2.T)) + (G1.V3 @ G1.Y3.T)
    assert np.allclose(A1_rec, A_full, atol=tol)
    assert np.allclose(L1_rec, L_rank_def, atol=tol)
    # gamma_check finite and positive
    assert np.all(np.isfinite(G1.gamma_check))
    assert np.all(G1.gamma_check > 0)

    # Case 2: L full rank, A rank-deficient by 1 (n_L = 0, n_A = 1)
    A_rank_def = _rank_deficient_matrix(m=N + 4, n=N, rank=N - 1, rng=rng)
    L_full = _rank_deficient_matrix(m=N + 2, n=N, rank=N, rng=rng)
    G2 = gsvd(A_rank_def, L_full, full_matrices=True)

    assert G2.n_A == 1
    assert G2.n_L == 0
    A2_rec = (G2.U1 @ G2.Y1.T) + (G2.U2 @ (np.diag(G2.c_check) @ G2.Y2.T))
    L2_rec = (G2.V2 @ (np.diag(G2.s_check) @ G2.Y2.T)) + (G2.V3 @ G2.Y3.T)
    assert np.allclose(A2_rec, A_rank_def, atol=tol)
    assert np.allclose(L2_rec, L_full, atol=tol)
    assert np.all(np.isfinite(G2.gamma_check))

    # Full decomposition consistency for both cases
    def check_full_blocks(G, A_mat, L_mat, tol_full=1e-7):
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

        assert np.allclose(UTAX, expected_A, atol=tol_full)
        assert np.allclose(VTLX, expected_L, atol=tol_full)

        Xinv = np.linalg.inv(G.X)
        A_rec_full = G.U @ expected_A @ Xinv
        L_rec_full = G.V @ expected_L @ Xinv
        assert np.allclose(A_rec_full, A_mat, atol=tol_full)
        assert np.allclose(L_rec_full, L_mat, atol=tol_full)

    check_full_blocks(G1, A_full, L_rank_def)
    check_full_blocks(G2, A_rank_def, L_full)

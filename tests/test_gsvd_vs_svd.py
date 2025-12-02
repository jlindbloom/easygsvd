import numpy as np

from easygsvd.gsvd import gsvd
from easygsvd.util import colspaces_equal


def _make_rank_deficient_pair(N=100, seed=0):
    """Return A (M x N) and L (K x N) with rank N-1 each, but ker(A) ∩ ker(L) = {0}."""
    rng = np.random.default_rng(seed)

    # Right-side orthonormal basis to define complementary kernels.
    Q_right, _ = np.linalg.qr(rng.standard_normal((N, N)))
    kerA = Q_right[:, :1]          # one-dimensional kernel for A
    kerL = Q_right[:, 1:2]         # one-dimensional kernel for L (independent of kerA)
    rest = Q_right[:, 2:]          # shared subspace so stacked matrix has full column rank

    V_A = np.hstack([kerL, rest])  # columns span kerA^⊥
    V_L = np.hstack([kerA, rest])  # columns span kerL^⊥

    r = N - 1
    # Sample M and K around 100; allow below N but ensure >= r for QR.
    M = int(rng.integers(80, 131))
    K = int(rng.integers(80, 131))
    M = max(M, r)  # guarantee enough rows
    K = max(K, r)
    U_A, _ = np.linalg.qr(rng.standard_normal((M, r)))
    U_L, _ = np.linalg.qr(rng.standard_normal((K, r)))
    s_A = rng.uniform(0.5, 2.0, size=r)
    s_L = rng.uniform(0.5, 2.0, size=r)

    A = U_A @ (s_A[:, None] * V_A.T)  # M x N, rank r
    L = U_L @ (s_L[:, None] * V_L.T)  # K x N, rank r
    return A, L


def test_gsvd_matches_svd_of_a_l_oblique_inverse():
    # Run multiple random trials to reduce the chance of a fluke pass.
    for seed in range(50):
        A, L = _make_rank_deficient_pair(N=100, seed=seed)

        result = gsvd(A, L)
        L_oblique_pinv = result.get_L_oblique_pinv(matrix=True)
        AL_product = A @ L_oblique_pinv

        # SVD of A @ L_A^\dagger
        U, svals, Vt = np.linalg.svd(AL_product, full_matrices=False)

        # GSVD predicts AL_product = U2 diag(gamma_check) V2^T
        reconstructed = result.U2 @ (np.diag(result.gamma_check) @ result.V2.T)

        # Leading singular values should match gamma_check; any extras should be ~0.
        g = result.gamma_check
        assert np.allclose(AL_product, reconstructed, atol=1e-12)
        assert np.allclose(svals[: g.size], g, rtol=1e-11, atol=1e-12)
        if svals.size > g.size:
            assert np.allclose(svals[g.size :], 0.0, atol=1e-12)

        assert colspaces_equal(U[:, : g.size], result.U2, tol=1e-10)
        assert colspaces_equal(Vt.T[:, : g.size], result.V2, tol=1e-10)

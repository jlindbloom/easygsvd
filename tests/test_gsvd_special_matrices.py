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


def test_diagonal_pair_recovers_known_c_s_and_reconstruction():
    rng = np.random.default_rng(0)
    N = 100
    c_true = rng.uniform(0.2, 1.0, size=N)
    s_true = np.sqrt(1.0 - c_true**2)

    A = np.diag(c_true)
    L = np.diag(s_true)

    G = gsvd(A, L, full_matrices=False)

    c_sorted = np.sort(c_true)[::-1]
    s_sorted = np.sort(s_true)[::-1]
    assert np.allclose(np.sort(G.c)[::-1], c_sorted, atol=1e-12, rtol=0)
    assert np.allclose(np.sort(G.s)[::-1], s_sorted, atol=1e-12, rtol=0)

    # Economic reconstruction should be exact
    A_recon = (G.U1 @ G.Y1.T) + (G.U2 @ (np.diag(G.c_check) @ G.Y2.T))
    L_recon = (G.V2 @ (np.diag(G.s_check) @ G.Y2.T)) + (G.V3 @ G.Y3.T)
    assert np.allclose(A_recon, A, atol=1e-12)
    assert np.allclose(L_recon, L, atol=1e-12)


def test_scaling_relation_for_gamma():
    rng = np.random.default_rng(1)
    N = 100

    for seed in range(20):
        A, L = _random_full_column_rank_pair(N=N, seed=seed)
        alpha = rng.uniform(0.5, 2.0)
        beta = rng.uniform(0.5, 2.0)

        G_orig = gsvd(A, L, full_matrices=False)
        G_scaled = gsvd(alpha * A, beta * L, full_matrices=False)

        g_orig = np.sort(G_orig.gamma_check)[::-1]
        g_scaled = np.sort(G_scaled.gamma_check)[::-1]
        expected = np.abs(alpha / beta) * g_orig
        assert np.allclose(g_scaled, expected, rtol=1e-10, atol=1e-10)


def test_orthogonal_pre_post_transforms_preserve_gsvd_invariants():
    N = 100
    tol = 1e-7

    for seed in range(20):
        rng = np.random.default_rng(seed)
        A, L = _random_full_column_rank_pair(N=N, seed=seed)

        QA, _ = np.linalg.qr(rng.standard_normal((A.shape[0], A.shape[0])))
        QL, _ = np.linalg.qr(rng.standard_normal((L.shape[0], L.shape[0])))

        G0 = gsvd(A, L, full_matrices=False)
        G1 = gsvd(QA @ A, QL @ L, full_matrices=False)

        # Gamma invariance
        g0 = np.sort(G0.gamma_check)[::-1]
        g1 = np.sort(G1.gamma_check)[::-1]
        assert np.allclose(g0, g1, atol=tol, rtol=tol)

        # X subspace invariance
        assert colspaces_equal(G0.X, G1.X, tol=1e-8)

        # c and s invariance (sorted)
        assert np.allclose(np.sort(G0.c)[::-1], np.sort(G1.c)[::-1], atol=tol, rtol=tol)
        assert np.allclose(np.sort(G0.s)[::-1], np.sort(G1.s)[::-1], atol=tol, rtol=tol)

        # Uhat and Vhat rotate by QA/QL
        assert colspaces_equal(QA @ G0.Uhat, G1.Uhat, tol=1e-8)
        assert colspaces_equal(QL @ G0.Vhat, G1.Vhat, tol=1e-8)
